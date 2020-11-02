import itertools
from collections import Counter, defaultdict
import numpy as np

from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import Task, Struct
from lsst.afw.geom import SpanSet
from lsst.afw.display import Display
from lsst.afw.math import GaussianFunction1D, SeparableKernel, convolve, ConvolutionControl

from pfs.datamodel import FiberStatus
from pfs.drp.stella.traces import findTracePeaks, centroidTrace, TracePeak
from pfs.drp.stella.fitPolynomial import FitPolynomialTask
from pfs.drp.stella.fiberProfile import FiberProfile
from pfs.drp.stella.fiberProfileSet import FiberProfileSet

import lsstDebug

__all__ = ("BuildFiberProfilesConfig", "BuildFiberProfilesTask")

backend = "ds9"
colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "orange"]


class BuildFiberProfilesConfig(Config):
    """Configuration for BuildFiberProfilesTask"""
    mask = ListField(dtype=str, default=["CR", "BAD", "NO_DATA"], doc="Mask planes to ignore")
    doBlindFind = Field(dtype=bool, default=True, doc="Find traces without using DetectorMap?")
    columnFwhm = Field(dtype=float, default=2.0, doc="Typical FWHM across columns (spatial dimension)")
    rowFwhm = Field(dtype=float, default=3.0, doc="Typical FWHM across rows (spectral dimension)")
    convolveGrowMask = Field(dtype=int, default=1, doc="Number of pixels to grow mask for convolution")
    kernelSize = Field(dtype=float, default=4.0, doc="Size of convolution kernel, relative to sigma")
    findThreshold = Field(dtype=float, default=150.0, doc="Threshold value for finding traces")
    pruneMaxWidth = Field(dtype=int, default=25, doc="Maximum width of peak span to avoid pruning")
    associationDepth = Field(dtype=int, default=10, doc="Depth of trace association (rows)")
    pruneMinLength = Field(dtype=int, default=1000, doc="Minimum length of trace to avoid pruning")
    pruneMinFrac = Field(dtype=float, default=0.7, doc="Minimum detection fraction of trace to avoid pruning")
    centroidRadius = Field(dtype=int, default=5, doc="Radius about the peak for centroiding")
    centerFit = ConfigurableField(target=FitPolynomialTask, doc="Fit polynomial to trace centroids")
    profileSwath = Field(dtype=float, default=300, doc="Length of swaths to use for calculating profile")
    profileRadius = Field(dtype=int, default=5, doc="Radius about the peak for profile")
    profileOversample = Field(dtype=int, default=10, doc="Oversample factor for profile")
    profileRejIter = Field(dtype=int, default=1, doc="Rejection iterations for profile")
    profileRejThresh = Field(dtype=float, default=3.0, doc="Rejection threshold (sigma) for profile")


class BuildFiberProfilesTask(Task):
    """Build a FiberTraceSet from an image"""
    ConfigClass = BuildFiberProfilesConfig
    _DefaultName = "BuildFiberProfiles"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("centerFit")
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, exposure, detectorMap=None, pfsConfig=None):
        """Build a FiberProfileSet from an image

        We find traces on the image, centroid those traces, measure the fiber
        profile, and construct the FiberTraces.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure from which to build FiberTraces.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Mapping from x,y to fiberId,row.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end fiber configuration.

        Returns
        -------
        profiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of each fiber.
        centers : `dict` mapping `int` to callable
            Callable for each fiber (indexed by fiberId) that provides the
            center of the trace as a function of row.
        """
        if self.config.doBlindFind or detectorMap is None or pfsConfig is None:
            convolved = self.convolveImage(exposure.maskedImage)
            traces = self.findPeaks(convolved)
            self.log.debug("Found %d peaks", sum([len(pp) for pp in traces]))
            traces = self.prunePeaks(traces)
            self.log.debug("Pruned to %d peaks", sum([len(pp) for pp in traces]))
            traces = self.associatePeaks(traces, convolved.getWidth())
            self.log.debug("Associated into %d traces", len(traces))
            traces = self.pruneTraces(traces, exposure.getHeight())
            self.log.debug("Pruned to %d traces", len(traces))
        else:
            traces = self.generateTraces(detectorMap, pfsConfig)
        profiles = FiberProfileSet.makeEmpty(exposure.getInfo().getVisitInfo(), exposure.getMetadata())
        centers = {}
        for ii, tt in enumerate(traces):
            self.centroidTrace(exposure.maskedImage, tt)
            fit = self.fitTraceCenters(tt, exposure.getHeight())
            # Not using the real fiberId here; we'll sort that out in the identifyFibers method
            profiles[ii] = self.calculateProfile(exposure.maskedImage, fit.func)
            centers[ii] = fit.func

        if detectorMap is not None:
            identifications = self.identifyFibers(profiles, centers, detectorMap, pfsConfig)
            profiles = identifications.profiles
            centers = identifications.centers

        self.log.info("Profiled %d fibers", len(profiles))
        return Struct(profiles=profiles, centers=centers)

    def convolveImage(self, maskedImage):
        """Convolve image by Gaussian kernels in x and y

        The convolution kernel size can be specified separately for the columns
        and rows in the config.

        The boundary is unconvolved, and is set to ``NaN``.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image to convolve.

        Returns
        -------
        convolved : `lsst.afw.image.MaskedImage`
            Convolved image.
        """
        def fwhmToSigma(fwhm):
            """Convert FWHM to sigma for a Gaussian"""
            return fwhm/(2*np.sqrt(2*np.log(2)))

        def sigmaToSize(sigma):
            """Determine kernel size from Gaussian sigma"""
            return 2*int(self.config.kernelSize*sigma) + 1

        xSigma = fwhmToSigma(self.config.columnFwhm)
        ySigma = fwhmToSigma(self.config.rowFwhm)

        kernel = SeparableKernel(sigmaToSize(xSigma), sigmaToSize(ySigma),
                                 GaussianFunction1D(xSigma), GaussianFunction1D(ySigma))

        convolvedImage = maskedImage.Factory(maskedImage.getBBox())
        convolve(convolvedImage, maskedImage, kernel, ConvolutionControl())

        # Redo the convolution of the mask plane, using a smaller kernel
        mask = convolvedImage.mask
        mask.array[:] = maskedImage.mask.array
        grow = self.config.convolveGrowMask
        for name in convolvedImage.mask.getMaskPlaneDict():
            bitmask = convolvedImage.mask.getPlaneBitMask(name)
            SpanSet.fromMask(mask, bitmask).dilated(grow).clippedTo(mask.getBBox()).setMask(mask, bitmask)

        if self.debugInfo.displayConvolved:
            Display(backend=backend, frame=1).mtv(convolvedImage)
            sigNoise = convolvedImage.clone()
            sigNoise.image.array /= np.sqrt(convolvedImage.variance.array)
            Display(backend=backend, frame=2).mtv(sigNoise)

        return convolvedImage

    def findPeaks(self, maskedImage):
        """Find peaks on an image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image on which to find peaks.

        Returns
        -------
        peaks : `list` of `list` of `pfs.drp.stella.TracePeak`
            List of peaks for each row of the image.
        """
        mask = maskedImage.mask
        peaks = findTracePeaks(maskedImage, self.config.findThreshold,
                               mask.getPlaneBitMask(self.config.mask))
        bitmask = 2**mask.addMaskPlane("FIBERTRACE")
        spans = SpanSet(sum([[pp.span for pp in rowPeaks] for rowPeaks in peaks], []))
        spans.setMask(mask, bitmask)

        if self.debugInfo.findPeaks:
            display = Display(backend=self.debugInfo.backend, frame=self.debugInfo.frame or 1)
            display.setMaskPlaneColor("FIBERTRACE", "blue")
            display.mtv(maskedImage)
            with display.Buffering():
                for rowPeaks in peaks:
                    for pp in rowPeaks:
                        display.dot("o", pp.peak, pp.row, ctype="yellow")
                        display.line([(pp.low, pp.row), (pp.high, pp.row)], ctype="yellow")

        return peaks

    def prunePeaks(self, peaks):
        """Prune the list of peaks

        Large peaks are removed.

        Parameters
        ----------
        peaks : iterable of `pfs.drp.stella.TracePeak`
            Peaks on the image.

        Returns
        -------
        pruned : `list` of `list` of `pfs.drp.stella.TracePeak`
            Pruned list of peaks for each row on the image.
        """
        pruned = []
        counts = defaultdict(int)
        for ii, rowPeaks in enumerate(peaks):
            accepted = []
            for pp in rowPeaks:
                width = pp.high - pp.low
                counts[width] += 1
                if width > self.config.pruneMaxWidth:
                    continue
                accepted.append(pp)
            pruned.append(accepted)
        self.log.debug("Peak width counts: %s", dict(counts))
        return pruned

    def associatePeaks(self, peaks, width):
        """Associate peaks into traces

        Groups the peaks from different rows.

        We associate a peak with a trace if the column of the peak has
        previously been assigned to a trace in the last ``associationDepth``
        rows.

        If there are multiple peaks in a row that want to be assigned to the
        same trace, we take the one with the peak closest to most recent peak
        from the trace, allowing only one-to-one matches.

        Parameters
        ----------
        peaks : `list` of `list` of `pfs.drp.stella.TracePeak`
            List of peaks for each row on the image.
        width : `int`
            Width of the image.

        Returns
        -------
        traces : `list` of `list` of `pfs.drp.stella.TracePeak`
            List of peaks for each trace candidate.
        """
        association = np.full((width, self.config.associationDepth), -1, dtype=int)  # -1 means no trace
        traces = []
        for yy, rowPeaks in enumerate(peaks):
            layer = yy % self.config.associationDepth
            candidates = defaultdict(list)
            unassociated = {}
            for pp in rowPeaks:
                indices = Counter(association[int(pp.peak)])
                if len(indices) == 1 and indices[-1] > 0:
                    # Newly identified trace
                    index = len(traces)
                    traces.append([pp])
                    association[pp.low:pp.high + 1, layer] = index
                    continue
                del indices[-1]
                unassociated[pp.peak] = pp
                for index in indices:
                    # Score is distance to most recent trace peak
                    score = np.abs(pp.peak - traces[index][-1].peak)
                    candidates[index].append((pp, score))

            # Assign peaks to existing traces
            # Work through the traces, first with those with clear associations, and then those with multiple
            # associations. For each trace, take the unassociated peak that has the highest score.
            for index in sorted(candidates, key=lambda ii: len(candidates[ii])):
                remaining = [cc for cc in candidates[index] if cc[0].peak in unassociated]
                if len(remaining) == 0:
                    continue
                best = min(remaining, key=lambda cc: cc[1])  # Lowest score
                pp = best[0]
                traces[index].append(pp)
                association[pp.low:pp.high + 1, layer] = index
                del unassociated[pp.peak]

            # Create new traces for peaks which don't have an association (because other peaks took away
            # all of this peak's associations).
            for pp in unassociated.values():
                index = len(traces)
                traces.append([pp])
                association[pp.low:pp.high + 1, layer] = index

        if self.debugInfo.associatePeaks:
            display = Display(backend=backend, frame=1)
            for tt, cc in zip(traces, itertools.cycle(colors)):
                with display.Buffering():
                    for pp in tt:
                        display.dot("+", pp.peak, pp.row, ctype=cc)

        return traces

    def pruneTraces(self, traces, height=None):
        """Prune the list of traces

        Traces that are too short or spotty are removed.

        Parameters
        ----------
        traces : `list` of `list` of `pfs.drp.stella.TracePeak`
            List of peaks for each trace candidate.
        height : `int`, optional.
            Height of the image. Not needed for the algorithm, but guards
            against pruning all traces if your image is too short.

        Returns
        -------
        accepted : `list` of `list` of `pfs.drp.stella.TracePeak`
            List of peaks for each trace.
        """
        if height is not None and height < self.config.pruneMinLength:
            raise RuntimeError(f"Image height ({height}) is shorter than pruneMinLength "
                               f"({self.config.pruneMinLength})")

        accepted = []
        for ii, peakList in enumerate(traces):
            num = len(peakList)
            lowRow = peakList[0].row
            highRow = peakList[-1].row
            assert num == 1 or all(pp.row > lowRow for pp in peakList[1:])
            assert num == 1 or all(pp.row < highRow for pp in peakList[:1])
            length = highRow - lowRow + 1
            numRows = len(set([pp.row for pp in peakList]))  # Weed out peaks in the same row
            if length < self.config.pruneMinLength or numRows/length < self.config.pruneMinFrac:
                self.log.trace("Pruning trace %d: %d vs %d, %f vs %f", ii, length, self.config.pruneMinLength,
                               numRows/length, self.config.pruneMinFrac)
                continue
            accepted.append(peakList)
        return accepted

    def generateTraces(self, detectorMap, pfsConfig):
        """Generate traces without looking at the image

        We know where the traces are from the ``detectorMap``, and we know which
        fibers are illuminated from the ``pfsConfig``.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to build FiberTraces.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Mapping from x,y to fiberId,row.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end fiber configuration.

        Returns
        -------
        traces : `list` of `list` of `pfs.drp.stella.TracePeak`
            List of peaks for each trace.
        """
        indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD, detectorMap.fiberId)
        traces = []
        for ii in indices:
            fiberId = detectorMap.fiberId[ii]
            xCenter = detectorMap.getXCenter(fiberId)
            traces.append([TracePeak(yy, int(xx), xx, int(xx)) for yy, xx in enumerate(xCenter)])
        return traces

    def centroidTrace(self, maskedImage, peakList):
        """Measure the centroids for peaks in a trace

        The centroid for each peak is updated with the measured value.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImageF`
            Image on which to measure centroids.
        peakList : iterable of `pfs.drp.stella.TracePeak`
            List of peaks to centroid. Peaks will be updated with the measured
            centroids.
        """
        badBitmask = maskedImage.mask.getPlaneBitMask(self.config.mask)
        centroidTrace(peakList, maskedImage, self.config.centroidRadius, badBitmask)
        if self.debugInfo.plotCentroids:
            import matplotlib.pyplot as plt
            plt.plot([pp.peak for pp in peakList], [pp.row for pp in peakList], 'k.')
            plt.show()

    def fitTraceCenters(self, peakList, height):
        """Fit a function of trace center as a function of row

        Parameters
        ----------
        peakList : iterable of `pfs.drp.stella.TracePeak`
            List of peaks with measured centers to fit.
        height : `int`
            Height of the image (pixels).

        Returns
        -------
        fit : `lsst.pipe.base.Struct`
            Polynomial fit results; the output of ``FitPolynomialTask.run``.
            The most important element is ``func``, a callable that provides the
            center of the trace as a function of row.
        """
        row = np.array([pp.row for pp in peakList])
        peak = np.array([pp.peak for pp in peakList])
        return self.centerFit.run(row, peak, xMin=0, xMax=height)

    def calculateProfile(self, maskedImage, centerFunc):
        """Measure the fiber profile

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image on which to measure the fiber profile.
        centerFunc : callable
            A callable that provides the center of the trace as a function of
            row.

        Returns
        -------
        profile : `pfs.drp.stella.fiberProfile.FiberProfile`
            Fiber profile.
        """
        profile = FiberProfile.fromImage(maskedImage, centerFunc, self.config.profileRadius,
                                         self.config.profileOversample, self.config.profileSwath,
                                         self.config.profileRejIter, self.config.profileRejThresh,
                                         self.config.mask)
        if self.debugInfo.plotProfile:
            profile.plot()
        return profile

    def identifyFibers(self, profiles, centers, detectorMap, pfsConfig=None):
        """Identify fibers that have been found and traced

        We compare the measured center positions in the middle of the detector
        with the predicted positions.

        Parameters
        ----------
        profiles : `pfs.drp.stella.FiberProfileSet`
            Fiber profiles, with temporary fiberId values.
        centers : `dict` mapping `int` to callable
            Callable for each fiber (indexed by temporary fiberId; same as for
            the ``profiles``) that provides the center of the trace as a
            function of row.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Mapping from x,y to fiberId,row.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end fiber configuration. Used to check that we've identified all
            the lit fibers.

        Returns
        -------
        profiles : `pfs.drp.stella.FiberProfileSet`
            Fiber profiles with correct fiberId values.
        centers : `dict` mapping `int` to callable
            Callable for each fiber (indexed by fiberId; same as for
            the ``profiles``) that provides the center of the trace as a
            function of row.s
        """
        if len(profiles) == 0:
            self.log.warn("Unable to identify fibers: no fiberTraces")
            return Struct(profiles=FiberProfileSet.makeEmpty(), centers={})
        middle = 0.5*(detectorMap.bbox.getMinY() + detectorMap.bbox.getMaxY())
        expectCenters = np.array([detectorMap.getXCenter(ff, middle) for ff in detectorMap.fiberId])
        assignments = {}
        used = set()
        for index in centers:
            best = detectorMap.fiberId[np.argmin(np.abs(expectCenters - centers[index](middle)))]
            if best in used:
                raise RuntimeError("Matched fiber to a used fiberId")
            assignments[index] = best
            used.add(best)
        self.log.debug("Identified %d fiberIds: %s", len(used), assignments)

        # Apply new fiberIds
        newProfiles = FiberProfileSet.makeEmpty(profiles.visitInfo, profiles.metadata)
        newCenters = {}
        for index, fiberId in assignments.items():
            newProfiles[fiberId] = profiles[index]
            newCenters[fiberId] = centers[index]
        profiles = newProfiles
        centers = newCenters

        if pfsConfig is not None:
            indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD, detectorMap.fiberId)
            notFound = set(detectorMap.fiberId[indices]) - used
            if notFound:
                self.log.warn("Failed to identify %d fiberIds: %s", len(notFound), sorted(notFound))

        if self.debugInfo.identifyFibers:
            display = Display(backend=backend, frame=1)
            for fiberId in profiles:
                display.dot(str(fiberId), centers[fiberId](middle), middle, ctype="green")
            for fiberId in detectorMap.fiberId:
                if fiberId not in used:
                    display.dot(str(fiberId), detectorMap.getXCenter(fiberId, middle), middle, ctype="red")

        return Struct(profiles=profiles, centers=centers)
