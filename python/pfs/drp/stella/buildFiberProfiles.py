import itertools
from collections import Counter, defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np

from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import Task, Struct
from lsst.afw.geom import SpanSet
from lsst.afw.image import ExposureF, MaskedImageF
from lsst.afw.display import Display

from pfs.datamodel import FiberStatus, CalibIdentity
from pfs.drp.stella.traces import findTracePeaks, centroidPeak, TracePeak
from pfs.drp.stella.fitPolynomial import FitPolynomialTask
from pfs.drp.stella.fiberProfile import FiberProfile
from pfs.drp.stella.fiberProfileSet import FiberProfileSet
from pfs.drp.stella.images import convolveImage
from pfs.drp.stella import DetectorMap
from .datamodel import PfsConfig

import lsstDebug
from pfs.drp.stella.utils.psf import fwhmToSigma

__all__ = ("BuildFiberProfilesConfig", "BuildFiberProfilesTask")

colors = ["red", "green", "blue", "cyan", "magenta", "yellow", "orange"]


class BuildFiberProfilesConfig(Config):
    """Configuration for BuildFiberProfilesTask"""
    mask = ListField(dtype=str, default=["CR", "BAD", "SAT", "NO_DATA"], doc="Mask planes to ignore")
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
    centerFit = ConfigurableField(target=FitPolynomialTask, doc="Fit polynomial to trace centroids")
    badFibers = ListField(dtype=int, default=[], doc="Fibers to ignore (e.g., bad but not recorded as such")
    profileSwath = Field(dtype=float, default=300, doc="Length of swaths to use for calculating profile")
    profileRadius = Field(dtype=int, default=5, doc="Radius about the peak for profile")
    profileOversample = Field(dtype=int, default=10, doc="Oversample factor for profile")
    profileRejIter = Field(dtype=int, default=1, doc="Rejection iterations for profile")
    profileRejThresh = Field(dtype=float, default=3.0, doc="Rejection threshold (sigma) for profile")
    profileTol = Field(dtype=float, default=1.0e-4, doc="Tolerance for matrix inversion when fitting profile")
    extractFwhm = Field(dtype=float, default=1.5, doc="FWHM for spectral extraction")
    extractIter = Field(dtype=int, default=2, doc="Number of iterations for spectral extraction loop")


class BuildFiberProfilesTask(Task):
    """Build a FiberTraceSet from an image"""
    ConfigClass = BuildFiberProfilesConfig
    _DefaultName = "buildFiberProfiles"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("centerFit")
        self.debugInfo = lsstDebug.Info(__name__)

    def run(
        self,
        exposure: ExposureF,
        identity: CalibIdentity,
        *,
        detectorMap: Optional[DetectorMap] = None,
        pfsConfig: Optional[PfsConfig] = None,
    ):
        """Build a FiberProfileSet from an image

        We find traces on the image, centroid those traces, and measure the
        fiber profiles.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure from which to build fiber profiles.
        identity : `pfs.datamodel.CalibIdentity`
            Identifying information for the calibration.
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

        See also
        --------
        BuildFiberProfilesTask.runMultiple
        """
        return self.runSingle(exposure, identity, detectorMap, pfsConfig)

    def runSingle(
        self,
        exposure: ExposureF,
        identity: CalibIdentity,
        detectorMap: Optional[DetectorMap] = None,
        pfsConfig: Optional[PfsConfig] = None,
    ):
        """Build a FiberProfileSet from a single image

        We find traces on the image, centroid those traces, measure the fiber
        profile, and construct the fiber profiles.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure from which to build fiber profiles.
        identity : `pfs.datamodel.CalibIdentity`
            Identifying information for the calibration.
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

        See also
        --------
        BuildFiberProfilesTask.runMultiple
        """
        centers = self.findTraces(exposure.maskedImage, detectorMap, pfsConfig)
        profiles = FiberProfileSet.makeEmpty(
            identity, exposure.getInfo().getVisitInfo(), exposure.getMetadata()
        )
        for ff in centers:
            try:
                profiles[ff] = self.calculateProfile(exposure.maskedImage, centers[ff])
            except RuntimeError as exc:
                self.log.debug("Failed to calculate profile for fiber %d: %s", ff, exc)

        self.log.info("Profiled %d fibers", len(profiles))
        return Struct(profiles=profiles, centers=centers)

    def runMultiple(
        self,
        exposureList: Sequence[ExposureF],
        identity: CalibIdentity,
        detectorMapList: Sequence[DetectorMap],
        pfsConfigList: Optional[Sequence[PfsConfig]] = None,
    ) -> Struct:
        """Build a FiberProfileSet from an image

        We find traces on the image, centroid those traces, measure the fiber
        profile, and construct the FiberTraces.

        Parameters
        ----------
        exposureList : iterable of `lsst.afw.image.Exposure`
            Exposures from which to build fiber profiles.
        identity : `pfs.datamodel.CalibIdentity`
            Identifying information for the calibration.
        detectorMapList : iterable of `pfs.drp.stella.DetectorMap`
            Mappings from x,y to fiberId,row.
        pfsConfigList : iterable of `pfs.datamodel.PfsConfig`, optional
            Top-end fiber configurations.

        Returns
        -------
        profiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of each fiber.
        centerList : iterable of `numpy.ndarray` of `float` with shape ``(Nfibers, Nrows)``
            Arrays that provide the center of the trace as a function of
            row for each fiber in each image.
        spectraList : iterable of `numpy.ndarray` of `float` with shape ``(Nfibers, Nrows)``
            Arrays that provide the flux of the trace as a function of row for
            each fiber in each image.
        """
        num = len(exposureList)
        imageList = [exp.maskedImage for exp in exposureList]
        dims = imageList[0].getDimensions()
        for img in imageList[1:]:
            if img.getDimensions() != dims:
                raise RuntimeError(f"Dimensions mismatch: {img.getDimensions()} vs {dims}")
        if len(imageList) != len(detectorMapList):
            raise RuntimeError(f"Length mismatch for exposures ({len(imageList)}) and "
                               f"detectorMaps ({len(detectorMapList)})")
        if pfsConfigList is not None and len(imageList) != len(pfsConfigList):
            raise RuntimeError(f"Length mismatch for exposures ({len(imageList)}) and "
                               f"pfsConfigs ({len(pfsConfigList)})")

        rows = np.arange(0, dims.getY(), dtype=float)

        centersList: List[np.ndarray] = []
        normList: List[np.ndarray] = []
        fibers: Optional[Set[int]] = None
        for ii in range(num):
            image = imageList[ii]
            detectorMap = detectorMapList[ii]
            pfsConfig = pfsConfigList[ii] if pfsConfigList is not None else None
            centers = self.findTraces(image, detectorMap, pfsConfig)
            if fibers is None:
                fibers = set(centers.keys())
            elif set(centers.keys()) != fibers:
                raise RuntimeError(f"FiberId mismatch: {set(centers.keys())} vs {fibers}")
            centersArray = np.empty((len(centers), image.getHeight()), dtype=float)
            norm = np.zeros_like(centersArray, dtype=np.float32)
            prof = FiberProfile.makeGaussian(fwhmToSigma(self.config.extractFwhm), image.getHeight(),
                                             self.config.profileRadius, self.config.profileOversample)
            badBitMask = image.mask.getPlaneBitMask(self.config.mask)
            for jj, ff in enumerate(sorted(centers.keys())):
                centersArray[jj] = centers[ff](rows)
                spectrum = prof.extractSpectrum(image, detectorMap, ff, badBitMask)
                norm[jj] = spectrum.flux

            centersList.append(centersArray)
            normList.append(norm)

        if fibers is None:
            raise RuntimeError("No fibers")
        fiberId = np.array(list(sorted(fibers)), dtype=int)

        self.log.info("Starting initial profile extraction (sigma=%f)...",
                      fwhmToSigma(self.config.extractFwhm))
        profiles = FiberProfileSet.fromImages(
            identity,
            imageList,
            fiberId,
            centersList,
            normList,
            self.config.profileRadius,
            self.config.profileOversample,
            self.config.profileSwath,
            self.config.profileRejIter,
            self.config.profileRejThresh,
            self.config.profileTol,
            self.config.mask,
            exposureList[0].getInfo().getVisitInfo(),
            exposureList[0].getMetadata(),
        )

        if self.debugInfo.plotProfile:
            for pp in profiles:
                pp.plot()

        # Iterate with the revised profiles
        for ii in range(self.config.extractIter):
            # Generate profiles with which to measure the flux
            # These are Gaussian approximations to the profiles we've measured, to remove any nastiness in the
            # measured profiles that might slow down convergence.
            fluxProfiles = FiberProfileSet.makeEmpty(profiles.visitInfo, profiles.metadata)
            width = np.array([profiles[ff].calculateStatistics().width for ff in profiles])
            select = np.isfinite(width) & (width > 0)
            sigma = np.median(np.sqrt(width[select]))
            for ff in profiles:
                fluxProfiles[ff] = FiberProfile.makeGaussian(
                    sigma, image.getHeight(), self.config.profileRadius, self.config.profileOversample
                )

            for jj in range(num):
                badBitMask = imageList[jj].mask.getPlaneBitMask(self.config.mask)
                spectra = fluxProfiles.extractSpectra(imageList[jj], detectorMapList[jj], badBitMask)
                normList[jj] = spectra.getAllFluxes()

            self.log.info("Starting profile extraction iteration %d (sigma=%f)...", ii + 1, sigma)
            profiles = FiberProfileSet.fromImages(
                identity,
                imageList,
                fiberId,
                centersList,
                normList,
                self.config.profileRadius,
                self.config.profileOversample,
                self.config.profileSwath,
                self.config.profileRejIter,
                self.config.profileRejThresh,
                self.config.profileTol,
                self.config.mask,
                exposureList[0].getInfo().getVisitInfo(),
                exposureList[0].getMetadata(),
            )

            if self.debugInfo.plotProfile:
                for pp in profiles:
                    pp.plot()

        return Struct(profiles=profiles, centers=centersList, norm=normList)

    def findTraces(
        self,
        image: MaskedImageF,
        detectorMap: Optional[DetectorMap] = None,
        pfsConfig: Optional[PfsConfig] = None,
    ) -> Dict[int, Callable[[float], float]]:
        """Find and centroid traces on an image

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image from which to build FiberTraces.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Mapping from x,y to fiberId,row.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end fiber configuration.

        Returns
        -------
        centers : `dict` mapping `int` to callable
            Callable for each fiber (indexed by fiberId) that provides the
            center of the trace as a function of row.
        """
        if self.config.doBlindFind or detectorMap is None:
            convolved = self.convolveImage(image)
            peaks = self.findPeaks(convolved)
            self.log.debug("Found %d peaks", sum([len(pp) for pp in peaks]))
            peaks = self.prunePeaks(peaks)
            self.log.debug("Pruned to %d peaks", sum([len(pp) for pp in peaks]))
            traces = self.associatePeaks(peaks, convolved.getWidth())
            self.log.debug("Associated into %d traces", len(traces))
            traces = self.pruneTraces(traces, image.getHeight())
            self.log.debug("Pruned to %d traces", len(traces))
            traces = {ii: tt for ii, tt in enumerate(traces)}
        else:
            traces = self.generateTraces(detectorMap, pfsConfig)

        if self.config.doBlindFind:
            # Centroid the traces.
            # If we don't have a detectorMap, we will fit a functional form to get the centers for the
            # profiles. If we do have a detectorMap, this will be used for identifying fibers.
            newTraces = {}
            for ff, tt in traces.items():
                tt = self.centroidTrace(image, tt)
                if len(tt) < self.config.pruneMinLength:
                    continue
                newTraces[ff] = tt
            traces = newTraces
            del newTraces
            self.log.debug("Centroided %d traces", len(traces))

            centers = {ff: self.fitTraceCenters(tt, image.getHeight()).func for ff, tt in traces.items()}
            if detectorMap is not None:
                identifications = self.identifyFibers(centers, detectorMap, pfsConfig)
                traces = {identifications[ff]: tt for ff, tt in traces.items()}
                centers = {identifications[ff]: cc for ff, cc in centers.items()}
            self.log.debug("Fit centers for %d traces", len(centers))
        else:
            # Use centers directly from detectorMap. This yields more accurate centers for the profiles.
            def centerFunc(fiberId, yy):
                """Return xCenter from the detectorMap

                Parameters
                ----------
                fiberId : `int`
                    Fiber identifier.
                yy : `numpy.ndarray`
                    Row values.

                Returns
                -------
                xCenter : `numpy.ndarray` of `float`
                    xCenter values.
                """
                return detectorMap.getXCenter(fiberId, yy.astype(np.float64))

            centers = {ff: partial(centerFunc, ff) for ff in traces}

        self.log.info("Found and centroided %d traces", len(centers))
        return centers

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
        convolvedImage = convolveImage(maskedImage, self.config.columnFwhm, self.config.rowFwhm,
                                       self.config.convolveGrowMask, self.config.kernelSize)

        if self.debugInfo.displayConvolved:
            Display(frame=1).mtv(convolvedImage)
            sigNoise = convolvedImage.clone()
            sigNoise.image.array /= np.sqrt(convolvedImage.variance.array)
            Display(frame=2).mtv(sigNoise)

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
            display = Display(frame=self.debugInfo.frame or 1)
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
                    # Score is negative number of past associations and distance to most recent trace peak
                    numAssoc = indices[index]
                    distance = np.abs(pp.peak - traces[index][-1].peak)
                    candidates[index].append((pp, (-numAssoc, distance)))

            # Assign peaks to existing traces
            # Work through the traces, first with those with clear associations, and then those with multiple
            # associations. For each trace, take the unassociated peak that has the lowest score.
            for index in sorted(candidates,
                                key=lambda ii: (len(candidates[ii]), min(cc[1] for cc in candidates[ii]))):
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
            display = Display(frame=1)
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

        numTraces = len(traces)
        length = np.empty(numTraces, dtype=int)
        frac = np.empty(numTraces, dtype=float)
        accepted = []
        for ii, peakList in enumerate(traces):
            num = len(peakList)
            lowRow = peakList[0].row
            highRow = peakList[-1].row
            assert num == 1 or all(pp.row > lowRow for pp in peakList[1:])
            assert num == 1 or all(pp.row < highRow for pp in peakList[:1])
            length[ii] = highRow - lowRow + 1
            numRows = len(set([pp.row for pp in peakList]))  # Weed out peaks in the same row
            frac[ii] = numRows/length[ii]
            if length[ii] < self.config.pruneMinLength or frac[ii] < self.config.pruneMinFrac:
                self.log.trace("Pruning trace %d: %d vs %d, %f vs %f", ii, length[ii],
                               self.config.pruneMinLength, frac[ii], self.config.pruneMinFrac)
                continue
            accepted.append(peakList)

        if self.debugInfo.pruneTraces:
            import matplotlib.pyplot as plt
            plt.plot(length, frac, "k.")
            plt.xlabel("Trace length (pixels)")
            plt.ylabel("Coverage fraction")
            plt.axvline(self.config.pruneMinLength)
            plt.axhline(self.config.pruneMinFrac)
            plt.show()

        return accepted

    def generateTraces(self, detectorMap, pfsConfig=None):
        """Generate traces without looking at the image

        We know where the traces are from the ``detectorMap``, and we know which
        fibers are illuminated from the ``pfsConfig``.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Mapping from x,y to fiberId,row.
        pfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end fiber configuration.

        Returns
        -------
        traces : `dict` of `list` of `pfs.drp.stella.TracePeak`
            Peaks for each trace, indexed by fiberId.
        """
        if pfsConfig is not None:
            fiberId = pfsConfig.fiberId
        else:
            fiberId = detectorMap.fiberId
        if self.config.badFibers:
            fiberId = np.array(list(set(fiberId) - set(self.config.badFibers)))
        traces = {}
        for ff in fiberId:
            xCenter = detectorMap.getXCenter(ff)
            traces[ff] = [TracePeak(yy, int(xx), xx, int(xx)) for yy, xx in enumerate(xCenter)]
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

        Returns
        -------
        newPeakList : `list` of `pfs.drp.stella.TracePeak`
            List of peaks with centroids measured. Peaks without a good centroid
            measurement will be removed.
        """
        badBitmask = maskedImage.mask.getPlaneBitMask(self.config.mask)
        sigma = fwhmToSigma(self.config.columnFwhm)
        newPeakList = []
        for pp in peakList:
            centroidPeak(pp, maskedImage, sigma, badBitmask)
            if not np.isfinite(pp.peak):
                continue
            newPeakList.append(pp)

        if self.debugInfo.plotCentroids:
            import matplotlib.pyplot as plt
            plt.plot([pp.peak for pp in newPeakList], [pp.row for pp in newPeakList], 'k.')
            plt.show()
        return newPeakList

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

    def identifyFibers(self, centers, detectorMap, pfsConfig=None):
        """Identify fibers that have been found and traced

        We compare the measured center positions in the middle of the detector
        with the predicted positions.

        Parameters
        ----------
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
        identifications : `dict` mapping `int` to `int`
            Mapping of ``centers`` index to fiberId.
        """
        if len(centers) == 0:
            self.log.warning("Unable to identify fibers: no traces found")
            return centers
        middle = 0.5*(detectorMap.bbox.getMinY() + detectorMap.bbox.getMaxY())
        expectCenters = np.array([detectorMap.getXCenter(ff, middle) for ff in detectorMap.fiberId])
        assignments = {}
        used = {}
        for index in centers:
            bestIndex = np.argmin(np.abs(expectCenters - centers[index](middle)))
            best = detectorMap.fiberId[bestIndex]
            assert best not in used, "Matched fiber to a used fiberId"
            assignments[index] = best
            used[best] = index
            expectCenters[bestIndex] = np.inf  # Prevent reuse
        self.log.debug("Identified %d fiberIds: %s", len(used), assignments)

        if pfsConfig is not None:
            indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD, detectorMap.fiberId)
            notFound = set(detectorMap.fiberId[indices]) - set(used.keys())
            if notFound:
                self.log.warn("Failed to identify %d fiberIds: %s", len(notFound), sorted(notFound))

        if self.debugInfo.identifyFibers:
            display = Display(frame=1)
            for index in centers:
                fiberId = assignments[index]
                display.dot(str(fiberId), centers[index](middle), middle, ctype="green")
            for fiberId in detectorMap.fiberId:
                if fiberId not in used:
                    display.dot(str(fiberId), detectorMap.getXCenter(fiberId, middle), middle, ctype="red")

        return assignments
