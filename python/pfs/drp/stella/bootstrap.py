from types import SimpleNamespace
from operator import attrgetter
from datetime import datetime
from collections import defaultdict
import numpy as np
from astropy.modeling.models import Gaussian1D, Chebyshev2D
from astropy.modeling.fitting import LinearLSQFitter, LevMarLSQFitter

from lsst.pipe.base import CmdLineTask, TaskRunner, ArgumentParser, Struct
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.obs.pfs.isrTask import PfsIsrTask

from pfs.datamodel import FiberStatus, CalibIdentity
from pfs.datamodel.pfsConfig import TargetType
from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus
from .buildFiberProfiles import BuildFiberProfilesTask
from .findLines import FindLinesTask
from .readLineList import ReadLineListTask
from .constructSpectralCalibs import setCalibHeader
from . import SplinedDetectorMap

import lsstDebug


class BootstrapConfig(Config):
    """Configuration for BootstrapTask"""
    isr = ConfigurableField(target=PfsIsrTask, doc="Instrumental signature removal")
    profiles = ConfigurableField(target=BuildFiberProfilesTask, doc="Fiber profiles")
    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read linelist")
    minArcLineIntensity = Field(dtype=float, default=0, doc="Minimum 'NIST' intensity to use emission lines")
    mask = ListField(dtype=str, default=["NO_DATA", "BAD", "SAT", "CR", "BAD_FLAT"],
                     doc="Mask pixels to ignore in extracting spectra")
    findLines = ConfigurableField(target=FindLinesTask, doc="Find arc lines")
    matchRadius = Field(dtype=float, default=1.0, doc="Line matching radius (nm)")
    badLineStatus = ListField(dtype=str, default=["NOT_VISIBLE", "BLEND"],
                              doc="Reference line status flags indicating that line should be excluded")
    spatialOrder = Field(dtype=int, default=1, doc="Polynomial order in the spatial dimension")
    spectralOrder = Field(dtype=int, default=1, doc="Polynomial order in the spectral dimension")
    rejIterations = Field(dtype=int, default=3, doc="Number of fitting iterations")
    rejThreshold = Field(dtype=float, default=3.0, doc="Rejection threshold (stdev)")
    allowSplit = Field(dtype=bool, default=True, doc="Allow split detectors for brm arms (not n)?")
    rowForCenter = Field(dtype=float, default=2048, doc="Row for xCenter calculation; used if allowSplit")
    midLine = Field(dtype=float, default=2048,
                    doc="Column defining the division between left and right amps; used if allowSplit")
    fiberStatus = ListField(dtype=str, default=["GOOD", "BROKENFIBER"], doc="Fiber statuses to allow")
    targetType = ListField(dtype=str,
                           default=["SCIENCE", "SKY", "FLUXSTD", "SUNSS_IMAGING", "SUNSS_DIFFUSE", "DCB",
                                    "HOME"],
                           doc="Target types to allow")
    spatialOffset = Field(dtype=float, default=0.0, doc="Offset to apply to spatial dimension")
    spectralOffset = Field(dtype=float, default=0.0, doc="Offset to apply to spectral dimension")
    badFibers = ListField(dtype=int, default=[], doc="Fibers to ignore (e.g., bad but not recorded as such")

    def setDefaults(self):
        super().setDefaults()
        self.profiles.doBlindFind = True  # We can't trust the detectorMap


class BootstrapRunner(TaskRunner):
    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        """Produce list of targets

        We only want to operate on a single flat and single arc, together.
        """
        flatArms = defaultdict(dict)
        for ref in parsedCmd.flatId.refList:
            arm = ref.dataId["arm"]
            spec = ref.dataId["spectrograph"]
            if arm in flatArms[spec]:
                raise RuntimeError(f"Multiple flat exposures specified for arm={arm} spectrograph={spec}")
            flatArms[spec][arm] = ref

        arcArms = defaultdict(dict)
        for ref in parsedCmd.arcId.refList:
            arm = ref.dataId["arm"]
            spec = ref.dataId["spectrograph"]
            if arm in arcArms[spec]:
                raise RuntimeError(f"Multiple arc exposures specified for arm={arm} spectrograph={spec}")
            arcArms[spec][arm] = ref

        flats = set([(ss, aa) for ss in flatArms for aa in flatArms[ss]])
        arcs = set([(ss, aa) for ss in arcArms for aa in arcArms[ss]])
        missingArcs = flats - arcs
        missingFlats = arcs - flats
        if missingArcs:
            missing = "; ".join(f"arm={aa} spectrograph={ss}" for ss, aa in missingArcs)
            raise RuntimeError(f"No arcs provided for flats: {missing}")
        if missingFlats:
            missing = "; ".join(f"arm={aa} spectrograph={ss}" for ss, aa in missingFlats)
            raise RuntimeError(f"No flats provided for arcs: {missing}")
        assert flats == arcs

        return [(flatArms[ss][aa], dict(arcRef=arcArms[ss][aa])) for ss, aa in flats]


class BootstrapTask(CmdLineTask):
    """Bootstrap a detectorMap

    We have a reasonable detectorMap from the optical model + 2D simulator.
    However, the as-built spectrograph may have a slightly different mapping
    of fiberId,wavelength to x,y, or the slit may have been repositioned.
    Here, we use a quartz flat and an arc to fit out differences between the
    expected and actual detectorMap.
    """
    _DefaultName = "bootstrap"
    ConfigClass = BootstrapConfig
    RunnerClass = BootstrapRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("profiles")
        self.makeSubtask("readLineList")
        self.makeSubtask("findLines")

    @classmethod
    def _makeArgumentParser(cls):
        """Build a suitable ArgumentParser"""
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--flatId", "raw", help="data ID for flat, e.g., visit=12345")
        parser.add_id_argument("--arcId", "raw", help="data ID for arc, e.g., visit=54321")
        return parser

    def runDataRef(self, flatRef, arcRef):
        """Fit out differences between the expected and actual detectorMap

        We use a quartz flat to find and trace fibers. The resulting fiberTrace
        is used to extract the arc spectra. We find lines on the extracted arc
        spectra, and match them to the reference lines, and then fit the
        positions of the arc lines. The updated detectorMap is written out.

        Parameters
        ----------
        flatRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for a quartz flat.
        arcRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for an arc.
        """
        flatConfig = flatRef.get("pfsConfig")
        arcConfig = arcRef.get("pfsConfig")
        if not np.all(flatConfig.fiberId == arcConfig.fiberId):
            raise RuntimeError("Mismatch between fibers for flat (%s) and arc (%s)" %
                               (flatConfig.fiberId, arcConfig.fiberId))
        traces = self.traceFibers(flatRef, flatConfig)
        lineResults = self.findArcLines(arcRef, traces)

        self.visualize(lineResults.exposure, [ss.fiberId for ss in lineResults.spectra],
                       lineResults.detectorMap, lineResults.refLines, frame=1)

        matches = self.matchArcLines(lineResults.lines, lineResults.refLines, lineResults.detectorMap)
        fiberIdLists = self.selectFiberIds(lineResults.detectorMap, arcRef.dataId["arm"])
        for fiberId in fiberIdLists:
            self.fitDetectorMap(matches, lineResults.detectorMap, fiberId)

        self.visualize(lineResults.exposure, [ss.fiberId for ss in lineResults.spectra],
                       lineResults.detectorMap, lineResults.refLines, frame=2)

        self.setCalibHeader(lineResults.detectorMap.metadata, arcRef.dataId)
        arcRef.put(lineResults.detectorMap, "detectorMap", visit0=arcRef.dataId["visit"])

    def traceFibers(self, flatRef, pfsConfig):
        """Trace fibers on the quartz flat

        We need to make sure we find as many fibers on the flat as we expect
        from the ``pfsConfig``, and assign them the correct ``fiberId``s.

        Parameters
        ----------
        flatRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for a quartz flat.

        Returns
        -------
        traces : `pfs.drp.stella.FiberTraceSet`
            Set of fiber traces.
        """
        exposure = self.isr.runDataRef(flatRef).exposure
        detMap = flatRef.get("detectorMap")
        identity = CalibIdentity.fromDict(flatRef.dataId)
        result = self.profiles.run(exposure, identity, detectorMap=detMap, pfsConfig=pfsConfig)
        traces = result.profiles.makeFiberTraces(exposure.getDimensions(), result.centers)
        traces.sortTracesByXCenter()  # Organised left to right
        self.log.info("Found %d fibers on flat", len(traces))
        fiberStatus = [FiberStatus.fromString(fs) for fs in self.config.fiberStatus]
        targetType = [TargetType.fromString(fs) for fs in self.config.targetType]
        fiberId = detMap.fiberId
        if self.config.badFibers:
            fiberId = np.array(list(set(fiberId) - set(self.config.badFibers)))
        select = pfsConfig.getSelection(fiberId=fiberId, fiberStatus=fiberStatus,
                                        targetType=targetType)
        numSelected = select.sum()
        if len(traces) != numSelected:
            raise RuntimeError("Insufficient traces (%d) found vs expected number of fibers (%d)" %
                               (len(traces), numSelected))
        fiberId = np.sort(pfsConfig.fiberId[select])
        # Assign fiberId from pfsConfig to the fiberTraces, but we have to get the order right!
        # The fiber trace numbers from the left, but the pfsConfig may number from the right.
        middle = 0.5*exposure.getHeight()
        centers = np.array([detMap.getXCenter(ff, middle) for ff in fiberId])
        increasing = np.all(centers[1:] - centers[:-1] > 0)
        decreasing = np.all(centers[1:] - centers[:-1] < 0)
        assert increasing or decreasing
        for tt, ff in zip(traces, fiberId if increasing else reversed(fiberId)):
            tt.fiberId = ff
        return traces

    def findArcLines(self, arcRef, traces):
        """Find lines on the extracted arc spectra

        The x and y centroids are done separately, for convenience: the x
        centroid from the fiber trace, and the y centroid from the extracted
        spectrum.

        Parameters
        ----------
        arcExposure : `lsst.daf.persistence.ButlerDataRef`
            Data reference for arc.
        traces : `pfs.drp.stella.FiberTraceSet`
            Set of fiber traces.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra
        lines : `list` of `list` of `types.SimpleNamespace`
            List of lines found on each fiber, each with attributes:

        ``fiberId``
            Fiber identifier (`int`).
        ``x``
            Center in x (`float`).
        ``y``
            Center in y (`float`).
        ``flux``
            Peak flux (`float`).

        detectorMap : `pfs.drp.stella.DetectorMap`
            Map of fiberId,wavelength to x,y.
        """
        exposure = self.isr.runDataRef(arcRef).exposure
        detMap = arcRef.get("detectorMap")

        if hasattr(detMap, "getBase"):
            detMap = detMap.getBase()

        detMap.applySlitOffset(self.config.spatialOffset, self.config.spectralOffset)
        refLines = self.readLineList.run(metadata=exposure.getMetadata())
        badBitMask = exposure.mask.getPlaneBitMask(self.config.mask)
        spectra = traces.extractSpectra(exposure.maskedImage, badBitMask)
        yCenters = [self.findLines.runCentroids(ss).centroids for ss in spectra]
        xCenters = [self.centroidTrace(tt, yList) for tt, yList in zip(traces, yCenters)]
        lines = [[SimpleNamespace(fiberId=spectrum.fiberId, x=xx, y=yy, flux=spectrum.spectrum[int(yy + 0.5)])
                  for xx, yy in zip(xList, yList)]
                 for xList, yList, spectrum in zip(xCenters, yCenters, spectra)]
        self.log.info("Found %d lines in %d traces", sum(len(ll) for ll in lines), len(lines))
        return Struct(spectra=spectra, lines=lines, detectorMap=detMap, exposure=exposure, refLines=refLines)

    def centroidTrace(self, trace, rows):
        """Centroid the trace

        Could do an x,y centroid on the arc image, but we have a y centroid
        already, so we really just need to know where it is in x.

        Parameters
        ----------
        trace : `pfs.drp.stella.FiberTrace`
            Fiber trace.
        rows : iterable of `float`
            Floating-point row value at which to centroid the trace.
        """
        image = trace.getTrace()
        indices = np.arange(image.getBBox().getMinY(), image.getBBox().getMaxY() + 1)
        values = np.array([np.interp(rows, indices, image.image.array[:, ii])
                           for ii in range(image.getWidth())]).T

        def fitGaussianCentroid(data):
            """Fit a Gaussian and returns the centroid

            Parameters
            ----------
            data : array-like
                Data to centroid.

            Returns
            -------
            centroid : `float`
                Centroid of array.
            """
            num = len(data)
            center = np.argmax(data)
            amplitude = data[center]
            width = 1.0
            model = Gaussian1D(amplitude, center, width, bounds={"mean": (0, num - 1)})
            fitter = LevMarLSQFitter()
            fit = fitter(model, np.arange(num, dtype=float), data)
            return fit.mean.value

        return [fitGaussianCentroid(data) + image.getX0() for data in values]

    def matchArcLines(self, obsLines, refLines, detectorMap):
        """Match observed arc lines with the reference list

        For each observed line, we take the brightest reference line within the
        ``matchRadius`` (configuration parameter). We can get away with such a
        simple matching algorithm because we intend to use only the brightest
        lines from arcs with a simple/sparse spectrum.

        Parameters
        ----------
        obsLines : `list` of `list` of `types.SimpleNamespace`
            Observed lines in each spectrum (returned from ``findArcLines``);
            should have attributes ``flux`` and ``y``.
        refLines : `list` of `pfs.drp.stella.ReferenceLine`
            Reference lines.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Map of fiberId,wavelength to x,y.

        Returns
        -------
        matches : `list` of `types.SimpleNamespace`
            Matched lines, with attributes:

        ``obs``
            Observed line (`types.SimpleNamespace`; returned from
            ``findArcLines``).
        ``ref``
            Reference line (`pfs.drp.stella.ReferenceLine`).
        """
        matches = []
        badLineStatus = ReferenceLineStatus.fromNames(*self.config.badLineStatus)
        refLines = ReferenceLineSet.fromRows([rl for rl in refLines if (rl.status & badLineStatus) == 0])
        for obs in obsLines:
            used = set()
            obs = sorted(obs, key=attrgetter("flux"), reverse=True)  # Brightest first
            for line in obs:
                wl = detectorMap.findWavelength(line.fiberId, line.y)
                candidates = [ref for ref in refLines if
                              ref.wavelength not in used and
                              abs(ref.wavelength - wl) < self.config.matchRadius]
                if not candidates:
                    continue
                ref = max(candidates, key=attrgetter("intensity"))
                matches.append(SimpleNamespace(obs=line, ref=ref))
                used.add(ref.wavelength)
        self.log.info("Matched %d lines", len(matches))
        return matches

    def selectFiberIds(self, detectorMap, arm):
        """Select lists of fiberIds to fit/update

        For the b, r and m arms, we want to fit/update the left and right
        halves separately, because there are two detectors. For the n arm,
        there's a single detector.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Map of fiberId,wavelength to x,y.
        arm : `str`
            Spectrograph arm (one of b, r, m, n).

        Returns
        -------
        fiberIdLists : `list` of `set` of `int`
            Lists of fiberIds to fit/update.
        """
        if not self.config.allowSplit or arm == "n":
            return [None]
        xCenter = [detectorMap.getXCenter(fiberId, self.config.rowForCenter) for
                   fiberId in detectorMap.fiberId]
        return [set([fiberId for fiberId, xCenter in zip(detectorMap.fiberId, xCenter) if
                     xCenter < self.config.midLine]),
                set([fiberId for fiberId, xCenter in zip(detectorMap.fiberId, xCenter) if
                     xCenter >= self.config.midLine]),
                ]

    def fitDetectorMap(self, matches, detectorMap, fiberId=None, doUpdate=True):
        """Fit the observed line locations and update the detectorMap

        We fit a model of where the lines are, based on where the lines were
        expected to be.

        Parameters
        ----------
        matches : `list` of `types.SimpleNamespace`
            Matched lines, with ``obs`` and ``ref`` attributes, from
            ``matchArcLines``.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Map of fiberId,wavelength to x,y.
        fiberId : iterable of `int`
            List of fiberIds to fit/update. If ``None``, will fit/update all.
        doUpdate : `bool`
            Update the ``detectorMap``?
        """
        xDomain = [detectorMap.bbox.getMinX(), detectorMap.bbox.getMaxX()]
        yDomain = [detectorMap.bbox.getMinY(), detectorMap.bbox.getMaxY()]
        if fiberId:
            fiberId = set(fiberId)
        xx = np.array([detectorMap.findPoint(mm.obs.fiberId, mm.ref.wavelength)
                       for mm in matches if not fiberId or mm.obs.fiberId in fiberId]).T  # Where it should be
        yy = np.array([[mm.obs.x, mm.obs.y] for mm in matches if
                       not fiberId or mm.obs.fiberId in fiberId]).T  # Where it is
        diff = yy - xx

        if lsstDebug.Info(__name__).plotShifts:
            import matplotlib.pyplot as plt
            import matplotlib.cm
            from matplotlib.colors import Normalize
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            cmap = matplotlib.cm.rainbow
            mean = diff.mean(axis=1)
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            magnitude = np.hypot(diff[0] - mean[0], diff[1] - mean[1])
            norm = Normalize()
            norm.autoscale(magnitude)
            axes.quiver(xx[0], xx[1], diff[0] - mean[0], diff[1] - mean[1], color=cmap(norm(magnitude)))
            axes.set_xlabel("Spatial")
            axes.set_ylabel("Spectral")
            colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            colors.set_array([])
            fig.colorbar(colors, cax=cax, orientation='vertical')
            plt.show()

        self.log.info("Median difference from detectorMap: %f,%f pixels",
                      np.median(diff[0]), np.median(diff[1]))

        result = fitChebyshev2D(xx, yy, self.config.spatialOrder, self.config.spectralOrder,
                                xDomain=xDomain, yDomain=yDomain, rejIterations=self.config.rejIterations,
                                rejThreshold=self.config.rejThreshold)
        self.log.info("Fit %d/%d points, rms: x=%f y=%f total=%f pixels",
                      result.used.sum(), len(result.used), result.xRms, result.yRms, result.rms)
        good = result.used
        fitSpatial = result.xFit
        fitSpectral = result.yFit

        if lsstDebug.Info(__name__).plotResiduals:
            import matplotlib.pyplot as plt
            import matplotlib.cm
            from matplotlib.colors import Normalize
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fig, axes = plt.subplots(2, 2)

            dSpatial = yy[0] - fitSpatial(xx[0], xx[1])
            dSpectral = yy[1] - fitSpectral(xx[0], xx[1])

            axes[0, 0].scatter(yy[0][good], dSpatial[good], color="k", marker=".")
            axes[0, 0].scatter(yy[0][~good], dSpatial[~good], color="r", marker=".")
            axes[0, 0].set_xlabel("Spatial")
            axes[0, 0].set_ylabel(r"$\Delta$ Spatial")

            axes[0, 1].scatter(yy[0][good], dSpectral[good], color="k", marker=".")
            axes[0, 1].scatter(yy[0][~good], dSpectral[~good], color="r", marker=".")
            axes[0, 1].set_xlabel("Spatial")
            axes[0, 1].set_ylabel(r"$\Delta$ Spectral")

            axes[1, 0].scatter(yy[1][good], dSpatial[good], color="k", marker=".")
            axes[1, 0].scatter(yy[1][~good], dSpatial[~good], color="r", marker=".")
            axes[1, 0].set_xlabel("Spectral")
            axes[1, 0].set_ylabel(r"$\Delta$ Spatial")

            axes[1, 1].scatter(yy[1][good], dSpectral[good], color="k", marker=".")
            axes[1, 1].scatter(yy[1][~good], dSpectral[~good], color="r", marker=".")
            axes[1, 1].set_xlabel("Spectral")
            axes[1, 1].set_ylabel(r"$\Delta$ Spectral")

            plt.subplots_adjust()
            plt.show()

            cmap = matplotlib.cm.rainbow
            mean = diff.mean(axis=1)
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            divider = make_axes_locatable(axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            magnitude = np.hypot(diff[0] - mean[0], diff[1] - mean[1])
            norm = Normalize()
            norm.autoscale(magnitude[good])
            axes.quiver(xx[0][good], xx[1][good], (diff[0] - mean[0])[good], (diff[1] - mean[1])[good],
                        color=cmap(norm(magnitude[good])))
            axes.set_xlabel("Spatial")
            axes.set_ylabel("Spectral")
            colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            colors.set_array([])
            fig.colorbar(colors, cax=cax, orientation='vertical')
            plt.show()

        # Update the detectorMap
        if doUpdate:
            self.log.info("Updating detectorMap...")
            if not isinstance(detectorMap, SplinedDetectorMap):
                raise RuntimeError("Can only update a SplinedDetectorMap")
            for ff in detectorMap.fiberId:
                if fiberId and ff not in fiberId:
                    continue
                wavelengthFunc = detectorMap.getWavelengthSpline(ff)  # x=rows --> y=wavelength
                centerFunc = detectorMap.getXCenterSpline(ff)  # x=rows --> y=xCenter

                wavelength = wavelengthFunc.getY()
                wlRows = wavelengthFunc.getX()
                wlCols = centerFunc(wlRows)
                wlRowsFixed = fitSpectral(wlCols, wlRows)
                detectorMap.setWavelength(ff, wlRowsFixed, wavelength)

                cenRows = centerFunc.getX()
                cenCols = centerFunc.getY()
                cenColsFixed = fitSpatial(cenCols, cenRows)
                cenRowsFixed = fitSpectral(cenCols, cenRows)
                detectorMap.setXCenter(ff, cenRowsFixed, cenColsFixed)

    def visualize(self, image, fiberId, detectorMap, refLines, frame=1):
        """Visualize arc lines on an image

        Requires that ``lsstDebug`` has been set up, and the ``visualize``
        parameter set to a true value.

        Displays the image, and the position of arc lines.

        Parameters
        ----------
        image : `lsst.afw.image.Image` or `lsst.afw.image.Exposure`
            Image to display.
        fiberId : iterable of `int`
            Fiber identifiers.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Map of fiberId,wavelength to x,y.
        refLines : iterable of `pfs.drp.stella.ReferenceLine`
            Reference lines.
        frame : `int`, optional
            Display frame to use.
        """
        if not lsstDebug.Info(__name__).visualize:
            return
        from lsst.afw.display import Display
        top = 50
        disp = Display(frame)
        disp.mtv(image)

        wlArrays = np.array([detectorMap.getWavelength(ff) for ff in fiberId])
        minWl = wlArrays.min()
        maxWl = wlArrays.max()
        refLines = [rl for rl in refLines if rl.wavelength > minWl and rl.wavelength < maxWl]
        refLines = sorted(refLines, key=attrgetter("intensity"), reverse=True)[:top]  # Brightest
        wavelengths = [rl.wavelength for rl in refLines]
        detectorMap.display(disp, fiberId, wavelengths, plotTraces=False)

    def setCalibHeader(self, metadata, dataId):
        """Set the header

        Parameters
        ----------
        metadata : `lsst.daf.base.PropertyList`
            FITS header to update.
        dataId : `dict` (`str`: POD)
            Data identifier.
        """
        keywords = ("arm", "spectrograph", "ccd", "filter", "calibDate", "calibTime", "visit0")
        mapping = dict(visit0="visit", calibDate="dateObs", calibTime="taiObs")
        outputId = {key: dataId[mapping.get(key, key)] for key in keywords}
        setCalibHeader(metadata, "detectorMap", [dataId["visit"]], outputId)

        date = datetime.now().isoformat()
        history = f"bootstrap on {date} with arc={dataId['visit']}"
        metadata.add("HISTORY", history)

    def _getMetadataName(self):
        return None


def fitChebyshev2D(xx, yy, xOrder, yOrder, xDomain=None, yDomain=None, rejIterations=3, rejThreshold=3.0):
    """Fit linked 2D Cheyshev polynomials

    This function exists because
    `astropy.modeling.fitting.FittingWithOutlierRejection` doesn't work easily
    for 2 dimensions.

    Parameters
    ----------
    xx : array-like, (2, N)
        Input positions.
    yy : array-like, (2, N)
        Observed positions.
    xOrder, yOrder : `int`
        Polynomial order in x and y.
    xDomain, yDomain : array-like, (2)
        Minimum and maximum values for x and y.
    rejIterations : `int`
        Number of rejection iterations.
    rejThreshold : `float`
        Rejection threshold (stdev).

    Returns
    -------
    xRms, yRms : `float`
        RMS residual in x and y.
    residuals : array-like of `float`, (2, N)
        Fit residuals.
    rms : `float`
        RMS residual offset.
    xFit, yFit : `astropy.modeling.models.Chebyshev2D`
        Fit functions for x and y.
    used : array-like of `bool`, (N)
        Array indicating which positions were used (those that were not used
        were rejected).
    """
    assert len(xx) == len(yy)
    model = Chebyshev2D(xOrder, yOrder, x_domain=xDomain, y_domain=yDomain)
    fitter = LinearLSQFitter()
    good = np.all(np.isfinite(xx), axis=0) & np.all(np.isfinite(yy), axis=0)
    for ii in range(rejIterations):
        xFit = fitter(model, xx[0][good], xx[1][good], yy[0][good])
        yFit = fitter(model, xx[0][good], xx[1][good], yy[1][good])

        xResid = yy[0] - xFit(xx[0], xx[1])
        yResid = yy[1] - yFit(xx[0], xx[1])
        residuals = np.hypot(xResid, yResid)

        stdev = residuals[good].std()
        good = np.abs(residuals) < rejThreshold*stdev

    xFit = fitter(model, xx[0][good], xx[1][good], yy[0][good])
    yFit = fitter(model, xx[0][good], xx[1][good], yy[1][good])
    xResid = yy[0] - xFit(xx[0], xx[1])
    yResid = yy[1] - yFit(xx[0], xx[1])
    residuals = np.hypot(xResid, yResid)

    return SimpleNamespace(xRms=xResid[good].std(), yRms=yResid[good].std(), residuals=residuals,
                           rms=residuals[good].std(), xFit=xFit, yFit=yFit, used=good)
