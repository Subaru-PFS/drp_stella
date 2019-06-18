import os
from types import SimpleNamespace
from operator import attrgetter
import numpy as np
from astropy.modeling.models import Gaussian1D, Chebyshev2D
from astropy.modeling.fitting import LinearLSQFitter, LevMarLSQFitter

from lsst.pipe.base import CmdLineTask, TaskRunner, ArgumentParser, Struct
from lsst.pex.config import Config, Field, ConfigurableField
from lsst.ip.isr import IsrTask
from lsst.utils import getPackageDir

from lsst.obs.pfs.utils import getLampElements

from .findAndTraceAperturesTask import FindAndTraceAperturesTask
from .findLines import FindLinesTask
from .utils import readLineListFile

import lsstDebug


class BootstrapConfig(Config):
    """Configuration for BootstrapTask"""
    isr = ConfigurableField(target=IsrTask, doc="Instrumental signature removal")
    trace = ConfigurableField(target=FindAndTraceAperturesTask, doc="Task to trace apertures")
    minArcLineIntensity = Field(dtype=float, default=0, doc="Minimum 'NIST' intensity to use emission lines")
    findLines = ConfigurableField(target=FindLinesTask, doc="Find arc lines")
    matchRadius = Field(dtype=float, default=5.0, doc="Line matching radius (nm)")
    spatialOrder = Field(dtype=int, default=2, doc="Polynomial order in the spatial dimension")
    spectralOrder = Field(dtype=int, default=2, doc="Polynomial order in the spectral dimension")
    rejIterations = Field(dtype=int, default=3, doc="Number of fitting iterations")
    rejThreshold = Field(dtype=float, default=3.0, doc="Rejection threshold (stdev)")


class BootstrapRunner(TaskRunner):
    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        """Produce list of targets

        We only want to operate on a single flat and single arc, together.
        """
        if len(parsedCmd.flatId.refList) != 1 or len(parsedCmd.arcId.refList) != 1:
            raise RuntimeError("Did not specify a single flat (%d) and a single arc (%d)" %
                               (len(parsedCmd.flatId.refList), len(parsedCmd.arcId.refList)))
        args = parsedCmd.flatId.refList[0]
        kwargs["arcRef"] = parsedCmd.arcId.refList[0]
        kwargs["lineListFilename"] = parsedCmd.lineList
        return [(args, kwargs)]


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
        self.makeSubtask("trace")
        self.makeSubtask("findLines")

    @classmethod
    def _makeArgumentParser(cls):
        """Build a suitable ArgumentParser"""
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--flatId", "raw", help="data ID for flat, e.g., visit=12345")
        parser.add_id_argument("--arcId", "raw", help="data ID for arc, e.g., visit=54321")
        parser.add_argument("--lineList", help="Reference line list",
                            default=os.path.join(getPackageDir("obs_pfs"),
                                                 "pfs", "lineLists", "ArCdHgKrNeXe.txt"))
        return parser

    def run(self, flatRef, arcRef, lineListFilename):
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
        lineListFilename : `str`
            Filename for a reference arc line list.
        """
        flatConfig = flatRef.get("pfsConfig")
        arcConfig = arcRef.get("pfsConfig")
        if not np.all(flatConfig.fiberId == arcConfig.fiberId):
            raise RuntimeError("Mismatch between fibers for flat (%s) and arc (%s)" %
                               (flatConfig.fiberId, arcConfig.fiberId))
        traces = self.traceFibers(flatRef, flatConfig)
        refLines = self.readLines(arcRef, lineListFilename)
        lineResults = self.findArcLines(arcRef, traces)
        matches = self.matchArcLines(lineResults.lines, refLines, lineResults.detectorMap)
        self.fitDetectorMap(matches, lineResults.detectorMap)
        arcRef.put(lineResults.detectorMap, "detectormap")

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
        detMap = flatRef.get("detectormap")
        traces = self.trace.run(exposure.maskedImage, detMap)
        if len(traces) != len(pfsConfig.fiberId):
            raise RuntimeError("Mismatch between number of traces (%d) and number of fibers (%d)" %
                               (len(traces), len(pfsConfig.fiberId)))
        self.log.info("Found %d fibers on flat", len(traces))
        # Assign fiberId from pfsConfig to the fiberTraces, but we have to get the order right!
        # The fiber trace numbers from the left, but the pfsConfig may number from the right.
        middle = 0.5*exposure.getHeight()
        centers = np.array([detMap.getXCenter(ff, middle) for ff in pfsConfig.fiberId])
        increasing = np.all(centers[1:] - centers[:-1] > 0)
        decreasing = np.all(centers[1:] - centers[:-1] < 0)
        assert increasing or decreasing
        for tt, fiberId in zip(traces, pfsConfig.fiberId if increasing else reversed(pfsConfig.fiberId)):
            tt.fiberId = fiberId
        return traces

    def readLines(self, arcRef, lineListFilename):
        """Read reference lines

        Parameters
        ----------
        arcRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for arc.

        Returns
        -------
        lines : `list` of `pfs.drp.stella.ReferenceLine`
            Reference lines.
        """
        metadata = arcRef.get("raw_md")
        lamps = getLampElements(metadata)
        if not lamps:
            raise RuntimeError("No lamps found from metadata")
        return readLineListFile(lineListFilename, lamps, minIntensity=self.config.minArcLineIntensity)

    def findArcLines(self, arcRef, traces):
        """Find lines on the extracted arc spectra

        The x and y centroids are done separately, for convenience: the x
        centroid from the fiber trace, and the y centroid from the extracted
        spectrum.

        Parameters
        ----------
        arcRef : `lsst.daf.persistence.ButlerDataRef`
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
        detMap = arcRef.get("detectormap")
        spectra = traces.extractSpectra(exposure.maskedImage, detMap, True)
        yCenters = [self.findLines.run(ss).lines for ss in spectra]
        xCenters = [self.centroidTrace(tt, yList) for tt, yList in zip(traces, yCenters)]
        lines = [[SimpleNamespace(fiberId=spectrum.fiberId, x=xx, y=yy, flux=spectrum.spectrum[int(yy + 0.5)])
                  for xx, yy in zip(xList, yList)]
                 for xList, yList, spectrum in zip(xCenters, yCenters, spectra)]
        self.log.info("Found %d lines in %d traces", sum(len(ll) for ll in lines), len(lines))
        return Struct(spectra=spectra, lines=lines, detectorMap=detMap)

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
                ref = max(candidates, key=attrgetter("guessedIntensity"))
                matches.append(SimpleNamespace(obs=line, ref=ref))
                used.add(ref.wavelength)
        self.log.info("Matched %d lines", len(matches))
        return matches

    def fitDetectorMap(self, matches, detectorMap, doUpdate=True):
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
        doUpdate : `bool`
            Update the ``detectorMap``?
        """
        xDomain = [detectorMap.bbox.getMinX(), detectorMap.bbox.getMaxX()]
        yDomain = [detectorMap.bbox.getMinY(), detectorMap.bbox.getMaxY()]
        xx = np.array([detectorMap.findPoint(mm.obs.fiberId, mm.ref.wavelength)
                       for mm in matches]).T  # Where it should be
        yy = np.array([[mm.obs.x, mm.obs.y] for mm in matches]).T  # Where it is

        if lsstDebug.Info(__name__).plotShifts:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            axes.quiver(xx[0], xx[1], yy[0] - xx[0], yy[1] - yy[1])
            axes.set_xlabel("Spatial")
            axes.set_ylabel("Spectral")
            plt.show()

        diff = yy - xx
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
            fig, axes = plt.subplots(2, 2)

            dSpatial = yy[0] - fitSpatial(xx[0], xx[1])
            dSpectral = yy[1] - fitSpectral(xx[0], xx[1])

            axes[0, 0].scatter(yy[0][good], dSpatial[good], color="k")
            axes[0, 0].scatter(yy[0][~good], dSpatial[~good], color="r")
            axes[0, 0].set_xlabel("Spatial")
            axes[0, 0].set_ylabel(r"\Delta Spatial")

            axes[0, 1].scatter(yy[0][good], dSpectral[good], color="k")
            axes[0, 1].scatter(yy[0][~good], dSpectral[~good], color="r")
            axes[0, 1].set_xlabel("Spatial")
            axes[0, 1].set_ylabel(r"\Delta Spectral")

            axes[1, 0].scatter(yy[1][good], dSpatial[good], color="k")
            axes[1, 0].scatter(yy[1][~good], dSpatial[~good], color="r")
            axes[1, 0].set_xlabel("Spectral")
            axes[1, 0].set_ylabel(r"\Delta Spatial")

            axes[1, 1].scatter(yy[1][good], dSpectral[good], color="k")
            axes[1, 1].scatter(yy[1][~good], dSpectral[~good], color="r")
            axes[1, 1].set_xlabel("Spectral")
            axes[1, 0].set_ylabel(r"\Delta Spectral")

            plt.subplots_adjust()
            plt.show()

            fig = plt.figure()
            axes = fig.add_subplot(1, 1, 1)
            axes.quiver(xx[0][good], xx[1][good], yy[0][good] - xx[0][good], yy[1][good] - yy[1][good],
                        color="k")
            axes.quiver(xx[0][~good], xx[1][~good], yy[0][~good] - xx[0][~good], yy[1][~good] - yy[1][~good],
                        color="r")
            axes.set_xlabel("Spatial")
            axes.set_ylabel("Spectral")
            plt.show()

        # Update the detectorMap
        if doUpdate:
            self.log.info("Updating detectorMap...")
            rows = np.arange(detectorMap.bbox.getMaxY() + 1, dtype=np.float32)
            for fiberId in detectorMap.fiberIds:
                wavelength = detectorMap.getWavelength(fiberId)
                assert len(wavelength) == len(rows)
                center = detectorMap.getXCenter(fiberId)
                assert len(center) == len(rows)
                spatial = fitSpatial(center, rows)
                spectral = fitSpectral(center, rows)
                detectorMap.setXCenter(fiberId, rows, spatial.astype(np.float32))
                detectorMap.setWavelength(fiberId, spectral.astype(np.float32), wavelength.astype(np.float32))

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
    good = np.ones(xx.shape[1], dtype=bool)
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
