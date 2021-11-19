import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.optimize

import lsstDebug

from lsst.utils import getPackageDir
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task, Struct
from lsst.geom import Box2D

from pfs.drp.stella import DetectorMap, DoubleDetectorMap, DoubleDistortion
from .arcLine import ArcLineSet
from .referenceLine import ReferenceLineStatus
from .utils.math import robustRms


__all__ = ("FitDistortedDetectorMapConfig", "FitDistortedDetectorMapTask", "FittingError")


@dataclass
class ArcLineResiduals:
    """Residuals in arc line positions

    Analagous to `ArcLine`, this stores the position measurement of a single
    arc line, but the ``x,y`` positions are relative to a detectorMap. The
    original ``x,y`` positions are stored as ``xOrig,yOrig``.

    Parameters
    ----------
    fiberId : `int`
        Fiber identifier.
    wavelength : `float`
        Reference line wavelength (nm).
    x, y : `float`
        Differential position relative to an external detectorMap.
    xOrig, yOrig : `float`
        Measured position.
    xBase, yBase : `float`
        Expected position from base detectorMap.
    xErr, yErr : `float`
        Error in measured position.
    intensity : `float`
        Measured intensity (arbitrary units).
    intensityErr : `float`
        Error in measured intensity (arbitrary units).
    flag : `bool`
        Measurement flag (``True`` indicates an error in measurement).
    status : `pfs.drp.stella.ReferenceLine.Status`
        Flags whether the lines are fitted, clipped or reserved etc.
    description : `str`
        Line description (e.g., ionic species)
    """
    fiberId: int
    wavelength: float
    x: float
    y: float
    xOrig: float
    yOrig: float
    xBase: float
    yBase: float
    xErr: float
    yErr: float
    intensity: float
    intensityErr: float
    flag: bool
    status: int
    description: str


class ArcLineResidualsSet(ArcLineSet):
    """A list of `ArcLineResiduals`

    Analagous to `ArcLineSet`, this stores the position measurement of a list
    of arc lines, but the ``x,y`` positions are relative to a detectorMap. The
    original ``x,y`` positions are stored as ``xOrig,yOrig``.

    Parameters
    ----------
    lines : `list` of `ArcLineResiduals`
        List of lines in the spectra.
    """
    RowClass = ArcLineResiduals
    schema = (("fiberId", np.int32),
              ("wavelength", float),
              ("x", float),
              ("y", float),
              ("xOrig", float),
              ("yOrig", float),
              ("xBase", float),
              ("yBase", float),
              ("xErr", float),
              ("yErr", float),
              ("intensity", float),
              ("intensityErr", float),
              ("flag", bool),
              ("status", np.int32),
              ("description", str),
              )

    def append(self, fiberId, wavelength, x, y, xOrig, yOrig, xBase, yBase, xErr, yErr,
               intensity, intensityErr, flag, status, description):
        """Append to the list of lines

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.
        wavelength : `float`
            Reference line wavelength (nm).
        x, y : `float`
            Differential position relative to an external detectorMap.
        xOrig, yOrig : `float`
            Measured position.
        xBase, yBase : `float`
            Expected position from base detectorMap.
        xErr, yErr : `float`
            Error in measured position.
        intensity : `float`
            Measured intensity (arbitrary units).
        intensityErr : `float`
            Error in measured intensity (arbitrary units).
        flag : `bool`
            Measurement flag (``True`` indicates an error in measurement).
        status : `pfs.drp.stella.ReferenceLine.Status`
            Flags whether the lines are fitted, clipped or reserved etc.
        description : `str`
            Line description (e.g., ionic species)
        """
        self.data = self.data.append(dict(fiberId=fiberId, wavelength=wavelength, x=x, y=y,
                                          xOrig=xOrig, yOrig=yOrig, xBase=xBase, yBase=yBase,
                                          xErr=xErr, yErr=yErr, intensity=intensity,
                                          intensityErr=intensityErr, flag=flag, status=status,
                                          description=description), ignore_index=True)

    @property
    def xOrig(self):
        """Array of original x position (`numpy.ndarray` of `float`)"""
        return self.data["xOrig"].values

    @property
    def yOrig(self):
        """Array of original y position (`numpy.ndarray` of `float`)"""
        return self.data["yOrig"].values

    @property
    def xBase(self):
        """Array of expected x position (`numpy.ndarray` of `float`)"""
        return self.data["xBase"].values

    @property
    def yBase(self):
        """Array of expected y position (`numpy.ndarray` of `float`)"""
        return self.data["yBase"].values

    @classmethod
    def readFits(cls, filename):
        """Read from FITS file

        Not implemented, because we don't expect to write this.
        """
        raise NotImplementedError("Not implemented")

    def writeFits(self, filename):
        """Write to FITS file

        Not implemented, because we don't expect to write this.
        """
        raise NotImplementedError("Not implemented")


def fitStraightLine(xx, yy):
    """Fit a straight line, y = slope*x + intercept

    Parameters
    ----------
    xx : `numpy.ndarray` of `float`, size ``N``
        Ordinate.
    yy : `numpy.ndarray` of `float`, size ``N``
        Co-ordinate.

    Returns
    -------
    slope : `float`
        Slope of line.
    intercept : `float`
        Intercept of line.
    xMean : `float`
        Mean of x values.
    yMean : `float`
        Mean of y values.
    """
    xMean = xx.mean()
    yMean = yy.mean()
    dx = xx - xMean
    dy = yy - yMean
    xySum = np.sum(dx*dy)
    xxSum = np.sum(dx**2)
    slope = xySum/xxSum
    intercept = yMean - slope*xMean
    return Struct(slope=slope, intercept=intercept, xMean=xMean, yMean=yMean)


def calculateFitStatistics(fit, lines, selection, numParameters, soften=(0.0, 0.0), maxSoften=1.0, **kwargs):
    """Calculate statistics of the distortion fit

    Parameters
    ----------
    fit : `numpy.ndarray` of `float`, shape ``(N, 2)``
        Fit positions.
    lines : `pfs.drp.stella.ArcLineSet`
        Arc line measurements.
    selection : `numpy.ndarray` of `bool`
        Flags indicating which of the ``lines`` are to be used in the
        calculation.
    numParameters : `int`
        Number of parameters in fit.
    soften : `tuple` (`float`, `float`), optional
        Systematic error in x and y to add in quadrature to measured errors
        (pixels).
    maxSoften : `float`
        Maximum softening value to consider.
    **kwargs
        Additional elements to add to results.

    Returns
    -------
    xResid, yResid : `numpy.ndarray` of `float`
        Fit residual in x,y for each of the ``lines`` (pixels).
    xRms, yRms : `float`
        Weighted RMS residual in x,y (pixels).
    xRobustRms, yRobustRms : `float`
        Robust RMS (from IQR) residual in x,y (pixels).
    chi2 : `float`
        Fit chi^2.
    dof : `float`
        Degrees of freedom.
    numParameters : `int`
        Number of parameters in fit.
    selection : `numpy.ndarray` of `bool`
        Selection used in calculating statistics.
    soften : `tuple` (`float`, `float`), optional
        Systematic error in x and y that was applied to measured errors
        (pixels) in chi^2 calculation.
    xSoften, ySoften : `float`
        Calculated systematic errors required to soften errors to attain
        chi^2/dof = 1.
    """
    xResid = (lines.x - fit[:, 0])
    yResid = (lines.y - fit[:, 1])

    xRobustRms = robustRms(xResid[selection])
    yRobustRms = robustRms(yResid[selection])

    xResid2 = xResid[selection]**2
    yResid2 = yResid[selection]**2
    xErr2 = lines.xErr[selection]**2 + soften[0]**2
    yErr2 = lines.yErr[selection]**2 + soften[1]**2

    xWeight = 1.0/xErr2
    yWeight = 1.0/yErr2
    xWeightedRms = np.sqrt(np.sum(xWeight*xResid2)/np.sum(xWeight))
    yWeightedRms = np.sqrt(np.sum(yWeight*yResid2)/np.sum(yWeight))

    chi2 = np.sum(xResid2/xErr2 + yResid2/yErr2)
    dof = 2*selection.sum() - numParameters

    def calculateSoftening(residuals, errors, dof):
        """Calculate systematic error that softens chi^2/dof to 1

        Parameters
        ----------
        residuals : `numpy.ndarray` of `float`
            Residual values.
        errors : `numpy.ndarray` of `float`
            Estimated error values.
        dof : `int`
            Number of degrees of freedom.

        Returns
        -------
        softenChi2 : callable
            Function that calculates chi^2 given a softening parameter.
        """
        residuals2 = residuals**2
        errors2 = errors**2

        def softenChi2(soften):
            """Return chi^2/dof (minus 1) with the softening applied

            This allows us to find the softening parameter that results in
            chi^2/dof = 1 by bisection.

            Parameters
            ----------
            soften : `float`
                Systematic error to add in quadrature to measured errors
                (pixels).

            Returns
            -------
            chi2 : `float`
                chi^2/dof - 1
            """
            return np.sum(residuals2/(soften**2 + errors2))/dof - 1

        if softenChi2(0.0) < 0:
            return 0.0
        if softenChi2(maxSoften) > 0:
            return np.nan
        return scipy.optimize.bisect(softenChi2, 0.0, maxSoften)

    dimDof = selection.sum() - numParameters/2  # Assume equipartition of number of parameters between dims
    xSoften = calculateSoftening(xResid[selection], lines.xErr[selection], dimDof)
    ySoften = calculateSoftening(yResid[selection], lines.yErr[selection], dimDof)

    return Struct(xResid=xResid, yResid=yResid, xRms=xWeightedRms, yRms=yWeightedRms,
                  xRobustRms=xRobustRms, yRobustRms=yRobustRms, chi2=chi2, dof=dof,
                  numParameters=numParameters, selection=selection, soften=soften,
                  xSoften=xSoften, ySoften=ySoften, **kwargs)


def addColorbar(figure, axes, cmap, norm, label=None):
    """Add colorbar to a plot

    Parameters
    ----------
    figure : `matplotlib.pyplot.Figure`
        Figure containing the axes.
    axes : `matplotlib.pyplot.Axes`
        Axes with the plot.
    cmap : `matplotlib.colors.Colormap`
        Color map.
    norm : `matplot.colors.Normalize`
        Normalization for color map.
    label : `str`
        Label to apply to colorbar.

    Returns
    -------
    colorbar : `matplotlib.colorbar.Colorbar`
        The colorbar.
    """
    import matplotlib.cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size='5%', pad=0.05)
    colors = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    colors.set_array([])
    figure.colorbar(colors, cax=cax, orientation="vertical", label=label)


class FittingError(RuntimeError):
    """Error in fitting distortion model"""
    pass


class FitDistortedDetectorMapConfig(Config):
    """Configuration for FitDistortedDetectorMapTask"""
    lineFlags = ListField(dtype=str, default=["BAD"], doc="ReferenceLineStatus flags for lines to ignore")
    iterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejection = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    order = Field(dtype=int, default=4, doc="Distortion order")
    reserveFraction = Field(dtype=float, default=0.1, doc="Fraction of lines to reserve in the final fit")
    soften = Field(dtype=float, default=0.03, doc="Systematic error to apply")
    lsqThreshold = Field(dtype=float, default=1.0e-6, doc="Eigenvaluethreshold for solving least-squares")
    doSlitOffsets = Field(dtype=bool, default=False, doc="Fit for new slit offsets?")
    base = Field(dtype=str,
                 doc="Template for base detectorMap; should include '%%(arm)s' and '%%(spectrograph)s'",
                 default=os.path.join(getPackageDir("drp_pfs_data"), "detectorMap",
                                      "detectorMap-sim-%(arm)s%(spectrograph)s.fits")
                 )
    minSignalToNoise = Field(dtype=float, default=20.0,
                             doc="Minimum (intensity) signal-to-noise ratio of lines to fit")
    minNumWavelengths = Field(dtype=int, default=3, doc="Required minimum number of discrete wavelengths")


class FitDistortedDetectorMapTask(Task):
    ConfigClass = FitDistortedDetectorMapConfig
    _DefaultName = "fitDetectorMap"
    DetectorMap = DoubleDetectorMap
    Distortion = DoubleDistortion

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, dataId, bbox, lines, visitInfo, metadata=None,
            spatialOffsets=None, spectralOffsets=None, base=None):
        """Fit a DistortionBasedDetectorMap to arc line measurements

        Parameters
        ----------
        dataId : `dict`
            Data identifier. Should contain at least ``arm`` (`str`; one of
            ``b``, ``r``, ``n``, ``m``) and ``spectrograph`` (`int`).
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information for exposure.
        metadata : `lsst.daf.base.PropertyList`, optional
            DetectorMap metadata (FITS header).
        spatialOffsets, spectralOffsets : `numpy.ndarray` of `float`
            Spatial and spectral offsets to use, if ``doSlitOffsets=False``.
        base : `pfs.drp.stella.SplinedDetectorMap`, optional
            Base detectorMap. If not provided, one pointed to in the config will
            be read in.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        model : `pfs.drp.stella.GlobalDetectorModel`
            Model that was fit to the data.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Residual RMS in x,y (pixels)
        chi2 : `float`
            Fit chi^2.
        soften : `tuple` (`float`, `float`), optional
            Systematic error in x and y that was applied to measured errors
            (pixels).
        used : `numpy.ndarray` of `bool`
            Array indicating which lines were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Array indicating which lines were reserved from the fit.

        Raises
        ------
        FittingError
            If the data is not of sufficient quality to fit.
        """
        if base is None:
            base = self.getBaseDetectorMap(dataId)
        self.copySlitOffsets(base, spatialOffsets, spectralOffsets)
        residuals = self.calculateBaseResiduals(base, lines)
        dispersion = base.getDispersion(base.fiberId[len(base)//2])
        results = self.fitDistortion(bbox, residuals, dispersion, seed=visitInfo.getExposureId())
        detectorMap = self.DetectorMap(base, results.distortion, visitInfo, metadata)
        numParameters = results.numParameters
        if self.config.doSlitOffsets:
            offsets = self.measureSlitOffsets(detectorMap, self.calculateBaseResiduals(detectorMap, lines),
                                              results.selection)
            numParameters += offsets.numParameters

        results = calculateFitStatistics(detectorMap.findPoint(lines.fiberId, lines.wavelength),
                                         lines, results.selection, numParameters,
                                         detectorMap=detectorMap)
        self.log.info("Final result: chi2=%f dof=%d xRMS=%f yRMS=%f xSoften=%f ySoften=%f from %d lines",
                      results.chi2, results.dof, results.xRms, results.yRms,
                      results.xSoften, results.ySoften, results.selection.sum())
        if self.debugInfo.lineQa:
            self.lineQa(lines, detectorMap)
        if self.debugInfo.wlResid:
            self.plotWavelengthResiduals(detectorMap, lines, results.selection, results.reserved)
        return results

    def getBaseDetectorMap(self, dataId):
        """Provide the detectorMap on which this will be based

        We retrieve the detectorMap by filename (through the config). This
        might be upgraded later to use the butler.

        Parameters
        ----------
        dataId : `dict`
            Data identifier. Should include ``arm`` (`str`) and ``spectrograph``
            (`int`) in order to allow selection of different detectorMaps for
            different gratings.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Base detectorMap.
        """
        filename = self.config.base % dataId
        return DetectorMap.readFits(filename)

    def getGoodLines(self, lines: ArcLineSet):
        """Return a boolean array indicating which lines are good.

        Parameters
        ----------
        lines : `ArcLineSet`
            Line measurements.

        Returns
        -------
        good : `numpy.ndarray` of `bool`
            Boolean array indicating which lines are good.
        """
        self.log.debug("%d lines in list", len(lines))
        good = lines.flag == 0
        self.log.debug("%d good lines after measurement flags", good.sum())
        good &= (lines.status & ReferenceLineStatus.fromNames(*self.config.lineFlags)) == 0
        self.log.debug("%d good lines after line status", good.sum())
        good &= np.isfinite(lines.x) & np.isfinite(lines.y)
        good &= np.isfinite(lines.xErr) & np.isfinite(lines.yErr)
        self.log.debug("%d good lines after finite positions", good.sum())
        if self.config.minSignalToNoise > 0:
            good &= np.isfinite(lines.intensity) & np.isfinite(lines.intensityErr)
            self.log.debug("%d good lines after finite intensities", good.sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                good &= (lines.intensity/lines.intensityErr) > self.config.minSignalToNoise
            self.log.debug("%d good lines after signal-to-noise", good.sum())
        return good

    def measureSlitOffsets(self, detectorMap, lines, select):
        """Measure slit offsets for base detectorMap

        The detectorMap is modified in-place.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Base detectorMap.
        lines : `ArcLineSet`
            Original line measurements (NOT the residuals).
        select : `numpy.ndarray` of `bool`
            Boolean array indicating which lines to use.

        Returns
        -------
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Weighted RMS residual in x,y (pixels).
        xRobustRms, yRobustRms : `float`
            Robust RMS (from IQR) residual in x,y (pixels).
        chi2 : `float`
            Fit chi^2.
        dof : `float`
            Degrees of freedom.
        soften : `tuple` (`float`, `float`), optional
            Systematic error in x and y that was applied to measured errors
            (pixels).
        """
        sysErr = self.config.soften
        numFibers = len(detectorMap)
        fiberId = lines.fiberId
        xy = detectorMap.findPoint(fiberId, lines.wavelength)
        dx = xy[:, 0] - lines.xOrig
        dy = xy[:, 1] - lines.yOrig
        xErr = np.hypot(lines.xErr, sysErr)
        yErr = np.hypot(lines.yErr, sysErr)
        fit = np.full((len(lines), 2), np.nan, dtype=float)

        use = select.copy()
        for ii in range(self.config.iterations):
            spatial = np.zeros(numFibers, dtype=float)
            spectral = np.zeros(numFibers, dtype=float)
            noMeasurements = set()
            for jj, ff in enumerate(detectorMap.getFiberId()):
                choose = use & (fiberId == ff)
                if not np.any(choose):
                    noMeasurements.add(ff)
                    continue
                # Robust measurement
                spatial[jj] = np.median(dx[choose])
                spectral[jj] = np.median(dy[choose])
                fit[choose, 0] = spatial[jj]
                fit[choose, 1] = spectral[jj]

            result = calculateFitStatistics(fit, lines, use, 2*(numFibers - len(noMeasurements)),
                                            (sysErr, sysErr))
            self.log.debug(
                "Slit offsets iteration %d: chi2=%f dof=%d xRMS=%f yRMS=%f from %d/%d lines",
                ii, result.chi2, result.dof, result.xRms, result.yRms, use.sum(), select.sum()
            )
            self.log.debug("Unable to measure slit offsets for %d fiberIds: %s",
                           len(noMeasurements), sorted(noMeasurements))
            self.log.debug("Spatial offsets: %s", spatial)
            self.log.debug("Spectral offsets: %s", spectral)

            with np.errstate(invalid="ignore"):
                newUse = (select & (np.abs(result.xResid/xErr) < self.config.rejection) &
                          (np.abs(result.yResid/yErr) < self.config.rejection))
            self.log.debug("Rejecting %d/%d lines in iteration %d", use.sum() - newUse.sum(), use.sum(), ii)
            if np.all(newUse == use):
                # Converged
                break
            use = newUse

        # Final fit, with more precise measurement
        spatial = np.zeros(numFibers, dtype=float)
        spectral = np.zeros(numFibers, dtype=float)
        fit = np.zeros_like(xy)
        noMeasurements = set()
        for jj, ff in enumerate(detectorMap.getFiberId()):
            choose = use & (fiberId == ff)
            if not np.any(choose):
                noMeasurements.add(ff)
                continue
            with np.errstate(divide="ignore"):
                spatial[jj] = -np.average(dx[choose], weights=1.0/xErr[choose]**2)
                spectral[jj] = -np.average(dy[choose], weights=1.0/yErr[choose]**2)
                fit[choose, 0] = spatial[jj]
                fit[choose, 1] = spectral[jj]

        result = calculateFitStatistics(fit, lines, use, 2*(numFibers - len(noMeasurements)),
                                        (sysErr, sysErr))
        self.log.info(
            "Slit offsets measurement: chi2=%f dof=%d xRMS=%f yRMS=%f xSoften=%f ySoften=%f from %d/%d lines",
            result.chi2, result.dof, result.xRms, result.yRms, result.xSoften, result.ySoften,
            use.sum(), select.sum()
        )
        self.log.info("Unable to measure slit offsets for %d fiberIds: %s",
                      len(noMeasurements), sorted(noMeasurements))
        self.log.debug("Spatial offsets: %s", spatial)
        self.log.debug("Spectral offsets: %s", spectral)

        detectorMap.setSlitOffsets(detectorMap.getSpatialOffsets() + spatial,
                                   detectorMap.getSpectralOffsets() + spectral)
        return result

    def copySlitOffsets(self, detectorMap, spatialOffsets, spectralOffsets):
        """Set specified slit offsets

        Parameters
        ----------
        base : `pfs.drp.stella.SplinedDetectorMap`, optional
            Base detectorMap. If not provided, one pointed to in the config will
            be read in.
        spatialOffsets, spectralOffsets : `numpy.ndarray` of `float`
            Spatial and spectral offsets to use.
        """
        if spatialOffsets is None or spectralOffsets is None:
            self.log.warn("No slit offsets provided")
            return
        if (np.all(spatialOffsets == 0) or np.all(spectralOffsets == 0)) and not self.config.doSlitOffsets:
            self.log.warn("All provided slit offsets are zero; consider using doSlitOffsets=True")
        detectorMap.setSlitOffsets(spatialOffsets, spectralOffsets)

    def calculateBaseResiduals(self, detectorMap, lines):
        """Calculate position residuals w.r.t. base detectorMap

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Base detectorMap.
        lines : `ArcLineSet`
            Original line measurements (NOT the residuals).

        Returns
        -------
        residuals : `ArcLineResidualsSet`
            Arc line position residuals.
        """
        points = detectorMap.findPoint(lines.fiberId, lines.wavelength)
        return ArcLineResidualsSet.fromArrays(
            fiberId=lines.fiberId,
            wavelength=lines.wavelength,
            x=lines.x - points[:, 0],
            y=lines.y - points[:, 1],
            xOrig=lines.x,
            yOrig=lines.y,
            xBase=points[:, 0],
            yBase=points[:, 1],
            xErr=lines.xErr,
            yErr=lines.yErr,
            intensity=lines.intensity,
            intensityErr=lines.intensityErr,
            flag=lines.flag,
            status=lines.status,
            description=lines.description,
        )

    def fitDistortion(self, bbox, lines, dispersion, seed=0, fitStatic=True, Distortion=None):
        """Fit a distortion model

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        dispersion : `float`
            Wavelength dispersion (nm/pixel); for interpreting RMS in logging.
        seed : `int`
            Seed for random number generator used for selecting reserved lines.
        fitStatic : `bool`, optional
            Fit static components to the distortion model?
        Distortion : subclass of `pfs.drp.stella.BaseDistortion`
            Class to use for distortion. If ``None``, uses ``self.Distortion``.

        Returns
        -------
        distortion : Distortion
            Model that was fit to the data.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Residual RMS in x,y (pixels)
        chi2 : `float`
            Fit chi^2.
        soften : `tuple` (`float`, `float`), optional
            Systematic error in x and y that was applied to measured errors
            (pixels).
        used : `numpy.ndarray` of `bool`
            Array indicating which lines were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Array indicating which lines were reserved from the fit.

        Raises
        ------
        FittingError
            If the data is not of sufficient quality to fit.
        """
        good = self.getGoodLines(lines)
        numGood = good.sum()

        rng = np.random.RandomState(seed)
        numReserved = int(self.config.reserveFraction*numGood + 0.5)
        reservedIndices = rng.choice(np.arange(numGood, dtype=int), replace=False, size=numReserved)
        reserved = np.zeros_like(good)
        select = np.zeros(numGood, dtype=bool)
        select[reservedIndices] = True
        reserved[good] = select
        xErr = np.hypot(lines.xErr, self.config.soften)
        yErr = np.hypot(lines.yErr, self.config.soften)

        used = good & ~reserved
        result = None
        for ii in range(self.config.iterations):
            result = self.fitModel(bbox, lines, used, fitStatic=fitStatic, Distortion=Distortion)
            self.log.debug(
                "Fit iteration %d: chi2=%f dof=%d xRMS=%f yRMS=%f (%f nm) from %d/%d lines",
                ii, result.chi2, result.dof, result.xRms, result.yRms, result.yRms*dispersion, used.sum(),
                numGood - numReserved
            )
            self.log.debug("Fit iteration %d: %s", ii, result.distortion)
            if self.debugInfo.plot:
                self.plotModel(lines, used, result)
            if self.debugInfo.residuals:
                self.plotResiduals(result.distortion, lines, used, reserved)
            with np.errstate(invalid="ignore"):
                newUsed = (good & ~reserved & (np.abs(result.xResid/xErr) < self.config.rejection) &
                           (np.abs(result.yResid/yErr) < self.config.rejection))
            self.log.debug("Rejecting %d/%d lines in iteration %d", used.sum() - newUsed.sum(),
                           used.sum(), ii)
            if np.all(newUsed == used):
                # Converged
                break
            used = newUsed

        result = self.fitModel(bbox, lines, used, fitStatic=fitStatic, Distortion=Distortion)
        self.log.info("Final fit: "
                      "chi2=%f dof=%d xRMS=%f yRMS=%f (%f nm) xSoften=%f ySoften=%f from %d/%d lines",
                      result.chi2, result.dof, result.xRms, result.yRms, result.yRms*dispersion,
                      result.xSoften, result.ySoften, used.sum(), numGood - numReserved)
        reservedStats = calculateFitStatistics(result.distortion(lines.xBase, lines.yBase), lines, reserved,
                                               result.distortion.getNumParameters(),
                                               (self.config.soften, self.config.soften),
                                               distortion=result.distortion)
        self.log.info("Fit quality from reserved lines: "
                      "chi2=%f xRMS=%f yRMS=%f (%f nm) xSoften=%f ySoften=%f from %d lines (%.1f%%)",
                      reservedStats.chi2, reservedStats.xRobustRms, reservedStats.yRobustRms,
                      reservedStats.yRobustRms*dispersion, reservedStats.xSoften, reservedStats.ySoften,
                      reserved.sum(), reserved.sum()/numGood*100)
        self.log.debug("    Final fit model: %s", result.distortion)

        soften = (result.xSoften, result.ySoften)
        result = self.fitModel(bbox, lines, used, soften, fitStatic=fitStatic, Distortion=Distortion)
        self.log.info("Softened fit: "
                      "chi2=%f dof=%d xRMS=%f yRMS=%f (%f nm) xSoften=%f ySoften=%f from %d lines",
                      result.chi2, result.dof, result.xRms, result.yRms, result.yRms*dispersion,
                      result.xSoften, result.ySoften, select.sum())

        reservedStats = calculateFitStatistics(result.distortion(lines.xBase, lines.yBase), lines, reserved,
                                               result.distortion.getNumParameters(), soften,
                                               distortion=result.distortion)
        self.log.info("Softened fit quality from reserved lines: "
                      "chi2=%f xRMS=%f yRMS=%f (%f nm) xSoften=%f ySoften=%f from %d lines",
                      reservedStats.chi2, reservedStats.xRobustRms, reservedStats.yRobustRms,
                      reservedStats.yRobustRms*dispersion, reservedStats.xSoften, reservedStats.ySoften,
                      reserved.sum())
        self.log.debug("    Softened fit model: %s", result.distortion)

        result.reserved = reserved
        if self.debugInfo.plot:
            self.plotModel(lines, used, result)
        if self.debugInfo.distortion:
            self.plotDistortion(result.distortion, lines, used)
        if self.debugInfo.residuals:
            self.plotResiduals(result.distortion, lines, used, reserved)
        return result

    def fitModel(self, bbox, lines, select, soften=None, fitStatic=True, Distortion=None):
        """Fit a model to the arc lines

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        select : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        soften : `tuple` (`float`, `float`), optional
            Systematic error in x and y to add in quadrature to measured errors
            (pixels).
        fitStatic : `bool`, optional
            Fit static components to the distortion model?
        Distortion : subclass of `pfs.drp.stella.BaseDistortion`
            Class to use for distortion. If ``None``, uses ``self.Distortion``.

        Returns
        -------
        distortion : `pfs.drp.stella.DoubleDistortion`
            Distortion model that was fit to the data.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Residual RMS in x,y (pixels)
        chi2 : `float`
            Fit chi^2.
        soften : `float`
            Systematic error that was applied to measured errors (pixels).

        Raises
        ------
        FittingError
            If the data is not of sufficient quality to fit.
        """
        if not np.any(select):
            raise FittingError("No selected lines to fit")
        numWavelengths = len(set(lines.wavelength[select]))
        if numWavelengths < self.config.minNumWavelengths:
            raise FittingError(f"Insufficient discrete wavelengths ({numWavelengths} vs "
                               f"{self.config.minNumWavelengths} required)")
        if Distortion is None:
            Distortion = self.Distortion
        if soften is None:
            soften = (self.config.soften, self.config.soften)
        xSoften, ySoften = soften

        xx = lines.x[select].astype(float)
        yy = lines.y[select].astype(float)
        xBase = lines.xBase[select]
        yBase = lines.yBase[select]
        xErr = np.hypot(lines.xErr[select].astype(float), xSoften)
        yErr = np.hypot(lines.yErr[select].astype(float), ySoften)

        distortion = Distortion.fit(self.config.order, Box2D(bbox), xBase, yBase,
                                    xx, yy, xErr, yErr, fitStatic, self.config.lsqThreshold)
        return calculateFitStatistics(distortion(lines.xBase, lines.yBase), lines, select,
                                      distortion.getNumParameters(), soften, distortion=distortion)

    def lineQa(self, lines, detectorMap):
        """Check the quality of the model fit by looking at the lines

        We match the same lines from different fibers and compare the RMS with
        the measurement errors. The results are logged and plotted.

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        """
        matches = defaultdict(list)
        num = len(lines.fiberId)
        for ii in range(num):
            matches[lines.wavelength[ii]].append(ii)
        model = detectorMap.model
        stdev = []
        error = []
        for wl in sorted(matches.keys()):
            if len(matches[wl]) < 10:
                continue
            indices = np.array(matches[wl])
            fit = np.array([detectorMap.findWavelength(lines.fiberId[ii], lines.y[ii]) for ii in indices])
            resid = (fit - wl)/model.getScaling().dispersion
            self.log.info("Line %f: rms residual=%f, mean error=%f num=%d",
                          wl, robustRms(resid), np.median(lines.yErr[indices]), len(indices))
            stdev.append(robustRms(resid))
            error.append(np.median(lines.yErr[indices]))

        import matplotlib.pyplot as plt
        plt.plot(error, stdev, 'ko')
        plt.xlabel("Median centroid error")
        plt.ylabel("Fit RMS")
        plt.show()

    def plotModel(self, lines, good, result):
        """Plot the model fit result

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        good : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        result : `lsst.pipe.base.Struct`
            Results from ``fitModel``.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from matplotlib.colors import Normalize
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        xErr = lines.xErr
        yErr = lines.yErr
        if result.soften > 0:
            xErr = np.hypot(result.soften, xErr)
            yErr = np.hypot(result.soften, yErr)

        xResid = result.xResid/xErr
        yResid = result.yResid/yErr

        fig, axes = plt.subplots(2, 3)
        axes[0, 0].scatter(lines.x[good], xResid[good], marker=".", color="b")
        axes[0, 0].scatter(lines.x[~good], xResid[~good], marker=".", color="r")
        axes[0, 0].set_xlabel("Spatial")
        axes[0, 0].set_ylabel(r"\Delta Spatial (\sigma)")
        axes[0, 0].set_title("Spatial residuals")

        axes[1, 0].scatter(lines.y[good], yResid[good], marker=".", color="b")
        axes[1, 0].scatter(lines.y[~good], yResid[~good], marker=".", color="r")
        axes[1, 0].set_xlabel("Spectral")
        axes[1, 0].set_ylabel(r"\Delta Spectral (\sigma)")
        axes[1, 0].set_title("Spectral residuals")

        numFibers = result.distortion.getNumFibers()
        numLines = min(10, numFibers)
        fiberId = result.distortion.getFiberId()[np.linspace(0, numFibers, numLines, False, dtype=int)]
        wavelength = np.linspace(lines.wavelength.min(), lines.wavelength.max(), numLines)
        ff, wl = np.meshgrid(fiberId, wavelength)
        ff = ff.flatten()
        wl = wl.flatten()
        xy = result.distortion(ff, wl)
        xx = xy[:, 0]
        yy = xy[:, 1]

        for ii in range(numLines):
            select = ff == fiberId[ii]
            axes[0, 1].plot(xx[select], yy[select], 'k-')
            axes[1, 1].plot(xx[select], yy[select], 'k-')
            select = wl == wavelength[ii]
            axes[0, 1].plot(xx[select], yy[select], 'k-')
            axes[1, 1].plot(xx[select], yy[select], 'k-')
        axes[0, 1].scatter(lines.x[good], lines.y[good], marker=".", color="b")
        axes[0, 1].set_title("Good")
        axes[0, 1].set_xlabel("Spatial")
        axes[0, 1].set_ylabel("Spectral")
        axes[1, 1].scatter(lines.x[~good], lines.y[~good], marker=".", color="r")
        axes[1, 1].set_title("Rejected")
        axes[1, 1].set_xlabel("Spatial")
        axes[1, 1].set_ylabel("Spectral")

        cmap = matplotlib.cm.bwr

        norm = Normalize()
        norm.autoscale(xResid[good])
        ax = axes[0, 2]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax.scatter(lines.x[good], lines.y[good], marker=".", color=cmap(norm(xResid[good])))
        colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colors.set_array([])
        fig.colorbar(colors, cax=cax, orientation="vertical", label=r"\Delta Spatial (\sigma)")
        ax.set_title("Spatial offset")
        ax.set_xlabel("Spatial")
        ax.set_ylabel("Spectral")

        norm = Normalize()
        norm.autoscale(yResid[good])
        ax = axes[1, 2]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax.scatter(lines.x[good], lines.y[good], marker=".", color=cmap(norm(yResid[good])))
        colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colors.set_array([])
        fig.colorbar(colors, cax=cax, orientation="vertical", label=r"\Delta Spectral (\sigma)")
        ax.set_title("Spectral offset")
        ax.set_xlabel("Spatial")
        ax.set_ylabel("Spectral")

        fig.tight_layout()
        plt.show()

    def plotDistortion(self, distortion, lines, select):
        """Plot distortion field

        We plot the x and y distortions as a function of xi,eta.

        Parameters
        ----------
        distortion : `pfs.drp.stella.BaseDistortion`
            Distortion model.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        select : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from matplotlib.colors import Normalize
        from pfs.drp.stella.math import evaluatePolynomial, evaluateAffineTransform

        numSamples = 1000
        cmap = matplotlib.cm.rainbow

        xyRange = distortion.getRange()
        xyModel = np.meshgrid(np.linspace(xyRange.getMinX(), xyRange.getMaxX(), numSamples),
                              np.linspace(xyRange.getMinY(), xyRange.getMaxY(), numSamples),
                              sparse=False)
        xModel = xyModel[0].flatten()
        yModel = xyModel[1].flatten()

        def calculateNorm(xx, yy):
            # Coordinates for plotting
            xNorm = (xx - xyRange.getMinX())/(xyRange.getMaxX() - xyRange.getMinX())
            yNorm = (yy - xyRange.getMinY())/(xyRange.getMaxY() - xyRange.getMinY())
            return xNorm, yNorm

        def getDistortion(poly):
            """Evaluate the polynomial without the linear terms

            Parameters
            ----------
            poly : `lsst.afw.math.Chebyshev1Function2D`
                Polynomial with distortion.

            Returns
            -------
            distortion : `numpy.ndarray` of `float`, shape ``(numSamples,numSamples)``
                Image of the distortion.
            """
            params = np.array(poly.getParameters())
            params[:3] = 0.0
            distortion = type(poly)(params, poly.getXYRange())
            return evaluatePolynomial(distortion, xModel, yModel).reshape(numSamples, numSamples)

        xDistortion = getDistortion(distortion.getXDistortion())
        yDistortion = getDistortion(distortion.getYDistortion())

        xObs, yObs = lines.xBase[select], lines.yBase[select]
        xObsNorm, yObsNorm = calculateNorm(xObs, yObs)

        def removeLinear(values, poly):
            params = np.array(poly.getParameters())
            params[3:] = 0.0
            linear = type(poly)(params, poly.getXYRange())
            return values - evaluatePolynomial(linear, xObs, yObs)

        # For the observed positions, we need to remove the linear part of the distortion and the
        # affine transformation for the right CCD.
        xObs = removeLinear(lines.x[select], distortion.getXDistortion())
        yObs = removeLinear(lines.y[select], distortion.getYDistortion())

        onRightCcd = distortion.getOnRightCcd(lines.xBase[select])
        rightCcd = evaluateAffineTransform(distortion.getRightCcd(), xObs[onRightCcd], yObs[onRightCcd])
        xObs[onRightCcd] -= rightCcd[0]
        yObs[onRightCcd] -= rightCcd[1]

        fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
        for ax, image, values, dim in zip(axes, (xDistortion, yDistortion), (xObs, yObs), ("x", "y")):
            norm = Normalize()
            norm.autoscale(image)
            ax[0].imshow(image, cmap=cmap, norm=norm, origin="lower", extent=(0, 1, 0, 1))
            ax[0].set_xticks((0, 1))
            ax[0].set_yticks((0, 1))
            ax[0].set_title(f"Model {dim}")
            ax[1].scatter(xObsNorm, yObsNorm, marker=".", alpha=0.2, color=cmap(norm(values)))
            ax[1].set_title(f"Observed {dim}")
            ax[1].set_aspect("equal")
            addColorbar(fig, ax[0], cmap, norm, f"{dim} distortion (pixels)")
            addColorbar(fig, ax[1], cmap, norm, f"{dim} distortion (pixels)")

        axes[0][0].set_ylabel("Normalized y (wavelength)")
        axes[1][0].set_ylabel("Normalized y (wavelength)")
        axes[1][0].set_xlabel("Normalized x (fiberId)")
        axes[1][1].set_xlabel("Normalized x (fiberId)")

        fig.tight_layout()
        fig.suptitle("Distortion field")
        plt.show()

    def plotResiduals(self, distortion, lines, used, reserved):
        """Plot fit residuals

        We plot the x and y residuals as a function of fiberId,wavelength

        Parameters
        ----------
        model : `pfs.drp.stella.BaseDistortion`
            Model containing distortion.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        used : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` were reserved from the fit.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from matplotlib.colors import Normalize

        xy = distortion(lines.xBase, lines.yBase)
        dx = lines.x - xy[:, 0]
        dy = lines.y - xy[:, 1]
        good = self.getGoodLines(lines)

        def calculateNormalization(values, nSigma=4.0):
            """Calculate normalization to apply to values

            We generate a normalisation from median +/- nSigma*sigma, where
            sigma is estimated from the IQR.

            Parameters
            ----------
            values : array_like
                Values from which to calculate normalization.
            nSigma : `float`, optional
                Number of sigma either side of the median for range.

            Returns
            -------
            norm : `matplotlib.colors.Normalize`
                Normalization to apply to values.
            """
            lq, median, uq = np.percentile(values, (25.0, 50.0, 75.0))
            sigma = 0.741*(uq - lq)
            return Normalize(median - nSigma*sigma, median + nSigma*sigma)

        cmap = matplotlib.cm.rainbow
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

        for ax, select, label in zip(
            axes.T,
            [(used & ~reserved), reserved, (good & ~used & ~reserved & np.isfinite(dx) & np.isfinite(dy))],
            ["Used", "Reserved", "Rejected"],
        ):
            ax[0].set_title(label)
            if not np.any(select):
                continue
            xNorm = calculateNormalization(dx[select])
            yNorm = calculateNormalization(dy[select])
            ax[0].scatter(lines.fiberId[select], lines.wavelength[select], marker=".", alpha=0.2,
                          color=cmap(xNorm(dx[select])))
            ax[1].scatter(lines.fiberId[select], lines.wavelength[select], marker=".", alpha=0.2,
                          color=cmap(yNorm(dx[select])))
            addColorbar(fig, ax[0], cmap, xNorm, "x residual (pixels)")
            addColorbar(fig, ax[1], cmap, yNorm, "y residual (pixels)")

        for ax in axes[1, :]:
            ax.set_xlabel("fiberId")
        for ax in axes[:, 0]:
            ax.set_ylabel("Wavelength (nm)")

        fig.tight_layout()
        fig.suptitle("Residuals")
        plt.show()

    def plotWavelengthResiduals(self, detectorMap, lines, used, reserved):
        """Plot wavelength residuals

        We plot wavelength residuals as a function of row for a uniform
        selection of fibers.

        The number of rows and columns (and therefore the number of fibers) is
        controlled by ``wlResidRows`` and ``wlResidCols`` debug parameters.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured line positions.
        used : `numpy.ndarray` of `bool`
            Boolean array indicating which lines were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Boolean array indicating which lines were reserved from the fit.
        """
        import matplotlib.pyplot as plt
        from matplotlib.font_manager import FontProperties

        fiberId = np.array(list(sorted(set(lines.fiberId))))
        numFibers = len(fiberId)

        numCols = self.debugInfo.wlResidCols or 2
        numRows = self.debugInfo.wlResidRows or 3
        numPlots = min(numCols*numRows, numFibers)

        indices = np.linspace(0, numFibers, numPlots, False, dtype=int)
        rejected = ~used & ~reserved

        fig, axes = plt.subplots(nrows=numRows, ncols=numCols, sharex=True, sharey=True,
                                 gridspec_kw=dict(wspace=0.0, hspace=0.0))
        for ax, index in zip(axes.flatten(), indices):
            ff = fiberId[index]
            select = lines.fiberId == ff

            for group, color, label in zip((used, rejected, reserved),
                                           ("k", "r", "b"),
                                           ("Used", "Rejected", "Reserved")):
                subset = select & group
                if not np.any(subset):
                    continue
                wlActual = lines.wavelength[subset]
                wlFit = detectorMap.findWavelength(ff, lines.y[subset])
                residual = wlFit - wlActual

                ax.scatter(lines.y[subset], residual, color=color, marker=".", label=label, alpha=0.5)
            ax.text(0.05, 0.8, f"fiberId={ff}", ha="left", transform=ax.transAxes)
            ax.axhline(0, ls=':', color='black')

        font = FontProperties()
        font.set_size('xx-small')

        legend = axes.flatten()[0].legend(prop=font)
        for lh in legend.legendHandles:
            lh.set_alpha(1)
        for ax in axes.flatten():
            ax.set_xlabel("Row (pixels)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Residual (nm)")

        fig.suptitle("Wavelength residuals")
        plt.show()
