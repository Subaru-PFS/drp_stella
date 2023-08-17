import os
from collections import defaultdict, Counter
from typing import Optional, Type

import numpy as np
import scipy.optimize

import lsstDebug

from lsst.utils import getPackageDir
from lsst.pex.config import Config, Field, ListField, DictField
from lsst.pipe.base import Task, Struct
from lsst.geom import Box2D

from pfs.datamodel.pfsTable import PfsTable, Column
from pfs.drp.stella import DetectorMap
from pfs.drp.stella import DoubleDetectorMap, DoubleDistortion
from pfs.drp.stella import PolynomialDetectorMap, PolynomialDistortion
from .applyExclusionZone import getExclusionZone
from .arcLine import ArcLineSet
from .referenceLine import ReferenceLineStatus
from .utils.math import robustRms
from .table import Table


__all__ = ("FitDistortedDetectorMapConfig", "FitDistortedDetectorMapTask", "FittingError")


class LineResiduals(PfsTable):
    """Table of residuals of line measurements

    Parameters
    ----------
    fiberId : `np.array` of `int`
        Fiber identifiers.
    wavelength : `np.array` of `float`
        Reference line wavelengths (nm).
    x, y : `np.array` of `float`
        Differential positions relative to an external detectorMap.
    xErr, yErr : `np.array` of `float`
        Errors in measured positions.
    flux : `np.array` of `float`
        Measured fluxes (arbitrary units).
    fluxErr : `np.array` of `float`
        Errors in measured fluxes (arbitrary units).
    flag : `np.array` of `bool`
        Measurement flags (``True`` indicates an error in measurement).
    status : `np.array` of `int`
        Flags whether the lines are fitted, clipped or reserved etc.
    description : `np.array` of `str`
        Line descriptions (e.g., ionic species)
    xOrig, yOrig : `np.array` of `float`
        Original measured positions (pixels).
    xBase, yBase : `np.array` of `float`
        Expected position from base detectorMap.
    """
    schema = ArcLineSet.DamdClass.schema + [
        Column("xOrig", np.float64, "Original measured x position (pixels)", np.nan),
        Column("yOrig", np.float64, "Original measured y position (pixels)", np.nan),
        Column("xBase", np.float64, "Expected x position from base detectorMap (pixels)", np.nan),
        Column("yBase", np.float64, "Expected y position from base detectorMap (pixels)", np.nan),
    ]
    fitsExtName = "RESID"


class ArcLineResidualsSet(Table):
    """A list of `ArcLineResiduals`

    Analagous to `ArcLineSet`, this stores the position measurement of a list
    of arc lines, but the ``x,y`` positions are relative to a detectorMap. The
    original ``x,y`` positions are stored as ``xOrig,yOrig``.

    Parameters
    ----------
    data : `pandas.DataClass`
        Table data.
    """
    DamdClass = LineResiduals


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
    num : `int`
        Number of points selected.
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

    xSelection = selection & np.isfinite(xResid) & np.isfinite(yResid)
    ySelection = xSelection & (lines.description != "Trace")
    del selection
    xNum = xSelection.sum()
    yNum = ySelection.sum()
    num = xNum + yNum

    xRobustRms = robustRms(xResid[xSelection]) if xNum > 0 else np.nan
    yRobustRms = robustRms(yResid[ySelection]) if yNum > 0 else np.nan

    xResid2 = xResid[xSelection]**2
    yResid2 = yResid[ySelection]**2
    xErr2 = lines.xErr[xSelection]**2 + soften[0]**2
    yErr2 = lines.yErr[ySelection]**2 + soften[1]**2

    xWeight = 1.0/xErr2
    yWeight = 1.0/yErr2
    with np.errstate(invalid="ignore", divide="ignore"):
        xWeightedRms = np.sqrt(np.sum(xWeight*xResid2)/np.sum(xWeight))
        yWeightedRms = np.sqrt(np.sum(yWeight*yResid2)/np.sum(yWeight))

    chi2 = np.sum(xResid2/xErr2) + np.sum(yResid2/yErr2)
    dof = num - numParameters

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
        if residuals.size == 0:
            return 0.0
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

    dimDof = num - numParameters/2  # Assume equipartition of number of parameters between dimensions
    xSoften = calculateSoftening(xResid[xSelection], lines.xErr[xSelection], dimDof)
    ySoften = calculateSoftening(yResid[ySelection], lines.yErr[ySelection], dimDof)

    return Struct(xResid=xResid, yResid=yResid, xRms=xWeightedRms, yRms=yWeightedRms,
                  xRobustRms=xRobustRms, yRobustRms=yRobustRms, chi2=chi2, dof=dof, num=num,
                  numParameters=numParameters, selection=xSelection, soften=soften,
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
    traceIterations = Field(dtype=int, default=2, doc="Number of iterations for updating trace wavleengths")
    iterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejection = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    order = Field(dtype=int, default=7, doc="Distortion order")
    reserveFraction = Field(dtype=float, default=0.1, doc="Fraction of lines to reserve in the final fit")
    soften = Field(dtype=float, default=0.03, doc="Systematic error to apply")
    lsqThreshold = Field(dtype=float, default=1.0e-6, doc="Eigenvaluethreshold for solving least-squares")
    doSlitOffsets = Field(dtype=bool, default=False, doc="Fit for new slit offsets?")
    slitOffsetIterations = Field(dtype=int, default=3, doc="Number of iterations for measuring slit offsets")
    base = Field(dtype=str,
                 doc="Template for base detectorMap; should include '%%(arm)s' and '%%(spectrograph)s'",
                 default=os.path.join(getPackageDir("drp_pfs_data"), "detectorMap",
                                      "detectorMap-sim-%(arm)s%(spectrograph)s.fits")
                 )
    minSignalToNoise = Field(dtype=float, default=20.0,
                             doc="Minimum (flux) signal-to-noise ratio of lines to fit")
    maxCentroidError = Field(dtype=float, default=0.15, doc="Maximum centroid error (pixels) of lines to fit")
    maxRejectionFrac = Field(
        dtype=float,
        default=0.3,
        doc="Maximum fraction of lines that may be rejected in a single iteration",
    )
    minNumWavelengths = Field(dtype=int, default=3, doc="Required minimum number of discrete wavelengths")
    weightings = DictField(keytype=str, itemtype=float, default={},
                           doc="Weightings to apply to different species. Default weighting is 1.0.")
    qaNumFibers = Field(dtype=int, default=5, doc="Number of fibers to use for QA")
    exclusionRadius = Field(dtype=float, default=4,
                            doc="Exclusion radius to apply to reference lines (pixels)")
    doRejectBadLines = Field(dtype=bool, default=False,
                             doc="Reject reference lines for all fibers that have a bad mean residual?")


class FitDistortedDetectorMapTask(Task):
    ConfigClass = FitDistortedDetectorMapConfig
    _DefaultName = "fitDetectorMap"

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
            Arc line measurements. The ``status`` member will be updated to
            indicate which lines were used and reserved.
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
        num : `int`
            Number of points selected.
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
        reserved : `numpy.ndarray` of `bool`
            Array indicating which lines were reserved from the fit.

        Raises
        ------
        FittingError
            If the data is not of sufficient quality to fit.
        """
        if base is None:
            base = self.getBaseDetectorMap(dataId)
        if self.config.doSlitOffsets:
            base.setSlitOffsets(np.zeros(len(base)), np.zeros(len(base)))
        else:
            self.copySlitOffsets(base, spatialOffsets, spectralOffsets)
        DetectorMap = self.getDetectorMapClass(dataId["arm"])
        Distortion = self.getDistortionClass(dataId["arm"])
        for ii in range(self.config.traceIterations):
            self.log.debug("Commencing trace iteration %d", ii)
            residuals = self.calculateBaseResiduals(base, lines)
            weights = self.calculateWeights(lines)
            dispersion = base.getDispersionAtCenter(base.fiberId[len(base)//2])
            results = self.fitDistortion(
                bbox, residuals, weights, dispersion, seed=visitInfo.id, Distortion=Distortion
            )
            reserved = results.reserved
            detectorMap = DetectorMap(base, results.distortion, visitInfo, metadata)
            numParameters = results.numParameters
            if self.config.doSlitOffsets:
                detectorMap.setSlitOffsets(np.zeros(len(base)), np.zeros(len(base)))
                for _ in range(self.config.slitOffsetIterations):
                    offsets = self.measureSlitOffsets(detectorMap, lines, results.selection, weights)
                numParameters += offsets.numParameters
            if not self.updateTraceWavelengths(lines, detectorMap):
                break

        results = self.measureQuality(lines, detectorMap, results.selection, numParameters)
        results.detectorMap = detectorMap
        results.reserved = reserved

        lines.status[results.selection] |= ReferenceLineStatus.DETECTORMAP_USED
        lines.status[results.reserved] |= ReferenceLineStatus.DETECTORMAP_RESERVED

        if self.debugInfo.finalResiduals:
            self.plotResiduals(residuals, results.xResid, results.yResid, results.selection, results.reserved)

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

    def getDetectorMapClass(self, arm: str) -> Type[DetectorMap]:
        """Return the class to use for the detectorMap

        Parameters
        ----------
        arm : `str`
            Spectrograph arm in use (one of ``b``, ``r``, ``n``, ``m``).

        Returns
        -------
        detectorMapClass : `type`
            Class to use for the detectorMap.
        """
        return PolynomialDetectorMap if arm == "n" else DoubleDetectorMap

    def getDistortionClass(self, arm: str) -> Type:
        """Return the class to use for the distortion

        Parameters
        ----------
        arm : `str`
            Spectrograph arm in use (one of ``b``, ``r``, ``n``, ``m``).

        Returns
        -------
        distortionClass : `type`
            Class to use for the distortion.
        """
        return PolynomialDistortion if arm == "n" else DoubleDistortion

    def getGoodLines(self, lines: ArcLineSet, dispersion: Optional[float] = None) -> np.ndarray:
        """Return a boolean array indicating which lines are good.

        Parameters
        ----------
        lines : `ArcLineSet`
            Line measurements.
        dispersion : `float`, optional
            Dispersion (nm/pixel) to use for applying exclusion zone.

        Returns
        -------
        good : `numpy.ndarray` of `bool`
            Boolean array indicating which lines are good.
        """
        def getCounts():
            """Provide a list of counts of different species"""
            if self.log.isEnabledFor(self.log.DEBUG):
                counts = Counter(lines.description[good])
                return ", ".join(f"{key}: {counts[key]}" for key in sorted(counts))
            return ""

        isTrace = lines.description == "Trace"
        self.log.debug("%d lines in list", len(lines))
        good = lines.flag == 0
        self.log.debug("%d good lines after measurement flags (%s)", good.sum(), getCounts())
        good &= (lines.status & ReferenceLineStatus.fromNames(*self.config.lineFlags)) == 0
        self.log.debug("%d good lines after line status (%s)", good.sum(), getCounts())
        good &= np.isfinite(lines.x) & np.isfinite(lines.y)
        good &= np.isfinite(lines.xErr) & np.isfinite(lines.yErr)
        self.log.debug("%d good lines after finite positions (%s)", good.sum(), getCounts())
        if self.config.minSignalToNoise > 0:
            good &= np.isfinite(lines.flux) & np.isfinite(lines.fluxErr)
            self.log.debug("%d good lines after finite intensities (%s)", good.sum(), getCounts())
            with np.errstate(invalid="ignore", divide="ignore"):
                good &= (lines.flux/lines.fluxErr) > self.config.minSignalToNoise
            self.log.debug("%d good lines after signal-to-noise (%s)", good.sum(), getCounts())
        if self.config.maxCentroidError > 0:
            maxCentroidError = self.config.maxCentroidError
            good &= (lines.xErr > 0) & (lines.xErr < maxCentroidError)
            good &= ((lines.yErr > 0) & (lines.yErr < maxCentroidError)) | isTrace
            self.log.debug("%d good lines after centroid errors (%s)", good.sum(), getCounts())
        if dispersion is not None and self.config.exclusionRadius > 0 and not np.all(isTrace):
            wavelength = np.unique(lines.wavelength[~isTrace])
            status = [np.bitwise_or.reduce(lines.status[lines.wavelength == wl]) for wl in wavelength]
            exclusionRadius = dispersion*self.config.exclusionRadius
            exclude = getExclusionZone(wavelength, exclusionRadius, np.array(status))
            good &= np.isin(lines.wavelength, wavelength[exclude], invert=True) | isTrace
            self.log.debug("%d good lines after %.2f nm exclusion zone (%s)",
                           good.sum(), exclusionRadius, getCounts())
        return good

    def measureSlitOffsets(self, detectorMap, lines, select, weights):
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
        weights : `numpy.ndarray` of `float`
            Weights for each line. This excludes the weighting that comes from
            the measurement errors, and should be the square root of the usual
            weighting applied to the Fisher matrix.

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
        xy = np.full((len(lines), 2), np.nan, dtype=float)
        isTrace = lines.description == "Trace"
        notTrace = ~isTrace

        xy[notTrace] = detectorMap.findPoint(fiberId[notTrace], lines.wavelength[notTrace])
        xy[isTrace, 0] = detectorMap.getXCenter(fiberId[isTrace], lines.y[isTrace])
        xy[isTrace, 1] = lines.y[isTrace]
        dx = xy[:, 0] - lines.x
        dy = xy[:, 1] - lines.y
        xErr = np.hypot(lines.xErr, sysErr)
        yErr = np.hypot(lines.yErr, sysErr)
        fit = np.full((len(lines), 2), np.nan, dtype=float)

        use = select & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(xErr) & np.isfinite(yErr)

        # Check for fibers that have all measurements rejected in previous fits.
        # For those fibers, restore all measurements, just for this exercise.
        for ff in set(fiberId):
            thisFiber = fiberId == ff
            if not np.any(use & thisFiber):
                use[thisFiber] = np.isfinite(dx[thisFiber]) & np.isfinite(dy[thisFiber])
                use[thisFiber] &= np.isfinite(xErr[thisFiber]) & np.isfinite(yErr[thisFiber])

        for ii in range(self.config.iterations):
            spatial = np.zeros(numFibers, dtype=float)
            spectral = np.zeros(numFibers, dtype=float)
            noMeasurements = set()
            for jj, ff in enumerate(detectorMap.getFiberId()):
                choose = use & (fiberId == ff)
                if not np.any(choose & notTrace):
                    noMeasurements.add(ff)
                    continue
                # Robust measurement
                spatial[jj] = -np.median(dx[choose])
                spectral[jj] = -np.median(dy[choose & notTrace])
                fit[choose, 0] = xy[choose, 0] + spatial[jj]
                fit[choose, 1] = xy[choose, 1] + spectral[jj]

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

            newUse = select & self.rejectOutliers(result, xErr, yErr)

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
            yChoose = choose & notTrace
            if not np.any(choose) or not np.any(yChoose):
                noMeasurements.add(ff)
                continue
            with np.errstate(divide="ignore"):
                spatial[jj] = -np.average(dx[choose], weights=(weights[choose]/xErr[choose])**2)
                spectral[jj] = -np.average(dy[yChoose], weights=(weights[yChoose]/yErr[yChoose])**2)
                fit[choose, 0] = xy[choose, 0] + spatial[jj]
                fit[choose, 1] = xy[choose, 1] + spectral[jj]

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

    def updateTraceWavelengths(self, lines: ArcLineSet, detectorMap: DetectorMap) -> bool:
        """Update trace wavelengths

        Trace wavelengths are approximate only, and based on the original
        detectorMap (which we are correcting). If there is a big shift, the
        inaccuracy of the trace wavelengths can adversely affect the quality of
        the fit. We therefore update the trace wavelengths with the new
        detectorMap and re-fit.

        Parameters
        ----------
        lines : `ArcLineSet`
            Measured line positions. May also include traces, the wavelengths of
            which will be updated in-place.
        detectorMap : `DetectorMap`
            Best current estimate of the mapping between fiberId,wavelength and
            x,y.

        Returns
        -------
        anyTrace : `bool`
            Any traces got updated?
        """
        isTrace = lines.description == "Trace"
        if not np.any(isTrace):
            return False
        lines.wavelength[isTrace] = detectorMap.findWavelength(lines.fiberId[isTrace], lines.y[isTrace])
        return True

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
            self.log.warning("No slit offsets provided")
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
        xx = lines.x
        yy = lines.y
        dx = lines.x - points[:, 0]
        dy = lines.y - points[:, 1]

        if self.debugInfo.baseResiduals:
            import matplotlib.pyplot as plt
            import matplotlib.cm
            from matplotlib.colors import Normalize
            cmap = matplotlib.cm.rainbow
            fig, axes = plt.subplots(ncols=3)

            good = self.getGoodLines(lines, detectorMap.getDispersionAtCenter())
            good &= np.all(np.isfinite(points), axis=1)
            magnitude = np.hypot(dx[good], dy[good])
            norm = Normalize()
            norm.autoscale(magnitude)
            axes[0].quiver(
                xx[good],
                yy[good],
                dx[good],
                dy[good],
                color=cmap(norm(magnitude)),
                scale=1,
                angles="xy",
                scale_units="xy",
            )
            axes[0].set_xlabel("Spatial (pixels)")
            axes[0].set_ylabel("Spectral (pixels)")
            axes[0].set_title("Offsets")
            addColorbar(fig, axes[0], cmap, norm, "Offset (pixels)")

            norm = Normalize()
            norm.autoscale(dx[good])
            axes[1].scatter(
                lines.fiberId[good],
                lines.wavelength[good],
                color=cmap(norm(dx[good])),
                marker=".",
            )
            axes[1].set_xlabel("fiberId")
            axes[1].set_ylabel("Wavelength (nm)")
            axes[1].set_title("Spatial offset")
            addColorbar(fig, axes[1], cmap, norm, "Spatial offset (pixels)")
            norm = Normalize()
            norm.autoscale(dy[good])
            axes[2].scatter(
                lines.fiberId[good],
                lines.wavelength[good],
                color=cmap(norm(dy[good])),
                marker=".",
            )
            axes[2].set_xlabel("fiberId")
            axes[2].set_ylabel("Wavelength (nm)")
            axes[2].set_title("Spectral offset")
            addColorbar(fig, axes[2], cmap, norm, "Spectral offset (pixels)")
            fig.subplots_adjust(wspace=0.75)
            plt.show()

        return ArcLineResidualsSet.fromColumns(
            fiberId=lines.fiberId,
            wavelength=lines.wavelength,
            x=dx,
            y=dy,
            xOrig=xx,
            yOrig=yy,
            xBase=points[:, 0],
            yBase=points[:, 1],
            xErr=lines.xErr,
            yErr=lines.yErr,
            xx=lines.xx,
            yy=lines.yy,
            xy=lines.xy,
            flux=lines.flux,
            fluxErr=lines.fluxErr,
            fluxNorm=lines.fluxNorm,
            flag=lines.flag,
            status=lines.status,
            description=lines.description,
            transition=lines.transition,
            source=lines.source,
        )

    def calculateWeights(self, lines: ArcLineSet):
        """Calculate weights for line measurements

        Weights are applied based on the species.

        Parameters
        ----------
        lines : `ArcLineSet`
            Line measurements.

        Returns
        -------
        weights : `numpy.ndarray` of `float`
            Weights for each line measurement.
        """
        weights = np.ones(len(lines), dtype=float)
        for species in self.config.weightings:
            selectSpecies = lines.description == species
            if np.any(selectSpecies):
                weights[selectSpecies] = self.config.weightings[species]
        return weights

    def fitDistortion(self, bbox, lines, weights, dispersion, seed=0, fitStatic=True, Distortion=None):
        """Fit a distortion model

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        weights : `numpy.ndarray` of `float`
            Weights for each line. This excludes the weighting that comes from
            the measurement errors, and should be the square root of the usual
            weighting applied to the Fisher matrix.
        dispersion : `float`
            Wavelength dispersion (nm/pixel); for interpreting RMS in logging.
        seed : `int`
            Seed for random number generator used for selecting reserved lines.
        fitStatic : `bool`, optional
            Fit static components to the distortion model?
        Distortion : subclass of `pfs.drp.stella.BaseDistortion`
            Class to use for distortion. If ``None``, uses `DoubleDistortion`.

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
        if Distortion is None:
            Distortion = DoubleDistortion
        good = self.getGoodLines(lines, dispersion)
        numGood = good.sum()

        rng = np.random.RandomState(seed)
        numReserved = int(self.config.reserveFraction*numGood + 0.5)
        reservedIndices = rng.choice(np.arange(numGood, dtype=int), replace=False, size=numReserved)
        reserved = np.zeros_like(good)
        select = np.zeros(numGood, dtype=bool)
        select[reservedIndices] = True
        reserved[good] = select

        # Errors used in rejection
        xErr = np.hypot(lines.xErr, self.config.soften)
        yErr = np.hypot(lines.yErr, self.config.soften)
        rejection = self.config.rejection

        used = good & ~reserved
        result = None
        for ii in range(self.config.iterations):
            result = self.fitModel(bbox, lines, used, weights, fitStatic=fitStatic, Distortion=Distortion)
            self.log.debug(
                "Fit iteration %d: chi2=%f dof=%d xRMS=%f yRMS=%f (%f nm) from %d/%d lines",
                ii, result.chi2, result.dof, result.xRms, result.yRms, result.yRms*dispersion, used.sum(),
                numGood - numReserved
            )
            self.log.debug("Fit iteration %d: %s", ii, result.distortion)
            if self.debugInfo.plot:
                self.plotModel(lines, used, result)
            if self.debugInfo.residuals:
                self.plotResiduals(lines, result.xResid, result.yResid, used, reserved)
            with np.errstate(invalid="ignore"):
                newUsed = good & ~reserved
                if self.config.doSlitOffsets:
                    # There may be fiber-specific offsets, so do rejection for individual fibers
                    for fiberId in set(lines.fiberId[used]):
                        choose = used & (lines.fiberId == fiberId)
                        yChoose = choose & (lines.description != "Trace")
                        dx = np.median(result.xResid[choose]) if np.any(choose) else np.nan
                        dy = np.median(result.yResid[yChoose]) if np.any(yChoose) else np.nan
                        newUsed[choose] &= ((np.abs((result.xResid[choose] - dx)/xErr[choose]) < rejection) &
                                            (np.abs((result.yResid[choose] - dy)/yErr[choose]) < rejection))
                else:
                    keep = self.rejectOutliers(result, xErr, yErr)
                    if self.config.doRejectBadLines:
                        keep &= self.rejectBadLines(result, lines)
                    newUsed &= keep

            self.log.debug("Rejecting %d/%d lines in iteration %d", used.sum() - newUsed.sum(),
                           used.sum(), ii)
            if np.all(newUsed == used):
                # Converged
                break
            used = newUsed

        result = self.fitModel(bbox, lines, used, weights, fitStatic=fitStatic, Distortion=Distortion)
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
        if not np.all(np.isfinite(soften)):
            self.log.warn("Non-finite softening, probably a bad fit")
        else:
            result = self.fitModel(
                bbox, lines, used, weights, soften, fitStatic=fitStatic, Distortion=Distortion
            )
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
            self.plotResiduals(lines, result.xResid, result.yResid, used, reserved)
        return result

    def fitModel(self, bbox, lines, select, weights, soften=None, fitStatic=True, Distortion=None):
        """Fit a model to the arc lines

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        select : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        weights : `numpy.ndarray` of `float`
            Weights for each line. This excludes the weighting that comes from
            the measurement errors, and should be the square root of the usual
            weighting applied to the Fisher matrix.
        soften : `tuple` (`float`, `float`), optional
            Systematic error in x and y to add in quadrature to measured errors
            (pixels).
        fitStatic : `bool`, optional
            Fit static components to the distortion model?
        Distortion : subclass of `pfs.drp.stella.BaseDistortion`
            Class to use for distortion. If ``None``, uses `DoubleDistortion`.

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
            Distortion = DoubleDistortion
        if soften is None:
            soften = (self.config.soften, self.config.soften)
        xSoften, ySoften = soften

        xx = lines.x[select].astype(float)
        yy = lines.y[select].astype(float)
        xBase = lines.xBase[select]
        yBase = lines.yBase[select]
        with np.errstate(invalid="ignore", divide="ignore"):
            xErr = np.hypot(lines.xErr[select].astype(float), xSoften)/weights[select]
            yErr = np.where(lines.description[select] == "Trace", np.inf,
                            np.hypot(lines.yErr[select].astype(float), ySoften)/weights[select])

        distortion = Distortion.fit(self.config.order, Box2D(bbox), xBase, yBase,
                                    xx, yy, xErr, yErr, fitStatic, self.config.lsqThreshold)

        return calculateFitStatistics(distortion(lines.xBase, lines.yBase), lines, select,
                                      distortion.getNumParameters(), soften, distortion=distortion)

    def measureQuality(self, lines: ArcLineSet, detectorMap: DetectorMap, selection, numParameters) -> Struct:
        """Measure and log fit quality information

        Returned results apply to the entire fit, but we calculate and log
        statistics for smaller sub-sections of the data.

        Parameters
        ----------
        lines : `ArcLineSet`
            Line measurements.
        detectorMap : `DetectorMap`
            Final fit detectorMap, mapping fiberId,wavelength to x,y.
        selection : `numpy.ndarray` of `bool`
            Boolean array indicating which lines are good (not rejected during
            fit).
        numParameters : `int`
            Number of parameters in fitting.

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
        num : `int`
            Number of points selected.
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
        fitPosition = detectorMap.findPoint(lines.fiberId, lines.wavelength)
        soften = (self.config.soften, self.config.soften)
        results = calculateFitStatistics(fitPosition, lines, selection, numParameters, soften,
                                         detectorMap=detectorMap)
        self.log.info("Final result: chi2=%f dof=%d xRMS=%f yRMS=%f xSoften=%f ySoften=%f from %d lines",
                      results.chi2, results.dof, results.xRms, results.yRms,
                      results.xSoften, results.ySoften, results.selection.sum())

        for descr in sorted(set(lines.description)):
            choose = selection & (lines.description == descr)
            stats = calculateFitStatistics(fitPosition, lines, choose, 0, soften)
            self.log.info("Stats for %s: chi2=%f dof=%d xRMS=%f yRMS=%f xSoften=%f ySoften=%f from %d lines",
                          descr, stats.chi2, stats.dof, stats.xRms, stats.yRms,
                          stats.xSoften, stats.ySoften, stats.selection.sum())

        fiberId = np.array(sorted(set(lines.fiberId[selection])))
        for ff in fiberId[np.linspace(0, len(fiberId) - 1, self.config.qaNumFibers, dtype=int)]:
            choose = selection & (lines.fiberId == ff)
            stats = calculateFitStatistics(fitPosition, lines, choose, 0, soften)
            self.log.info("Stats for fiberId=%d: chi2=%f dof=%d xRMS=%f yRMS=%f xSoften=%f ySoften=%f "
                          "from %d lines (%s)",
                          ff, stats.chi2, stats.dof, stats.xRms, stats.yRms,
                          stats.xSoften, stats.ySoften, stats.selection.sum(),
                          ", ".join(f"{cc[1]} {cc[0]}" for cc in Counter(lines.description[choose]).items()))

        if self.log.isEnabledFor(self.log.DEBUG):
            good = self.getGoodLines(lines) & (lines.description != "Trace")
            for wl in sorted(set(lines.wavelength[good].tolist())):
                choose = good & (lines.wavelength == wl)
                description = ", ".join(set(lines.description[choose]))
                stats = calculateFitStatistics(fitPosition, lines, choose, 0, soften)
                self.log.info("Stats for wavelength=%f (%s): chi2=%f dof=%d xRMS=%f yRMS=%f "
                              "xSoften=%f ySoften=%f from %d fibers",
                              wl, description, stats.chi2, stats.dof, stats.xRms, stats.yRms,
                              stats.xSoften, stats.ySoften, stats.selection.sum())

        return results

    def rejectOutliers(self, fitStats: Struct, xErr: np.ndarray, yErr: np.ndarray) -> np.ndarray:
        """Reject outliers from distortion fit

        Parameters
        ----------
        fitStats : `lsst.pipe.base.Struct`
            Fit statistics; the output of ``calculateFitStatistics``.
        xErr, yErr : `np.ndarray`
            Errors in x and y (pixels).

        Returns
        -------
        keep : `np.ndarray` of `bool`
            Array indicating which points should be kept.
        """
        xSoften = fitStats.xRobustRms if np.isfinite(fitStats.xRobustRms) else 0.0
        ySoften = fitStats.yRobustRms if np.isfinite(fitStats.yRobustRms) else 0.0
        xResid = np.abs(fitStats.xResid/np.hypot(xErr, xSoften))
        yResid = np.abs(fitStats.yResid/np.hypot(yErr, ySoften))
        keep = (xResid < self.config.rejection) & (yResid < self.config.rejection)
        minKeepFrac = 1.0 - self.config.maxRejectionFrac
        if keep.sum() < minKeepFrac*fitStats.selection.sum():
            xResidLimit = np.percentile(xResid[fitStats.selection], minKeepFrac*100)
            yResidLimit = np.percentile(yResid[fitStats.selection], minKeepFrac*100)
            keep = (xResid < xResidLimit) & (yResid < yResidLimit)
            self.log.debug(
                "Standard rejection limit (%f) too severe; using %f, %f",
                self.config.rejection,
                xResidLimit,
                yResidLimit
            )
        return keep

    def rejectBadLines(self, fitStats: Struct, lines: ArcLineSet) -> np.ndarray:
        """Reject bad lines from distortion fit

        If a particular reference line has a mean spectral residual across all
        fibers exceeding the spectral RMS, it is rejected for all fibers. This
        can help rejecting bad entries in the line list that haven't been
        manually flagged.

        Parameters
        ----------
        fitStats : `lsst.pipe.base.Struct`
            Fit statistics; the output of ``calculateFitStatistics``.
        lines : `lsst.pipe.tasks.ArcLineSet`
            Arc line measurements.

        Returns
        -------
        keep : `np.ndarray` of `bool`
            Array indicating which points should be kept.
        """
        select = fitStats.selection & (lines.description != "Trace")
        lineMeanResid = {}
        for wl in np.unique(lines.wavelength[select]):
            line = select & (lines.wavelength == wl)
            lineMeanResid[wl] = np.mean(fitStats.yResid[line])

        lineMeanResidWl = np.array(list(lineMeanResid.keys()))
        lineMeanResidValues = np.array(list(lineMeanResid.values()))
        badLines = lineMeanResidWl[np.abs(lineMeanResidValues) > self.config.rejection*fitStats.yRobustRms]
        self.log.debug("Rejecting lines at wavelengths: %s", badLines)

        return np.isin(lines.wavelength, badLines, invert=True)

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

    def plotResiduals(self, lines, dx, dy, used, reserved):
        """Plot fit residuals

        We plot the x and y residuals as a function of fiberId,wavelength

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineResidualsSet`
            Arc line measurement residuals (w.r.t. base detectorMap).
        dx, dy : `numpy.ndarray` of `float`, shape ``(N,)``
            Residuals in x,y for each line.
        used : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` were reserved from the fit.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from matplotlib.colors import Normalize

        good = self.getGoodLines(lines) & np.isfinite(dx) & np.isfinite(dy)

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
            lq, median, uq = np.percentile(values[np.isfinite(values)], (25.0, 50.0, 75.0))
            sigma = 0.741*(uq - lq)
            return Normalize(median - nSigma*sigma, median + nSigma*sigma)

        cmap = matplotlib.cm.rainbow
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

        isTrace = lines.description == "Trace"

        for ax, select, label in zip(
            axes.T,
            [(used & ~reserved), reserved, (good & ~used & ~reserved)],
            ["Used", "Reserved", "Rejected"],
        ):
            ax[0].set_title(label)
            select &= ~isTrace
            if not np.any(select):
                continue
            xNorm = calculateNormalization(dx[select])
            yNorm = calculateNormalization(dy[select])
            ax[0].scatter(lines.fiberId[select], lines.wavelength[select], marker=".", alpha=0.2,
                          color=cmap(xNorm(dx[select])))
            ax[1].scatter(lines.fiberId[select], lines.wavelength[select], marker=".", alpha=0.2,
                          color=cmap(yNorm(dy[select])))
            addColorbar(fig, ax[0], cmap, xNorm, "x residual (pixels)")
            addColorbar(fig, ax[1], cmap, yNorm, "y residual (pixels)")

        for ax in axes[1, :]:
            ax.set_xlabel("fiberId")
        for ax in axes[:, 0]:
            ax.set_ylabel("Wavelength (nm)")

        fig.tight_layout()
        fig.suptitle("Line residuals")

        if np.any(isTrace):
            fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
            fiberId = set(lines.fiberId[isTrace])
            fiberNorm = Normalize(lines.fiberId.min(), lines.fiberId.max())
            residNorm = calculateNormalization(dx[isTrace & used])
            for ff in fiberId:
                select = isTrace & good & used & ~reserved & (lines.fiberId == ff)
                rejected = isTrace & good & ~used & ~reserved & (lines.fiberId == ff)
                with np.errstate(invalid="ignore"):
                    axes[0].scatter(lines.xOrig[select], lines.yOrig[select], marker=".",
                                    color=cmap(residNorm(dx[select])))
                    if np.any(rejected):
                        axes[1].scatter(lines.xOrig[rejected], lines.yOrig[rejected], marker=".",
                                        color=cmap(residNorm(dx[rejected])))
                axes[2].plot(dx[select], lines.yOrig[select], ls="-", color=cmap(fiberNorm(ff)), alpha=0.2)
            addColorbar(fig, axes[0], cmap, residNorm, "x residual (pixels)")
            addColorbar(fig, axes[1], cmap, residNorm, "x residual (pixels)")
            addColorbar(fig, axes[2], cmap, fiberNorm, "fiberId")
            axes[0].set_xlabel("Column (pixels)")
            axes[0].set_ylabel("Row (pixels)")
            axes[0].set_title("Used")
            axes[1].set_xlim(axes[0].get_xlim())
            axes[1].set_ylim(axes[0].get_ylim())
            axes[1].set_xlabel("Column (pixels)")
            axes[1].set_ylabel("Row (pixels)")
            axes[1].set_title("Rejected")
            axes[2].set_xlabel("x residual (pixels)")
            axes[2].set_ylabel("Row (pixels)")
            axes[2].set_title("Residuals")
            fig.tight_layout()
            fig.suptitle("Trace residuals")

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
        rejected = self.getGoodLines(lines, detectorMap.getDispersionAtCenter()) & ~used & ~reserved

        fig, axes = plt.subplots(nrows=numRows, ncols=numCols, sharex=True, sharey=True,
                                 gridspec_kw=dict(wspace=0.0, hspace=0.0))
        for ax, index in zip(axes.flatten(), indices):
            ff = fiberId[index]
            select = (lines.fiberId == ff) & (lines.description != "Trace")

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
