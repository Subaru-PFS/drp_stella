import os
from collections import defaultdict
from deprecated import deprecated

import numpy as np
import scipy.optimize

import lsstDebug

from lsst.utils import getPackageDir
from lsst.pex.config import Config, Field, DictField, ListField
from lsst.pipe.base import Task, Struct
from lsst.afw.math import LeastSquares

from pfs.drp.stella import DetectorMap, DifferentialDetectorMap
from .GlobalDetectorModel import GlobalDetectorModel, GlobalDetectorModelScaling
from .referenceLine import ReferenceLineStatus
from .utils.math import robustRms
from .fitDistortedDetectorMap import ArcLineResidualsSet


__all__ = ("FitDifferentialDetectorMapConfig", "FitDifferentialDetectorMapTask")


def rmsPixelsToVelocity(rms, model):
    """Convert an RMS in pixels to velocity in km/s

    Parameters
    ----------
    rms : `float`
        RMS in the spectral dimension (pixels).
    model : `pfs.drp.stella.GlobalDetectorModel`
        Model of the focal plane.

    Returns
    -------
    rms : `float`
        Velocity RMS (km/s).
    """
    return 3.0e5*rms*model.getScaling().dispersion/model.getScaling().wavelengthCenter


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


def calculateFitStatistics(model, lines, selection, soften=0.0):
    """Calculate statistics of the model fit

    Parameters
    ----------
    model : `pfs.drp.stella.GlobalDetectorModel`
        Model that was fit to the data.
    lines : `pfs.drp.stella.ArcLineSet`
        Arc line measurements.
    selection : `numpy.ndarray` of `bool`
        Flags indicating which of the ``lines`` are to be used in the
        calculation.
    soften : `float`, optional
        Systematic error to add in quadrature to measured errors (pixels).

    Returns
    -------
    model : `pfs.drp.stella.GlobalDetectorModel`
        Model that was fit to the data.
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
    soften : `float`
        Systematic error that was applied to measured errors (pixels).
    """
    fit = model(lines.fiberId.astype(np.int32), lines.wavelength.astype(float))
    xResid = (lines.x - fit[:, 0])
    yResid = (lines.y - fit[:, 1])

    xRobustRms = robustRms(xResid[selection])
    yRobustRms = robustRms(yResid[selection])

    xResid2 = xResid[selection]**2
    yResid2 = yResid[selection]**2
    xErr2 = lines.xErr[selection]**2 + soften**2
    yErr2 = lines.yErr[selection]**2 + soften**2

    xWeight = 1.0/xErr2
    yWeight = 1.0/yErr2
    xWeightedRms = np.sqrt(np.sum(xWeight*xResid2)/np.sum(xWeight))
    yWeightedRms = np.sqrt(np.sum(yWeight*yResid2)/np.sum(yWeight))

    chi2 = np.sum(xResid2/xErr2 + yResid2/yErr2)
    dof = 2*selection.sum() - model.getNumParameters(model.getDistortionOrder())
    return Struct(model=model, xResid=xResid, yResid=yResid, xRms=xWeightedRms, yRms=yWeightedRms,
                  xRobustRms=xRobustRms, yRobustRms=yRobustRms, chi2=chi2, dof=dof, soften=soften)


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


class FitDifferentialDetectorMapConfig(Config):
    """Configuration for FitDifferentialDetectorMapTask"""
    lineFlags = ListField(dtype=str, default=["BAD"], doc="ReferenceLineStatus flags for lines to ignore")
    iterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejection = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    order = Field(dtype=int, default=4, doc="Distortion order")
    reserveFraction = Field(dtype=float, default=0.1, doc="Fraction of lines to reserve in the final fit")
    soften = Field(dtype=float, default=0.01, doc="Systematic error to apply")
    buffer = Field(dtype=float, default=0.05, doc="Buffer for xi,eta range")
    forceSingleCcd = Field(dtype=bool, default=False,
                           doc="Force a single CCD? This might be useful for a sparse fiber density")
    fiberCenter = DictField(keytype=int, itemtype=float, doc="Central fiberId, separating CCDs",
                            default={1: 326, 2: 326, 3: 326, 4: 326})
    doSlitOffsets = Field(dtype=bool, default=False, doc="Fit for new slit offsets?")
    base = Field(dtype=str,
                 doc="Template for base detectorMap; should include '%%(arm)s' and '%%(spectrograph)s'",
                 default=os.path.join(getPackageDir("drp_pfs_data"), "detectorMap",
                                      "detectorMap-sim-%(arm)s%(spectrograph)s.fits")
                 )


class FitDifferentialDetectorMapTask(Task):
    ConfigClass = FitDifferentialDetectorMapConfig
    _DefaultName = "fitDetectorMap"

    @deprecated(reason="FitDifferentialDetectorMapTask has been replaced by FitDistortedDetectorMapTask")
    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, dataId, bbox, lines, visitInfo, metadata=None,
            spatialOffsets=None, spectralOffsets=None, base=None):
        """Fit a DifferentialDetectorMap to arc line measurements

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
        detectorMap : `pfs.drp.stella.DifferentialDetectorMap`
            Mapping of fiberId,wavelength to x,y.
        model : `pfs.drp.stella.GlobalDetectorModel`
            Model that was fit to the data.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Residual RMS in x,y (pixels)
        chi2 : `float`
            Fit chi^2.
        soften : `float`
            Systematic error that was applied to measured errors (pixels).
        used : `numpy.ndarray` of `bool`
            Array indicating which lines were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Array indicating which lines were reserved from the fit.
        """
        arm = dataId["arm"]
        spectrograph = dataId["spectrograph"]
        doFitHighCcd = (arm != "n") and not self.config.forceSingleCcd
        fiberCenter = self.config.fiberCenter[spectrograph] if doFitHighCcd else 0
        if base is None:
            base = self.getBaseDetectorMap(dataId)
        if self.config.doSlitOffsets:
            self.log.warning("Not measuring slit offsets: deprecated")
        elif spatialOffsets is not None and spectralOffsets is not None:
            self.copySlitOffsets(base, spatialOffsets, spectralOffsets)
        residuals = self.calculateBaseResiduals(base, lines)
        minMaxFiberId = (base.fiberId.min(), base.fiberId.max())
        results = self.fitGlobalDetectorModel(bbox, residuals, doFitHighCcd, fiberCenter,
                                              seed=visitInfo.id, minMaxFiberId=minMaxFiberId)
        results.detectorMap = DifferentialDetectorMap(base, results.model, visitInfo, metadata)

        if self.debugInfo.lineQa:
            self.lineQa(lines, results.detectorMap)
        if self.debugInfo.wlResid:
            self.plotWavelengthResiduals(results.detectorMap, lines, results.used, results.reserved)
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

    def measureSlitOffsets(self, detectorMap, lines):
        """Measure slit offsets for base detectorMap

        The detectorMap is modified in-place.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Base detectorMap.
        lines : `ArcLineSet`
            Original line measurements (NOT the residuals).
        """
        good = (lines.flag == 0)
        sysErr = self.config.soften
        detectorMap.measureSlitOffsets(
            lines.fiberId[good], lines.wavelength[good],
            lines.x[good], lines.y[good],
            np.hypot(lines.xErr[good], sysErr), np.hypot(lines.yErr[good], sysErr)
        )

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
            raise RuntimeError("Slit offsets not provided")
        if np.all(spatialOffsets == 0) or np.all(spectralOffsets == 0):
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
        return ArcLineResidualsSet.fromColumns(
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

    def fitScaling(self, bbox, lines, select, minMaxFiberId=None):
        """Determine scaling for GlobalDetectorModel

        These are not fit parameters, but merely provides convenient scaling
        factors so that the fit parameters, and especially the slit offsets, are
        in units roughly approximating pixels.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineResidualsSet`
            Arc line measurements.
        select : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        minMaxFiberId : `tuple` (`int`, `int`), optional
            Minimum and maximum fiber identifiers to cover. If not provided, use
            ``lines.fiberId``.

        Returns
        -------
        scaling : `pfs.drp.stella.GlobalDetectorModelScaling`
            Scaling for model.
        """
        minFiberId = minMaxFiberId[0] if minMaxFiberId is not None else lines.fiberId.min()
        maxFiberId = minMaxFiberId[1] if minMaxFiberId is not None else lines.fiberId.max()
        fiberFit = fitStraightLine(lines.fiberId[select], lines.xOrig[select])
        wlFit = fitStraightLine(lines.yOrig[select], lines.wavelength[select])
        return GlobalDetectorModelScaling(
            fiberPitch=np.abs(fiberFit.slope),  # pixels per fiber
            dispersion=wlFit.slope,  # nm per pixel,
            wavelengthCenter=wlFit.slope*bbox.getHeight()/2 + wlFit.intercept,
            minFiberId=minFiberId,
            maxFiberId=maxFiberId,
            height=bbox.getHeight(),
            buffer=self.config.buffer,
        )

    def fitGlobalDetectorModel(self, bbox, lines, doFitHighCcd, fiberCenter=0, seed=0, minMaxFiberId=None):
        """Fit a distortion model to the entire detector

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        doFitHighCcd : `bool`
            Fit an affine transformation for the high-fiberId CCD?
        fiberCenter : `float`
            Central fiberId, separating low- and high-fiberId CCDs.
        seed : `int`
            Seed for random number generator used for selecting reserved lines.
        minMaxFiberId : `tuple` (`int`, `int`), optional
            Minimum and maximum fiber identifiers to cover. If not provided, use
            ``lines.fiberId``.

        Returns
        -------
        model : `pfs.drp.stella.GlobalDetectorModel`
            Model that was fit to the data.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Residual RMS in x,y (pixels)
        chi2 : `float`
            Fit chi^2.
        soften : `float`
            Systematic error that was applied to measured errors (pixels).
        used : `numpy.ndarray` of `bool`
            Array indicating which lines were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Array indicating which lines were reserved from the fit.
        """
        numLines = len(lines)

        good = lines.flag == 0
        good &= (lines.status & ReferenceLineStatus.fromNames(*self.config.lineFlags)) == 0
        good &= np.isfinite(lines.x) & np.isfinite(lines.y)
        good &= np.isfinite(lines.xErr) & np.isfinite(lines.yErr)
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
            result = self.fitModel(bbox, lines, used, doFitHighCcd, fiberCenter, minMaxFiberId=minMaxFiberId)
            self.log.debug(
                "Fit iteration %d: chi2=%f dof=%d xRMS=%f yRMS=%f (%f nm, %f km/s) from %d/%d lines",
                ii, result.chi2, result.dof, result.xRms, result.yRms,
                result.yRms*result.model.getScaling().dispersion,
                rmsPixelsToVelocity(result.yRms, result.model), used.sum(),
                numLines - numReserved
            )
            self.log.debug("Fit iteration %d: %s", ii, result.model)
            if self.debugInfo.plot:
                self.plotModel(lines, used, result)
            if self.debugInfo.residuals:
                self.plotResiduals(result.model, lines, used, reserved)
            with np.errstate(invalid="ignore"):
                newUsed = (good & ~reserved & (np.abs(result.xResid/xErr) < self.config.rejection) &
                           (np.abs(result.yResid/yErr) < self.config.rejection))
            self.log.debug("Rejecting %d/%d lines in iteration %d", used.sum() - newUsed.sum(),
                           used.sum(), ii)
            if np.all(newUsed == used):
                # Converged
                break
            used = newUsed

        result = self.fitModel(bbox, lines, used, doFitHighCcd, fiberCenter, minMaxFiberId=minMaxFiberId)
        self.log.info("Final fit: chi2=%f dof=%d xRMS=%f yRMS=%f (%f nm, %f km/s) from %d/%d lines",
                      result.chi2, result.dof, result.xRms, result.yRms,
                      result.yRms*result.model.getScaling().dispersion,
                      rmsPixelsToVelocity(result.yRms, result.model), used.sum(), numGood - numReserved)
        reservedStats = calculateFitStatistics(result.model, lines, reserved, self.config.soften)
        self.log.info("Fit quality from reserved lines: "
                      "chi2=%f xRMS=%f yRMS=%f (%f nm, %f km/s) from %d lines (%.1f%%)",
                      reservedStats.chi2, reservedStats.xRobustRms, reservedStats.yRobustRms,
                      reservedStats.yRobustRms*result.model.getScaling().dispersion,
                      rmsPixelsToVelocity(reservedStats.yRobustRms, result.model), reserved.sum(),
                      reserved.sum()/numGood*100)
        self.log.debug("    Final fit model: %s", result.model)

        result = self.fitSoftenedModel(bbox, lines, used, reserved, result, doFitHighCcd, fiberCenter,
                                       minMaxFiberId)
        result.used = used
        result.reserved = reserved
        if self.debugInfo.plot:
            self.plotModel(lines, used, result)
        if self.debugInfo.distortion:
            self.plotDistortion(result.model, lines, used)
        if self.debugInfo.residuals:
            self.plotResiduals(result.model, lines, used, reserved)
        return result

    def fitModel(self, bbox, lines, select, doFitHighCcd, fiberCenter=0, soften=None, minMaxFiberId=None):
        """Fit a model to the arc lines

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        select : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        doFitHighCcd : `bool`
            Fit affine transformation for high-fiberId CCD?
        fiberCenter : `float`
            Central fiberId, separating low- and high-fiberId CCDs.
        soften : `float`, optional
            Systematic error to add in quadrature to measured errors (pixels).
        minMaxFiberId : `tuple` (`int`, `int`), optional
            Minimum and maximum fiber identifiers to cover. If not provided, use
            ``lines.fiberId``.

        Returns
        -------
        model : `pfs.drp.stella.GlobalDetectorModel`
            Model that was fit to the data.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Residual RMS in x,y (pixels)
        chi2 : `float`
            Fit chi^2.
        soften : `float`
            Systematic error that was applied to measured errors (pixels).
        """
        numDistortion = GlobalDetectorModel.getNumDistortion(self.config.order)

        if soften is None:
            soften = self.config.soften

        scaling = self.fitScaling(bbox, lines, select, minMaxFiberId)

        fiberId = lines.fiberId[select].astype(np.int32)  # Overwrite fiberId variable
        wavelength = lines.wavelength[select].astype(float)
        xx = lines.x[select].astype(float)
        yy = lines.y[select].astype(float)
        onHighCcd = lines.fiberId[select] > fiberCenter
        xErr = np.hypot(lines.xErr[select].astype(float), soften)
        yErr = np.hypot(lines.yErr[select].astype(float), soften)

        # Set up the least-squares equations
        xiEta = scaling(fiberId, wavelength)
        xi = xiEta[:, 0]
        eta = xiEta[:, 1]
        design = GlobalDetectorModel.calculateDesignMatrix(self.config.order, scaling.getRange(), xiEta)

        def fitDimension(values, errors):
            """Fit a distortion polynomial in a single dimension

            Parameters
            ----------
            values : `numpy.ndarray` of `float`, shape ``(N,)``
                Values to fit.
            errors : `numpy.ndarray` of `float`, shape ``(N,)``
                Errors in values.

            Returns
            -------
            distortion : `numpy.ndarray` of `float`, shape ``(M,)``
                Coefficients of distortion polynomial.
            offset : `float`
                Constant offset for high-fiberId CCD.
            xiRot, etaRot : `float`
                Linear terms in xi and eta for high-fiberId CCD.
            """
            numExtra = 3 if doFitHighCcd else 1
            dm = np.zeros((len(values), numDistortion + numExtra), dtype=float)  # design matrix
            dm[:, :-numExtra] = design/errors[:, np.newaxis]  # Weighting
            # Insert additional terms for CCD offset and rotation
            # We always fit the offset: there's a discontinuity in fiberIds at the center
            dm[:, -1] = np.where(onHighCcd, 1.0/errors, 0.0)  # CCD offset
            if doFitHighCcd:
                dm[:, -3] = np.where(onHighCcd, xi/errors, 0.0)  # CCD rotation
                dm[:, -2] = np.where(onHighCcd, eta/errors, 0.0)  # CCD rotation

            fisher = np.matmul(dm.T, dm)
            rhs = np.matmul(dm.T, values/errors)
            equation = LeastSquares.fromNormalEquations(fisher, rhs)
            solution = equation.getSolution()
            return Struct(
                distortion=solution[:-numExtra].copy(),
                offset=solution[-1],
                xiRot=solution[-3] if doFitHighCcd else 0.0,
                etaRot=solution[-2] if doFitHighCcd else 0.0,
            )

        xParams = fitDimension(xx, xErr)
        yParams = fitDimension(yy, yErr)

        highCcd = GlobalDetectorModel.makeHighCcdCoefficients(
            xParams.offset, yParams.offset,
            xParams.xiRot, xParams.etaRot,
            yParams.xiRot, yParams.etaRot
        )

        model = GlobalDetectorModel(self.config.order, scaling, fiberCenter,
                                    xParams.distortion, yParams.distortion, highCcd)
        return calculateFitStatistics(model, lines, select, soften)

    def fitSoftenedModel(self, bbox, lines, select, reserved, result, doFitHighCcd, fiberCenter=0,
                         minMaxFiberId=None):
        """Fit with errors adjusted so that chi^2/dof <= 1

        This provides a measure of the systematic error.

        Parameters
        ----------
        arm : `str`
            Spectrograph arm (``b``, ``r``, ``n``, ``m``).
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        select : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        reserved : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` have been reserved for QA.
        result : `lsst.pipe.base.Struct`
            Results from ``fitModel``.
        doFitHighCcd : `bool`
            Fit affine transformation for high-fiberId CCD?
        fiberCenter : `float`
            Central fiberId, separating low- and high-fiberId CCDs.
        minMaxFiberId : `tuple` (`int`, `int`), optional
            Minimum and maximum fiber identifiers to cover. If not provided, use
            ``lines.fiberId``.

        Returns
        -------
        model : `pfs.drp.stella.GlobalDetectorModel`
            Model that was fit to the data.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Residual RMS in x,y (pixels)
        chi2 : `float`
            Fit chi^2.
        soften : `float`
            Systematic error that was applied to measured errors (pixels).
        """
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
            colChi2 = np.sum(result.xResid[select]**2/(soften**2 + lines.xErr[select]**2))
            rowChi2 = np.sum(result.yResid[select]**2/(soften**2 + lines.yErr[select]**2))
            return (colChi2 + rowChi2)/result.dof - 1

        if softenChi2(0.0) <= 0:
            self.log.info("No softening necessary")
            return result

        soften = scipy.optimize.bisect(softenChi2, 0.0, 1.0)
        self.log.info("Softening errors by %f pixels (%f nm, %f km/s) to yield chi^2/dof=1", soften,
                      soften*result.model.getScaling().dispersion,
                      rmsPixelsToVelocity(soften, result.model))

        result = self.fitModel(bbox, lines, select, doFitHighCcd, fiberCenter, soften, minMaxFiberId)
        self.log.info("Softened fit: chi2=%f dof=%d xRMS=%f yRMS=%f (%f nm, %f km/s) from %d/%d lines",
                      result.chi2, result.dof, result.xRms, result.yRms,
                      result.yRms*result.model.getScaling().dispersion,
                      rmsPixelsToVelocity(result.yRms, result.model), select.sum(), len(select))

        reservedStats = calculateFitStatistics(result.model, lines, reserved, soften)
        self.log.info("Softened fit quality from reserved lines: "
                      "chi2=%f xRMS=%f yRMS=%f (%f nm, %f km/s) from %d lines",
                      reservedStats.chi2, reservedStats.xRobustRms, reservedStats.yRobustRms,
                      reservedStats.yRobustRms*result.model.getScaling().dispersion,
                      rmsPixelsToVelocity(reservedStats.yRobustRms, result.model), reserved.sum())
        self.log.debug("    Softened fit model: %s", result.model)
        return result

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

        numFibers = result.model.getNumFibers()
        numLines = min(10, numFibers)
        fiberId = result.model.getFiberId()[np.linspace(0, numFibers, numLines, False, dtype=int)]
        wavelength = np.linspace(lines.wavelength.min(), lines.wavelength.max(), numLines)
        ff, wl = np.meshgrid(fiberId, wavelength)
        ff = ff.flatten()
        wl = wl.flatten()
        xy = result.model(ff, wl)
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

    def plotDistortion(self, model, lines, select):
        """Plot distortion field

        We plot the x and y distortions as a function of xi,eta.

        Parameters
        ----------
        model : `pfs.drp.stella.GlobalDetectorModel`
            Model containing distortion.
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

        xiEtaRange = model.getScaling().getRange()
        xiEtaModel = np.meshgrid(np.linspace(xiEtaRange.getMinX(), xiEtaRange.getMaxX(), numSamples),
                                 np.linspace(xiEtaRange.getMinY(), xiEtaRange.getMaxY(), numSamples),
                                 sparse=False)
        xiModel = xiEtaModel[0].flatten()
        etaModel = xiEtaModel[1].flatten()

        def calculateXiEta(fiberId, wavelength):
            # Coordinates for input to polynomial
            xiEta = model.getScaling()(fiberId.astype(np.int32), wavelength.astype(float))
            return xiEta[:, 0].copy(), xiEta[:, 1].copy()  # Copy to force C-contiguous

        def calculateXiEtaNorm(xi, eta):
            # Coordinates for plotting
            xiNorm = (xi - xiEtaRange.getMinX())/(xiEtaRange.getMaxX() - xiEtaRange.getMinX())
            etaNorm = (eta - xiEtaRange.getMinY())/(xiEtaRange.getMaxY() - xiEtaRange.getMinY())
            return xiNorm, etaNorm

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
            return evaluatePolynomial(distortion, xiModel, etaModel).reshape(numSamples, numSamples)

        xDistortion = getDistortion(model.getXDistortion())
        yDistortion = getDistortion(model.getYDistortion())

        fiberId = model.getFiberId()
        wavelength = np.full_like(fiberId, model.getScaling().wavelengthCenter, dtype=float)
        fiberNorm = calculateXiEtaNorm(*calculateXiEta(fiberId, wavelength))[0]

        xiObs, etaObs = calculateXiEta(lines.fiberId[select], lines.wavelength[select])
        xiObsNorm, etaObsNorm = calculateXiEtaNorm(xiObs, etaObs)

        def removeLinear(values, poly):
            params = np.array(poly.getParameters())
            params[3:] = 0.0
            linear = type(poly)(params, poly.getXYRange())
            return values - evaluatePolynomial(linear, xiObs, etaObs)

        # For the observed positions, we need to remove the linear part of the distortion and the
        # affine transformation for the high-fiberId CCD.
        xObs = removeLinear(lines.x[select], model.getXDistortion())
        yObs = removeLinear(lines.y[select], model.getYDistortion())

        onHighCcd = model.getOnHighCcd(lines.fiberId[select])
        highCcd = evaluateAffineTransform(model.getHighCcd(), xiObs[onHighCcd], etaObs[onHighCcd])
        xObs[onHighCcd] -= highCcd[0]
        yObs[onHighCcd] -= highCcd[1]

        fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)
        for ax, image, values, dim in zip(axes, (xDistortion, yDistortion), (xObs, yObs), ("x", "y")):
            norm = Normalize()
            norm.autoscale(image)
            ax[0].imshow(image, cmap=cmap, norm=norm, origin="lower", extent=(0, 1, 0, 1))
            for ff in fiberNorm:
                ax[0].axvline(ff, color="k", alpha=0.25)
            ax[0].set_xticks((0, 1))
            ax[0].set_yticks((0, 1))
            ax[0].set_title(f"Model {dim}")
            ax[1].scatter(xiObsNorm, etaObsNorm, marker=".", alpha=0.2, color=cmap(norm(values)))
            ax[1].set_title(f"Observed {dim}")
            ax[1].set_aspect("equal")
            addColorbar(fig, ax[0], cmap, norm, f"{dim} distortion (pixels)")
            addColorbar(fig, ax[1], cmap, norm, f"{dim} distortion (pixels)")

        axes[0][0].set_ylabel("Normalized eta (wavelength)")
        axes[1][0].set_ylabel("Normalized eta (wavelength)")
        axes[1][0].set_xlabel("Normalized xi (fiberId)")
        axes[1][1].set_xlabel("Normalized xi (fiberId)")

        fig.tight_layout()
        fig.suptitle("Distortion field")
        plt.show()

    def plotResiduals(self, model, lines, good, reserved):
        """Plot fit residuals

        We plot the x and y residuals as a function of fiberId,wavelength

        Parameters
        ----------
        model : `pfs.drp.stella.GlobalDetectorModel`
            Model containing distortion.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        good : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` were reserved from the fit.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from matplotlib.colors import Normalize

        xy = model(lines.fiberId.astype(np.int32), lines.wavelength.astype(float))
        dx = lines.x - xy[:, 0]
        dy = lines.y - xy[:, 1]

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
            [(good & ~reserved), reserved, (~good & ~reserved)],
            ["Used", "Reserved", "Bad"],
        ):
            if not np.any(select):
                continue
            xNorm = calculateNormalization(dx[select])
            yNorm = calculateNormalization(dy[select])
            ax[0].scatter(lines.fiberId[select], lines.wavelength[select], marker=".", alpha=0.2,
                          color=cmap(xNorm(dx[select])))
            ax[1].scatter(lines.fiberId[select], lines.wavelength[select], marker=".", alpha=0.2,
                          color=cmap(yNorm(dx[select])))
            ax[0].set_title(label)
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
