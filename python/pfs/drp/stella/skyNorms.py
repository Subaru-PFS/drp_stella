from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lsst.pex.config import Field, ConfigurableField, ListField
from lsst.pipe.base import Struct

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from pfs.datamodel import PfsFiberArraySet, PfsConfig
from .focalPlaneFunction import FocalPlaneFunction, ConstantPerFiber
from .fitContinuum import FitContinuumTask
from .fitFocalPlane import FitBlockedOversampledSplineTask, FitConstantPerFiberTask
from .fitFocalPlane import FitFocalPlanePolynomialTask
from .math import calculateMedian
from .readLineList import ReadLineListTask
from .referenceLine import ReferenceLineSet
from .selectFibers import SelectFibersTask
from .utils.math import robustRms


def fitScales(data: np.ndarray, model: np.ndarray, variance: np.ndarray) -> np.ndarray:
    """Fit model scale factors to spectra

    We fit a single scale factor for each spectrum.

    The inputs may be masked arrays.

    Parameters
    ----------
    data : `numpy.ndarray`, shape ``(numSpectra, numSamples)``
        Data to fit.
    model : `numpy.ndarray`, shape ``(numSpectra, numSamples)``
        Model to fit (evaluated at the same points as ``data``).
    variance : `numpy.ndarray`, shape ``(numSpectra, numSamples)``
        Variance of the data.

    Returns
    -------
    factor : `numpy.ndarray`, shape ``(numSpectra,)``
        Factor by which to multiply the model to best fit the data.
    """
    if data.shape != model.shape or data.shape != variance.shape:
        raise ValueError("data, model, and variance must have the same shape")
    modelDotModel = np.ma.sum(model**2/variance, axis=1).filled(0.0)
    dataDotModel = np.ma.sum(data*model/variance, axis=1).filled(0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.array(dataDotModel/modelDotModel)


class MeasureSkyNormsConnections(
    PipelineTaskConnections, dimensions=("instrument", "visit", "arm", "spectrograph")
):
    """Connections for MeasureSkyNormsTask"""
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )

    skyNorms = OutputConnection(
        name="skyNorms",
        doc="Sky normalizations",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    focalPlaneFit = OutputConnection(
        name="skyNorms_focalPlane",
        doc="Focal plane fit to sky normalizations",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class MeasureSkyNormsConfig(PipelineTaskConfig, pipelineConnections=MeasureSkyNormsConnections):
    """Configuration for MeasureSkyNormsTask"""
    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read line list")
    selectSky = ConfigurableField(target=SelectFibersTask, doc="Select fibers for 1d sky subtraction")
    fitSkyModel = ConfigurableField(
        target=FitBlockedOversampledSplineTask, doc="Fit sky model splines over the focal plane"
    )
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum")
    minSignalToNoise = Field(dtype=float, default=5.0, doc="Minimum signal-to-noise for data to fit")
    mask = ListField(
        dtype=str,
        default=["BAD", "CR", "SAT", "BAD_FLAT", "BAD_FIBERNORMS", "SUSPECT", "NO_DATA"],
        doc="Mask bits to reject",
    )
    minRatio = Field(dtype=float, default=0.5, doc="Minimum data/sky ratio for good data")
    maxRatio = Field(dtype=float, default=2.0, doc="Maximum data/sky ratio for good data")
    rejectRatio = Field(dtype=float, default=0.1, doc="Rejection limit of data/sky ratio")
    iterations = Field(dtype=int, default=3, doc="Number of fitting iterations")
    rejection = Field(dtype=float, default=3.0, doc="Rejection limit for residuals")
    fitFocalPlane = ConfigurableField(
        target=FitFocalPlanePolynomialTask, doc="Fit polynomial over the focal plane"
    )

    def setDefaults(self):
        super().setDefaults()
        self.readLineList.minIntensity = 100  # Don't worry about faint lines for continuum fitting
        self.selectSky.targetType = ("SKY", "SUNSS_DIFFUSE", "HOME")
        self.selectSky.targetType = ("SKY", "SUNSS_DIFFUSE", "HOME")
        # Scale back rejection because otherwise everything gets rejected
        self.fitSkyModel.rejIterations = 1
        self.fitSkyModel.rejThreshold = 4.0
        self.fitSkyModel.mask = ["NO_DATA", "BAD_FLAT", "BAD_FIBERNORMS", "SUSPECT"]


class MeasureSkyNormsTask(PipelineTask):
    """Merge all extracted spectra from a single exposure"""
    _DefaultName = "measureSkyNorms"
    ConfigClass = MeasureSkyNormsConfig

    config: MeasureSkyNormsConfig
    readLineList: ReadLineListTask
    selectSky: SelectFibersTask
    fitSkyModel: FitBlockedOversampledSplineTask
    fitContinuum: FitContinuumTask
    fitFocalPlane: FitFocalPlanePolynomialTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")
        self.makeSubtask("selectSky")
        self.makeSubtask("fitSkyModel")
        self.makeSubtask("fitContinuum")
        self.makeSubtask("fitFocalPlane")

    def run(
        self, pfsArm: PfsFiberArraySet, pfsConfig: PfsConfig, skyNorms: FocalPlaneFunction | None = None
    ) -> Struct:
        """Measure sky normalizations

        Parameters
        ----------
        pfsArm : `PfsFiberArraySet`
            Extracted spectra from arm.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        skyNorms : `pfs.drp.stella.FocalPlaneFunction`, optional
            Common-mode sky normalizations to remove before measuring the
            residual.

        Returns
        -------
        sky : `FocalPlaneFunction`
            Sky spectral model.
        skyNorms : `ConstantPerFiber`
            Normalizations of sky data.
        armContinuum : `np.ndarray`
            Continuum fit to the ``pfsArm``.
        skyContinuum : `np.ndarray`
            Continuum fit to the sky spectral model.
        refLines : `ReferenceLineSet`
            Sky line list, used for fitting continuum.
        focalPlaneFit : `FocalPlaneFunction`
            Fit to the sky normalizations over the focal plane.
        """
        sky = self.measureSky(pfsArm, pfsConfig)
        skyData = sky(pfsArm.wavelength, pfsConfig.select(fiberId=pfsArm.fiberId))
        if skyNorms is not None:
            normData = skyNorms(pfsArm.wavelength, pfsConfig)
            skyData.values *= normData.values
            skyData.variances *= normData.values**2
            skyData.masks |= normData.masks

        refLines = self.readLineList.run(metadata=pfsArm.metadata)
        result = self.measureNormalizations(pfsArm, skyData, refLines)
        result.focalPlaneFit = self.fitFocalPlane.fitArrays(
            pfsArm.fiberId,
            np.full((len(pfsArm), 1), np.nan, dtype=float),
            result.values[:, None],
            np.isnan(result.values)[:, None],
            result.rms[:, None]**2,
            pfsConfig.pfiCenter,
        )
        focalPlaneEval = result.focalPlaneFit.eval(pfsConfig.pfiCenter)
        result.skyNorms.values -= focalPlaneEval.values

        result.sky = sky
        result.refLines = refLines

        return result

    def measureSky(self, pfsArm: PfsFiberArraySet, pfsConfig: PfsConfig) -> FocalPlaneFunction:
        """Measure sky spectra

        Parameters
        ----------
        pfsArm : `PfsFiberArraySet`
            Spectra from which to subtract sky model.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.

        Returns
        -------
        sky1d : `FocalPlaneFunction`
            Sky model.
        """
        skyConfig = self.selectSky.run(pfsConfig.select(fiberId=pfsArm.fiberId))
        skySpectra = pfsArm.select(pfsConfig, fiberId=skyConfig.fiberId)
        if len(skySpectra) == 0:
            raise RuntimeError("No sky spectra to use for sky subtraction")
        return self.fitSkyModel.run(skySpectra, skyConfig)

    def measureNormalizations(
        self,
        pfsArm: PfsFiberArraySet,
        skyData: Struct,
        refLines: ReferenceLineSet | None = None,
    ) -> Struct:
        """Measure normalizations for sky model

        We currently measure a single normalization factor for each fiber, but
        this could be generalized to a more complex model.

        Parameters
        ----------
        pfsArm : `PfsFiberArraySet`
            Spectra from which to subtract sky model.
        skyData : `Struct`
            Realized sky model spectra.
        refLines : `ReferenceLineSet`, optional
            Sky line list, used for fitting continuum.

        Returns
        -------
        skyNorms : `ConstantPerFiber`
            Normalizations of sky data for each fiber.
        values : `numpy.ndarray`, shape ``(numFibers,)``
            Normalization values for each fiber. This data is contained in the
            ``skyNorms``, but this is a more convenient form.
        rms : `numpy.ndarray`, shape ``(numFibers,)``
            RMS scatter of normalization values for each fiber. This data is
            contained in the ``skyNorms``, but this is a more convenient form.
        armContinuum : `np.ndarray`
            Continuum fit to the ``pfsArm``.
        skyContinuum : `np.ndarray`
            Continuum fit to the sky model.
        """
        armContinuum = self.fitContinuum.run(pfsArm, refLines)

        covar = np.zeros_like(pfsArm.covar)
        covar[:, 0, :] = skyData.variances
        skySpectra = PfsFiberArraySet(
            pfsArm.identity,
            pfsArm.fiberId,
            pfsArm.wavelength,
            skyData.values,
            skyData.masks,
            np.zeros_like(pfsArm.flux),
            np.ones_like(pfsArm.norm),
            covar,
            pfsArm.flags,
            pfsArm.metadata,
        )
        skyContinuum = self.fitContinuum.run(skySpectra, refLines)

        reject = ((pfsArm.mask & pfsArm.flags.get(*self.config.mask)) != 0)
        norm = pfsArm.norm
        with np.errstate(invalid="ignore", divide="ignore"):
            flux = pfsArm.flux/norm - armContinuum
            var = pfsArm.variance/norm**2
            err = np.sqrt(var)
            reject |= (flux/err < self.config.minSignalToNoise)
            skyFlux = skySpectra.flux - skyContinuum  # norm is unity: flux is already normalized
            ratio = flux/skyFlux
            reject |= (ratio < self.config.minRatio) | (ratio > self.config.maxRatio)
            factor = np.array([calculateMedian(rat, rej) for rat, rej in zip(ratio, reject)])
            reject |= np.abs(ratio - factor[:, None]) > self.config.rejectRatio

        data = np.ma.masked_where(reject, flux)
        model = np.ma.masked_where(reject, skyFlux)
        variance = np.ma.masked_where(reject, var)

        for iteration in range(self.config.iterations):
            factor = fitScales(data, model, variance)
            self.log.debug(
                "Iteration %d: factor = %.3f +/- %.3f",
                iteration,
                np.nanmedian(np.array(factor)),
                robustRms(factor, True),
            )
            residuals = data - factor[:, None]*model
            reject = np.abs(residuals) > self.config.rejection*np.sqrt(variance)
            data.mask |= reject
            model.mask |= reject

        # Final fit after rejection iterations
        factor = fitScales(data, model, variance)
        self.log.info(
            "Sky norms = %.3f +/- %.3f",
            np.nanmedian(np.array(factor)),
            robustRms(factor, True),
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            residuals = data - factor[:, None]*model
            rms = np.array([robustRms(residuals[ii].compressed()) for ii, ff in enumerate(pfsArm.fiberId)])

        return Struct(
            skyNorms=ConstantPerFiber(pfsArm.fiberId, factor, rms),
            values=factor,
            rms=rms,
            armContinuum=armContinuum,
            skyContinuum=skyContinuum,
            data=data,
            model=model,
            variance=variance,
        )


class CombineSkyNormsConnections(PipelineTaskConnections, dimensions=("instrument", "arm", "spectrograph")):
    """Connections for CombineSkyNormsTask"""
    skyNormsList = InputConnection(
        name="skyNorms",
        doc="Sky normalizations",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    combined = OutputConnection(
        name="skyNorms_calib",
        doc="Sky normalizations for calibration",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )


class CombineSkyNormsConfig(PipelineTaskConfig, pipelineConnections=CombineSkyNormsConnections):
    """Configuration for CombineSkyNormsTask"""
    fit = ConfigurableField(target=FitConstantPerFiberTask, doc="Fit constant per fiber")


class CombineSkyNormsTask(PipelineTask):
    """Combine sky normalizations from multiple visits"""
    _DefaultName = "combineSkyNorms"
    ConfigClass = CombineSkyNormsConfig

    config: CombineSkyNormsConfig
    fit: FitConstantPerFiberTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fit")

    def run(self, skyNormsList: Sequence[FocalPlaneFunction]) -> Struct:
        """Combine sky normalizations from multiple visits

        Parameters
        ----------
        skyNormsList : collection of `FocalPlaneFunction`
            Sky normalizations from multiple visits.

        Returns
        -------
        combined : `FocalPlaneFunction`
            Combined sky normalizations.
        """
        fiberId = skyNormsList[0].fiberId
        for skyNorms in skyNormsList:
            if not np.array_equal(skyNorms.fiberId, fiberId):
                raise RuntimeError("Sky normalizations from different visits have different fiberId arrays")
        numFibers = len(fiberId)
        numMeasurements = len(skyNormsList)
        shape = (numFibers, numMeasurements)

        wavelength = np.full(shape, np.nan, dtype=float)
        values = np.full(shape, np.nan, dtype=float)
        masks = np.zeros(shape, dtype=bool)
        variances = np.full(shape, np.nan, dtype=float)
        positions = np.full((numFibers, 2), np.nan, dtype=float)
        for ii, skyNorms in enumerate(skyNormsList):
            result = skyNorms.eval(fiberId)
            values[:, ii] = result.values
            masks[:, ii] = result.masks
            variances[:, ii] = result.variances

        combined = self.fit.fitArrays(fiberId, wavelength, values, masks, variances, positions)
        return Struct(combined=combined)
