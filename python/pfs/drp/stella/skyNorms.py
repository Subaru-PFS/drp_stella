from __future__ import annotations

from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

from lsst.pex.config import Config, ConfigurableField, Field, ListField
from lsst.pipe.base import Struct, Task

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from pfs.datamodel import PfsFiberArraySet, PfsConfig
from .focalPlaneFunction import FocalPlaneFunction, ConstantPerFiber, FocalPlanePolynomial
from .fitContinuum import FitContinuumTask
from .fitFocalPlane import FitBlockedOversampledSplineTask, FitConstantPerFiberTask
from .fitFocalPlane import FitFocalPlanePolynomialTask
from .math import calculateMedian
from .readLineList import ReadLineListTask
from .referenceLine import ReferenceLineSet
from .selectFibers import SelectFibersTask
from .utils.math import robustRms

if TYPE_CHECKING:
    from lsst.pipe.base import QuantumContext
    from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection


def applySkyNorms(pfsArm: PfsFiberArraySet, skyNorms: FocalPlaneFunction) -> None:
    """Apply sky normalizations to a pfsArm in-place

    Parameters
    ----------
    pfsArm : `PfsFiberArraySet`
        Spectra to which to apply sky normalizations.
    skyNorms : `ConstantPerFiber`
        Sky normalizations to apply.
    """
    data = skyNorms.eval(pfsArm.fiberId)
    pfsArm.norm *= data.values
    # Dropping the variance on the floor for now
    pfsArm.masks[data.masks[:, None]] |= pfsArm.flags.add("BAD_SKY_NORMS")


def fitScales(data: list[np.ndarray], model: list[np.ndarray], variance: list[np.ndarray]) -> np.ndarray:
    """Fit model scale factors to spectra

    We fit a single scale factor for each fiber.

    The inputs may be masked arrays.

    Parameters
    ----------
    data : `list` of `numpy.ndarray`, shape ``(numSpectra, numSamples)``
        Data to fit.
    model : `list` of `numpy.ndarray`, shape ``(numSpectra, numSamples)``
        Model to fit (evaluated at the same points as ``data``).
    variance : `list` of `numpy.ndarray`, shape ``(numSpectra, numSamples)``
        Variance of the data.

    Returns
    -------
    factor : `numpy.ndarray`, shape ``(numSpectra,)``
        Factor by which to multiply the model to best fit the data.
    """
    for dd, mm, vv in zip(data, model, variance):
        if dd.shape != mm.shape or dd.shape != vv.shape:
            raise ValueError("data, model, and variance must have the same shape")
    modelDotModel = reduce(
        np.add,
        (np.ma.sum(mm**2/vv, axis=1).filled(0.0) for mm, vv in zip(model, variance)),
    )
    dataDotModel = reduce(
        np.add,
        (np.ma.sum(dd*mm/vv, axis=1).filled(0.0) for dd, mm, vv in zip(data, model, variance)),
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.array(dataDotModel/modelDotModel)


class MeasureSkyNormsConnections(
    PipelineTaskConnections, dimensions=("instrument", "visit", "spectrograph")
):
    """Connections for MeasureSkyNormsTask"""
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
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
        dimensions=("instrument", "visit", "spectrograph"),
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")
        self.makeSubtask("selectSky")
        self.makeSubtask("fitSkyModel")
        self.makeSubtask("fitContinuum")

    def run(
        self,
        pfsArm: list[PfsFiberArraySet],
        pfsConfig: PfsConfig,
        skyNorms: FocalPlaneFunction | None = None,
    ) -> Struct:
        """Measure sky normalizations

        Parameters
        ----------
        pfsArm : list of `PfsFiberArraySet`
            Extracted spectra from each arm within a spectrograph module.
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
        """
        fiberId = pfsArm[0].fiberId
        for arm in pfsArm:
            if not np.array_equal(arm.fiberId, fiberId):
                raise RuntimeError("Mismatched fiberId arrays")
        pfsConfig = pfsConfig.select(fiberId=fiberId)

        if skyNorms is not None:
            for arm in pfsArm:
                applySkyNorms(arm, skyNorms)

        skyModels = [self.measureSky(arm, pfsConfig) for arm in pfsArm]
        skyData = [sm(arm.wavelength, pfsConfig) for sm, arm in zip(skyModels, pfsArm)]
        refLines = [self.readLineList.run(metadata=arm.metadata) for arm in pfsArm]

        result = self.measureNormalizations(pfsArm, skyData, refLines)

        result.sky = skyModels
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
        pfsArmList: list[PfsFiberArraySet],
        skyDataList: list[Struct],
        refLinesList: list[ReferenceLineSet | None] | None = None,
    ) -> Struct:
        """Measure normalizations for sky model

        We currently measure a single normalization factor for each fiber, but
        this could be generalized to a more complex model.

        Parameters
        ----------
        pfsArmList : `list` of `PfsFiberArraySet`
            Spectra from which to subtract sky model.
        skyDataList : `list` of `Struct`
            Realized sky model spectra for each pfsArm.
        refLinesList : `list` of `ReferenceLineSet`, optional
            Sky line list for each pfsArm; used for fitting continuum.

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
        num = len(pfsArmList)
        if refLinesList is None:
            refLinesList = [None]*num
        assert refLinesList is not None
        numFibers = len(pfsArmList[0])
        for pfsArm in pfsArmList:
            if len(pfsArm) != numFibers:
                raise RuntimeError("Mismatched number of fibers")

        armContinuumList = []
        skyContinuumList = []
        rejectList = []
        dataList = []
        modelList = []
        varianceList = []

        for pfsArm, skyData, refLines in zip(pfsArmList, skyDataList, refLinesList):
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

            armContinuumList.append(armContinuum)
            skyContinuumList.append(skyContinuum)
            rejectList.append(reject)
            dataList.append(np.ma.masked_where(reject, flux))
            modelList.append(np.ma.masked_where(reject, skyFlux))
            varianceList.append(np.ma.masked_where(reject, var))

        for iteration in range(self.config.iterations):
            factor = fitScales(dataList, modelList, varianceList)
            self.log.debug(
                "Iteration %d: factor = %.3f +/- %.3f",
                iteration,
                np.nanmedian(np.array(factor)),
                robustRms(factor, True),
            )
            for dd, mm, vv in zip(dataList, modelList, varianceList):
                residuals = dd - factor[:, None]*mm
                reject = np.abs(residuals) > self.config.rejection*np.sqrt(vv)
                dd.mask |= reject
                mm.mask |= reject

        # Final fit after rejection iterations
        factor = fitScales(dataList, modelList, varianceList)
        self.log.info(
            "Sky norms = %.3f +/- %.3f",
            np.nanmedian(np.array(factor)),
            robustRms(factor, True),
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            residuals = [dd - factor[:, None]*mm for dd, mm in zip(dataList, modelList)]
            rms = np.array([
                robustRms(np.concatenate([
                    res[ii].compressed() for res in residuals
                ])) for ii in range(numFibers)
            ])

        return Struct(
            skyNorms=ConstantPerFiber(pfsArmList[0].fiberId, factor, rms),
            values=factor,
            rms=rms,
            armContinuum=armContinuumList,
            skyContinuum=skyContinuumList,
            data=dataList,
            model=modelList,
            variance=varianceList,
        )


class SkyNormsFocalPlaneCorrectionConfig(Config):
    fit = ConfigurableField(target=FitFocalPlanePolynomialTask, doc="Fit focal plane model")


class SkyNormsFocalPlaneCorrectionTask(Task):
    """Apply focal plane correction to sky normalizations

    This task applies a position-dependent correction to sky normalizations,
    accounting for vignetting and fiber throughput variations.
    """
    ConfigClass = SkyNormsFocalPlaneCorrectionConfig
    _DefaultName = "skyNormsFocalPlaneCorrection"

    config: SkyNormsFocalPlaneCorrectionConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fit")

    def run(self, skyNorms: ConstantPerFiber, pfsConfig: PfsConfig) -> Struct:
        """Apply focal plane correction to sky normalizations

        Parameters
        ----------
        skyNorms : `ConstantPerFiber`
            Sky normalizations to correct.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration of fibers on the focal plane.

        Returns
        -------
        fit : `FocalPlanePolynomial`
            Focal plane fit.
        values : `list` of `Struct`
            Values of focal plane fit for each of the skyNorms.
        """
        fit = self.fitFocalPlane(skyNorms, pfsConfig)
        values = fit.eval(pfsConfig.select(fiberId=skyNorms.fiberId).pfiCenter)
        return Struct(fit=fit, values=values)

    def fitFocalPlane(self, skyNorms: ConstantPerFiber, pfsConfig: PfsConfig) -> FocalPlanePolynomial:
        """Fit focal plane model to sky normalizations

        Parameters
        ----------
        skyNorms : `list` of `ConstantPerFiber`
            Sky normalizations to fit.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration of fibers on the focal plane.

        Returns
        -------
        result : `FocalPlanePolynomial`
            Result of fitting the focal plane model.
        """
        numFibers = len(skyNorms)
        return self.config.fit.fitArrays(
            skyNorms.fiberId,
            np.full(numFibers, np.nan, dtype=float),
            skyNorms.value[:, None],
            np.isfinite(skyNorms.value)[:, None],
            skyNorms.rms[:, None]**2,
            pfsConfig.select(fiberId=skyNorms.fiberId).pfiCenter,
        )


class CombineSkyNormsConnections(PipelineTaskConnections, dimensions=("instrument",)):
    """Connections for CombineSkyNormsTask"""
    pfsConfigList = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        multiple=True,
    )
    skyNormsList = InputConnection(
        name="skyNorms",
        doc="Sky normalizations",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "spectrograph"),
        multiple=True,
    )

    combined = OutputConnection(
        name="skyNorms_calib",
        doc="Sky normalizations for calibration",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument",),
        isCalibration=True,
    )


class CombineSkyNormsConfig(PipelineTaskConfig, pipelineConnections=CombineSkyNormsConnections):
    """Configuration for CombineSkyNormsTask"""
    focalPlane = ConfigurableField(target=SkyNormsFocalPlaneCorrectionTask, doc="Focal plane correction")
    fitConstant = ConfigurableField(target=FitConstantPerFiberTask, doc="Fit constant per fiber")


class CombineSkyNormsTask(PipelineTask):
    """Combine sky normalizations from multiple visits"""
    _DefaultName = "combineSkyNorms"
    ConfigClass = CombineSkyNormsConfig

    config: CombineSkyNormsConfig
    focalPlane: SkyNormsFocalPlaneCorrectionTask
    fitConstant: FitConstantPerFiberTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("focalPlane")
        self.makeSubtask("fitConstant")

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection
    ) -> None:
        skyNormsList = defaultdict(list)
        pfsConfigList = {ref.dataId["visit"]: butler.get(ref) for ref in inputRefs["pfsConfig"]}
        for inputRef in inputRefs["skyNormsList"]:
            skyNorms = butler.get(inputRef)
            visit = inputRef.dataId["visit"]
            skyNormsList[visit].append(skyNorms)
        visitList = sorted(skyNormsList)
        outputs = self.run(
            [ConstantPerFiber.concatenate(*skyNormsList[vv]) for vv in visitList],
            [pfsConfigList[vv] for vv in visitList],
        )
        butler.put(outputs, outputRefs)

    def run(
        self, skyNormsList: list[ConstantPerFiber], pfsConfigList: list[PfsConfig]
    ) -> Struct:
        """Combine sky normalizations from multiple visits

        Parameters
        ----------
        skyNormsList : `list` of `ConstantPerFiber`
            Sky normalizations for each visit.
        pfsConfigList : `list` of `PfsConfig`
            PFS fiber configuration for each visit.

        Returns
        -------
        combined : `FocalPlaneFunction`
            Combined sky normalizations.
        """
        fiberId = pfsConfigList[0].fiberId
        for pfsConfig in pfsConfigList:
            if not np.array_equal(pfsConfig.fiberId, fiberId):
                raise RuntimeError("Mismatched fiberId arrays")

        numFibers = len(fiberId)
        numMeasurements = len(skyNormsList)
        shape = (numFibers, numMeasurements)

        values = np.full(shape, np.nan, dtype=float)
        variances = np.full(shape, np.nan, dtype=float)

        for ii, (skyNorms, pfsConfig) in enumerate(zip(skyNormsList, pfsConfigList)):
            focalPlane = self.focalPlane.run(skyNorms, pfsConfig)
            indices = np.searchsorted(fiberId, skyNorms.fiberId)
            if not np.all(fiberId[indices] == skyNorms.fiberId):
                raise RuntimeError("Unable to match fiberId arrays")
            values[indices, ii] = skyNorms.value - focalPlane.values.reshape(-1)
            variances[indices, ii] = skyNorms.rms**2

        masks = ~np.isfinite(values) | ~np.isfinite(variances)
        wavelength = np.full(shape, np.nan, dtype=float)  # Not used
        positions = np.full((numFibers, 2), np.nan, dtype=float)  # Not used
        combined = self.fitConstant.fitArrays(fiberId, wavelength, values, masks, variances, positions)
        return Struct(combined=combined)
