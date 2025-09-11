from __future__ import annotations

from collections import defaultdict
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

from lsst.pex.config import ConfigurableField, Field, ListField
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

if TYPE_CHECKING:
    from lsst.pipe.base import QuantumContext
    from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection


def fitScales(
    data: list[np.ndarray], model: list[np.ndarray], variance: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
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
    variance : `numpy.ndarray`, shape ``(numSpectra,)``
        Variance of the factor.
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
        return np.array(dataDotModel/modelDotModel), modelDotModel**-1


class MeasureSkyNormsConnections(
    PipelineTaskConnections, dimensions=("instrument", "visit")
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
        dimensions=("instrument", "visit"),
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
    doFitFocalPlane = Field(dtype=bool, default=True, doc="Fit focal plane model to sky norms?")
    fitFocalPlane = ConfigurableField(target=FitFocalPlanePolynomialTask, doc="Fit focal plane model")

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
    """Measure sky norms for a single exposure

    This task operates on the entire exposure at once, so that we can also
    remove a fit over the focal plane.
    """
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

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection
    ) -> None:
        pfsArmList = defaultdict(list)
        for inputRef in inputRefs.pfsArm:
            pfsArm = butler.get(inputRef)
            spectrograph = inputRef.dataId["spectrograph"]
            pfsArmList[spectrograph].append(pfsArm)
        pfsConfig = butler.get(inputRefs.pfsConfig)
        outputs = self.run(pfsArmList, pfsConfig)
        butler.put(outputs.skyNorms, outputRefs.skyNorms)

    def run(self, pfsArmList: dict[int, list[PfsFiberArraySet]], pfsConfig: PfsConfig) -> Struct:
        """Measure sky normalizations for the exposure

        Parameters
        ----------
        pfsArmList : `dict` mapping `int` to `list` of `PfsFiberArraySet`
            Extracted spectra from each arm within a spectrograph module,
            indexed by spectrograph number.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.

        Returns
        -------
        skyNorms : `ConstantPerFiber`
            Normalizations of sky data, after removing focal plane fit.
        perSpectrograph : `dict` mapping `int` to `Struct`
            Results for each spectrograph, indexed by spectrograph number.
        focalPlane : `FocalPlanePolynomial`
            Focal plane fit to sky normalizations.
        """
        results = {ss: self.runSpectrograph(pfsArmList[ss], pfsConfig) for ss in pfsArmList}
        allSkyNorms = ConstantPerFiber.concatenate(*(results[ss].skyNorms for ss in results))

        focalPlane = None
        if self.config.doFitFocalPlane:
            corrected = self.correctFocalPlane(allSkyNorms, pfsConfig)
            allSkyNorms = corrected.skyNorms
            focalPlane = corrected.fit

        return Struct(skyNorms=allSkyNorms, perSpectrograph=results, focalPlane=focalPlane)

    def runSpectrograph(self, pfsArmList: list[PfsFiberArraySet], pfsConfig: PfsConfig) -> Struct:
        """Measure sky normalizations for a single spectrograph

        Parameters
        ----------
        pfsArmList : `list` of `PfsFiberArraySet`
            Extracted spectra from each arm within a spectrograph module.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.

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
        fiberId = pfsArmList[0].fiberId
        for pfsArm in pfsArmList:
            if not np.array_equal(pfsArm.fiberId, fiberId):
                raise RuntimeError("Mismatched fiberId arrays")
        pfsConfig = pfsConfig.select(fiberId=fiberId)

        skyModels = [self.measureSky(pfsArm, pfsConfig) for pfsArm in pfsArmList]
        skyData = [sm(pfsArm.wavelength, pfsConfig) for sm, pfsArm in zip(skyModels, pfsArmList)]
        refLines = [self.readLineList.run(metadata=pfsArm.metadata) for pfsArm in pfsArmList]

        result = self.measureNormalizations(pfsArmList, skyData, refLines)

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
            factor, _ = fitScales(dataList, modelList, varianceList)
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
        factor, _ = fitScales(dataList, modelList, varianceList)
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

    def correctFocalPlane(self, skyNorms: ConstantPerFiber, pfsConfig: PfsConfig) -> Struct:
        """Fit focal plane model to sky normalizations

        Parameters
        ----------
        skyNorms : `list` of `ConstantPerFiber`
            Sky normalizations to fit.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration of fibers on the focal plane.

        Returns
        -------
        fit : `FocalPlanePolynomial`
            Result of fitting the focal plane model.
        values : `np.ndarray`, shape ``(numFibers,)``
            Focal plane correction values for each fiber.
        skyNorms : `ConstantPerFiber`
            Sky normalizations after removing focal plane fit.
        """
        numFibers = len(skyNorms)
        positions = pfsConfig.select(fiberId=skyNorms.fiberId).pfiCenter

        fit = self.fitFocalPlane.fitArrays(
            skyNorms.fiberId,
            np.full((numFibers, 1), np.nan, dtype=float),
            skyNorms.value[:, None],
            (~np.isfinite(skyNorms.value) | np.any(~np.isfinite(positions), axis=1))[:, None],
            skyNorms.rms[:, None]**2,
            positions,
        )
        correction = fit.eval(positions).values
        corrected = ConstantPerFiber(
            skyNorms.fiberId,
            skyNorms.value - correction.reshape(-1),
            skyNorms.rms,
        )
        return Struct(fit=fit, values=correction, skyNorms=corrected)


class CombineSkyNormsConnections(PipelineTaskConnections, dimensions=("instrument",)):
    """Connections for CombineSkyNormsTask"""
    skyNormsList = InputConnection(
        name="skyNorms",
        doc="Sky normalizations",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit"),
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

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection
    ) -> None:
        skyNormsList = defaultdict(list)
        for inputRef in inputRefs.skyNormsList:
            skyNorms = butler.get(inputRef)
            visit = inputRef.dataId["visit"]
            skyNormsList[visit].append(skyNorms)
        visitList = sorted(skyNormsList)
        outputs = self.run([ConstantPerFiber.concatenate(*skyNormsList[vv]) for vv in visitList])
        butler.put(outputs, outputRefs)

    def run(
        self, skyNormsList: list[ConstantPerFiber]
    ) -> Struct:
        """Combine sky normalizations from multiple visits

        Parameters
        ----------
        skyNormsList : `list` of `ConstantPerFiber`
            Sky normalizations for each visit.

        Returns
        -------
        combined : `FocalPlaneFunction`
            Combined sky normalizations.
        """
        fiberId = skyNormsList[0].fiberId
        for skyNorms in skyNormsList:
            if not np.array_equal(skyNorms.fiberId, fiberId):
                raise RuntimeError("Mismatched fiberId arrays")

        numFibers = len(fiberId)
        numMeasurements = len(skyNormsList)
        shape = (numFibers, numMeasurements)

        values = np.full(shape, np.nan, dtype=float)
        variances = np.full(shape, np.nan, dtype=float)

        for ii, skyNorms in enumerate(skyNormsList):
            values[:, ii] = skyNorms.value
            variances[:, ii] = skyNorms.rms**2

        masks = ~np.isfinite(values) | ~np.isfinite(variances)
        wavelength = np.full(shape, np.nan, dtype=float)  # Not used
        positions = np.full((numFibers, 2), np.nan, dtype=float)  # Not used
        combined = self.fit.fitArrays(fiberId, wavelength, values, masks, variances, positions)
        return Struct(combined=combined)
