from collections.abc import Collection
from functools import reduce

import numpy as np

from lsst.pex.config import Field, ConfigurableField, ListField
from lsst.pipe.base import Struct

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from pfs.datamodel import PfsFiberArraySet, PfsConfig
from .focalPlaneFunction import FocalPlaneFunction, PolynomialPerFiber
from .fitContinuum import FitContinuumTask
from .fitFocalPlane import FitBlockedOversampledSplineTask
from .gen3 import readDatasetRefs
from .math import calculateMedian
from .readLineList import ReadLineListTask
from .referenceLine import ReferenceLineSet
from .selectFibers import SelectFibersTask
from .utils.math import robustRms


class MeasureSkyNormsConnections(PipelineTaskConnections, dimensions=("instrument", "arm", "spectrograph")):
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
        multiple=True,
    )

    skyNorms = OutputConnection(
        name="skyNorms",
        doc="Sky normalizations",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
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
        dtype=str, default=["BAD", "CR", "SAT", "BAD_FLAT", "NO_DATA"], doc="Mask bits to reject"
    )
    minRatio = Field(dtype=float, default=0.5, doc="Minimum data/sky ratio for good data")
    maxRatio = Field(dtype=float, default=2.0, doc="Maximum data/sky ratio for good data")
    rejectRatio = Field(dtype=float, default=0.1, doc="Rejection limit of data/sky ratio")

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

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Run the task with the given quantum context and input/output references"""
        refs = readDatasetRefs(butler, inputRefs, "pfsArm", "pfsConfig")
        result = self.run(refs.pfsArm, refs.pfsConfig)
        butler.put(result.skyNorms, outputRefs.skyNorms)

    def runSingle(
        self, pfsArm: PfsFiberArraySet, pfsConfig: PfsConfig, skyNorms: FocalPlaneFunction | None = None
    ) -> Struct:
        """Measure sky normalizations for a single arm

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
        skyNorms : `PolynomialPerFiber`
            Normalizations of sky data.
        armContinuum : `np.ndarray`
            Continuum fit to the ``pfsArm``.
        skyContinuum : `np.ndarray`
            Continuum fit to the sky spectral model.
        refLines : `ReferenceLineSet`
            Sky line list, used for fitting continuum.
        """
        result = self.run([pfsArm], [pfsConfig], skyNorms)
        result.sky = result.sky[0]
        result.armContinuum = result.armContinuum[0]
        result.skyContinuum = result.skyContinuum[0]
        return result

    def run(
        self,
        pfsArmList: Collection[PfsFiberArraySet],
        pfsConfigList: Collection[PfsConfig],
        skyNorms: FocalPlaneFunction | None = None,
    ) -> Struct:
        """Measure sky normalizations

        This operates on multiple instances of the same arm+spectrograph.

        Parameters
        ----------
        pfsArmList : collection of `pfs.datamodel.pfsFiberArraySet`
            Extracted spectra from arm.
        pfsConfigList : collection of `pfs.datamodel.PfsConfig`
            PFS fiber configurations.
        skyNorms : `pfs.drp.stella.FocalPlaneFunction`, optional
            Common-mode sky normalizations to remove before measuring the
            residual.

        Returns
        -------
        sky : list of `FocalPlaneFunction`
            Sky spectral model for each arm.
        skyNorms : list of `PolynomialPerFiber`
            Normalizations of sky data for each arm.
        armContinuum : list of `np.ndarray`
            Continuum fit to the ``pfsArm``.
        skyContinuum : list of `np.ndarray`
            Continuum fit to the sky spectral model.
        refLines : `ReferenceLineSet`
            Sky line list, used for fitting continuum.
        """
        fiberId: np.ndarray | None = None
        skyList: list[FocalPlaneFunction] = []  # Sky model for each arm
        skyDataList: list[Struct] = []  # Realized sky spectra for each arm
        for pfsArm, pfsConfig in zip(pfsArmList, pfsConfigList):
            if fiberId is None:
                fiberId = pfsArm.fiberId
            elif not np.array_equal(fiberId, pfsArm.fiberId):
                raise ValueError("fiberId mismatch")
            sky = self.measureSky(pfsArm, pfsConfig)
            skyList.append(sky)
            skyData = sky(pfsArm.wavelength, pfsConfig)
            if skyNorms is not None:
                normData = skyNorms(pfsArm.wavelength, pfsConfig)
                skyData.values *= normData.values[:, None]
                skyData.variances *= normData.values[:, None]**2
                skyData.masks |= normData.masks[:, None]

            skyDataList.append(skyData)

        refLines = self.readLineList.run(metadata=pfsArmList[0].metadata)
        result = self.measureNormalizations(pfsArmList, skyDataList, skyNorms, refLines)

        result.sky = skyList
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
        pfsArmList: Collection[PfsFiberArraySet],
        skyDataList: Collection[Struct],
        refLines: ReferenceLineSet | None = None,
    ) -> PolynomialPerFiber:
        """Measure normalizations for sky model

        We currently measure a single normalization factor for each fiber, but
        this could be generalized to a more complex model.

        Parameters
        ----------
        pfsArmList : collection of `PfsFiberArraySet`
            Spectra from which to subtract sky model.
        skyDataList : collection of `Struct`
            Sky spectra, from model realized for each ``pfsArm``.
        refLines : `ReferenceLineSet`, optional
            Sky line list, used for fitting continuum.

        Returns
        -------
        skyNorms : `PolynomialPerFiber`
            Normalizations of sky data for each fiber.
        armContinuum : `np.ndarray`
            Continuum fit to the ``pfsArm``.
        skyContinuum : `np.ndarray`
            Continuum fit to the sky model.
        """
        if len(pfsArmList) != len(skyDataList):
            raise ValueError("pfsArm and skyModel must have the same length")

        minWavelength = min(pfsArm.wavelength.min() for pfsArm in pfsArmList)
        maxWavelength = max(pfsArm.wavelength.max() for pfsArm in pfsArmList)

        armContinuumList: list[np.ndarray] = []
        skyContinuumList: list[np.ndarray] = []
        data: list[np.ma.MaskedArray] = []
        model: list[np.ma.MaskedArray] = []
        variance: list[np.ndarray] = []
        good: list[np.ndarray] = []
        for pfsArm, sky in zip(pfsArmList, skyDataList):
            armContinuum = self.fitContinuum.run(pfsArm, refLines)

            covar = np.zeros_like(pfsArm.covar)
            covar[:, 0, :] = sky.variances
            skySpectra = PfsFiberArraySet(
                pfsArm.identity,
                pfsArm.fiberId,
                pfsArm.wavelength,
                sky.values,
                sky.masks,
                np.zeros_like(pfsArm.flux),
                np.ones_like(pfsArm.norm),
                covar,
                pfsArm.flags,
                pfsArm.metadata,
            )
            skyContinuum = self.fitContinuum.run(skySpectra, refLines)

            norm = pfsArm.norm
            flux = pfsArm.flux/norm - armContinuum
            skyFlux = skySpectra.flux - skyContinuum  # norm is unity as skySpectra.flux is already normalized
            select = ((pfsArm.mask & pfsArm.flags.get(*self.config.mask)) == 0)
            good.append(select)

            with np.errstate(invalid="ignore", divide="ignore"):
                reject = ~select & (flux/np.sqrt(pfsArm.variance) < self.config.minSignalToNoise)
                ratio = flux/skyFlux
                reject |= (ratio < self.config.minRatio) | (ratio > self.config.maxRatio)
                factor = np.array([calculateMedian(rat, rej) for rat, rej in zip(ratio, reject)])
                reject |= np.abs(ratio - factor[:, None]) > self.config.rejectRatio

            armContinuumList.append(armContinuum)
            skyContinuumList.append(skyContinuum)
            data.append(np.ma.masked_where(reject, flux))
            model.append(np.ma.masked_where(reject, skyFlux))
            variance.append(pfsArm.variance)

        def doFit(data: list[np.ndarray], model: list[np.ndarray], variance: list[np.ndarray]) -> np.ndarray:
            """Fit model to data

            The model is very simple: a single scale factor for each spectrum.

            Parameters
            ----------
            data : `list` of `numpy.ndarray`, shape ``(numSpectra, numSamples)``
                Data to fit.
            model : `list` of `numpy.ndarray`, shape ``(numSpectra, numSamples)``
                Model to fit (evaluated at the same points as ``data``).
            variance : `list` of `numpy.ndarray`
                Variance of the data.

            Returns
            -------
            factor : `numpy.ndarray`, shape ``(numSpectra,)``
                Factor by which to multiply the model to best fit the data.
            """
            for dd, mm, vv in zip(data, model, variance):
                assert dd.shape == mm.shape and dd.shape == vv.shape
            modelDotModel = reduce(
                np.ma.add, (np.ma.sum(mm**2/vv, axis=1) for mm, vv in zip(model, variance))
            )
            dataDotModel = reduce(
                np.ma.add, (np.ma.sum(dd*mm/vv, axis=1) for dd, mm, vv in zip(data, model, variance))
            )
            return np.array(dataDotModel/modelDotModel)

        for iteration in range(self.config.iterations):
            factor = doFit(data, model, variance)
            self.log.debug(
                "Iteration %d: factor = %.3f +/- %.3f",
                iteration,
                np.nanmedian(np.array(factor)),
                robustRms(factor[np.isfinite(factor)]),
            )
            for dd, mm, vv, gg in zip(data, model, variance, good):
                residuals = dd - factor[:, None]*mm
                with np.errstate(invalid="ignore", divide="ignore"):
                    rr = np.array(residuals)/np.sqrt(vv)
                self.log.debug("    RMS normalized residuals = %.1f", robustRms(rr[gg]))
                reject = np.abs(residuals) > self.config.rejection*np.sqrt(vv)
                dd.mask |= reject
                mm.mask |= reject

        # Final fit after rejection iterations
        factor = doFit(data, model, variance)
        self.log.info(
            "Sky norms = %.3f +/- %.3f",
            np.nanmedian(np.array(factor)),
            robustRms(factor[np.isfinite(factor)]),
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = {
                ff: robustRms((np.array(dd - factor[:, None]*mm)/np.sqrt(vv))[gg])
                for ff, dd, mm, vv, gg in zip(pfsArm.fiberId, data, model, variance, good)
            }

        coeffs = dict(zip(pfsArm.fiberId, factor))
        return Struct(
            skyNorms=PolynomialPerFiber(coeffs, rms, minWavelength, maxWavelength),
            armContinuum=armContinuumList,
            skyContinuum=skyContinuumList,
        )
