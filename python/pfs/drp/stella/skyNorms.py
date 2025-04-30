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
        values : `numpy.ndarray`, shape ``(numFibers,)``
            Normalization values for each fiber. This data is contained in the
            ``skyNorms``, but this is a more convenient form.
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
        skyNormValues: list[np.ndarray] = []  # Sky normalization values for each arm
        for pfsArm, pfsConfig in zip(pfsArmList, pfsConfigList):
            if fiberId is None:
                fiberId = pfsArm.fiberId
            elif not np.array_equal(fiberId, pfsArm.fiberId):
                raise ValueError("fiberId mismatch")
            sky = self.measureSky(pfsArm, pfsConfig)
            skyList.append(sky)
            skyData = sky(pfsArm.wavelength, pfsConfig.select(fiberId=pfsArm.fiberId))
            if skyNorms is not None:
                skyNormValues.append(
                    skyNorms(
                        np.full(
                            (len(pfsConfig), 1),
                            0.5*(skyNorms.minWavelength + skyNorms.maxWavelength),
                            dtype=float,
                        ),
                        pfsConfig,
                    ).values
                )
                normData = skyNorms(pfsArm.wavelength, pfsConfig)
                skyData.values *= normData.values
                skyData.variances *= normData.values**2
                skyData.masks |= normData.masks

            skyDataList.append(skyData)

        if skyNorms is not None:
            skyNormsValues = np.concatenate(skyNormValues)
            self.log.info(
                "Applying prior sky norms = %.3f +/- %.3f",
                np.nanmedian(skyNormsValues),
                robustRms(skyNormsValues, True),
            )

        refLines = self.readLineList.run(metadata=pfsArmList[0].metadata)
        result = self.measureNormalizations(pfsArmList, skyDataList, refLines)

        if skyNorms is not None:
            before = result.values/skyNormsValues
            self.log.info(
                "With prior sky norms: RMS %.3f --> %.3f",
                robustRms(before, True),
                robustRms(result.values, True),
            )

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
        values : `numpy.ndarray`, shape ``(numFibers,)``
            Normalization values for each fiber. This data is contained in the
            ``skyNorms``, but this is a more convenient form.
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
        fiberId = pfsArmList[0].fiberId
        for pfsArm, sky in zip(pfsArmList, skyDataList):
            if not np.array_equal(fiberId, pfsArm.fiberId):
                raise ValueError("fiberId mismatch")
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
            data.append(np.ma.masked_where(reject, flux))
            model.append(np.ma.masked_where(reject, skyFlux))
            variance.append(np.ma.masked_where(reject, var))

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
                np.add,
                (np.ma.sum(mm**2/vv, axis=1).filled(0.0) for mm, vv in zip(model, variance)),
            )
            dataDotModel = reduce(
                np.add,
                (np.ma.sum(dd*mm/vv, axis=1).filled(0.0) for dd, mm, vv in zip(data, model, variance)),
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.array(dataDotModel/modelDotModel)

        for iteration in range(self.config.iterations):
            factor = doFit(data, model, variance)
            self.log.debug(
                "Iteration %d: factor = %.3f +/- %.3f",
                iteration,
                np.nanmedian(np.array(factor)),
                robustRms(factor, True),
            )
            for dd, mm, vv in zip(data, model, variance):
                residuals = dd - factor[:, None]*mm
                reject = np.abs(residuals) > self.config.rejection*np.sqrt(vv)
                dd.mask |= reject
                mm.mask |= reject

        # Final fit after rejection iterations
        factor = doFit(data, model, variance)
        self.log.info(
            "Sky norms = %.3f +/- %.3f",
            np.nanmedian(np.array(factor)),
            robustRms(factor, True),
        )

        if len(pfsArmList) > 1:
            byExposure = np.array(
                [doFit([dd], [mm], [vv]) for dd, mm, vv in zip(data, model, variance)]
            )
            rms = {ff: robustRms(byExposure[:, ii], True) for ii, ff in enumerate(fiberId)}
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                residuals = [dd - factor[:, None]*mm for dd, mm in zip(data, model)]
                rms = {
                    ff: robustRms(np.concatenate([res[ii].compressed() for res in residuals]))
                    for ii, ff in enumerate(pfsArm.fiberId)
                }

        coeffs = {ff: np.array([xx]) for ff, xx in zip(fiberId, factor)}
        return Struct(
            skyNorms=PolynomialPerFiber(coeffs, rms, minWavelength, maxWavelength),
            values=factor,
            armContinuum=armContinuumList,
            skyContinuum=skyContinuumList,
        )
