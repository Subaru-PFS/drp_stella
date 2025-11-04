import lsstDebug
from lsst.pipe.base import QuantumContext
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from lsst.pex.config import ConfigurableField, ChoiceField, Field, ListField
from lsst.utils import getPackageDir
from pfs.datamodel.identity import Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.observations import Observations
from pfs.datamodel.pfsConfig import FiberStatus, PfsConfig, TargetType
from pfs.datamodel.pfsFiberArray import PfsFiberArray
from pfs.datamodel.pfsFluxReference import PfsFluxReference
from pfs.drp.stella.fluxModelInterpolator import FluxModelInterpolator
from pfs.drp.stella import ReferenceLine, ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella import SpectrumSet
from pfs.drp.stella.datamodel import PfsFiberArraySet, PfsMerged, PfsSimpleSpectrum, PfsSingle
from pfs.drp.stella.dustMap import DustMap
from pfs.drp.stella.estimateRadialVelocity import EstimateRadialVelocityTask
from pfs.drp.stella.extinctionCurve import F99ExtinctionCurve
from pfs.drp.stella.fitBroadbandSED import FitBroadbandSEDTask
from pfs.drp.stella.fitContinuum import FitContinuumTask
from pfs.drp.stella.fluxModelSet import FluxModelSet
from pfs.drp.stella.interpolate import interpolateFlux, interpolateMask
from pfs.drp.stella.lsf import GaussianLsf, Lsf, LsfDict
from pfs.drp.stella.utils.math import ChisqList
from pfs.drp.stella.utils import debugging

from astropy import constants as const
from deprecated import deprecated
import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.spatial

import copy
import dataclasses
import math
import os
import warnings

from typing import Literal, overload
from collections.abc import Generator, Mapping, Sequence

from numpy.typing import NDArray


__all__ = ["FitFluxReferenceConnections", "FitFluxReferenceConfig", "FitFluxReferenceTask"]


class FitFluxReferenceConnections(PipelineTaskConnections, dimensions=("instrument", "visit")):
    """Connections for FitFluxReferenceTask"""

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )
    pfsMerged = InputConnection(
        name="pfsMerged",
        doc="Merged spectra from exposure",
        storageClass="PfsMerged",
        dimensions=("instrument", "visit"),
    )
    pfsMergedLsf = InputConnection(
        name="pfsMergedLsf",
        doc="Line-spread function of merged spectra",
        storageClass="LsfDict",
        dimensions=("instrument", "visit"),
    )

    reference = OutputConnection(
        name="pfsFluxReference",
        doc="Fit reference spectrum of flux standards",
        storageClass="PfsFluxReference",
        dimensions=("instrument", "visit"),
    )


class FitFluxReferenceConfig(PipelineTaskConfig, pipelineConnections=FitFluxReferenceConnections):
    """Configuration for FitFluxReferenceTask"""

    fitBroadbandSED = ConfigurableField(
        target=FitBroadbandSEDTask, doc="Get probabilities of SEDs from broadband photometries."
    )
    fitObsContinuum = ConfigurableField(
        target=FitContinuumTask, doc="Fit a model to observed spectrum's continuum"
    )
    fitModelContinuum = ConfigurableField(
        target=FitContinuumTask, doc="Fit a model to model spectrum's continuum"
    )
    fitDownsampledContinuum = ConfigurableField(
        target=FitContinuumTask, doc="Fit a model to downsampled model spectrum's continuum"
    )
    estimateRadialVelocity = ConfigurableField(
        target=EstimateRadialVelocityTask, doc="Estimate radial velocity."
    )
    minBroadbandFluxes = Field(
        dtype=int,
        default=2,
        doc="min.number of required broadband fluxes for an observed flux standard to be fitted to.",
    )
    minWavelength = Field(
        dtype=float,
        default=480.0,
        doc="min of the wavelength range in which observation spectra are compared to models.",
    )
    maxWavelength = Field(
        dtype=float,
        default=1200.0,
        doc="max of the wavelength range in which observation spectra are compared to models.",
    )
    lrIgnoredRangesLeft = ListField(
        dtype=float,
        default=[685.0, 715.0, 755.0, 810.0, 895.0, 1100.0],
        doc="Left ends of wavelength ranges ignored (because e.g. of strong atmospheric absorption)"
        " when comparing low-resolution observation spectra to models.",
    )
    lrIgnoredRangesRight = ListField(
        dtype=float,
        default=[695.0, 735.0, 770.0, 835.0, 985.0, 1200.0],
        doc="Right ends of wavelength ranges ignored (because e.g. of strong atmospheric absorption)"
        " when comparing low-resolution observation spectra to models.",
    )
    mrIgnoredRangesLeft = ListField(
        dtype=float,
        default=[625.0, 755.0, 810.0, 885.0, 1100.0],
        doc="Left ends of wavelength ranges ignored (because e.g. of strong atmospheric absorption)"
        " when comparing middle-resolution observation spectra to models.",
    )
    mrIgnoredRangesRight = ListField(
        dtype=float,
        default=[735.0, 770.0, 835.0, 985.0, 1200.0],
        doc="Right ends of wavelength ranges ignored (because e.g. of strong atmospheric absorption)"
        " when comparing middle-resolution observation spectra to models.",
    )
    cutoffSNR = Field(
        dtype=float,
        default=10,
        doc="Minimally required pixel-wise S/N of observed spectra."
        " Spectra with S/N worse than this value will be discarded.",
    )
    cutoffSNRRangeLeft = Field(
        dtype=float,
        default=840,
        doc="Left edge of the wavelength range in which S/N is averaged to be compared with ``cutoffSNR``",
    )
    cutoffSNRRangeRight = Field(
        dtype=float,
        default=880,
        doc="Right edge of the wavelength range in which S/N is averaged to be compared with ``cutoffSNR``",
    )
    badMask = ListField(
        dtype=str, default=["BAD", "SAT", "CR", "NO_DATA", "SUSPECT"], doc="Mask planes for bad pixels"
    )
    modelSNR = Field(
        dtype=float,
        default=400,
        doc="Supposed S/N of model spectra."
        " Used in making up the variance of the flux for algorithms that require it."
        " It is not that the model spectra are affected by this amount of noise,"
        " nor is it that any artificial noise will be added to the model spectra.",
    )
    Rv = Field(dtype=float, default=3.1, doc="Ratio of total to selective extinction at V, Rv = A(V)/E(B-V).")
    broadbandFluxType = ChoiceField(
        doc="Type of broadband fluxes to use.",
        dtype=str,
        allowed={
            "fiber": "Use `psfConfig.fiberFlux`.",
            "psf": "Use `psfConfig.psfFlux`.",
            "total": "Use `psfConfig.totalFlux`.",
        },
        default="psf",
        optional=False,
    )
    fabricatedBroadbandFluxErrSNR = Field(
        dtype=float,
        default=0,
        doc="If positive, fabricate flux errors in pfsConfig if all of them are NaN"
        " (for old engineering data). The fabricated flux errors are such that S/N is this much.",
    )
    minTeff = Field(
        doc="FLUXSTD target gets fitted only if effective temperature, in K, from broadband fluxes"
        " is above this value.",
        dtype=float,
        default=5600,
    )
    maxTeff = Field(
        doc="FLUXSTD target gets fitted only if effective temperature, in K, from broadband fluxes"
        " is below this value.",
        dtype=float,
        default=7900,
    )
    minimizationMethod = Field(
        dtype=str,
        default="Powell",
        doc="Method with which to find the minimum of the objective function of spectrum fitting."
        " Valid values are 'brute-force' or any valid arguments for `scipy.optimize.minimize()`",
    )
    priorCutoff = Field(
        dtype=float,
        default=0.01,
        doc="(Valid if minimizationMethod='brute-force') Cut-off prior probability relative to max(prior)."
        " In comparing a flux model to an observed flux,"
        " the model is immediately rejected if its prior probability"
        " is less than `priorCutoff*max(prior)`.",
    )
    paramScale = ListField(
        dtype=float,
        default=[1e-3, 1, 1, 1],
        doc="(Valid if minimizationMethod!='brute-force')"
        " Values by which to multiply (teff, logg, m, alpha)"
        " to adjust them to be as large as each other.",
    )
    paramOrder = ListField(
        dtype=int,
        default=[2, 3, 0, 1],
        doc="(Valid if minimizationMethod!='brute-force')"
        " Order of (teff (0), logg (1), m (2), alpha (3)) in the parameter list of the objective function.",
    )
    modelOversamplingFactor = Field(
        dtype=float,
        default=2.0,
        doc="Before a model is compared to an observed spectrum,"
        " the model is downsampled so that its wavelength resolution will be"
        " about this much times as fine as obs.spec.'s wavelength resolution."
        " Disabled if 0.",
    )
    numPCAComponents = Field(
        dtype=int,
        default=512,
        doc="(Valid if minimizationMethod!='brute-force')"
        " Number of PCA components to use during fitting."
        " The final products are made of all components regardless of this setting.",
    )
    paramMargin = ListField(
        dtype=float,
        default=[100, 0, 0, 0],
        doc="A model parameter (teff, logg, m, alpha) is considered valid"
        " if it is in the domain and it is this much away from the boundary of the domain"
        " in each dimension",
    )

    def setDefaults(self) -> None:
        super().setDefaults()

        self.estimateRadialVelocity.mask = [
            "BAD",
            "SAT",
            "CR",
            "NO_DATA",
            "BAD_FIBERNORMS",
            "BAD_FLAT",
            "BAD_SKY",
            "EDGE",
            "ATMOSPHERE",
        ]
        self.fitObsContinuum.mask = ["BAD", "SAT", "CR", "NO_DATA", "BAD_FIBERNORMS", "BAD_FLAT", "BAD_SKY"]

        # Not sure these paramaters are good.
        self.fitObsContinuum.numKnots = 50
        self.fitObsContinuum.doMaskLines = True
        self.fitObsContinuum.maskLineRadius = 25
        self.fitModelContinuum.numKnots = 50
        self.fitModelContinuum.doMaskLines = True
        self.fitModelContinuum.maskLineRadius = 25
        self.fitDownsampledContinuum.numKnots = 50
        self.fitDownsampledContinuum.doMaskLines = True
        self.fitDownsampledContinuum.maskLineRadius = 25


class FitFluxReferenceTask(PipelineTask):
    """Construct reference for flux calibration."""

    ConfigClass = FitFluxReferenceConfig
    _DefaultName = "fitFluxReference"

    fitBroadbandSED: FitBroadbandSEDTask
    fitObsContinuum: FitContinuumTask
    fitModelContinuum: FitContinuumTask
    fitDownsampledContinuum: FitContinuumTask
    estimateRadialVelocity: EstimateRadialVelocityTask

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.makeSubtask("fitBroadbandSED")
        self.makeSubtask("fitObsContinuum")
        self.makeSubtask("fitModelContinuum")
        self.makeSubtask("fitDownsampledContinuum")
        self.makeSubtask("estimateRadialVelocity")

        self.debugInfo = lsstDebug.Info(__name__)

        self.fluxModelSet = FluxModelSet(getPackageDir("fluxmodeldata"))
        self.modelInterpolator = FluxModelInterpolator.fromFluxModelData(getPackageDir("fluxmodeldata"))
        self.extinctionMap = DustMap()

        self.fitFlagNames = MaskHelper()

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `QuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        inputs = butler.get(inputRefs)
        reference = self.run(**inputs)
        butler.put(reference, outputRefs.reference)

    def run(
        self, pfsConfig: PfsConfig, pfsMerged: PfsFiberArraySet, pfsMergedLsf: LsfDict
    ) -> PfsFluxReference:
        """Create flux reference from ``pfsMerged``
        and corresponding ``pfsConfig``.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end,
            in which information of broad band fluxes count.
        pfsMerged : `pfs.drp.stella.datamodel.PfsFiberArraySet`
            Typically an instance of `PfsMerged`.
        pfsMergedLsf : `dict` [`int`, `pfs.drp.stella.Lsf`]
            Combined line-spread functions indexed by fiberId.

        Returns
        -------
        pfsFluxReference : `pfs.datamodel.pfsFluxReference.PfsFluxReference`
            Reference spectra for flux calibration
        """
        pfsConfig = pfsConfig.select(targetType=TargetType.FLUXSTD)
        originalFiberId = np.copy(pfsConfig.fiberId)
        fitFlag: dict[int, int] = {}  # mapping fiberId -> flag indicating fit status

        removeBadFluxes(pfsConfig, self.config.broadbandFluxType, self.config.fabricatedBroadbandFluxErrSNR)

        self.log.info("Number of FLUXSTD: %d", len(pfsConfig))

        def selectPfsConfig(pfsConfig: PfsConfig, flagName: str, isGood: Sequence[bool]) -> PfsConfig:
            """Select fibers in pfsConfig for which ``isGood`` is ``True``.
            Fibers filtered out (``isGood`` is ``False``) will be registered
            on ``fitFlag`` (nonlocal variable).

            Parameters
            ----------
            pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
                Configuration of the PFS top-end.
            flagName : `str`
                Fibers filtered out will be registered on ``fitFlag``
                in this name.
            isGood : `list` of `bool`
                Boolean flags indicating whether fibers are good or not.

            Returns
            -------
            pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
                ``pfsConfig`` that contains only those fibers
                for which ``isGood`` is ``True``.
            """
            isGood = np.asarray(isGood, dtype=bool)
            flag = self.fitFlagNames.add(flagName)
            fitFlag.update((fiberId, flag) for fiberId in pfsConfig.fiberId[~isGood])
            goodConfig = pfsConfig[isGood]
            self.log.info("Number of FLUXSTD that are not %s: %d", flagName, len(goodConfig))
            return goodConfig

        pfsConfig = selectPfsConfig(pfsConfig, "ABSENT_FIBER", np.isin(pfsConfig.fiberId, pfsMerged.fiberId))
        pfsConfig = selectPfsConfig(pfsConfig, "BAD_FIBER", (pfsConfig.fiberStatus == FiberStatus.GOOD))
        pfsConfig = selectPfsConfig(
            pfsConfig,
            "DEFICIENT_BBFLUXES",
            [len(filterNames) >= self.config.minBroadbandFluxes for filterNames in pfsConfig.filterNames],
        )

        pfsMerged = removeBadSpectra(
            pfsMerged,
            self.config.cutoffSNR,
            (self.config.cutoffSNRRangeLeft, self.config.cutoffSNRRangeRight),
        )
        pfsConfig = selectPfsConfig(pfsConfig, "LOW_SNR_FIBER", np.isin(pfsConfig.fiberId, pfsMerged.fiberId))

        # Apply the Galactic extinction correction to observed broad-band fluxes in pfsConfig
        pfsConfigCorr = self.correctExtinction(copy.deepcopy(pfsConfig))

        # Prior PDF from broad-band typing, where the continuous spectrum matters.
        bbPdfs = self.fitBroadbandSED.run(pfsConfigCorr)
        if self.debugInfo.doWritePrior:
            debugging.writeExtraData(
                f"fitFluxReference-output/prior-{pfsMerged.filename}.pickle",
                prior=bbPdfs,
            )

        pfsConfig = selectPfsConfig(
            pfsConfig,
            "FITBBSED_FAILED",
            [((fiberId in bbPdfs) and np.all(np.isfinite(bbPdfs[fiberId]))) for fiberId in pfsConfig.fiberId],
        )

        paramsFromBB = self.findRoughlyBestModel(bbPdfs, paramOnly=True)
        pfsConfig = selectPfsConfig(
            pfsConfig,
            "FITBBSED_TEFF_OUTOFRANGE",
            [
                (
                    (fiberId in paramsFromBB)
                    and self.config.minTeff <= paramsFromBB[fiberId].param.teff <= self.config.maxTeff
                )
                for fiberId in pfsConfig.fiberId
            ],
        )

        # Extract just those fibers from pfsMerged
        # whose fiberId still remain in pfsConfig.
        pfsMerged = pfsMerged[np.isin(pfsMerged.fiberId, pfsConfig.fiberId)]
        self.log.info("Number of observed FLUXSTD to be fitted to: %d", len(pfsMerged))

        if len(pfsMerged) == 0:
            raise RuntimeError("No observed FLUXSTD can be fitted a model to.")

        pfsMerged /= pfsMerged.norm
        pfsMerged.norm[...] = 1.0

        pfsMerged = self.computeContinuum(pfsMerged, mode="observed").whiten(pfsMerged)
        pfsMerged = self.maskUninterestingRegions(pfsMerged)

        if self.debugInfo.doWriteWhitenedFlux:
            pfsMerged.writeFits(f"fitFluxReference-output/whitened-{pfsMerged.filename}")

        radialVelocities = self.getRadialVelocities(pfsConfig, pfsMerged, pfsMergedLsf, bbPdfs)

        if self.debugInfo.doWriteCrossCorr:
            debugging.writeExtraData(
                f"fitFluxReference-output/crossCorr-{pfsMerged.filename}.pickle",
                crossCorr={fiberId: record.crossCorr for fiberId, record in radialVelocities.items()},
            )

        flag = self.fitFlagNames.add("ESTIMATERADIALVELOCITY_FAILED")
        for fiberId in pfsConfig.fiberId:
            velocity = radialVelocities.get(fiberId)
            if velocity is None or velocity.fail or not np.isfinite(velocity.velocity):
                fitFlag[fiberId] = flag

        # Likelihoods from spectral fitting, where line spectra matter.
        self.log.info("Fitting models to spectra (takes some time)...")
        bestParams = self.fitModelsToSpectra(pfsConfig, pfsMerged, pfsMergedLsf, radialVelocities, bbPdfs)

        flag = self.fitFlagNames.add("FITMODELS_FAILED")
        for fiberId in pfsConfig.fiberId:
            param = bestParams.get(fiberId)
            if param is None or not param.success:
                if fitFlag.get(fiberId, 0) == 0:
                    fitFlag[fiberId] = flag

        flag = self.fitFlagNames.add("FITMODELS_OUTOFRANGE")
        for fiberId in pfsConfig.fiberId:
            param = bestParams.get(fiberId)
            if param is not None and param.success and not self.isParamInDomain(param.param):
                if fitFlag.get(fiberId, 0) == 0:
                    fitFlag[fiberId] = flag

        self.log.info("Making reference spectra by interpolation")
        bestModels = self.makeReferenceSpectra(
            pfsConfig, {fiberId: p.param for fiberId, p in bestParams.items()}
        )

        if not bestModels:
            raise RuntimeError("Fitting none of FLUXSTD succeeded.")

        flag = self.fitFlagNames.add("MAKEREFERENCESPECTRA_FAILED")
        for fiberId in pfsConfig.fiberId:
            bestModel = bestModels.get(fiberId)
            if bestModel is None:
                if fitFlag.get(fiberId, 0) == 0:
                    fitFlag[fiberId] = flag

        # We want a `wavelength` array. Any one in `bestModels` will do.
        for bestModel in bestModels.values():
            wavelength = bestModel.spectrum.wavelength
            break

        flux = np.full(shape=(len(originalFiberId), len(wavelength)), fill_value=np.nan, dtype=np.float32)
        with np.errstate(invalid="ignore"):
            fitParams = np.full(
                shape=(len(originalFiberId),),
                fill_value=np.nan,
                dtype=[
                    ("teff", np.float32),
                    ("logg", np.float32),
                    ("m", np.float32),
                    ("alpha", np.float32),
                    ("radial_velocity", np.float32),
                    ("radial_velocity_err", np.float32),
                    ("flux_scaling_chi2", np.float32),
                    ("flux_scaling_dof", np.int32),
                ],
            )

        fiberIdToIndex = {value: key for key, value in enumerate(originalFiberId)}

        for fiberId, bestModel in bestModels.items():
            velocity = radialVelocities[fiberId]
            index = fiberIdToIndex[fiberId]
            flux[index, :] = bestModel.spectrum.flux
            fitParams["teff"][index] = bestModel.param.teff
            fitParams["logg"][index] = bestModel.param.logg
            fitParams["m"][index] = bestModel.param.m
            fitParams["alpha"][index] = bestModel.param.alpha
            fitParams["radial_velocity"][index] = velocity.velocity
            fitParams["radial_velocity_err"][index] = velocity.error
            fitParams["flux_scaling_chi2"][index] = bestModel.fluxScalingChi2
            fitParams["flux_scaling_dof"][index] = bestModel.fluxScalingDof

        fitFlagArray = np.zeros(shape=(len(originalFiberId),), dtype=np.int32)

        for fiberId, flag in fitFlag.items():
            index = fiberIdToIndex[fiberId]
            fitFlagArray[index] = flag

        return PfsFluxReference(
            identity=pfsMerged.identity,
            fiberId=originalFiberId,
            wavelength=wavelength,
            flux=flux,
            metadata={},
            fitFlag=fitFlagArray,
            fitFlagNames=self.fitFlagNames,
            fitParams=fitParams,
        )

    def getRadialVelocities(
        self,
        pfsConfig: PfsConfig,
        pfsMerged: PfsFiberArraySet,
        pfsMergedLsf: LsfDict,
        bbPdfs: dict[int, NDArray[np.float64]],
    ) -> dict[int, Struct]:
        """Estimate the radial velocity for each fiber in ``pfsMerged``.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
        pfsMerged : `pfs.drp.stella.datamodel.PfsFiberArraySet`
            Typically an instance of PfsMerged.
            It must have been whitened.
        pfsMergedLsf : `dict` [`int`. `pfs.drp.stella.Lsf`]
            Combined line-spread functions indexed by fiberId.
        bbPdfs : `dict` [`int`, `NDArray` [`np.float64`]]
            ``bbPdfs[fiberId]``, if exists, is the probability distribution
            of the fiber ``fiberId`` being of each model type,
            determined by broad-band photometries.

        Returns
        -------
        radialVelocities : `dict [`int`, `lsst.pipe.base.Struct`]
            Mapping from ``fiberId`` to radial velocity.
            Each value has ``velocity``, ``error``,
            ``crossCorr``, and ``fail`` as its member.
            See ``EstimateRadialVelocityTask``.
        """
        # Find the best model from broad bands.
        # This model is used as the reference for cross-correlation calculation
        bestModels = self.findRoughlyBestModel(bbPdfs)

        radialVelocities: dict[int, Struct] = {}
        for iFiber, spectrum in enumerate(fibers(pfsConfig, pfsMerged)):
            fiberId = pfsConfig.fiberId[iFiber]
            model = bestModels.get(fiberId)
            if model is None or model.spectrum is None:
                continue
            modelSpectrum = convolveLsf(model.spectrum, pfsMergedLsf[fiberId], spectrum.wavelength)
            isBad = modelSpectrum.flux <= 0
            modelSpectrum.mask[isBad] |= modelSpectrum.flags.add("NO_DATA")
            modelSpectrum = self.computeContinuum(modelSpectrum, mode="model").whiten(modelSpectrum)
            radialVelocities[fiberId] = self.estimateRadialVelocity.run(spectrum, modelSpectrum)

        return radialVelocities

    def computeContinuum(
        self,
        spectra: PfsSimpleSpectrum | PfsFiberArraySet,
        *,
        mode: Literal["observed", "model", "downsampled"],
    ) -> "Continuum":
        """Whiten one or more spectra.

        Parameters
        ----------
        spectra : `PfsSimpleSpectrum` | `PfsFiberArraySet`
            spectra to whiten.
        mode : {"observed", "model", "downsampled"}
            Whether ``spectra`` is from observation, from simulation,
            or downsampled from simulation.

        Returns
        -------
        continuum : `Continuum`
            Fitted continuum.
        """
        if mode == "observed":
            fitContinuum = self.fitObsContinuum
        if mode == "model":
            fitContinuum = self.fitModelContinuum
        if mode == "downsampled":
            fitContinuum = self.fitDownsampledContinuum

        # If `spectra` is actually a single spectrum,
        # we put it into PfsFiberArraySet
        if len(spectra.flux.shape) == 1:
            if not hasattr(spectra, "covar"):
                spectra = promoteSimpleSpectrumToFiberArray(spectra, snr=self.config.modelSNR)
            # This is temporary object, so any fiberId will do.
            spectra = promoteFiberArrayToFiberArraySet(spectra, fiberId=1)

        # This function actually works with `PfsFiberArraySet`
        # nonetheless for its name.
        # (PfsArm is a subclass of PfsFiberArraySet)
        specset = SpectrumSet.fromPfsArm(spectra)

        lines = ReferenceLineSet.fromRows(
            [
                ReferenceLine("Hbeta", 486.2721, 1.0, ReferenceLineStatus.GOOD, "", 0),
                ReferenceLine("Halpha", 656.4614, 1.0, ReferenceLineStatus.GOOD, "", 0),
            ]
        )

        # Get the continuum for each fiber
        continuum = fitContinuum.run(specset, lines=lines)
        return Continuum(continuum, np.array([], dtype=int))

    def fitModelsToSpectra(
        self,
        pfsConfig: PfsConfig,
        obsSpectra: PfsFiberArraySet,
        pfsMergedLsf: LsfDict,
        radialVelocities: dict[int, Struct],
        priorPdfs: dict[int, NDArray[np.float64]],
    ) -> dict[int, Struct]:
        """For each observed spectrum,
        get probability of each model fitting to the spectrum.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
        obsSpectra : `PfsFiberArraySet`
            Continuum-subtracted observed spectra
        pfsMergedLsf : `dict` [`int`, `pfs.drp.stella.Lsf`]
            Combined line-spread functions indexed by fiberId.
        radialVelocities : `dict` [`int`, `lsst.pipe.base.Struct`]
            Mapping from ``fiberId`` to radial velocity.
            Each value, has ``velocity``, ``error``, and ``fail``
            as its member. See ``EstimateRadialVelocityTask``.
        priorPdfs : `dict` [`int`, `NDArray` [`np.float64`]]
            For each ``priorPdfs[fiberId]`` in ``priorPdfs``,
            ``priorPdfs[fiberId][iSED]`` is the prior probability of the SED ``iSED``
            matching the spectrum ``fiberId``.

        Returns
        -------
        bestParams : `dict` [`int`, `Struct`]
            ``bestParams[fiberId]`` is the best-fit parameters for ``fiberId``.
            Each element ``bestParams[fiberId]``, consists of the following members:

              - param : `ModelParam`
                Best-fit parameter.
              - success : `bool`
                True if optimization has succeeded.

            The following members exist if self.config.minimizationMethod != 'brute-force'

              - chi2 : `float`
                chi square.
              - dof : `int`
                degree of freedom of the chi square.
        """
        if self.config.minimizationMethod.lower() == "brute-force":
            return self.fitModelsToSpectraBruteForce(
                pfsConfig, obsSpectra, pfsMergedLsf, radialVelocities, priorPdfs
            )

        paramScale = np.array(self.config.paramScale, dtype=float)
        paramOrder = np.array(self.config.paramOrder, dtype=int)
        # Inverse of `paramOrder`.
        paramOrderInv = np.array([i for j, i in sorted((j, i) for i, j in enumerate(paramOrder))], dtype=int)

        def paramToX(param: Sequence) -> NDArray:
            """Convert (teff, logg, m, alpha) to arguments of the objective.

            Parameters
            ----------
            param : `numpy.ndarray`, shape(4,)
                [teff, logg, m, alpha].

            Returns
            -------
            x : `numpy.ndarray`, shape(4,)
                An array that can be input to the objective function.
            """
            x = np.array(param, dtype=float)
            x *= paramScale
            return x[paramOrder]

        def xToParam(x: NDArray) -> NDArray:
            """Convert arguments of the objective to (teff, logg, m, alpha).

            Parameters
            ----------
            x : `numpy.ndarray`, shape(4,)
                An array that can be input to the objective function.

            Returns
            -------
            param : `numpy.ndarray`, shape(4,)
                [teff, logg, m, alpha].
            """
            param = np.copy(x[paramOrderInv])
            param /= paramScale
            return param

        param4d = np.lib.recfunctions.structured_to_unstructured(
            self.fluxModelSet.parameters[["teff", "logg", "m", "alpha"]]
        ).astype(float)

        minX = paramToX(np.min(param4d, axis=(0,)))
        maxX = paramToX(np.max(param4d, axis=(0,)))
        bounds = list(zip(minX, maxX))

        if hasattr(self.modelInterpolator, "getSmallerInterpolator"):
            modelInterpolator = self.modelInterpolator.getSmallerInterpolator(self.config.numPCAComponents)
        else:
            self.log.warn(
                "FluxModelInterpolator does not have getSmallerInterpolator() method."
                " Update fluxmodeldata package."
            )
            modelInterpolator = self.modelInterpolator

        bestParams: dict[int, Struct] = {}
        # This dict will be filled only if debug mode is on
        whitenedModels: dict[int, PfsSimpleSpectrum] = {}

        for iFiber, obsSpectrum in enumerate(fibers(pfsConfig, obsSpectra)):
            fiberId = pfsConfig.fiberId[iFiber]
            velocity = radialVelocities.get(fiberId)
            priorPdf = priorPdfs.get(fiberId)

            if velocity is None or velocity.fail or not np.isfinite(velocity.velocity):
                continue
            if priorPdf is None or not np.all(np.isfinite(priorPdf)):
                continue

            self.log.info("Fitting a model to spectrum: fiberId=%d", fiberId)

            beta = velocity.velocity / const.c.to("km/s").value
            doppler = np.sqrt((1.0 + beta) / (1.0 - beta))

            teffList, prior1dList = marginalizePdf(param4d, priorPdf, axis=(1, 2, 3))
            teffList = teffList.reshape(-1)  # Shape (N, 1) => (N,)
            prior1d = scipy.interpolate.Akima1DInterpolator(teffList, prior1dList)

            def objective(x: NDArray, returnChisq=False) -> float | Struct:
                """Objective function to minimize.

                Parameters
                ----------
                x : `numpy.ndarray`, shape (4,)
                    Array returned by ``paramToX([teff, logg, m, alpha])``
                returnChisq : `bool`
                    If true, this function returns (chi2, dof).
                    If false (default), this function returns an objective
                    taking both chi2 and prior into account.

                Returns
                -------
                objective : float
                    Badness of ``x``. (returned only if ``returnChisq=False``)
                chi2 : `float`
                    chi square. (returned only if ``returnChisq=True``)
                dof : `int`
                    degree of freedom. (returned only if ``returnChisq=True``)
                whitenedModel : `PfsSimpleSpectrum`
                    Best-fit model, whitened. (returned only if ``returnChisq=True``)
                """
                param = xToParam(x)  # Note: param = (teff, logg, m, alpha)
                prior = prior1d(param[0])
                if prior <= 0:
                    if returnChisq:
                        return Struct(chi2=np.inf, dof=0)
                    else:
                        return np.inf

                model = modelInterpolator.interpolate(*param)
                model.wavelength = model.wavelength * doppler
                model = convolveLsf(model, pfsMergedLsf[fiberId], obsSpectrum.wavelength)
                model = model.resample(obsSpectrum.wavelength)
                modelContinuum = self.computeContinuum(model, mode="observed")
                model = modelContinuum.whiten(model)
                chisq = calculateSpecChiSquare(obsSpectrum, model, self.getBadMask())
                if returnChisq:
                    # Add this member for debug output.
                    chisq.whitenedModel = model
                    return chisq
                else:
                    return chisq.chi2 - 2 * math.log(prior)

            x0 = paramToX(param4d[np.argmax(priorPdf)])

            with np.errstate(invalid="ignore"):
                result = scipy.optimize.minimize(
                    objective, x0, bounds=bounds, method=self.config.minimizationMethod
                )
            param = xToParam(result.x)
            chisq = objective(result.x, returnChisq=True)

            bestParams[fiberId] = Struct(
                param=ModelParam(*param),
                chi2=chisq.chi2,
                dof=chisq.dof,
                success=result.success and math.isfinite(chisq.chi2),
            )

            if self.debugInfo.doWriteWhitenedFlux:
                whitenedModels[fiberId] = chisq.whitenedModel

        if self.debugInfo.doWriteWhitenedFlux:
            debugging.writeExtraData(
                f"fitFluxReference-output/whitenedModel-{obsSpectra.filename}.pickle",
                whitenedModel=whitenedModels,
            )

        return bestParams

    def fitModelsToSpectraBruteForce(
        self,
        pfsConfig: PfsConfig,
        obsSpectra: PfsFiberArraySet,
        pfsMergedLsf: LsfDict,
        radialVelocities: dict[int, Struct],
        priorPdfs: dict[int, NDArray[np.float64]],
    ) -> dict[int, Struct]:
        """For each observed spectrum,
        get probability of each model fitting to the spectrum.

        This method does not use a wise optimizer but evaluates chi^2 for every
        pre-computed model template in ``FluxModelSet`` to return the global
        minimum. Note that this method, for speed, adopts some approximations
        that ``fitModelsToSpectra`` does not use.
        For example, this method assumes all fibers have the same average LSF.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
        obsSpectra : `PfsFiberArraySet`
            Continuum-subtracted observed spectra
        pfsMergedLsf : `dict` [`int`, `pfs.drp.stella.Lsf`]
            Combined line-spread functions indexed by fiberId.
        radialVelocities : `dict` [`int`, `lsst.pipe.base.Struct`]
            Mapping from ``fiberId`` to radial velocity.
            Each element, if not None, has ``velocity``, ``error``, and ``fail``
            as its member. See ``EstimateRadialVelocityTask``.
        priorPdfs : `dict` [`int`, `NDArray` [`np.float64`]]
            For each ``priorPdfs[fiberId]`` in ``priorPdfs``,
            ``priorPdfs[fiberId][iSED]`` is the prior probability of the SED ``iSED``
            matching the spectrum ``fiberId``.

        Returns
        -------
        bestParams : `dict` [`int`, `Struct`]
            ``bestParams[fiberId]`` is the best-fit parameters for ``fiberId``.
            Each element ``bestParams[fiberId]``, consists of the following members:

              - param : `ModelParam`
                Best-fit parameter.
              - success : `bool`
                True if optimization has succeeded.
        """
        # Get the list of valid fiberIds
        fiberIds: list[int] = []
        for fiberId in pfsConfig.fiberId:
            velocity = radialVelocities.get(fiberId)
            priorPdf = priorPdfs.get(fiberId)

            if velocity is None or velocity.fail or not np.isfinite(velocity.velocity):
                continue
            if priorPdf is None or not np.all(np.isfinite(priorPdf)):
                continue

            fiberIds.append(fiberId)

        nFibers = len(fiberIds)
        nModels = len(self.fluxModelSet.parameters)
        relativePriors = np.full(shape=(nModels, nFibers), fill_value=np.nan, dtype=float)
        for iFiber, fiberId in enumerate(fiberIds):
            pdf = priorPdfs[fiberId]
            relativePriors[:, iFiber] = pdf / np.max(pdf)

        # prepare arrays of chi-squares.
        chisqLists: dict[int, ChisqList] = {}
        for fiberId in fiberIds:
            chisq = np.full(
                shape=(len(self.fluxModelSet.parameters),),
                fill_value=np.inf,
                dtype=float,
            )
            chisqLists[fiberId] = ChisqList(chisq, 0)

        averageLsf = getAverageLsf([pfsMergedLsf[fiberId] for fiberId in fiberIds])

        # `fibers()` is not very fast.
        # We don't want to call it redundantly in the inner loop below.
        obsSpectrums = {
            fiberId: spectrum for fiberId, spectrum in zip(pfsConfig.fiberId, fibers(pfsConfig, obsSpectra))
        }

        for iModel, (param, priorPdf) in enumerate(zip(self.fluxModelSet.parameters, relativePriors)):
            # These things will be created afterward when they are actually required.
            model: PfsSimpleSpectrum | None = None
            modelContinuum: "Continuum" | None = None

            for iFiber, fiberId in enumerate(fiberIds):
                velocity = radialVelocities[fiberId]
                obsSpectrum = obsSpectrums[fiberId]
                prior = priorPdf[iFiber]

                if not (prior >= self.config.priorCutoff):
                    continue
                if model is None:
                    model = self.fluxModelSet.getSpectrum(
                        teff=param["teff"], logg=param["logg"], m=param["m"], alpha=param["alpha"]
                    )
                if modelContinuum is None:
                    convolvedModel = self.downsampleModel(
                        convolveLsf(model, averageLsf, obsSpectrum.wavelength),
                        obsSpectrum,
                    )
                    modelContinuum = self.computeContinuum(convolvedModel, mode="downsampled")

                convolvedModel = self.downsampleModel(
                    convolveLsf(model, pfsMergedLsf[fiberId], obsSpectrum.wavelength),
                    obsSpectrum,
                )
                convolvedModel = modelContinuum.whiten(convolvedModel)
                chisq = calculateSpecChiSquareWithVelocity(
                    obsSpectrum, convolvedModel, velocity.velocity, self.getBadMask()
                )
                chisqLists[fiberId].chisq[iModel] = chisq.chi2
                # `chisq.dof` depends only on `fiberId`, so we can overwrite it
                chisqLists[fiberId].dof = chisq.dof

        # Output best-fit model spectra in the state they were
        # when they were compared to the whitened observed spectra
        if self.debugInfo.doWriteWhitenedFlux:
            whitenedModels = {}
            for fiberId in fiberIds:
                velocity = radialVelocities[fiberId]
                obsSpectrum = obsSpectrums[fiberId]
                iModel = np.argmin(chisqLists[fiberId].chisq)
                param = self.fluxModelSet.parameters[iModel]
                model = self.fluxModelSet.getSpectrum(
                    teff=param["teff"], logg=param["logg"], m=param["m"], alpha=param["alpha"]
                )
                convolvedModel = self.downsampleModel(
                    convolveLsf(model, averageLsf, obsSpectrum.wavelength),
                    obsSpectrum,
                )
                modelContinuum = self.computeContinuum(convolvedModel, mode="downsampled")
                convolvedModel = self.downsampleModel(
                    convolveLsf(model, pfsMergedLsf[fiberId], obsSpectrum.wavelength),
                    obsSpectrum,
                )
                convolvedModel = modelContinuum.whiten(convolvedModel)
                shifted = dopplerShift(
                    convolvedModel.wavelength,
                    convolvedModel.flux,
                    convolvedModel.mask,
                    velocity.velocity,
                    convolvedModel.wavelength,
                )
                convolvedModel.flux[...] = shifted.flux
                convolvedModel.mask[...] = shifted.mask
                whitenedModels[fiberId] = convolvedModel

            debugging.writeExtraData(
                f"fitFluxReference-output/whitenedModel-{obsSpectra.filename}.pickle",
                whitenedModel=whitenedModels,
            )

        if self.debugInfo.doWriteChisq:
            debugging.writeExtraData(
                f"fitFluxReference-output/chisq-{obsSpectra.filename}.pickle",
                chisq=chisqLists,
            )

        # Posterior PDF
        pdfs = {fiberId: chisqLists[fiberId].toProbability(prior=priorPdfs[fiberId]) for fiberId in fiberIds}

        if self.debugInfo.doWritePosterior:
            debugging.writeExtraData(
                f"fitFluxReference-output/posterior-{obsSpectra.filename}.pickle",
                posterior=pdfs,
            )

        bestParams = self.findSubgridPeak(pdfs)
        return {fiberId: Struct(param=param, success=True) for fiberId, param in bestParams.items()}

    def findRoughlyBestModel(
        self, pdfs: dict[int, NDArray[np.float64]], *, paramOnly: bool = False
    ) -> dict[int, Struct]:
        """Get the model spectrum corresponding to ``argmax(pdf)``
        for ``pdf`` in ``pdfs.values()``.

        Parameters
        ----------
        pdfs : `dict` [`int`, `NDArray` [`np.float64`]]
            For each ``pdfs[fiberId]`` in ``pdfs``,
            ``pdfs[fiberId][iSED]`` is the probability of the SED ``iSED``
            matching the spectrum ``fiberId``.
        paramOnly : `bool`
            If True, the values of the returned dict, which are of `Struct` type,
            have ``param`` member only.

        Returns
        -------
        models : `dict` [`int`, `lsst.pipe.base.Struct`]
            Mapping from ``fiberId`` to a structure whose members are:

              - spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
                    Spectrum.
                    This member does not exist if ``paramOnly`` is ``True``.
              - param : `ModelParam`
                    Parameter of ``spectrum``.
        """
        onePlusEpsilon = float(np.nextafter(np.float32(1), np.float32(2)))
        models: dict[int, Struct] = {}
        for fiberId, pdf in pdfs.items():
            if np.max(pdf) <= np.min(pdf) * onePlusEpsilon:
                # If the PDF is uniform, we ourselves choose a parameter set
                # because this one is better than
                #     `self.fluxModelSet.parameters[np.argmax(pdf)]`
                # which is always `self.fluxModelSet.parameters[0]`.
                param = ModelParam(teff=7500, logg=4.5, m=0.0, alpha=0.0)
                self.log.warn("findRoughlyBestModel: Probability distribution is uniform.")
            else:
                param = ModelParam.fromDict(self.fluxModelSet.parameters[np.argmax(pdf)])

            if paramOnly:
                models[fiberId] = Struct(param=param)
            else:
                model = self.fluxModelSet.getSpectrum(**param.toDict())
                models[fiberId] = Struct(spectrum=model, param=param)

        return models

    def findSubgridPeak(self, pdfs: dict[int, NDArray[np.float64]]) -> dict[int, "ModelParam"]:
        """Get ``argmax(pdf)`` for ``pdf`` in ``pdfs`` to subgrid precision.

        A smooth surface is fit to the ``pdf``,
        and the ``argmax`` here actually means the top of the surface.

        Parameters
        ----------
        pdfs : `dict` [`int`, `NDArray` [`np.float64`]]
            For each ``pdfs[fiberId]`` in ``pdfs``,
            ``pdfs[fiberId][iSED]`` is the probability of the SED ``iSED``
            matching the spectrum ``fiberId``.

        Returns
        -------
        params : `dict` [`int`, `ModelParam`]
            Mapping from ``fiberId`` to the best-fit parameter for the spectrum.
        """
        paramNames = ["teff", "logg", "m"]
        fixedParamNames = ["alpha"]

        # Note: numPointsToFitTo[len(paramNames)]
        #        = number of points to fit a function to.
        numPointsToFitTo = [
            # Number of grid points within the (d-1)-dimensional sphere
            # of radius sqrt(d) in d-dimensional space.
            1,
            3,
            9,
            27,
            89,
            333,
        ]

        paramCatalog = self.fluxModelSet.parameters

        outputParams: dict[int, ModelParam] = {}
        for fiberId, pdf in pdfs.items():

            # Rough peak
            peakParam = paramCatalog[np.argmax(pdf)]

            paramToIndex = {
                param[len(fixedParamNames) :]: index
                for index, param in enumerate(
                    zip(*(paramCatalog[name] for name in fixedParamNames + paramNames))
                )
                if all(param[i] == peakParam[fpname] for i, fpname in enumerate(fixedParamNames))
            }

            # Axes of params are diff. from each other by orders of magnitude.
            # We use not the raw values of the params
            # but their indices (tics) to draw a sphere in the parameter space.
            #
            # Note: ticToParam[i] = [-0.2, -0.1, 0, 0.1, 0.2] etc.
            # is the list of values of paramNames[i].
            ticToParam = [
                sorted(set(param[i] for param in paramToIndex.keys())) for i in range(len(paramNames))
            ]
            # Note: tic = paramToTic[i][value] is the inverse function
            # of value = ticToParam[i][tic]
            paramToTic = [{value: tic for tic, value in enumerate(toParam)} for toParam in ticToParam]
            # Rough peak in terms of tics
            peakTic = tuple(paramToTic[i][peakParam[name]] for i, name in enumerate(paramNames))
            # Sampled points in terms of tics
            ticList = [tuple(paramToTic[i][p] for i, p in enumerate(param)) for param in paramToIndex.keys()]

            # We use some samples nearest to the rough peak.
            # Notice that the samples actually chosen are not necessarily
            # arranged neatly within a sphere---for example,
            # the rough peak may be on the border of the domain,
            # or there may be grid defects in the neighborhood of the peak.
            ticList.sort(key=lambda tpl: sum((x - y) ** 2 for x, y in zip(tpl, peakTic)))
            ticList = ticList[: numPointsToFitTo[len(paramNames)]]
            # Convert selected tics to indices in paramCatalog
            indices = np.array(
                [paramToIndex[tuple(ticToParam[i][t] for i, t in enumerate(tic))] for tic in ticList],
                dtype=int,
            )
            # Cut only necessary portions out of pdf and paramCatalog.
            cutProb = pdf[indices]
            cutParamCatalog = paramCatalog[indices]

            # Fit y = \sum_{i+j <= 2} coeff[i,j] x[i] x[j];
            # where y = pdf, x[0] = 1, x[1:] = (parameters)
            axisList = [1.0] + [cutParamCatalog[name] for name in paramNames]
            axisToIndex = {}
            M = np.empty(shape=(len(cutProb), len(axisList) * (len(axisList) + 1) // 2), dtype=float)
            k = 0
            for i in range(0, len(axisList)):
                for j in range(i, len(axisList)):
                    M[:, k] = axisList[i] * axisList[j]
                    axisToIndex[i, j] = k
                    k += 1

            coeff = np.linalg.lstsq(M, cutProb, rcond=None)[0]

            # Reorder the terms of the polynomial
            # from y = \sum_{i+j <= 2} coeff[i,j] x[i] x[j]
            # to y = x'^T A x' + b^T x' + c
            # where x' = x[1:]
            axisList = axisList[1:]
            A = np.empty(shape=(len(axisList), len(axisList)), dtype=float)
            for i in range(0, len(axisList)):
                A[i, i] = coeff[axisToIndex[i + 1, i + 1]]
                for j in range(i + 1, len(axisList)):
                    A[i, j] = A[j, i] = 0.5 * coeff[axisToIndex[i + 1, j + 1]]

            b = np.empty(shape=(len(axisList)), dtype=float)
            for i in range(0, len(axisList)):
                b[i] = coeff[axisToIndex[0, i + 1]]

            # Now we know the peak is at -(1/2) A^{-1} b,
            # but we have to be careful whether it is a valid solution.
            bestParam = None
            if np.linalg.det(A) > 0:
                peak = -0.5 * np.linalg.solve(A, b)
                # We employ this peak only if it is within the minimum box
                # that includes ticList.
                if all(
                    ticToParam[i][min(tic[i] for tic in ticList)]
                    <= p
                    <= ticToParam[i][max(tic[i] for tic in ticList)]
                    for i, p in enumerate(peak)
                ):
                    bestParam = tuple(peak) + tuple(peakParam[name] for name in fixedParamNames)

            if bestParam is None:
                bestParam = tuple(peakParam[name] for name in paramNames + fixedParamNames)

            outputParams[fiberId] = ModelParam(*bestParam)

        return outputParams

    def maskUninterestingRegions(self, spectra: PfsFiberArraySet) -> PfsFiberArraySet:
        """Mask regions to be ignored.

        Parameters
        ----------
        spectra : `pfs.drp.stella.datamodel.PfsFiberArraySet`
            A set of spectra.

        Returns
        -------
        spectra : `pfs.drp.stella.datamodel.PfsFiberArraySet`
            The same instance as the argument.
        """
        if isMidResolution(spectra):
            ignoredRangesLeft = self.config.mrIgnoredRangesLeft
            ignoredRangesRight = self.config.mrIgnoredRangesRight
        else:
            ignoredRangesLeft = self.config.lrIgnoredRangesLeft
            ignoredRangesRight = self.config.lrIgnoredRangesRight

        # Mask atmospheric absorption lines etc.
        wavelength = spectra.wavelength
        badMask = spectra.flags.add("ATMOSPHERE")

        for low, high in zip(ignoredRangesLeft, ignoredRangesRight):
            spectra.mask[...] |= np.where((low < wavelength) & (wavelength < high), badMask, 0).astype(
                spectra.mask.dtype
            )

        # Mask edge regions.
        spectra.mask[...] |= np.where(
            (self.config.minWavelength < spectra.wavelength)
            & (spectra.wavelength < self.config.maxWavelength),
            0,
            spectra.flags.add("EDGE"),
        ).astype(spectra.mask.dtype)

        return spectra

    def makeReferenceSpectra(
        self, pfsConfig: PfsConfig, params: dict[int, "ModelParam"]
    ) -> dict[int, Struct]:
        """Get the model spectrum corresponding to each ``param`` in ``params.values()``.

        The returned spectra are affected by galactic extinction
        and their flux values agree with ``pfsConfig.psfFlux``.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
        params : `dict` [`int`, `ModelParam`]
            Each ``params[fiberId]`` in ``params``, if not None,
            is the parameter for ``fiberId``.

        Returns
        -------
        models : `dict` [`int`, `Struct`]
            Mapping from ``fiberId`` to a structure whose members are:

              - spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
                    Spectrum.
              - param : `ModelParam`
                    Parameter of ``spectrum``.
              - fluxScalingChi2 : `float`
                    chi^2 of flux scaling problem.
              - fluxScalingDof  : `int`
                    Degree of freedom of flux scaling problem.
        """
        bestModels: dict[int, Struct] = {}

        for fiberConfig in fiberConfigs(pfsConfig):
            fiberId = int(fiberConfig.fiberId.reshape(()))
            param = params.get(fiberId)
            if param is None:
                continue

            model = Struct()
            model.param = param
            model.spectrum = self.modelInterpolator.interpolate(**model.param.toDict())

            ebv = self.extinctionMap(fiberConfig.ra[0], fiberConfig.dec[0])
            extinction = F99ExtinctionCurve(self.config.Rv)
            model.spectrum.flux *= extinction.attenuation(model.spectrum.wavelength, ebv)

            scaled = adjustAbsoluteScale(model.spectrum, fiberConfig, self.config.broadbandFluxType)
            model.spectrum = scaled.spectrum
            model.fluxScalingChi2 = scaled.chi2
            model.fluxScalingDof = scaled.dof
            bestModels[fiberId] = model

        return bestModels

    def correctExtinction(self, pfsConfig: PfsConfig) -> PfsConfig:
        """Remove galactic extinction from photometry data, in place.

        Extinction is estimated for an F0 star.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
            Flux values in this object are overwritten by this function.

        Returns
        -------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
            This is the same instance as the argument.
        """
        # F0 star
        f0Model = self.fluxModelSet.getSpectrum(teff=7500, logg=4.5, m=0.0, alpha=0.0)

        ebvs = self.extinctionMap(pfsConfig.ra, pfsConfig.dec)
        extinction = F99ExtinctionCurve(self.config.Rv)

        for i, ebv in enumerate(ebvs):
            attenuatedF0Model = copy.copy(f0Model)
            attenuatedF0Model.flux = f0Model.flux * extinction.attenuation(f0Model.wavelength, ebvs[i])

            filterNames = pfsConfig.filterNames[i]
            unattenuatedFlux = np.empty(len(filterNames), dtype=float)
            attenuatedFlux = np.empty(len(filterNames), dtype=float)

            for iFilter, filterName in enumerate(filterNames):
                unattenuatedFlux[iFilter] = FilterCurve(filterName).photometer(f0Model)
                attenuatedFlux[iFilter] = FilterCurve(filterName).photometer(attenuatedF0Model)

            fractionalExtinction = attenuatedFlux / unattenuatedFlux

            pfsConfig.fiberFlux[i] /= fractionalExtinction
            pfsConfig.fiberFluxErr[i] /= fractionalExtinction
            pfsConfig.psfFlux[i] /= fractionalExtinction
            pfsConfig.psfFluxErr[i] /= fractionalExtinction
            pfsConfig.totalFlux[i] /= fractionalExtinction
            pfsConfig.totalFluxErr[i] /= fractionalExtinction

        return pfsConfig

    def getBadMask(self) -> list[str]:
        """Get the list of bad masks.

        The bad masks are those specified by the task configuration,
        plus some additional masks used by this task.

        Returns
        -------
        mask : `list` of `str`
            Mask names.
        """
        badMask = set(self.config.badMask)
        badMask.update(["EDGE", "ATMOSPHERE"])
        return list(badMask)

    def downsampleModel(self, model: PfsSimpleSpectrum, obsSpectrum: PfsSimpleSpectrum) -> PfsSimpleSpectrum:
        """Downsample a model spectrum so that its resolution will not be
        unmeaningfully higher than the observed spectrum's.

        Parameters
        ----------
        model : `PfsSimpleSpectrum`
            Model spectrum.
        obsSpectrum : `PfsSimpleSpectrum`
            Observed spectrum.

        Returns
        -------
        downsampled : `PfsSimpleSpectrum`
            Downsampled model.
        """
        if self.config.modelOversamplingFactor <= 0:
            # Downsampling is disabled
            return model

        minWavelength = model.wavelength[0]
        maxWavelength = model.wavelength[-1]
        dlambdaModel = np.max(model.wavelength[1:] - model.wavelength[:-1])
        dlambdaObs = np.min(obsSpectrum.wavelength[1:] - obsSpectrum.wavelength[:-1])
        # This is \Delta\lambda_{new} that we aim at
        dlambdaNew = dlambdaObs / self.config.modelOversamplingFactor
        lenNew = 1 + int(math.ceil((maxWavelength - minWavelength) / dlambdaNew))
        # This is \Delta\lambda_{new} that will actually be realized
        dlambdaNew = (maxWavelength - minWavelength) / (lenNew - 1)

        if dlambdaNew <= dlambdaModel:
            return model

        wavelengthNew = np.linspace(minWavelength, maxWavelength, num=lenNew, endpoint=True)
        return model.resample(wavelengthNew)

    def isParamInDomain(self, param: "ModelParam") -> bool:
        """Is a model parameter contained in the (hypothetical) domain
        where model spectra are defined?

        Parameters
        ----------
        param : `ModelParam`
            parameter to test

        Returns
        -------
        contained : `bool`
            True if the model parameter is contained in the domain.
        """
        point = np.asarray(dataclasses.astuple(param), dtype=float)
        domain = np.lib.recfunctions.structured_to_unstructured(
            self.fluxModelSet.parameters[["teff", "logg", "m", "alpha"]]
        ).astype(float)
        margin = np.asarray(self.config.paramMargin, dtype=float)
        return isPointContainedInDomain(point, domain, margin)


@dataclasses.dataclass
class ModelParam:
    """Parameter set of a model spectra.

    Parameters
    ----------
    teff : `float`
        Effective temperature in K.
    logg : `float`
        Surface gravity in Log(g/cm/s^2).
    m : `float`
        Metalicity in M/H.
    alpha : `float`
        Alpha-elements abundance in alpha/Fe.
    """

    teff: float
    logg: float
    m: float
    alpha: float

    @classmethod
    def fromDict(cls, mapping: Mapping[str, float]) -> "ModelParam":
        """Construct an instance from a dict-like object.

        Parameters
        ----------
        mapping : `Mapping[str, float]`
            A dict-like object that contains "teff", "logg", "m", and "alpha".
            There may be other keys, which are ignored by this method.

        Returns
        -------
        instance : `ModelParam`
            Constructed instance.
        """
        return cls(
            teff=float(mapping["teff"]),
            logg=float(mapping["logg"]),
            m=float(mapping["m"]),
            alpha=float(mapping["alpha"]),
        )

    def toDict(self) -> dict[str, float]:
        """Convert ``self`` to a dictionary.

        Returns
        -------
        dic : `dict` [`str`, `float`]
            A dictionary that contains "teff", "logg", "m", and "alpha".
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Continuum:
    """Continuous spectra.

    This class is the return type of
    ``FitFluxReferenceTask.computeContinuum()``

    Parameters
    ----------
    flux : `numpy.array`
        Continuum. Shape (M, N).
        M is the number of spectra and N is the number of samples.
    invalidIndices : `numpy.array`
        List of indices in [0, M).
        ``flux[i]`` (``i`` in  ``invalidIndices``) is invalid.
    """

    flux: NDArray[np.float64]
    invalidIndices: NDArray[np.int64]

    @overload
    def whiten(self, spectra: PfsFiberArraySet) -> PfsFiberArraySet:
        ...

    @overload
    def whiten(self, spectra: PfsSimpleSpectrum) -> PfsSimpleSpectrum:
        ...

    def whiten(self, spectra):
        """Divide ``spectra`` by ``self`` to whiten them.

        Parameters
        ----------
        spectra : `PfsSimpleSpectrum` or `PfsFiberArraySet`
            spectra to whiten.

        Returns
        -------
        spectra : `PfsSimpleSpectrum` or `PfsFiberArraySet`
            The same instance as the argument.
        """
        if len(spectra.flux.shape) == 1:
            assert len(self.flux) == 1
            spectra /= self.flux[0, :]
            if 0 in self.invalidIndices:
                spectra.mask[:] |= spectra.flags.add("BAD")
        else:
            spectra /= self.flux
            spectra.mask[self.invalidIndices, :] |= spectra.flags.add("BAD")

        if hasattr(spectra, "norm"):
            spectra.norm[...] = 1.0

        return spectra


def convolveLsf(
    spectrum: PfsSimpleSpectrum, lsf: Lsf, lsfWavelength: NDArray[np.float64]
) -> PfsSimpleSpectrum:
    """Convolve LSF to spectrum.

    This function assumes that ``spectrum`` (synthetic spectrum) is sampled
    more finely than ``lsf`` (LSF of an observed spectrum) is.

    Parameters
    ----------
    spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
        Spectrum.
    lsf : `pfs.drp.stella.Lsf`
        Lsf.
    lsfWavelength : `numpy.array` of `float`
        Wavelength array of the spectrum in which ``lsf`` was measured.

    Returns
    -------
    spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
        New instance of spectrum,
        with the same sampling points as the input spectrum's.
    """
    spectrum = copy.deepcopy(spectrum)
    lsf = lsf.warp(lsfWavelength, spectrum.wavelength)
    spectrum.flux = lsf.convolve(spectrum.flux)
    return spectrum


@deprecated("No longer valid since PIPE2D-1643")
def getAverageLsf(lsfs: Sequence[Lsf]) -> Lsf:
    """Get the average LSF.

    Parameters
    ----------
    lsfs : `list` of `pfs.drp.stella.Lsf`
        List of LSFs to average.

    Returns
    -------
    lsf : `pfs.drp.stella.Lsf`
        Average LSF.
    """
    sigma = np.sqrt(sum(lsf.computeShape().getIxx() for lsf in lsfs) / len(lsfs))
    return GaussianLsf(lsfs[0].length, sigma)


def adjustAbsoluteScale(
    spectrum: PfsSimpleSpectrum, fiberConfig: PfsConfig, broadbandFluxType: Literal["fiber", "psf", "total"]
) -> Struct:
    """Multiply a constant to the spectrum
    so that its integrations will agree to broadband fluxes.

    Because the broadband fluxes (``fiberConfig.psfFlux``) are affected
    by the Galactic extinction, ``spectrum`` must also be reddened before
    the call to this function.

    Parameters
    ----------
    spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
        Spectrum.
    fiberConfig : `pfs.datamodel.pfsConfig.PfsConfig`
        PfsConfig that contains only a single fiber.
    broadbandFluxType : {"fiber", "psf", "total"}
        Type of broadband flux to use.

    Returns
    -------
    spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
        The same instance as the argument.
    chi2 : `float`
        chi^2 of the problem to determine the scaling constant.
    dof : `int`
        Degree of freedom of the problem to determine the scaling constant.
    """
    if broadbandFluxType == "fiber":
        obsFlux = fiberConfig.fiberFlux[0]
        obsFluxErr = fiberConfig.fiberFluxErr[0]
    elif broadbandFluxType == "psf":
        obsFlux = fiberConfig.psfFlux[0]
        obsFluxErr = fiberConfig.psfFluxErr[0]
    elif broadbandFluxType == "total":
        obsFlux = fiberConfig.totalFlux[0]
        obsFluxErr = fiberConfig.totalFluxErr[0]
    else:
        raise ValueError(f"`broadbandFluxType` must be one of fiber|psf|total." f" ('{broadbandFluxType}')")

    obsFlux = np.asarray(obsFlux, dtype=float)
    obsFluxErr = np.asarray(obsFluxErr, dtype=float)

    isgood = np.isfinite(obsFlux) & (obsFluxErr > 0)
    obsFlux = obsFlux[isgood]
    obsFluxErr = obsFluxErr[isgood]
    filterNames = [f for f, good in zip(fiberConfig.filterNames[0], isgood) if good]

    refFluxList = []
    for filterName in filterNames:
        flux = FilterCurve(filterName).photometer(spectrum)
        refFluxList.append(flux)

    refFlux = np.asarray(refFluxList, dtype=float)

    # This is the minimum point of chi^2(scale)
    scale = np.sum(obsFlux * refFlux / obsFluxErr**2) / np.sum((refFlux / obsFluxErr) ** 2)
    # chi^2(scale) at the minimum point
    chi2 = np.sum((obsFlux - scale * refFlux) ** 2 / obsFluxErr**2)

    spectrum.flux[:] *= scale
    return Struct(spectrum=spectrum, chi2=chi2, dof=len(obsFlux) - 1)


def calculateSpecChiSquare(
    obsSpectrum: PfsFiberArray, model: PfsSimpleSpectrum, badMask: Sequence[str]
) -> Struct:
    """Calculate chi square of spectral fitting
    between a single observed spectrum and a single model spectrum.

    ``obsSpectrum.wavelength`` must be the same array as ``model.wavelength``.

    Parameters
    ----------
    obsSpectrum : `pfs.datamodel.pfsFiberArray.PfsFiberArray`
        Observed spectrum.
    model : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
        Model spectrum.
    badMask : `list` [`str`]
        Mask names.

    Returns
    -------
    chi2 : `float`
        chi square.
    dof : `int`
        degree of freedom.
    """
    good = 0 == (model.mask & model.flags.get(*(m for m in badMask if m in model.flags)))
    good &= 0 == (obsSpectrum.mask & obsSpectrum.flags.get(*(m for m in badMask if m in obsSpectrum.flags)))

    modelFlux = model.flux[good]
    flux = np.copy(obsSpectrum.flux)[good]
    invVar = 1.0 / obsSpectrum.variance[good]

    chi2 = np.sum(np.square(flux - modelFlux) * invVar)
    # Degree of freedom must be decremented by 1 because we are comparing
    # whitenend spectra, ignoring their amplitudes.
    dof = np.count_nonzero(good) - 1
    return Struct(chi2=chi2, dof=dof)


def calculateSpecChiSquareWithVelocity(
    obsSpectrum: PfsFiberArray, model: PfsSimpleSpectrum, radialVelocity: float, badMask: Sequence[str]
) -> Struct:
    """Calculate chi square of spectral fitting
    between a single observed spectrum and a single model spectrum.

    Parameters
    ----------
    obsSpectrum : `pfs.datamodel.pfsFiberArray.PfsFiberArray`
        Observed spectrum.
    model : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
        Model spectrum.
    radialVelocity : `float`
        Radial velocity in km/s.
    badMask : `list` [`str`]
        Mask names.

    Returns
    -------
    chi2 : `float`
        chi square.
    dof : `int`
        degree of freedom.
    """
    shifted = dopplerShift(model.wavelength, model.flux, model.mask, radialVelocity, obsSpectrum.wavelength)
    good = 0 == (shifted.mask & model.flags.get(*(m for m in badMask if m in model.flags)))
    good &= 0 == (obsSpectrum.mask & obsSpectrum.flags.get(*(m for m in badMask if m in obsSpectrum.flags)))

    modelFlux = shifted.flux[good]
    flux = np.copy(obsSpectrum.flux)[good]
    invVar = 1.0 / obsSpectrum.variance[good]

    chi2 = np.sum(np.square(flux - modelFlux) * invVar)
    # Degree of freedom must be decremented by 1 because we are comparing
    # whitenend spectra, ignoring their amplitudes.
    dof = np.count_nonzero(good) - 1
    return Struct(chi2=chi2, dof=dof)


def dopplerShift(
    wavelength0: NDArray[np.float64],
    flux0: NDArray[np.float64],
    mask0: NDArray[np.integer],
    velocity: float,
    wavelength: NDArray[np.float64],
) -> Struct:
    """Apply Doppler shift to ``flux0`` (spectrum observed by an observer
    moving together with the source) and interpolate it at ``wavelength``

    Parameters
    ----------
    wavelength0 : `numpy.array` of `float`
        Wavelength, in nm, at which ``flux0`` is sampled.
    flux0 : `numpy.array` of `float`
        Spectrum observed by an observer moving together with the source.
    mask0 : `numpy.array` of `int`
        Mask associated with ``flux0``
    velocity : `float`
        Velocity, in km/s, at which the source is moving away.
    wavelength : `numpy.array` of `float`
        Wavelength, in nm, at which the returned spectrum is sampled.

    Returns
    -------
    flux : `numpy.array` of `float`
        Doppler-shifted spectrum.
    mask : `numpy.array` of `int`
        Mask associated with ``flux``
    """
    beta = velocity / const.c.to("km/s").value
    invDoppler = np.sqrt((1.0 - beta) / (1.0 + beta))
    shiftedWavelength = wavelength * invDoppler

    flux = interpolateFlux(wavelength0, flux0, shiftedWavelength)
    mask = interpolateMask(wavelength0, mask0, shiftedWavelength)

    return Struct(flux=flux, mask=mask)


def isPointContainedInDomain(
    point: NDArray[np.float64], domain: NDArray[np.float64], margin: NDArray[np.float64]
) -> bool:
    """Is a point in d-D space contained in a domain?

    The "domain" is a hypothetical region in which a given finite set of points
    are scattered. Not the domain itself but the scattered points are given.

    The domain is expected not to be convex at all.
    No algorithms for convex polygons are applicable.
    We define that a point is in the domain in the following way.
    (Explanation is done with d (= number of dimensions) = 4)

    "The k-th orthant around a point T = (t_0, t_1, t_2, t_3) with margin m
    = (m_1, m_2, m_3, m_4)" (0 <= k <= 15) is the set of points x
    = (x_0, x_1, x_2, x_3) such that
        - if a_i = 0, then x_i <= t_i - m_i
        - if a_i = 1, then x_i >= t_i + m_i
    for i = 0, 1, 2, 3; where k = a_0 a_1 a_2 a_3 in the binary notation.

    "A point P is in the domain defined by D, a finite set of points,
    with margin m" if:
        There exist 16 points T_{0}, ..., T_{15} in D such that
        P is in the k-th orthant around T_{k} with margin m, for k = 0,...,15.

    Parameters
    ----------
    point : `numpy.ndarray of `float`
        Point in a d-D space. Shape (d,).
    domain: `numpy.ndarray of `float`
        Finite set of points that define a domain. Shape (N, d).
    margin : `numpy.ndarray of `float`
        Margin. Shape (d,).

    Returns
    -------
    contained : `bool`
        True if the ``point`` is contained in ``domain`` with margin ``m``.
    """
    n, dimensions = domain.shape

    xMinusT = np.asarray(point, dtype=domain.dtype).reshape(1, dimensions) - domain

    # ray[i, j, k] will eventually be a boolean value.
    # If it is True, then, as for the j-th dimension, the point P is in a ray
    # toward -oo (k=0) or +oo (k=1) starting at the point T_{i} of the finite
    # set ``domain``, and the distance between P and T_{i} is larger than ``margin``.
    ray = np.empty(shape=(n, dimensions, 2), dtype=domain.dtype)
    ray[:] = xMinusT.reshape(n, dimensions, 1)
    ray[:, :, 0] *= -1.0
    ray = ray >= np.asarray(margin, dtype=domain.dtype).reshape(1, dimensions, 1)

    # From this on, ray[i, 2*j + k] has the same meaning as previous ray[i, j, k]
    ray = ray.reshape(n, -1)

    charToK = {"0": 0, "1": 1}
    for a in range(2**dimensions, 2 ** (dimensions + 1)):
        index = [2 * j + charToK[c] for j, c in enumerate(bin(a)[3:])]
        if not np.any(np.all(ray[:, index], axis=(1,))):
            return False

    return True


def marginalizePdf(
    x: NDArray[np.float64], pdf: NDArray[np.float64], axis: int | tuple[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Marginalize ``axis`` of an N-D probability density ``pdf``.

    Marginalized PDF is computed as if``x`` is arranged like a grid.
    Otherwise, this function still returns something like a probability density,
    but it is not strictly correct.

    Parameters
    ----------
    x : `NDArray` [`np.float64`]
        Sampling points of ``pdf``. Shape (M, N)
    pdf : `NDArray` [`np.float64`]
        ``pdf`` at ``x``. Shape (M,)
    axis : `int` | `tuple` [`int`]
        Axis to marginalize.

    Returns
    -------
    y : `NDArray` [`np.float64`]
        Sampling points of ``marginalizedPdf``. Shape (P, Q)
    marginalizedPdf : `NDArray` [`np.float64`]
        ``pdf`` at ``x``. Shape (P,)
    """
    if not isinstance(axis, tuple):
        axis = (axis,)

    nSamples, nAxes = x.shape

    axisSetToKeep = set(range(nAxes))
    axisSetToKeep.difference_update(axis)
    axisListToKeep = np.array(sorted(axisSetToKeep), dtype=int)

    xToKeep = x[:, axisListToKeep]
    sums = {}

    for i in range(nSamples):
        y = tuple(xToKeep[i])
        p = pdf[i]
        sums[y] = sums.get(y, 0.0) + p

    keyValueList = sorted(sums.items())
    y = np.array([key for key, value in keyValueList], dtype=float)
    marginalizedPdf = np.array([value for key, value in keyValueList], dtype=float)

    return y, marginalizedPdf


def isMidResolution(pfsMerged: PfsMerged) -> bool:
    """Return True if a part of spectra in ``pfsMerged`` are from
    mid-resolution arm.

    Parameters
    ----------
    pfsMerged : `PfsMerged`
        Merged spectra from exposure.

    Returns
    -------
    midres : `bool`
        True if a part of spectra in ``pfsMerged`` are from mid-resolution arm.
    """
    # `pfsMerged.identity.arm` is made from `pfsArm`s that are actually merged
    # into `pfsMerged`, in contrast to `pfsConfig.arms`. The latter comes from
    # `pfsDesign`.
    return "m" in pfsMerged.identity.arm


def removeBadFluxes(
    pfsConfig: PfsConfig,
    fluxType: Literal["fiber", "psf", "total"],
    fluxSNR: float,
) -> None:
    """Remove bad fluxes (fiberFlux, psfFlux, totalFlux) from ``pfsConfig``.

    Fluxes are bad
      - if they are not finite, or
      - if their errors are not positive.

    In checking these conditions, ``fluxType`` flux is used.
    The other fluxes than ``fluxType`` are also removed if ``fluxType`` flux
    is bad, for consistency.

    Parameters
    ----------
    pfsConfig : `PfsConfig`
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
        Configuration of the PFS top-end.
    broadbandFluxType : {"fiber", "psf", "total"}
        Type of broadband flux to use.
    fluxSNR : `float`
        If fluxErr for all bands are NaN, we fabricated fluxErr such that S/N
        will be this much. Pass 0 to disable this behavior.
    """
    filterNameLists = [list_ for list_ in pfsConfig.filterNames]
    # We make deepcopies for fear that they may be overwritten if fluxSNR > 0
    fiberFluxArrays = [np.copy(arr) for arr in pfsConfig.fiberFlux]
    fiberFluxErrArrays = [np.copy(arr) for arr in pfsConfig.fiberFluxErr]
    psfFluxArrays = [np.copy(arr) for arr in pfsConfig.psfFlux]
    psfFluxErrArrays = [np.copy(arr) for arr in pfsConfig.psfFluxErr]
    totalFluxArrays = [np.copy(arr) for arr in pfsConfig.totalFlux]
    totalFluxErrArrays = [np.copy(arr) for arr in pfsConfig.totalFluxErr]

    if fluxType == "fiber":
        refFluxArrays = fiberFluxArrays
        refFluxErrArrays = fiberFluxErrArrays
    elif fluxType == "psf":
        refFluxArrays = psfFluxArrays
        refFluxErrArrays = psfFluxErrArrays
    elif fluxType == "total":
        refFluxArrays = totalFluxArrays
        refFluxErrArrays = totalFluxErrArrays
    else:
        raise ValueError(f"`fluxType` must be one of fiber|psf|total. ('{fluxType}')")

    for i in range(len(filterNameLists)):
        filterNameList = filterNameLists[i]
        fiberFluxArray = fiberFluxArrays[i]
        fiberFluxErrArray = fiberFluxErrArrays[i]
        psfFluxArray = psfFluxArrays[i]
        psfFluxErrArray = psfFluxErrArrays[i]
        totalFluxArray = totalFluxArrays[i]
        totalFluxErrArray = totalFluxErrArrays[i]
        refFluxArray = refFluxArrays[i]
        refFluxErrArray = refFluxErrArrays[i]

        if fluxSNR > 0:
            # Notice that refFluxErrArray will also be modified
            # if {fiber,psf,total}FluxErrArray are modified.
            if np.all(np.isnan(fiberFluxErrArray)):
                fiberFluxErrArray[:] = fiberFluxArray / fluxSNR
            if np.all(np.isnan(psfFluxErrArray)):
                psfFluxErrArray[:] = psfFluxArray / fluxSNR
            if np.all(np.isnan(totalFluxErrArray)):
                totalFluxErrArray[:] = totalFluxArray / fluxSNR

        isGood = np.isfinite(refFluxArray) & (refFluxErrArray > 0)

        filterNameLists[i] = [x for x, good in zip(filterNameList, isGood) if good]
        fiberFluxArrays[i] = fiberFluxArray[isGood]
        fiberFluxErrArrays[i] = fiberFluxErrArray[isGood]
        psfFluxArrays[i] = psfFluxArray[isGood]
        psfFluxErrArrays[i] = psfFluxErrArray[isGood]
        totalFluxArrays[i] = totalFluxArray[isGood]
        totalFluxErrArrays[i] = totalFluxErrArray[isGood]

    pfsConfig.filterNames = filterNameLists
    pfsConfig.fiberFlux = fiberFluxArrays
    pfsConfig.fiberFluxErr = fiberFluxErrArrays
    pfsConfig.psfFlux = psfFluxArrays
    pfsConfig.psfFluxErr = psfFluxErrArrays
    pfsConfig.totalFlux = totalFluxArrays
    pfsConfig.totalFluxErr = totalFluxErrArrays


def removeBadSpectra(
    pfsMerged: PfsMerged,
    cutoffSNR: float,
    cutoffSNRRange: tuple[float, float],
) -> PfsMerged:
    """Remove bad spectra from ``pfsMerged``.

    Spectra are bad if their average pixel-wise S/N is worse than ``cutoffSNR``.

    Parameters
    ----------
    pfsMerged : `PfsMerged`
        Merged spectra from exposure.
    cutoffSNR : `float`
        Cut-off S/N.
    cutoffSNRRange : `tuple` [`float`, `float`]
        Wavelength range (nm) in which pixel-wise S/N is averaged to be compared
        with ``cutoffSNR``.

    Returns
    -------
    pfsMerged : `PfsMerged`
        PfsMerged from which bad spectra have been removed.
    """
    left, right = cutoffSNRRange
    good = np.zeros(shape=(len(pfsMerged),), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(pfsMerged)):
            sampleIndex = (left <= pfsMerged.wavelength[i]) & (pfsMerged.wavelength[i] <= right)
            snr = pfsMerged.flux[i, sampleIndex] / np.sqrt(pfsMerged.variance[i, sampleIndex])
            good[i] = np.nanmedian(snr) >= cutoffSNR

    return pfsMerged[good]


def promoteSimpleSpectrumToFiberArray(spectrum: PfsSimpleSpectrum, snr: float) -> PfsFiberArray:
    """Promote an instance of PfsSimpleSpectrum to PfsFiberArray.

    Parameters
    ----------
    spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
        A simple spectrum without additional information such as ``covar``.
    snr : `float`
        Signal to noise ratio from which to invent ``covar`` array.
        (variance = (median(flux) / snr)**2).
        Note that no actual noise will be added to the input flux.

    Returns
    -------
    fiberArraySet : `pfs.drp.stella.datamodel.PfsFiberArraySet`
        `PfsFiberArraySet` that contains only the input fiber.
    """
    observations = Observations(
        visit=np.zeros(0, dtype=int),
        arm=np.zeros(0, dtype="U0"),
        spectrograph=np.zeros(0, dtype=int),
        pfsDesignId=np.zeros(0, dtype=int),
        fiberId=np.zeros(0, dtype=int),
        pfiNominal=np.zeros(shape=(0, 2), dtype=float),
        pfiCenter=np.zeros(shape=(0, 2), dtype=float),
    )

    spectrum = PfsSingle(
        target=spectrum.target,
        observations=observations,
        wavelength=spectrum.wavelength,
        flux=spectrum.flux,
        mask=spectrum.mask,
        sky=np.zeros(shape=spectrum.flux.shape, dtype=float),
        covar=np.zeros(shape=(3,) + spectrum.flux.shape, dtype=float),
        covar2=np.zeros(shape=(0, 0), dtype=float),
        flags=spectrum.flags,
        metadata=spectrum.metadata,
    )

    noise = np.nanmedian(spectrum.flux) / snr
    spectrum.variance[:] = noise**2

    return spectrum


def promoteFiberArrayToFiberArraySet(spectrum: PfsFiberArray, fiberId: int) -> PfsFiberArraySet:
    """Promote an instance of PfsFiberArray to PfsFiberArraySet.

    Parameters
    ----------
    spectrum : `pfs.datamodel.pfsFiberArray.PfsFiberArray`
        spectrum observed with a fiber.
    fiberId : `int`
        ID of the fiber.
        .
    Returns
    -------
    fiberArraySet : `pfs.drp.stella.datamodel.PfsFiberArraySet`
        `PfsFiberArraySet` that contains only the input fiber.
    """
    return PfsMerged(
        identity=Identity(visit=0, arm="", spectrograph=1, pfsDesignId=0),
        fiberId=np.full(shape=(1,), fill_value=fiberId, dtype=int),
        wavelength=spectrum.wavelength.reshape(1, -1),
        flux=spectrum.flux.reshape(1, -1),
        mask=spectrum.mask.reshape(1, -1),
        sky=spectrum.sky.reshape(1, -1),
        norm=np.ones_like(spectrum.flux).reshape(1, -1),
        covar=spectrum.covar.reshape((1,) + spectrum.covar.shape),
        flags=spectrum.flags,
        metadata=spectrum.metadata,
    )


def fibers(pfsConfig: PfsConfig, fiberArraySet: PfsFiberArraySet) -> Generator[PfsFiberArray, None, None]:
    """Iterator that yields each fiber in `PfsFiberArraySet`.

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
        Configuration of the PFS top-end.
    fiberArraySet : `pfs.drp.stella.datamodel.PfsFiberArraySet`
        Set of spectra observed with a set of fibers

    Yields
    -------
    fiberArray : `pfs.datamodel.pfsFiberArray.PfsFiberArray`
        spectrum observed with a fiber.
    """
    for fiberId in pfsConfig.fiberId:
        yield fiberArraySet.extractFiber(PfsSingle, pfsConfig, fiberId)


def fiberConfigs(pfsConfig: PfsConfig) -> Generator[PfsConfig, None, None]:
    """Iterator that yields single-fiber PfsConfigs.

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
        Configuration of the PFS top-end.

    Yields
    ------
    fiberConfig : `pfs.datamodel.pfsConfig.PfsConfig`
        PfsConfig that holds only a single fiber
    """
    n = len(pfsConfig.fiberId)
    for i in range(n):
        index = np.zeros(n, dtype=bool)
        index[i] = True
        yield pfsConfig[index]


class TransmissionCurve:
    """A transmission curve

    Used for synthetic photometry as part of photometric calibration.

    The transmission curve is represented as a lookup table, with arryas of
    wavelength and the corresponding transmission.

    Parameters
    ----------
    wavelength : array_like
        Array of wavelength values, nm.
    transmission : array_like
        Array of corresponding transmission values.
    """

    def __init__(self, wavelength, transmission):
        if len(wavelength) != len(transmission):
            raise RuntimeError(
                "Mismatched lengths for wavelength and transmission: "
                f"{len(wavelength)} vs {len(transmission)}"
            )
        self.wavelength = wavelength
        self.transmission = transmission

    def interpolate(self, wavelength):
        """Interpolate the filter transmission curve at the provided wavelength

        Parameters
        ----------
        wavelength : array_like
            Wavelength at which to interpolate the transmission curve.

        Returns
        -------
        transmission : array_like
            Transmission at the provided wavelength.
        """
        return np.interp(wavelength, self.wavelength, self.transmission, 0.0, 0.0)

    def _integrate(self, specFlux=None, specWavelength=None, power=1):
        r"""Integrate the filter transmission curve for synthetic photometry

        The integral is:

        .. math:: \int S(\lambda) (F(\lambda) / \lambda d\lambda)^p

        where :math:`F(\lambda)` is the filter transmission curve,
        :math:`S(\lambda)` is the spectrum, and the extra `\lambda` term is due
        to using photon-counting detectors. :math:`p` (``power``) is usually 1,
        but it can be, say, 2 in the numerator of a variance formula.

        Parameters
        ----------
        specFlux : `numpy.ndarray`, optional
            Spectrum to integrate. If not provided, use unity.
        specWavelength : `numpy.ndarray`, optional
            Wavelength array for ``flux``. This is ignored if ``flux`` is ``None``.
        power : `int`
            :math:`p` in the integral (default: ``1``). This should usually be ``1``.
        """
        if specFlux is not None:
            x = specWavelength
            y = specFlux
            weight = self.interpolate(specWavelength) / specWavelength
        else:
            x = self.wavelength
            y = np.ones(shape=len(self.wavelength))
            weight = self.transmission / self.wavelength
        return _trapezoidal(x, y, weight, power=power)

    def integrate(self, spectrum=None, power=1):
        r"""Integrate the filter transmission curve for synthetic photometry

        The integral is:

        .. math:: \int S(\lambda) (F(\lambda) / \lambda d\lambda)^p

        where :math:`F(\lambda)` is the filter transmission curve,
        :math:`S(\lambda)` is the spectrum, and the extra `\lambda` term is due
        to using photon-counting detectors. :math:`p` (``power``) is usually 1,
        but it can be, say, 2 in the numerator of a variance formula.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsFiberArray`, optional
            Spectrum to integrate. If not provided, use unity.
        power : `int`
            :math:`p` in the integral (default: ``1``). This should usually be ``1``.
        """
        if spectrum is not None:
            return self._integrate(spectrum.flux, spectrum.wavelength, power=power)
        else:
            return self._integrate(None, None, power=power)

    @overload
    def photometer(self, spectrum: PfsSimpleSpectrum, doComputeError: Literal[False]) -> float:
        ...

    @overload
    def photometer(self, spectrum: PfsFiberArray, doComputeError: Literal[True]) -> tuple[float, float]:
        ...

    def photometer(self, spectrum, doComputeError=False):
        """Measure flux with this filter.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsSimpleSpectrum`
            Spectrum to integrate.
        doComputeError : `bool`
            Whether to compute an error bar (standard deviation).
            If ``doComputeError=True``, ``spectrum`` must be of
            `pfs.datamodel.PfsFiberArray` type.

        Returns
        -------
        flux : `float`
            Integrated flux.
        error : `float`
            Standard deviation of ``flux``. (Returned only if ``doComputeError=True``).
        """
        fluxNumer = self._integrate(spectrum.flux, spectrum.wavelength)
        fluxDenom = self._integrate(None, None)
        flux = fluxNumer / fluxDenom

        if doComputeError:
            varianceNumer = self._integrate(spectrum.covar[0, :], spectrum.wavelength, power=2)
            error = np.sqrt(varianceNumer) / fluxDenom
            return flux, error
        else:
            return flux


class FilterCurve(TransmissionCurve):
    """A filter transmission curve

    The bandpass is read from disk, and represents the transmission curve of
    the filter.

    Parameters
    ----------
    filterName : `str`
        Name of the filter. Must be one that is known.
    """

    filenames = {
        "g_hsc": "HSC/hsc_g_v2018.dat",
        # This is HSC-R (as opposed to HSC-R2)
        "r_old_hsc": "HSC/hsc_r_v2018.dat",
        "r2_hsc": "HSC/hsc_r2_v2018.dat",
        # This is HSC-I (as opposed to HSC-I2)
        "i_old_hsc": "HSC/hsc_i_v2018.dat",
        "i2_hsc": "HSC/hsc_i2_v2018.dat",
        "z_hsc": "HSC/hsc_z_v2018.dat",
        "y_hsc": "HSC/hsc_y_v2018.dat",
        "g_ps1": "PS1/PS1_g.dat",
        "r_ps1": "PS1/PS1_r.dat",
        "i_ps1": "PS1/PS1_i.dat",
        "z_ps1": "PS1/PS1_z.dat",
        "y_ps1": "PS1/PS1_y.dat",
        "bp_gaia": "Gaia/Gaia_Bp.txt",
        "rp_gaia": "Gaia/Gaia_Rp.txt",
        "g_gaia": "Gaia/Gaia_G.txt",
        "u_sdss": "SDSS/u_all_tel_atm13.dat",
        "g_sdss": "SDSS/g_all_tel_atm13.dat",
        "r_sdss": "SDSS/r_all_tel_atm13.dat",
        "i_sdss": "SDSS/i_all_tel_atm13.dat",
        "z_sdss": "SDSS/z_all_tel_atm13.dat",
        "nj_fake": "fake/fake_narrow_J.dat",
    }
    """Mapping of filter name to filename"""

    def __init__(self, filterName):
        if filterName not in self.filenames:
            raise RuntimeError(f"Unrecognised filter: {filterName}")

        filename = os.path.join(
            os.environ["OBS_PFS_DIR"], "pfs", "fluxCal", "bandpass", self.filenames[filterName]
        )
        data = np.genfromtxt(filename, dtype=[("wavelength", "f4"), ("flux", "f4")])
        wavelength = data["wavelength"]
        transmission = data["flux"]  # Relative transmission
        super().__init__(wavelength, transmission)


def _trapezoidal(x: np.ndarray, y: np.ndarray, weight: np.ndarray, power: float = 1) -> float:
    r"""Compute :math:`\int y(x) (w(x) dx)^p` with trapezoidal rule.

    :math:`p` is usually 1.

    :math:`p` can also be 2. If `y(x)` for each x is a stochastic variable,
    you can compute the statistical error of the integral,
    substituting 2 for :math:`p` and :math:`(\Delta y)^2` for :math:`y`.

    Other values of :math:`p` are accepted, but the return value will be
    nonsense mathematically.

    Parameters
    ----------
    x : `np.ndarray`
        Sampling points of :math:`x`. ``len(x)`` must be >= 3.
    y : `np.ndarray`
        Sampling points of :math:`y`.
    weight : `np.ndarray`
        Weight. This will be raised to the power of :math:`p` (``power``)
        unlike ``y``.
    power : `float`
        :math:`p`.

    Returns
    -------
    integral : float
        :math:`\int y(x) (w(x) dx)^p`
    """
    if power == 1:
        return 0.5 * (
            y[0] * weight[0] * (x[1] - x[0])
            + y[-1] * weight[-1] * (x[-1] - x[-2])
            + np.sum(y[1:-1] * weight[1:-1] * (x[2:] - x[:-2]))
        )
    else:
        return 0.5**power * (
            y[0] * (weight[0] * (x[1] - x[0])) ** power
            + y[-1] * (weight[-1] * (x[-1] - x[-2])) ** power
            + np.sum(y[1:-1] * (weight[1:-1] * (x[2:] - x[:-2])) ** power)
        )
