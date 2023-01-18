import lsstDebug
import lsst.daf.persistence
from lsst.pex.config import Config, ConfigurableField, ChoiceField, Field, ListField
from lsst.pipe.base import ArgumentParser, CmdLineTask, Struct
from lsst.utils import getPackageDir
from pfs.datamodel.identity import Identity
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.observations import Observations
from pfs.datamodel.pfsConfig import FiberStatus, PfsConfig, TargetType
from pfs.datamodel.pfsFiberArray import PfsFiberArray
from pfs.datamodel.pfsFluxReference import PfsFluxReference
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.drp.stella.fluxModelInterpolator import FluxModelInterpolator
from pfs.drp.stella import ReferenceLine, ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella import SpectrumSet
from pfs.drp.stella.datamodel import PfsFiberArraySet
from pfs.drp.stella.dustMap import DustMap
from pfs.drp.stella.estimateRadialVelocity import EstimateRadialVelocityTask
from pfs.drp.stella.extinctionCurve import F99ExtinctionCurve
from pfs.drp.stella.fitBroadbandSED import FitBroadbandSEDTask
from pfs.drp.stella.fitContinuum import FitContinuumTask
from pfs.drp.stella.fitReference import FilterCurve
from pfs.drp.stella.fluxModelSet import FluxModelSet
from pfs.drp.stella.interpolate import interpolateFlux
from pfs.drp.stella.lsf import GaussianLsf, Lsf, LsfDict, warpLsf
from pfs.drp.stella.utils import debugging

from astropy import constants as const
import numpy as np

import copy
import dataclasses

from typing import Literal, overload
from typing import Dict, List, Union
from typing import Generator, Sequence

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Sequence


__all__ = ["FitPfsFluxReferenceConfig", "FitPfsFluxReferenceTask"]


class FitPfsFluxReferenceConfig(Config):
    """Configuration for FitPfsFluxReferenceTask"""

    fitBroadbandSED = ConfigurableField(
        target=FitBroadbandSEDTask, doc="Get probabilities of SEDs from broadband photometries."
    )
    fitObsContinuum = ConfigurableField(
        target=FitContinuumTask, doc="Fit a model to observed spectrum's continuum"
    )
    fitModelContinuum = ConfigurableField(
        target=FitContinuumTask, doc="Fit a model to model spectrum's continuum"
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
        default=600.0,
        doc="min of the wavelength range in which observation spectra are compared to models.",
    )
    maxWavelength = Field(
        dtype=float,
        default=1200.0,
        doc="max of the wavelength range in which observation spectra are compared to models.",
    )
    ignoredRangesLeft = ListField(
        dtype=float,
        default=[685.0, 716.0, 759.0, 810.0, 895.0, 1100.0],
        doc="Left ends of wavelength ranges ignored (because e.g. of strong atmospheric absorption)"
        " when comparing observation spectra to models.",
    )
    ignoredRangesRight = ListField(
        dtype=float,
        default=[695.0, 735.0, 770.0, 835.0, 985.0, 1200.0],
        doc="Right ends of wavelength ranges ignored (because e.g. of strong atmospheric absorption)"
        " when comparing observation spectra to models.",
    )
    badMask = ListField(dtype=str, default=["BAD", "SAT", "CR", "NO_DATA"], doc="Mask planes for bad pixels")
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
    priorCutoff = Field(
        dtype=float,
        default=0.01,
        doc="Cut-off prior probability relative to max(prior)."
        " In comparing a flux model to an observed flux,"
        " the model is immediately rejected if its prior probability"
        " is less than `priorCutoff*max(prior)`.",
    )

    def setDefaults(self) -> None:
        super().setDefaults()

        self.estimateRadialVelocity.mask = ["BAD", "SAT", "CR", "NO_DATA", "EDGE", "ATMOSPHERE"]

        # Not sure these paramaters are good.
        self.fitObsContinuum.numKnots = 50
        self.fitObsContinuum.doMaskLines = True
        self.fitObsContinuum.maskLineRadius = 25
        self.fitModelContinuum.numKnots = 50
        self.fitModelContinuum.doMaskLines = True
        self.fitModelContinuum.maskLineRadius = 25


class FitPfsFluxReferenceTask(CmdLineTask):
    """Construct reference for flux calibration."""

    ConfigClass = FitPfsFluxReferenceConfig
    _DefaultName = "fitPfsFluxReference"

    fitBroadbandSED: FitBroadbandSEDTask
    fitObsContinuum: FitContinuumTask
    fitModelContinuum: FitContinuumTask
    estimateRadialVelocity: EstimateRadialVelocityTask

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.makeSubtask("fitBroadbandSED")
        self.makeSubtask("fitObsContinuum")
        self.makeSubtask("fitModelContinuum")
        self.makeSubtask("estimateRadialVelocity")

        self.debugInfo = lsstDebug.Info(__name__)

        self.fluxModelSet = FluxModelSet(getPackageDir("fluxmodeldata"))
        self.modelInterpolator = FluxModelInterpolator.fromFluxModelData(getPackageDir("fluxmodeldata"))
        self.extinctionMap = DustMap()

        self.fitFlagNames = MaskHelper()

    @classmethod
    def _makeArgumentParser(cls) -> ArgumentParser:
        """Make ArgumentParser"""
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsMerged", help="data IDs, e.g. --id exp=12345")
        return parser

    def _getMetadataName(self) -> None:
        return None

    def runDataRef(self, dataRef: lsst.daf.persistence.ButlerDataRef) -> None:
        """Run on an exposure

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
        """
        self.log.info("Processing %s", str(dataRef.dataId))

        merged = dataRef.get("pfsMerged")
        mergedLsf = dataRef.get("pfsMergedLsf")
        pfsConfig = dataRef.get("pfsConfig")
        reference = self.run(pfsConfig, merged, mergedLsf)
        dataRef.put(reference, "pfsFluxReference")

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
        pfsMergedLsf : `dict` (`int`: `pfs.drp.stella.Lsf`)
            Combined line-spread functions indexed by fiberId.

        Returns
        -------
        pfsFluxReference : `pfs.datamodel.pfsFluxReference.PfsFluxReference`
            Reference spectra for flux calibration
        """
        pfsConfig = pfsConfig.select(targetType=TargetType.FLUXSTD)
        originalFiberId = np.copy(pfsConfig.fiberId)
        fitFlag: Dict[int, int] = {}  # mapping fiberId -> flag indicating fit status

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

        # Apply the Galactic extinction correction to observed broad-band fluxes in pfsConfig
        pfsConfigCorr = self.correctExtinction(copy.deepcopy(pfsConfig))

        # Prior PDF from broad-band typing, where the continuous spectrum matters.
        bbPdfs = self.fitBroadbandSED.run(pfsConfigCorr)
        if self.debugInfo.doWritePrior:
            debugging.writeExtraData(
                f"fitPfsFluxReference-output/prior-{pfsMerged.filename}.pickle",
                fiberId=pfsConfigCorr.fiberId,
                prior=bbPdfs,
            )

        pfsConfig = selectPfsConfig(
            pfsConfig,
            "FITBBSED_FAILED",
            [(bbPdf is not None and np.all(np.isfinite(bbPdf))) for bbPdf in bbPdfs],
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
            pfsMerged.writeFits(f"fitPfsFluxReference-output/whitened-{pfsMerged.filename}")

        radialVelocities = self.getRadialVelocities(pfsConfig, pfsMerged, pfsMergedLsf, bbPdfs)

        if self.debugInfo.doWriteCrossCorr:
            debugging.writeExtraData(
                f"fitPfsFluxReference-output/crossCorr-{pfsMerged.filename}.pickle",
                fiberId=pfsConfigCorr.fiberId,
                crossCorr=[record.crossCorr for record in radialVelocities],
            )

        flag = self.fitFlagNames.add("ESTIMATERADIALVELOCITY_FAILED")
        for fiberId, velocity in zip(pfsConfig.fiberId, radialVelocities):
            if velocity is None or velocity.fail or not np.isfinite(velocity.velocity):
                fitFlag[fiberId] = flag

        # Likelihoods from spectral fitting, where line spectra matter.
        self.log.info("Fitting models to spectra (takes some time)...")
        likelihoods = self.fitModelsToSpectra(pfsConfig, pfsMerged, pfsMergedLsf, radialVelocities, bbPdfs)
        if self.debugInfo.doWriteLikelihood:
            debugging.writeExtraData(
                f"fitPfsFluxReference-output/likelihood-{pfsMerged.filename}.pickle",
                fiberId=pfsConfig.fiberId,
                likelihood=likelihoods,
            )

        flag = self.fitFlagNames.add("FITMODELS_FAILED")
        for fiberId, likelihood in zip(pfsConfig.fiberId, likelihoods):
            if likelihood is None or not np.all(np.isfinite(likelihood)):
                if fitFlag.get(fiberId, 0) == 0:
                    fitFlag[fiberId] = flag

        # Posterior PDF
        pdfs: List[Union[NDArray[np.float64], None]] = []
        for bbPdf, likelihood in zip(bbPdfs, likelihoods):
            if (bbPdf is None) or (likelihood is None):
                pdfs.append(None)
            else:
                pdf = bbPdf * likelihood
                pdf *= 1.0 / np.sum(pdf)
                pdfs.append(pdf)

        if self.debugInfo.doWritePosterior:
            debugging.writeExtraData(
                f"fitPfsFluxReference-output/posterior-{pfsMerged.filename}.pickle",
                fiberId=pfsConfig.fiberId,
                posterior=pdfs,
            )

        self.log.info("Making reference spectra by interpolation")
        bestModels = self.makeReferenceSpectra(pfsConfig, pdfs)

        flag = self.fitFlagNames.add("MAKEREFERENCESPECTRA_FAILED")
        for fiberId, bestModel in zip(pfsConfig.fiberId, bestModels):
            if bestModel is None:
                if fitFlag.get(fiberId, 0) == 0:
                    fitFlag[fiberId] = flag

        wavelength = bestModels[0].spectrum.wavelength
        flux = np.full(shape=(len(originalFiberId), len(wavelength)), fill_value=np.nan, dtype=np.float32)
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

        for fiberId, bestModel, velocity in zip(pfsConfig.fiberId, bestModels, radialVelocities):
            if fitFlag.get(fiberId, 0) == 0:
                index = fiberIdToIndex[fiberId]
                flux[index, :] = bestModel.spectrum.flux
                fitParams["teff"][index] = bestModel.param[0]
                fitParams["logg"][index] = bestModel.param[1]
                fitParams["m"][index] = bestModel.param[2]
                fitParams["alpha"][index] = bestModel.param[3]
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
        bbPdfs: Sequence[Union[NDArray[np.float64], None]],
    ) -> List[Union[Struct, None]]:
        """Estimate the radial velocity for each fiber in ``pfsMerged``.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
        pfsMerged : `pfs.drp.stella.datamodel.PfsFiberArraySet`
            Typically an instance of PfsMerged.
            It must have been whitened.
        pfsMergedLsf : `dict` (`int`: `pfs.drp.stella.Lsf`)
            Combined line-spread functions indexed by fiberId.
        bbPdfs : `List[Optional[numpy.array]]`
            `bbPdfs[i]`, if not None, is the probability distribution
            of `pfsConfig.fiberId[i]` being of each model type,
            determined by broad-band photometries.

        Returns
        -------
        radialVelocities : `List[Optional[lsst.pipe.base.Struct]]`
            Radial velocity for each fiber.
            Each element, if not None, has ``velocity``, ``error``,
            ``crossCorr``, and ``fail`` as its member.
            See ``EstimateRadialVelocityTask``.
        """
        # Find the best model from broad bands.
        # This model is used as the reference for cross-correlation calculation
        bestModels = self.findRoughlyBestModel(bbPdfs)

        radialVelocities: List[Union[Struct, None]] = []
        for iFiber, (spectrum, model) in enumerate(zip(fibers(pfsConfig, pfsMerged), bestModels)):
            if model.spectrum is None:
                radialVelocities.append(None)
                continue
            modelSpectrum = convolveLsf(
                model.spectrum, pfsMergedLsf[pfsConfig.fiberId[iFiber]], spectrum.wavelength
            )
            modelSpectrum = self.computeContinuum(modelSpectrum, mode="model").whiten(modelSpectrum)
            radialVelocities.append(self.estimateRadialVelocity.run(spectrum, modelSpectrum))

        return radialVelocities

    def computeContinuum(
        self, spectra: Union[PfsSimpleSpectrum, PfsFiberArraySet], *, mode: Literal["observed", "model"]
    ) -> "Continuum":
        """Whiten one or more spectra.

        Parameters
        ----------
        spectra : `PfsSimpleSpectrum` or `PfsFiberArraySet`
            spectra to whiten.
        mode : `str`
            "observed" or "model".
            Whether ``spectra`` is from observation or from simulation.

        Returns
        -------
        continuum : `Continuum`
            Fitted continuum.
        """
        if mode == "observed":
            fitContinuum = self.fitObsContinuum
        if mode == "model":
            fitContinuum = self.fitModelContinuum

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

        fiberIdToIndex = {fiberId: index for index, fiberId in enumerate(spectra.fiberId)}

        # Get the continuum for each fiber
        continuumList = fitContinuum.run(specset, lines=lines)

        continua = np.ones_like(spectra.flux)
        for continuum in continuumList:
            continua[fiberIdToIndex[continuum.fiberId], :] = continuum.flux

        absentIndex = np.array(
            [
                fiberIdToIndex[fiberId]
                for fiberId in set(spectra.fiberId) - set(continuum.fiberId for continuum in continuumList)
            ],
            dtype=int,
        )

        return Continuum(continua, absentIndex)

    def fitModelsToSpectra(
        self,
        pfsConfig: PfsConfig,
        obsSpectra: PfsFiberArraySet,
        pfsMergedLsf: LsfDict,
        radialVelocities: Sequence[Union[Struct, None]],
        priorPdfs: Sequence[Union[NDArray[np.float64], None]],
    ) -> List[Union[NDArray[np.float64], None]]:
        """For each observed spectrum,
        get probability of each model fitting to the spectrum.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
        obsSpectra : `PfsFiberArraySet`
            Continuum-subtracted observed spectra
        pfsMergedLsf : `dict` (`int`: `pfs.drp.stella.Lsf`)
            Combined line-spread functions indexed by fiberId.
        radialVelocities : `list` of `Optional[lsst.pipe.base.Struct]`
            Radial velocity for each fiber.
            Each element, if not None, has ``velocity``, ``error``, and ``fail``
            as its member. See ``EstimateRadialVelocityTask``.
        priorPdfs : `list` of `numpy.array` of `float`
            For each ``priorPdfs[iSpectrum]`` in ``priorPdfs``,
            ``priorPdfs[iSpectrum][iSED]`` is the prior probability of the SED ``iSED``
            matching the spectrum ``pfsConfig.fiberId[iSpectrum]``.
            ``priorPdfs[iSpectrum]`` can be ``None``,
            in which case the corresponding return value will be ``None``.

        Returns
        -------
        pdfs : `list` of `numpy.array` of `float`
            For each ``pdfs[iSpectrum]`` in ``pdfs``,
            ``pdfs[iSpectrum][iSED]`` is the likelihood
            (not multiplied by the prior) of the SED ``iSED``
            matching the spectrum ``pfsConfig.fiberId[iSpectrum]``.
            ``pdfs[iSpectrum]`` may be ``None``.
        """
        nFibers = len(priorPdfs)
        nModels = len(self.fluxModelSet.parameters)
        relativePriors = np.full(shape=(nModels, nFibers), fill_value=np.nan, dtype=float)
        for iFiber, pdf in enumerate(priorPdfs):
            if pdf is not None:
                relativePriors[:, iFiber] = pdf / np.max(pdf)

        # prepare an array of chi-squares.
        chisqs: List[Union[NDArray[np.float64], None]] = []
        for pdf in priorPdfs:
            if pdf is None:
                chisqs.append(None)
                continue
            chisqs.append(
                np.full(
                    shape=(len(self.fluxModelSet.parameters),),
                    fill_value=np.inf,
                    dtype=float,
                )
            )

        averageLsf = getAverageLsf([pfsMergedLsf[fiberId] for fiberId in pfsConfig.fiberId])

        for iModel, (param, priorPdf) in enumerate(zip(self.fluxModelSet.parameters, relativePriors)):
            model = self.fluxModelSet.getSpectrum(
                teff=param["teff"], logg=param["logg"], m=param["m"], alpha=param["alpha"]
            )
            # This one will be created afterward when it is actually required.
            modelContinuum: Union["Continuum", None] = None

            for iFiber, (obsSpectrum, velocity, prior) in enumerate(
                zip(fibers(pfsConfig, obsSpectra), radialVelocities, priorPdf)
            ):
                if velocity is None or velocity.fail or not np.isfinite(velocity.velocity):
                    continue
                if not (prior >= self.config.priorCutoff):
                    continue
                if modelContinuum is None:
                    convolvedModel = convolveLsf(model, averageLsf, obsSpectrum.wavelength)
                    modelContinuum = self.computeContinuum(convolvedModel, mode="model")

                convolvedModel = convolveLsf(
                    model, pfsMergedLsf[pfsConfig.fiberId[iFiber]], obsSpectrum.wavelength
                )
                convolvedModel = modelContinuum.whiten(convolvedModel)
                chisqs[iFiber][iModel] = calculateSpecChiSquare(
                    obsSpectrum, convolvedModel, velocity.velocity, self.getBadMask()
                )

        pdfs: List[Union[NDArray[np.float64], None]] = []
        for chisq in chisqs:
            if chisq is None:
                pdfs.append(None)
                continue
            chisq -= np.min(chisq)
            pdf = np.exp(chisq / (-2.0))
            pdf /= np.sum(pdf)
            pdfs.append(pdf)

        return pdfs

    def findRoughlyBestModel(
        self, pdfs: Sequence[Union[NDArray[np.float64], None]]
    ) -> List[Union[Struct, None]]:
        """Get the model spectrum corresponding to ``argmax(pdf)``
        for ``pdf`` in ``pdfs``.

        Parameters
        ----------
        pdfs : `list` of `numpy.array` of `float`
            For each ``pdfs[iSpectrum]`` in ``pdfs``,
            ``pdfs[iSpectrum][iSED]`` is the probability of the SED ``iSED``
            matching the spectrum ``iSpectrum``.
            ``pdfs[iSpectrum]`` can be ``None``.

        Returns
        -------
        models : `list` of `Optional[lsst.pipe.base.Struct]`
            The members of each element are:

            spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
                Spectrum.
            param : `tuple`
                Parameter (Teff, logg, M, alpha).
        """
        onePlusEpsilon = float(np.nextafter(np.float32(1), np.float32(2)))
        models: List[Union[Struct, None]] = []
        for pdf in pdfs:
            if pdf is None:
                models.append(Struct(spectrum=None, param=None))
                continue
            if np.max(pdf) <= np.min(pdf) * onePlusEpsilon:
                # If the PDF is uniform, we ourselves choose a parameter set
                # because this one is better than
                #     `self.fluxModelSet.parameters[np.argmax(pdf)]`
                # which is always `self.fluxModelSet.parameters[0]`.
                param = {"teff": 7500, "logg": 4.5, "m": 0.0, "alpha": 0.0}
                self.log.warn("findRoughlyBestModel: Probability distribution is uniform.")
            else:
                param = self.fluxModelSet.parameters[np.argmax(pdf)]
            model = self.fluxModelSet.getSpectrum(
                teff=param["teff"], logg=param["logg"], m=param["m"], alpha=param["alpha"]
            )
            models.append(
                Struct(spectrum=model, param=(param["teff"], param["logg"], param["m"], param["alpha"]))
            )

        return models

    def findBestModel(self, pdfs: Sequence[Union[NDArray[np.float64], None]]) -> List[Union[Struct, None]]:
        """Get the model spectrum corresponding to ``argmax(pdf)``
        for ``pdf`` in ``pdfs``. A smooth surface is fit to the ``pdf``,
        and the ``argmax`` here actually means the top of the surface.

        Parameters
        ----------
        pdfs : `list` of `numpy.array` of `float`
            For each ``pdfs[iSpectrum]`` in ``pdfs``,
            ``pdfs[iSpectrum][iSED]`` is the probability of the SED ``iSED``
            matching the spectrum ``iSpectrum``.
            ``pdfs[iSpectrum]`` can be ``None``.

        Returns
        -------
        models : `list` of `Optional[lsst.pipe.base.Struct]`
            The members of each element are:

            spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
                Spectrum.
            param : `tuple`
                Parameter (Teff, logg, M, alpha).
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

        models: List[Union[Struct, None]] = []
        for pdf in pdfs:
            if pdf is None:
                models.append(Struct(spectrum=None, param=None))
                continue

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

            spectrum = self.modelInterpolator.interpolate(*bestParam)
            models.append(Struct(spectrum=spectrum, param=bestParam))

        return models

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
        # Mask atmospheric absorption lines etc.
        wavelength = spectra.wavelength
        badMask = spectra.flags.add("ATMOSPHERE")

        for low, high in zip(self.config.ignoredRangesLeft, self.config.ignoredRangesRight):
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
        self, pfsConfig: PfsConfig, pdfs: Sequence[Union[NDArray[np.float64], None]]
    ) -> List[Union[Struct, None]]:
        """Get the model spectrum corresponding to ``argmax(pdf)``
        for ``pdf`` in ``pdfs``. (See ``self.findBestModel()``)

        This method is different from ``self.findBestModel()``
        in that the returned spectra are affected by galactic extinction
        and their flux values agree with ``pfsConfig.psfFlux``.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.pfsConfig.PfsConfig`
            Configuration of the PFS top-end.
        pdfs : `list` of `numpy.array` of `float`
            For each ``pdfs[iSpectrum]`` in ``pdfs``,
            ``pdfs[iSpectrum][iSED]`` is the probability of the SED ``iSED``
            matching the spectrum ``iSpectrum``.
            ``pdfs[iSpectrum]`` can be ``None``.

        Returns
        -------
        models : `list` of `Optional[lsst.pipe.base.Struct]`
            The members of each element are:

            spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
                Spectrum.
            param : `tuple`
                Parameter (Teff, logg, M, alpha).
            fluxScalingChi2 : `float`
                chi^2 of flux scaling problem.
            fluxScalingDof  : `int`
                Degree of freedom of flux scaling problem.
        """
        bestModels = self.findBestModel(pdfs)

        for model, fiberConfig in zip(bestModels, fiberConfigs(pfsConfig)):
            if model.spectrum is None:
                continue

            ebv = self.extinctionMap(fiberConfig.ra[0], fiberConfig.dec[0])
            extinction = F99ExtinctionCurve(self.config.Rv)
            model.spectrum.flux *= extinction.attenuation(model.spectrum.wavelength, ebv)

            scaled = adjustAbsoluteScale(model.spectrum, fiberConfig, self.config.broadbandFluxType)
            model.spectrum = scaled.spectrum
            model.fluxScalingChi2 = scaled.chi2
            model.fluxScalingDof = scaled.dof

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

    def getBadMask(self) -> List[str]:
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


@dataclasses.dataclass
class Continuum:
    """Continuous spectra.

    This class is the return type of
    ``FitPfsFluxReferenceTask.computeContinuum()``

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
    lsf = warpLsf(lsf, lsfWavelength, spectrum.wavelength)
    spectrum.flux = lsf.computeKernel((len(spectrum) - 1) / 2.0).convolve(spectrum.flux)
    return spectrum


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
    broadbandFluxType : `Literal["fiber", "psf", "total"]`
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
    obsSpectrum: PfsFiberArray, model: PfsSimpleSpectrum, radialVelocity: float, badMask: Sequence[str]
) -> float:
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
    badMask : `List[str]`
        Mask names.

    Returns
    -------
    chisq : `float`
        chi square.
    """
    beta = radialVelocity / const.c.to("km/s").value
    invDoppler = np.sqrt((1.0 - beta) / (1.0 + beta))

    good = 0 == (model.mask & model.flags.get(*(m for m in badMask if m in model.flags)))

    modelFlux = interpolateFlux(
        model.wavelength[good], model.flux[good], obsSpectrum.wavelength * invDoppler, jacobian=False
    )

    bad = 0 != (obsSpectrum.mask & obsSpectrum.flags.get(*(m for m in badMask if m in obsSpectrum.flags)))

    flux = np.copy(obsSpectrum.flux)
    # If not(any(isnan(flux))), we don't have to do this
    # (because invVar will be set to 0 for bad pixels.)
    # We set flux[bad] to a finite value to make sure we won't encounter nan.
    flux[bad] = 1.0

    invVar = 1.0 / obsSpectrum.covar[0, :]
    invVar[bad] = 0.0

    chisq = np.sum(np.square(flux - modelFlux) * invVar)

    return chisq


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

    spectrum = PfsFiberArray(
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
    spectrum.covar[0, :] = noise**2

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
    return PfsFiberArraySet(
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
        yield fiberArraySet.extractFiber(PfsFiberArray, pfsConfig, fiberId)


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
