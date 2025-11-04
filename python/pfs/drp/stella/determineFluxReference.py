import numpy as np

from lsst.pipe.base import Struct
from lsst.pex.config import ConfigurableField, Field

from pfs.datamodel.pfsConfig import FiberStatus, PfsConfig, TargetType
from pfs.datamodel.pfsFiberArray import PfsFiberArray
from pfs.datamodel.pfsFiberArraySet import PfsFiberArraySet
from pfs.datamodel.pfsFluxReference import PfsFluxReference

from .extinctionCurve import F99ExtinctionCurve
from .fitFluxReference import FitFluxReferenceTask, FitFluxReferenceConfig, ModelParam
from .fitFluxReference import removeBadFluxes, removeBadSpectra, fibers
from .fitFluxReference import convolveLsf, adjustAbsoluteScale
from .gaia import GaiaTask
from .lsf import LsfDict


class DetermineFluxReferenceConfig(FitFluxReferenceConfig):
    """Configuration for the DetermineFluxReferenceTask"""
    gaia = ConfigurableField(target=GaiaTask, doc="Task to access Gaia data")
    gaiaSearchRadius = Field(dtype=float, default=1.0, doc="Search radius for Gaia cone search, arcseconds")
    alphaDefault = Field(dtype=float, default=0.0, doc="Default value for alpha in the model parameters")


class DetermineFluxReferenceTask(FitFluxReferenceTask):
    """Quick and dirty determination of the flux reference"""

    ConfigClass = DetermineFluxReferenceConfig
    _DefaultName = "determineFluxReference"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("gaia")

    def run(
        self, pfsConfig: PfsConfig, pfsMerged: PfsFiberArraySet, pfsMergedLsf: LsfDict
    ) -> PfsFluxReference:
        """Perform a quick and dirty determination of the flux reference.

        We use Gaia to find the best model for each ``FLUXSTD`` fiber,
        and use that to measure the radial velocity. The Gaia model is then
        scaled to match the broadband flux of the fiber in the pfsConfig.

        Parameters
        ----------
        pfsConfig : `PfsConfig`
            PFS fiber configuration, containing only the fibers of interest.
        pfsMerged : `PfsFiberArraySet`
            Typically an instance of `PfsMerged`.
        pfsMergedLsf : `LsfDict`
            Combined line-spread functions indexed by fiberId.

        Returns
        -------
        pfsFluxReference : `PfsFluxReference`
            The flux reference, containing the flux for each ``FLUXSTD`` fiber.
        """

        self.log.info("Running %s", self.__class__.__name__)
        pfsConfig = pfsConfig.select(targetType=TargetType.FLUXSTD, fiberStatus=FiberStatus.GOOD)
        fiberId = pfsConfig.fiberId.copy()
        num = len(fiberId)
        fitFlagArray = np.zeros(shape=num, dtype=np.int32)

        removeBadFluxes(pfsConfig, self.config.broadbandFluxType, self.config.fabricatedBroadbandFluxErrSNR)
        self.log.info("Number of FLUXSTD: %d", len(pfsConfig))
        fitFlagArray[~np.isin(fiberId, pfsMerged.fiberId)] |= self.fitFlagNames.add("NO_MERGED_SPECTRUM")

        pfsMerged = removeBadSpectra(
            pfsMerged.select(fiberId=pfsConfig.fiberId),
            self.config.cutoffSNR,
            (self.config.cutoffSNRRangeLeft, self.config.cutoffSNRRangeRight),
        )
        isBadSnr = (fitFlagArray == 0) & ~np.isin(fiberId, pfsMerged.fiberId)
        fitFlagArray[isBadSnr] |= self.fitFlagNames.add("BAD_SNR")
        pfsConfig = pfsConfig.select(fiberId=pfsMerged.fiberId)
        assert np.array_equal(pfsConfig.fiberId, pfsMerged.fiberId)

        models = self.findRoughlyBestModel(pfsConfig)
        pfsMerged = pfsMerged.select(fiberId=list(models.keys()))
        self.log.info("Number of observed FLUXSTD with models: %d", len(pfsMerged))
        if len(pfsMerged) == 0:
            raise RuntimeError("No observed FLUXSTD can be fitted a model to.")

        pfsMerged /= pfsMerged.norm
        pfsMerged.norm[...] = 1.0

        pfsMerged = self.computeContinuum(pfsMerged, mode="observed").whiten(pfsMerged)
        pfsMerged = self.maskUninterestingRegions(pfsMerged)
        pfsConfig = pfsConfig.select(fiberId=pfsMerged.fiberId)

        if self.debugInfo.doWriteWhitenedFlux:
            pfsMerged.writeFits(f"fitFluxReference-output/whitened-{pfsMerged.filename}")

        radialVelocities = self.getRadialVelocities(pfsConfig, pfsMerged, pfsMergedLsf, models)
        badModel = self.fitFlagNames.add("BAD_MODEL")
        badRadialVelocity = self.fitFlagNames.add("BAD_RADIAL_VELOCITY")

        wavelength = None
        for mm in models.values():
            wl = mm.spectrum.wavelength
            if wavelength is None:
                wavelength = wl.copy()
            else:
                assert np.all(wavelength == wl)
        if wavelength is None:
            raise RuntimeError("No good models found; cannot determine flux reference.")

        # Prepare the output
        flux = np.zeros(shape=(num, wavelength.size), dtype=np.float32)
        fitParams = np.full(
            num,
            np.nan,
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
        for ii, ff in enumerate(fiberId):
            mm = models.get(ff, None)
            if mm is None:
                fitFlagArray[ii] |= badModel
                continue
            fitParams[ii]["teff"] = mm.param.teff
            fitParams[ii]["logg"] = mm.param.logg
            fitParams[ii]["m"] = mm.param.m
            fitParams[ii]["alpha"] = mm.param.alpha

            rv = radialVelocities.get(ff, None)
            if rv is None:
                fitFlagArray[ii] |= badRadialVelocity
                continue
            fitParams[ii]["radial_velocity"] = rv.velocity
            fitParams[ii]["radial_velocity_err"] = rv.error

            scaled = adjustAbsoluteScale(
                mm.spectrum, pfsConfig.select(fiberId=ff), self.config.broadbandFluxType
            )
            fitParams[ii]["flux_scaling_chi2"] = scaled.chi2
            fitParams[ii]["flux_scaling_dof"] = scaled.dof

            flux[ii, :] = scaled.spectrum.flux

        return PfsFluxReference(
            identity=pfsMerged.identity,
            fiberId=fiberId,
            wavelength=wavelength,
            flux=flux,
            metadata={},
            fitFlag=fitFlagArray,
            fitFlagNames=self.fitFlagNames,
            fitParams=fitParams,
        )

    def findRoughlyBestModel(self, pfsConfig: PfsConfig) -> dict[int, Struct]:  # type: ignore[override]
        """Find a roughly-best model for each fiberId in the PfsConfig.

        Parameters
        ----------
        pfsConfig : `PfsConfig`
            PFS fiber configuration, containing only the fibers of interest.

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
        models: dict[int, Struct] = {}
        for ii in range(len(pfsConfig)):
            gaia = self.gaia.run(pfsConfig.ra[ii], pfsConfig.dec[ii], radius=self.config.gaiaSearchRadius)
            if gaia is None:
                continue
            param = ModelParam(
                teff=gaia.teff_gspphot,
                logg=gaia.logg_gspphot,
                m=gaia.mh_gspphot,
                alpha=self.config.alphaDefault,
            )

            # Gaia gives us the extinction, A_0, at 541.4 nm for an assumed Fitzpatrick (1999) extinction
            # curve. This is approximately the A_V extinction: it is not exactly the same for wide
            # bandpasses and different source spectra, but for the V-band, the difference is "justifiably
            # neglected" (Gaia Data Release 3, 2023, A&A, 674, A26, section 4.2).
            # The conversion from A_V to E(B-V) is to divide by R_V = 3.1 (assuming the Fitzpatrick curve).
            ebv = gaia.azero_gspphot / self.config.Rv

            self.log.debug(
                "Found Gaia source for fiberId %d: source_id=%d, %s, E(B-V)=%f",
                pfsConfig.fiberId[ii],
                int(gaia.source_id),
                param,
                ebv,
            )
            spectrum = self.getModelSpectrum(param, ebv)
            models[pfsConfig.fiberId[ii]] = Struct(spectrum=spectrum, param=param)
        return models

    def getModelSpectrum(self, param: ModelParam, ebv: float) -> PfsFiberArray:
        """Get the model spectrum for the given parameters

        Parameters
        ----------
        param : `ModelParam`
            Model parameters.
        ebv : `float`
            E(B-V) extinction value.

        Returns
        -------
        spectrum : `PfsFiberArray`
            The model spectrum for the given parameters, with extinction applied.
        """
        spectrum = self.modelInterpolator.interpolate(**param.toDict())
        extinction = F99ExtinctionCurve(self.config.Rv)
        spectrum.flux *= extinction.attenuation(spectrum.wavelength, ebv)
        return spectrum

    def getRadialVelocities(  # type: ignore[override]
        self,
        pfsConfig: PfsConfig,
        pfsMerged: PfsFiberArraySet,
        pfsMergedLsf: LsfDict,
        models: dict[int, Struct],
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
        models : `dict` [`int`, `lsst.pipe.base.Struct`]
            Mapping from ``fiberId`` to a structure whose members are:

              - spectrum : `pfs.drp.stella.datamodel.pfsFiberArray.PfsSimpleSpectrum`
                    Spectrum of the model.
              - param : `ModelParam`
                    Parameters of the model.

        Returns
        -------
        radialVelocities : `dict [`int`, `lsst.pipe.base.Struct`]
            Mapping from ``fiberId`` to radial velocity.
            Each value has ``velocity``, ``error``,
            ``crossCorr``, and ``fail`` as its member.
            See ``EstimateRadialVelocityTask``.
        """
        radialVelocities: dict[int, Struct] = {}
        for iFiber, fiberSpectrum in enumerate(fibers(pfsConfig, pfsMerged)):
            fiberId = pfsConfig.fiberId[iFiber]
            modelSpectrum = models[fiberId].spectrum
            modelSpectrum = convolveLsf(modelSpectrum, pfsMergedLsf[fiberId], fiberSpectrum.wavelength)
            modelSpectrum = self.computeContinuum(modelSpectrum, mode="model").whiten(modelSpectrum)
            radialVelocities[fiberId] = self.estimateRadialVelocity.run(fiberSpectrum, modelSpectrum)
        return radialVelocities
