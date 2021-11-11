from collections import defaultdict
import os.path
import re
import numpy as np
from lsst.pex.config import Config, Field, ConfigurableField, ListField, ConfigField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, Struct

from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.wavelengthArray import WavelengthArray
from .datamodel import PfsConfig, PfsArm, PfsMerged
from pfs.datamodel import Identity, TargetType
from .fitFocalPlane import FitBlockedOversampledSplineTask
from .focalPlaneFunction import FocalPlaneFunction
from .utils import getPfsVersions
from .lsf import warpLsf, coaddLsf
from .SpectrumContinued import Spectrum
from .datamodel.interpolate import interpolateFlux, interpolateMask
from .fitContinuum import FitContinuumTask
from .subtractSky1d import subtractSky1d


class WavelengthSamplingConfig(Config):
    """Configuration for wavelength sampling"""
    minWavelength = Field(dtype=float, default=350, doc="Minimum wavelength (nm)")
    maxWavelength = Field(dtype=float, default=1270, doc="Maximum wavelength (nm)")
    length = Field(dtype=int, default=11376, doc="Length of wavelength array (sets the resolution)")

    @property
    def dWavelength(self):
        """Return the wavelength spacing (nm)"""
        return (self.maxWavelength - self.minWavelength)/(self.length - 1)

    @property
    def wavelength(self):
        """Return the appropriate wavelength vector"""
        return WavelengthArray(self.minWavelength, self.maxWavelength, self.length)


class MergeArmsConfig(Config):
    """Configuration for MergeArmsTask"""
    wavelength = ConfigField(dtype=WavelengthSamplingConfig, doc="Wavelength configuration")
    doSubtractSky1d = Field(dtype=bool, default=True, doc="Do 1D sky subtraction?")
    fitSkyModel = ConfigurableField(target=FitBlockedOversampledSplineTask,
                                    doc="Fit sky model over the focal plane")
    doBarycentricCorr = Field(dtype=bool, default=True, doc="Do barycentric correction?")
    mask = ListField(dtype=str, default=["NO_DATA", "CR", "INTRP", "SAT", "BAD_FLAT"],
                     doc="Mask values to reject when combining")
    pfsConfigFile = Field(dtype=str, default="", doc="""Full pathname of pfsCalib file to use.
    If of the form "pfsConfig-0x%x-%d.fits", the pfsDesignId and visit0 will be deduced from the filename;
    if not, the values 0x666 and 0 are used.""")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum to mean normalisation")

    def setDefaults(self):
        super().setDefaults()
        # Scale back rejection because otherwise everything gets rejected
        self.fitSkyModel.rejIterations = 1
        self.fitSkyModel.rejThreshold = 10.0


class MergeArmsRunner(TaskRunner):
    """Runner for MergeArmsTask"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """Produce list of targets for MergeArmsTask

        We want to operate on all data within a single exposure at once.
        """
        exposures = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.id.refList:
            visit = ref.dataId["visit"]
            spectrograph = ref.dataId["spectrograph"]
            exposures[visit][spectrograph].append(ref)
        return [(list(specs.values()), kwargs) for specs in exposures.values()]


class MergeArmsTask(CmdLineTask):
    """Merge all extracted spectra from a single exposure"""
    _DefaultName = "mergeArms"
    ConfigClass = MergeArmsConfig
    RunnerClass = MergeArmsRunner

    @classmethod
    def _makeArgumentParser(cls):
        """Make an ArgumentParser"""
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsArm",
                               help="data IDs, e.g. --id exp=12345 spectrograph=1..3")
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitSkyModel")
        self.makeSubtask("fitContinuum")

    def run(self, spectra, pfsConfig, lsfList):
        """Merge all extracted spectra from a single exposure

        Parameters
        ----------
        spectra : iterable of iterable of `pfs.datamodel.PsfArm`
            Extracted spectra from the different arms, for each spectrograph.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, fiber targets.
        lsfList : iterable of iterable of `pfs.drp.stella.Lsf`
            Line-spread functions from the different arms, for each
            spectrograph.

        Returns
        -------
        spectra : `pfs.datamodel.PfsMerged`
            Merged spectra.
        lsf : `pfs.drp.stella.Lsf`
            Merged line-spread function.
        sky1d : `list` of `pfs.drp.stella.FocalPlaneFunction`
            Sky models for each arm.
        """
        for spec, lsf in zip(spectra, lsfList):
            for armSpec, armLsf in zip(spec, lsf):
                if set(armSpec.fiberId) != set(armLsf):
                    msg = "Set of fiberIds of LSFs != fiberIds for spectra: "
                    onlyFiberId = set(armSpec.fiberId) - set(armLsf)
                    onlyLsf = set(armLsf) - set(armSpec.fiberId)
                    if onlyFiberId:
                        msg += f" Only in fiberId: {onlyFiberId} (fixing)"

                        for fid in onlyFiberId:
                            armLsf[fid] = None
                    if onlyLsf:
                        msg += f" Only in armPsf: {onlyLsf}"
                    self.log.warn(msg)

        norm = self.normalizeSpectra(sum(spectra, []))

        sky1d = []
        if self.config.doSubtractSky1d:
            # Do sky subtraction arm by arm for now; alternatives involve changing the run() API
            for ss in sum(spectra, []):
                sky1d.append(self.skySubtraction(ss, pfsConfig))

        spectrographs = [self.mergeSpectra(ss, norm) for ss in spectra]  # Merge in wavelength
        merged = PfsMerged.fromMerge(spectrographs, metadata=getPfsVersions())  # Merge across spectrographs

        lsfList = [self.mergeLsfs(ll, ss) for ll, ss in zip(lsfList, spectra)]
        mergedLsf = self.combineLsfs(lsfList)

        return Struct(spectra=merged, lsf=mergedLsf, sky1d=sky1d)

    def runDataRef(self, expSpecRefList):
        """Merge all extracted spectra from a single exposure

        Parameters
        ----------
        expSpecRefList : iterable of iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for each sensor, grouped by spectrograph.

        Returns
        -------
        spectra : `pfs.datamodel.PfsMerged`
            Merged spectra.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, fiber targets.
        lsf : `pfs.drp.stella.Lsf`
            Merged line-spread function.
        sky1d : `pfs.drp.stella.FocalPlaneFunction`
            1D sky model.
        """
        spectra = [[dataRef.get("pfsArm") for dataRef in specRefList] for
                   specRefList in expSpecRefList]
        lsfList = [[dataRef.get("pfsArmLsf") for dataRef in specRefList] for specRefList in expSpecRefList]
        if self.config.pfsConfigFile:
            try:
                pfsDesignId, visit0 = re.split(r"[-.]", os.path.split(self.config.pfsConfigFile)[1])[1:-1]

                pfsDesignId = int(pfsDesignId, 16)
                visit0 = int(visit0)
            except ValueError:
                pfsDesignId = 666
                visit0 = 0

            self.log.info("Reading pfsConfig for pfsDesignId=0x%x, visit0=%d", pfsDesignId, visit0)
            pfsConfig = PfsConfig._readImpl(self.config.pfsConfigFile,
                                            pfsDesignId=pfsDesignId, visit0=visit0)
        else:
            pfsConfig = expSpecRefList[0][0].get("pfsConfig")

        results = self.run(spectra, pfsConfig, lsfList)

        expSpecRefList[0][0].put(results.spectra, "pfsMerged")
        expSpecRefList[0][0].put(results.lsf, "pfsMergedLsf")
        if results.sky1d is not None:
            for sky1d, ref in zip(results.sky1d, expSpecRefList[0]):
                ref.put(sky1d, "sky1d")

        results.pfsConfig = pfsConfig
        return results

    def normalizeSpectra(self, spectra):
        """Calculate and apply a suitable target normalisation

        We want the merged spectra to have normalisations something close to
        counts, and to approximate the fluxes from the image (this isn't
        possible in the dichroic overlap, but is fairly well defined everywhere
        else). We do this by resampling the input normalisations to a common
        wavelength scale and taking a straight mean (without strong rejection,
        so the very different values around the dichroic won't cause problems).
        That will have some sharp features (due to, e.g., CRs that we've missed,
        or even real spectral features like absorption bands), but we'll smooth
        over that by fitting a continuum model. The continuum model will form
        the basis for our target normalisation: we'll resample back to the
        original wavelength frames and apply.

        Parameters
        ----------
        spectra : iterable of `pfs.datamodel.PsfArm`
            Extracted spectra from the different arms, for each spectrograph.
            The spectra will be modified in-place.

        Returns
        -------
        norm : `numpy.ndarray` of `float`
            Adopted normalisation.
        """
        wavelength = self.config.wavelength.wavelength
        resampled = []
        for ss in spectra:
            badBitmask = ss.flags.get(*self.config.mask)
            for ii in range(len(ss)):
                norm = interpolateFlux(ss.wavelength[ii], ss.norm[ii], wavelength, fill=np.nan)
                mask = interpolateMask(ss.wavelength[ii], ss.mask[ii], wavelength, fill=badBitmask)
                ignore = ((mask & badBitmask) != 0) | ~np.isfinite(norm)
                if np.all(ignore):
                    continue
                resampled.append(np.ma.masked_where(ignore, norm))

        mean = np.ma.mean(resampled, axis=0)
        spectrum = Spectrum(wavelength.size)
        spectrum.flux = mean.data
        spectrum.mask.array[0] = np.where(mean.mask, 0xFF, 0)
        continuum = self.fitContinuum.fitContinuum(spectrum)

        for ss in spectra:
            norm = interpolateFlux(wavelength, continuum, ss.wavelength)
            with np.errstate(invalid="ignore", divide="ignore"):
                nn = norm/ss.norm
                ss *= nn
                ss.mask[nn == 0] |= ss.flags["NO_DATA"]

        return continuum

    def mergeSpectra(self, spectraList, norm):
        """Combine all spectra from the same exposure

        All spectra should have the same fibers, so we simply iterate over the
        fibers, combining each spectrum from that fiber.

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsFiberArraySet`
            List of spectra to coadd.
        norm : `numpy.ndarray` of `float`
            Adopted normalisation.

        Returns
        -------
        result : `pfs.datamodel.PfsMerged`
            Merged spectra.
        """
        archetype = spectraList[0]
        identity = Identity.fromMerge([ss.identity for ss in spectraList])
        fiberId = archetype.fiberId
        if any(np.any(ss.fiberId != fiberId) for ss in spectraList):
            raise RuntimeError("Selection of fibers differs")
        wavelength = self.config.wavelength.wavelength
        resampled = [ss.resample(wavelength) for ss in spectraList]
        flags = MaskHelper.fromMerge([ss.flags for ss in spectraList])
        combination = self.combine(resampled, flags)
        if self.config.doBarycentricCorr:
            self.log.warn("Barycentric correction is not yet implemented.")

        return PfsMerged(identity, fiberId, combination.wavelength, combination.flux, combination.mask,
                         combination.sky, norm, combination.covar, flags, archetype.metadata)

    def combine(self, spectra, flags):
        """Combine spectra

        Parameters
        ----------
        spectra : iterable of `pfs.datamodel.PfsFiberArraySet`
            List of spectra to combine. These should already have been
            resampled to a common wavelength representation.
        flags : `pfs.datamodel.MaskHelper`
            Mask interpreter, for identifying bad pixels.
        norm : `numpy.ndarray` of `float`
            Adopted normalisation.

        Returns
        -------
        wavelength : `numpy.ndarray` of `float`
            Wavelengths for combined spectrum.
        flux : `numpy.ndarray` of `float`
            Flux measurements for combined spectrum.
        sky : `numpy.ndarray` of `float`
            Sky measurements for combined spectrum.
        covar : `numpy.ndarray` of `float`
            Covariance matrix for combined spectrum.
        mask : `numpy.ndarray` of `int`
            Mask for combined spectrum.
        """
        archetype = spectra[0]
        mask = np.zeros_like(archetype.mask)
        flux = np.zeros_like(archetype.flux)
        sky = np.zeros_like(archetype.sky)
        covar = np.zeros_like(archetype.covar)
        sumWeights = np.zeros_like(archetype.flux)

        for ss in spectra:
            with np.errstate(invalid="ignore"):
                good = ((ss.mask & ss.flags.get(*self.config.mask)) == 0) & (ss.covar[:, 0] > 0)
            weight = np.zeros_like(ss.flux)
            weight[good] = 1.0/ss.covar[:, 0][good]
            with np.errstate(invalid="ignore"):
                flux += ss.flux*weight
                sky += ss.sky*weight
            mask[good] |= ss.mask[good]
            sumWeights += weight

        good = sumWeights > 0
        flux[good] /= sumWeights[good]
        sky[good] /= sumWeights[good]
        covar[:, 0][good] = 1.0/sumWeights[good]
        covar[:, 0][~good] = np.inf
        covar[:, 1:2] = np.where(good, 0.0, np.inf)[:, np.newaxis]
        for ss in spectra:
            mask[~good] |= ss.mask[~good]
        mask[~good] |= flags["NO_DATA"]
        covar2 = np.zeros((1, 1), dtype=archetype.covar.dtype)
        return Struct(wavelength=archetype.wavelength, flux=flux, sky=sky, covar=covar,
                      mask=mask, covar2=covar2)

    def mergeLsfs(self, lsfList, spectraList):
        """Merge LSFs for different arms within a spectrograph

        Parameters
        ----------
        lsfList : iterable of `dict` (`int`: `pfs.drp.stella.Lsf`)
            Line-spread functions indexed by fiberId, for each arm.
        spectraList : iterable of `pfs.datamodel.PfsFiberArraySet`
            Spectra for each arm.

        Returns
        -------
        lsf : `dict` (`int`: `pfs.drp.stella.Lsf`)
            Merged line-spread functions indexed by fiberId.
        """
        fiberId = set(lsfList[0].keys())
        for lsf in lsfList:
            assert set(lsf.keys()) == fiberId
        wavelength = self.config.wavelength.wavelength
        warpedLsfList = []
        for lsf, spectra in zip(lsfList, spectraList):
            warpedLsf = {}
            for ii in range(len(spectra)):
                ff = spectra.fiberId[ii]
                warpedLsf[ff] = warpLsf(lsf.get(ff), spectra.wavelength[ii], wavelength)

            warpedLsfList.append(warpedLsf)

        return {ff: coaddLsf([ww[ff] for ww in warpedLsfList]) for ff in fiberId}

    def combineLsfs(self, lsfList):
        """Combine LSFs for different spectrographs

        The spectrographs have different fiberId values, so this is simply a
        matter of stuffing everything into a single object.

        Parameters
        ----------
        lsfList : iterable of `dict` (`int`: `pfs.drp.stella.Lsf`)
            Line-spread functions indexed by fiberId, for each spectrograph.

        Returns
        -------
        lsf : ``dict` (`int`: `pfs.drp.stella.Lsf`)
            Combined line-spread functions indexed by fiberId.
        """
        lsf = {}
        for ll in lsfList:
            lsf.update(ll)
        return lsf

    def skySubtraction(self, spectra: PfsArm, pfsConfig: PfsConfig) -> FocalPlaneFunction:
        """Fit and subtract sky model

        Parameters
        ----------
        spectra : `PfsArm`
            Spectra to which to fit and subtract a sky model.
        pfsConfig : `PfsConfig`
            Top-end configuration.

        Returns
        -------
        sky1d : `FocalPlaneFunction`
            Sky model.
        """
        skySpectra = spectra.select(pfsConfig, targetType=TargetType.SKY)
        skyConfig = pfsConfig.select(fiberId=skySpectra.fiberId)
        if len(skySpectra) == 0:
            raise RuntimeError("No sky spectra to use for sky subtraction")

        sky1d = self.fitSkyModel.run(skySpectra, skyConfig)
        subtractSky1d(spectra, pfsConfig, sky1d)
        return sky1d

    def _getMetadataName(self):
        return None
