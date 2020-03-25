from collections import defaultdict
import numpy as np
from lsst.pex.config import Config, Field, ConfigurableField, ListField, ConfigField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, Struct

from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.wavelengthArray import WavelengthArray
from .datamodel import PfsMerged
from pfs.datamodel import Identity
from .subtractSky1d import SubtractSky1dTask
from .utils import getPfsVersions


class WavelengthSamplingConfig(Config):
    """Configuration for wavelength sampling"""
    minWavelength = Field(dtype=float, default=350, doc="Minimum wavelength (nm)")
    maxWavelength = Field(dtype=float, default=1260, doc="Maximum wavelength (nm)")
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
    subtractSky1d = ConfigurableField(target=SubtractSky1dTask, doc="1d sky subtraction")
    doBarycentricCorr = Field(dtype=bool, default=True, doc="Do barycentric correction?")
    mask = ListField(dtype=str, default=["NO_DATA", "CR"], doc="Mask values to reject when combining")


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
        self.makeSubtask("subtractSky1d")

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
        """
        spectra = [[dataRef.get("pfsArm") for dataRef in specRefList] for
                   specRefList in expSpecRefList]
        # XXX fix when we have LSF implemented
        lsf = [[None for dataRef in specRefList] for specRefList in expSpecRefList]
        pfsConfig = expSpecRefList[0][0].get("pfsConfig")
        if self.config.doSubtractSky1d:
            sky1d = self.subtractSky1d.run(sum(spectra, []), pfsConfig, sum(lsf, []))
            expSpecRefList[0][0].put(sky1d, "sky1d")

        spectrographs = [self.mergeSpectra(ss) for ss in spectra]  # Merge in wavelength
        merged = PfsMerged.fromMerge(spectrographs, metadata=getPfsVersions())  # Merge across spectrographs
        expSpecRefList[0][0].put(merged, "pfsMerged")
        return Struct(spectra=merged, pfsConfig=pfsConfig)

    def mergeSpectra(self, spectraList):
        """Combine all spectra from the same exposure

        All spectra should have the same fibers, so we simply iterate over the
        fibers, combining each spectrum from that fiber.

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsFiberArraySet`
            List of spectra to coadd.

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
                         combination.sky, combination.covar, flags, archetype.metadata)

    def combine(self, spectra, flags):
        """Combine spectra

        Parameters
        ----------
        spectra : iterable of `pfs.datamodel.PfsFiberArraySet`
            List of spectra to combine. These should already have been
            resampled to a common wavelength representation.
        flags : `pfs.datamodel.MaskHelper`
            Mask interpreter, for identifying bad pixels.

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
        mask[~good] = flags["NO_DATA"]
        covar2 = np.zeros((1, 1), dtype=archetype.covar.dtype)
        return Struct(wavelength=archetype.wavelength, flux=flux, sky=sky, covar=covar,
                      mask=mask, covar2=covar2)

    def _getMetadataName(self):
        return None
