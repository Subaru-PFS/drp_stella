from collections import defaultdict
import os.path
import re
import numpy as np

import lsstDebug
from lsst.pex.config import Config, Field, ConfigurableField, ListField, ConfigField
from lsst.pipe.base import CmdLineTask, ArgumentParser, TaskRunner, Struct

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.wavelengthArray import WavelengthArray
from pfs.drp.stella.gen3 import DatasetRefList, zipDatasetRefs
from pfs.drp.stella.selectFibers import SelectFibersTask
from .datamodel import PfsConfig, PfsArm, PfsMerged
from pfs.datamodel import Identity
from .fitFocalPlane import FitBlockedOversampledSplineTask
from .focalPlaneFunction import FocalPlaneFunction
from .utils import getPfsVersions
from .lsf import LsfDict, warpLsf, coaddLsf
from .SpectrumContinued import Spectrum
from .interpolate import calculateDispersion, interpolateFlux, interpolateMask
from .fitContinuum import BaseFitContinuumTask, FitSplineContinuumTask
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


class MergeArmsConnections(PipelineTaskConnections, dimensions=("instrument", "exposure")):
    """Connections for MergeArmsTask"""
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )
    lsf = InputConnection(
        name="pfsArmLsf",
        doc="1D line-spread function",
        storageClass="LsfDict",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )

    pfsMerged = OutputConnection(
        name="pfsMerged",
        doc="Merged spectra from an exposure",
        storageClass="PfsMerged",
        dimensions=("instrument", "exposure"),
    )
    pfsMergedLsf = OutputConnection(
        name="pfsMergedLsf",
        doc="Line-spread function of merged spectra from an exposure",
        storageClass="LsfDict",
        dimensions=("instrument", "exposure"),
    )
    sky1d = OutputConnection(
        name="sky1d",
        doc="1d sky model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )


class MergeArmsConfig(PipelineTaskConfig, pipelineConnections=MergeArmsConnections):
    """Configuration for MergeArmsTask"""
    wavelength = ConfigField(dtype=WavelengthSamplingConfig, doc="Wavelength configuration")
    doSubtractSky1d = Field(dtype=bool, default=True, doc="Do 1D sky subtraction?")
    selectSky = ConfigurableField(target=SelectFibersTask, doc="Select fibers for 1d sky subtraction")
    fitSkyModel = ConfigurableField(target=FitBlockedOversampledSplineTask,
                                    doc="Fit sky model over the focal plane")
    doBarycentricCorr = Field(dtype=bool, default=True, doc="Do barycentric correction?")
    mask = ListField(dtype=str, default=["NO_DATA", "CR", "INTRP", "SAT", "BAD_FLAT"],
                     doc="Mask values to reject when combining")
    pfsConfigFile = Field(dtype=str, default="", doc="""Full pathname of pfsCalib file to use.
    If of the form "pfsConfig-0x%x-%d.fits", the pfsDesignId and visit0 will be deduced from the filename;
    if not, the values 0x666 and 0 are used.""")
    fitContinuum = ConfigurableField(target=FitSplineContinuumTask, doc="Fit continuum to mean normalisation")

    def setDefaults(self):
        super().setDefaults()
        self.selectSky.targetType = ("SKY", "SUNSS_IMAGING")
        # Scale back rejection because otherwise everything gets rejected
        self.fitSkyModel.rejIterations = 1
        self.fitSkyModel.rejThreshold = 10.0
        self.fitContinuum.numKnots = 100  # Increased number of knots because larger wavelength range
        self.fitContinuum.iterations = 0  # No rejection: normalisation doesn't need to be exact, just robust


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


class MergeArmsTask(CmdLineTask, PipelineTask):
    """Merge all extracted spectra from a single exposure"""
    _DefaultName = "mergeArms"
    ConfigClass = MergeArmsConfig
    RunnerClass = MergeArmsRunner

    selectSky: SelectFibersTask
    fitSkyModel: FitBlockedOversampledSplineTask
    fitContinuum: BaseFitContinuumTask

    @classmethod
    def _makeArgumentParser(cls):
        """Make an ArgumentParser"""
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsArm",
                               help="data IDs, e.g. --id exp=12345 spectrograph=1..3")
        return parser

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("selectSky")
        self.makeSubtask("fitSkyModel")
        self.makeSubtask("fitContinuum")
        self.debugInfo = lsstDebug.Info(__name__)

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
        allSpectra = sum(spectra, [])
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

        for spec in spectra:
            self.normalizeSpectra(spec)

        sky1d = []
        if self.config.doSubtractSky1d:
            # Do sky subtraction arm by arm for now; alternatives involve changing the run() API
            for ss in allSpectra:
                sky1d.append(self.skySubtraction(ss, pfsConfig))

        spectrographs = [self.mergeSpectra(ss) for ss in spectra]  # Merge in wavelength
        merged = PfsMerged.fromMerge(spectrographs, metadata=getPfsVersions())  # Merge across spectrographs

        lsfList = [self.mergeLsfs(ll, ss) for ll, ss in zip(lsfList, spectra)]
        mergedLsf = self.combineLsfs(lsfList)

        return Struct(pfsMerged=merged, pfsMergedLsf=mergedLsf, sky1d=sky1d)

    def runQuantum(
        self,
        butler: ButlerQuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `ButlerQuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        pfsArmList = defaultdict(list)
        lsfList = defaultdict(list)
        sky1dRefs = defaultdict(list)
        for pfsArm, lsf, sky1d in zipDatasetRefs(
            DatasetRefList.fromList(inputRefs.pfsArm),
            DatasetRefList.fromList(inputRefs.lsf),
            DatasetRefList.fromList(outputRefs.sky1d),
        ):
            dataId = pfsArm.dataId
            spectrograph = dataId["spectrograph"]
            pfsArmList[spectrograph].append(butler.get(pfsArm))
            lsfList[spectrograph].append(butler.get(lsf))
            sky1dRefs[spectrograph].append(sky1d)

        pfsConfig = butler.get(inputRefs.pfsConfig)
        outputs = self.run(list(pfsArmList.values()), pfsConfig, list(lsfList.values()))

        butler.put(outputs.pfsMerged, outputRefs.pfsMerged)
        butler.put(outputs.pfsMergedLsf, outputRefs.pfsMergedLsf)
        for sky1d, ref in zip(outputs.sky1d, sum(sky1dRefs.values(), [])):
            butler.put(sky1d, ref)

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

        expSpecRefList[0][0].put(results.pfsMerged, "pfsMerged")
        expSpecRefList[0][0].put(results.pfsMergedLsf, "pfsMergedLsf")
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
        wavelength scale and taking a straight sum (without strong rejection,
        so the very different values around the dichroic won't cause problems).
        That will have some sharp features (due to, e.g., CRs that we've missed,
        or even real spectral features like absorption bands), but we'll smooth
        over that by fitting a continuum model. The continuum model will form
        the basis for our target normalisation: we'll resample back to the
        original wavelength frames and apply.

        Parameters
        ----------
        spectra : iterable of `pfs.datamodel.PsfArm`
            Extracted spectra from the different arms, for a single
            spectrograph. The spectra will be modified in-place.

        Returns
        -------
        norm : `numpy.ndarray` of `float`
            Adopted normalisation.
        """
        wavelength = self.config.wavelength.wavelength
        fiberId = spectra[0].fiberId
        assert all(np.all(ss.fiberId == fiberId) for ss in spectra)  # Consistent fibers

        # Collect normalisations from each arm, interpolated to common wavelength sampling
        norm = np.zeros((fiberId.size, wavelength.size), dtype=np.float32)
        for ss in spectra:
            badBitmask = ss.flags.get(*self.config.mask)
            for ii in range(len(ss)):
                dispersion = calculateDispersion(ss.wavelength[ii])
                nn = interpolateFlux(ss.wavelength[ii], ss.norm[ii]/dispersion, wavelength, fill=0.0)
                mask = interpolateMask(ss.wavelength[ii], ss.mask[ii], wavelength, fill=badBitmask)
                ignore = ((mask & badBitmask) != 0) | ~np.isfinite(nn)
                if np.all(ignore):
                    continue
                nn[ignore] = 0.0
                norm[ii] += nn

        # Determine normalisation for each fiber
        specNorms = [np.zeros_like(ss.norm) for ss in spectra]
        for ii in range(fiberId.size):
            spectrum = Spectrum(wavelength.size)
            # Leave bad pixels at zero without masking. Small areas of bad pixels will get interpolated over
            # in the continuum fit. Large areas of bad pixels (like at the ends) will cause the fit to go to
            # zero, which is fine (in particular, it will keep the fit from going strongly negative at the
            # ends).
            spectrum.flux = norm[ii]
            spectrum.mask.array[0] = 0
            continuum = self.fitContinuum.runSingle(spectrum)

            if self.debugInfo.plotNorm and ii == (self.debugInfo.fiberIndex or 307):
                import matplotlib.pyplot as plt
                plt.plot(spectra[0].wavelength[ii], spectra[0].norm[ii], "b:")
                plt.plot(spectra[1].wavelength[ii], spectra[1].norm[ii], "r:")
                plt.plot(spectra[0].wavelength[ii], spectra[0].flux[ii], "b-")
                plt.plot(spectra[1].wavelength[ii], spectra[1].flux[ii], "r-")
                plt.plot(wavelength, norm[ii], "k:")
                plt.plot(wavelength, continuum, "k-")
                plt.suptitle(f"fiberId={spectra[0].fiberId[ii]}")
                plt.show()

            norm[ii] = continuum
            for jj, ss in enumerate(spectra):
                specNorms[jj][ii] = interpolateFlux(wavelength, continuum, ss.wavelength[ii])

            if self.debugInfo.plotNorm and ii == (self.debugInfo.fiberIndex or 307):
                import matplotlib.pyplot as plt
                plt.plot(spectra[0].wavelength[ii], spectra[0].norm[ii], "b:")
                plt.plot(spectra[1].wavelength[ii], spectra[1].norm[ii], "r:")
                plt.plot(spectra[0].wavelength[ii], specNorms[0][ii], "b-")
                plt.plot(spectra[1].wavelength[ii], specNorms[1][ii], "r-")
                plt.suptitle(f"fiberId={spectra[0].fiberId[ii]}")
                plt.show()

        # Apply normalisation
        # The new normalisation is in units of electrons/nm, so the renormalised fluxes are too.
        for ii, ss in enumerate(spectra):
            with np.errstate(invalid="ignore", divide="ignore"):
                nn = specNorms[ii]/ss.norm
                ss *= nn
                ss.mask[nn == 0] |= ss.flags["NO_DATA"]

        return norm

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
        resampled = [ss.resample(wavelength, jacobian=True) for ss in spectraList]
        flags = MaskHelper.fromMerge([ss.flags for ss in spectraList])
        combination = self.combine(resampled, flags)
        if self.config.doBarycentricCorr:
            self.log.warn("Barycentric correction is not yet implemented.")

        return PfsMerged(identity, fiberId, combination.wavelength, combination.flux, combination.mask,
                         combination.sky, combination.norm, combination.covar, flags, archetype.metadata)

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
            Normalised flux measurements for combined spectrum.
        sky : `numpy.ndarray` of `float`
            Sky measurements for combined spectrum.
        norm : `numpy.ndarray` of `float`
            Normalisation of combined spectrum.
        covar : `numpy.ndarray` of `float`
            Covariance matrix for combined spectrum.
        mask : `numpy.ndarray` of `int`
            Mask for combined spectrum.
        """
        archetype = spectra[0]
        mask = np.zeros_like(archetype.mask)
        flux = np.zeros_like(archetype.flux)
        sky = np.zeros_like(archetype.sky)
        norm = np.zeros_like(archetype.norm)
        covar = np.zeros_like(archetype.covar)
        sumWeights = np.zeros_like(archetype.flux)

        for ss in spectra:
            with np.errstate(invalid="ignore", divide="ignore"):
                variance = ss.variance/ss.norm**2
                good = ((ss.mask & ss.flags.get(*self.config.mask)) == 0) & (variance > 0)

            weight = np.zeros_like(ss.flux)
            weight[good] = 1.0/variance[good]
            with np.errstate(invalid="ignore"):
                flux[good] += ss.flux[good]*weight[good]/ss.norm[good]
                sky[good] += ss.sky[good]*weight[good]/ss.norm[good]
                norm[good] += ss.norm[good]*weight[good]
            mask[good] |= ss.mask[good]
            sumWeights += weight

        good = sumWeights > 0
        flux[good] /= sumWeights[good]
        sky[good] /= sumWeights[good]
        norm[good] /= sumWeights[good]
        covar[:, 0][good] = 1.0/sumWeights[good]
        covar[:, 0][~good] = np.inf
        covar[:, 1:] = np.where(good, 0.0, np.inf)[:, np.newaxis]

        for ss in spectra:
            mask[~good] |= ss.mask[~good]
        mask[~good] |= flags["NO_DATA"]
        covar2 = np.zeros((1, 1), dtype=archetype.covar.dtype)
        with np.errstate(invalid="ignore"):
            return Struct(wavelength=archetype.wavelength, flux=flux*norm, sky=sky*norm, norm=norm,
                          covar=covar*norm[:, np.newaxis, :]**2, mask=mask, covar2=covar2)

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

        return {ff: coaddLsf([ww.get(ff, None) for ww in warpedLsfList]) for ff in fiberId}

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
        lsf = LsfDict()
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

        if False:
            skyConfig = self.selectSky.run(pfsConfig.select(fiberId=spectra.fiberId))
            skySpectra = spectra.select(pfsConfig, fiberId=skyConfig.fiberId)
        else:
            flux = np.median(spectra.flux, axis=1)
            indices = np.argsort(flux)
            fiberId = spectra.fiberId[indices[:indices.size//2]]
            skyConfig = pfsConfig.select(fiberId=fiberId)
            skySpectra = spectra.select(pfsConfig, fiberId=fiberId)

        if len(skySpectra) == 0:
            raise RuntimeError("No sky spectra to use for sky subtraction")

        sky1d = self.fitSkyModel.run(skySpectra, skyConfig)
        subtractSky1d(spectra, pfsConfig, sky1d)
        return sky1d

    def _getMetadataName(self):
        return None
