from collections import defaultdict
import numpy as np

import lsstDebug
from lsst.pex.config import Field, ConfigurableField, ListField
from lsst.pipe.base import Struct

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from lsst.obs.pfs.utils import getLamps

from pfs.datamodel import TargetType
from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.utils import createHash
from pfs.drp.stella.gen3 import DatasetRefList, zipDatasetRefs
from pfs.drp.stella.selectFibers import SelectFibersTask
from .combineSpectra import combineSpectraSets
from .datamodel import PfsConfig, PfsArm, PfsMerged
from pfs.datamodel import Identity
from .fitFocalPlane import FitBlockedOversampledSplineTask
from .focalPlaneFunction import FocalPlaneFunction
from .utils import getPfsVersions
from .lsf import LsfDict, CoaddLsf
from .SpectrumContinued import Spectrum
from .interpolate import calculateDispersion, interpolateFlux, interpolateMask
from .fitContinuum import FitContinuumTask
from .subtractSky1d import subtractSky1d
from .barycentricCorrection import applyBarycentricCorrection
from .wavelengthSampling import WavelengthSamplingTask


class MergeArmsConnections(PipelineTaskConnections, dimensions=("instrument", "visit")):
    """Connections for MergeArmsTask"""
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
    lsf = InputConnection(
        name="pfsArmLsf",
        doc="1D line-spread function",
        storageClass="LsfDict",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    pfsMerged = OutputConnection(
        name="pfsMerged",
        doc="Merged spectra from an exposure",
        storageClass="PfsMerged",
        dimensions=("instrument", "visit"),
    )
    pfsMergedLsf = OutputConnection(
        name="pfsMergedLsf",
        doc="Line-spread function of merged spectra from an exposure",
        storageClass="LsfDict",
        dimensions=("instrument", "visit"),
    )
    sky1d = OutputConnection(
        name="sky1d",
        doc="1d sky model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )


class MergeArmsConfig(PipelineTaskConfig, pipelineConnections=MergeArmsConnections):
    """Configuration for MergeArmsTask"""
    wavelength = ConfigurableField(target=WavelengthSamplingTask, doc="Wavelength sampling")
    doSubtractSky1d = Field(dtype=bool, default=True, doc="Do 1D sky subtraction?")
    selectSky = ConfigurableField(target=SelectFibersTask, doc="Select fibers for 1d sky subtraction")
    fitSkyModel = ConfigurableField(target=FitBlockedOversampledSplineTask,
                                    doc="Fit sky model over the focal plane")
    mask = ListField(dtype=str, default=["NO_DATA"], doc="Mask values to reject when combining")
    suspect = ListField(dtype=str, default=["SUSPECT"], doc="Mask values to allow if we're desperate")
    pfsConfigFile = Field(dtype=str, default="", doc="""Full pathname of pfsCalib file to use.
    If of the form "pfsConfig-0x%x-%d.fits", the pfsDesignId and visit0 will be deduced from the filename;
    if not, the values 0x666 and 0 are used.""")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum to mean normalisation")
    notesCopyFirst = ListField(
        dtype=str,
        doc="Notes for which we simply copy the first value from one of the arms",
        default=["blackSpotId", "blackSpotDistance", "blackSpotCorrection", "barycentricCorrection"],
    )
    doBarycentricCorrection = Field(dtype=bool, default=True, doc="Apply barycentric correction to sky data?")

    def setDefaults(self):
        super().setDefaults()
        self.selectSky.targetType = ("SKY", "SUNSS_DIFFUSE", "HOME")
        # Scale back rejection because otherwise everything gets rejected
        self.fitSkyModel.rejIterations = 1
        self.fitSkyModel.rejThreshold = 4.0
        self.fitSkyModel.mask = ["NO_DATA", "BAD_FLAT", "BAD_FIBERNORMS", "SUSPECT"]
        self.fitContinuum.numKnots = 100  # Increased number of knots because larger wavelength range
        self.fitContinuum.iterations = 0  # No rejection: normalisation doesn't need to be exact, just robust


class MergeArmsTask(PipelineTask):
    """Merge all extracted spectra from a single exposure"""
    _DefaultName = "mergeArms"
    ConfigClass = MergeArmsConfig

    selectSky: SelectFibersTask
    fitSkyModel: FitBlockedOversampledSplineTask
    fitContinuum: FitContinuumTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("wavelength")
        self.makeSubtask("selectSky")
        self.makeSubtask("fitSkyModel")
        self.makeSubtask("fitContinuum")
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, spectra, pfsConfig, lsfList, haveMedRes: bool = False):
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
        haveMedRes : `bool`
            Do we have medium-resolution data?

        Returns
        -------
        spectra : `pfs.datamodel.PfsMerged`
            Merged spectra.
        lsf : `pfs.drp.stella.Lsf`
            Merged line-spread function.
        sky1d : `list` of `pfs.drp.stella.FocalPlaneFunction`
            Sky models for each arm.
        """
        wavelength = self.wavelength.run(haveMedRes)

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
            self.normalizeSpectra(spec, wavelength)

        md = allSpectra[0].metadata
        haveSky = not getLamps(md)  # No lamps probably means sky exposure, but might be a dark
        if haveSky and md.get("DATA-TYP", "").lower().strip() == "dark":
            haveSky = False
        sky1d = []
        if self.config.doSubtractSky1d:
            if haveSky:
                # Do sky subtraction arm by arm for now; alternatives involve changing the run() API
                for ss in allSpectra:
                    sky1d.append(self.skySubtraction(ss, pfsConfig))
            else:
                self.log.warn("Skipping sky subtraction for lamp exposure")
                sky1d = [None]*len(allSpectra)

        # Now that the sky subtraction is done, we can apply the barycentric correction
        if self.config.doBarycentricCorrection:
            if haveSky:
                noBarycentricFibers = set()
                for ss in allSpectra:
                    noBarycentricFibers |= applyBarycentricCorrection(ss)
                self.log.info("Applied barycentric correction")
                if noBarycentricFibers:
                    noTarget = [TargetType.fromString(tt) for tt in (
                        "SKY",
                        "SUNSS_DIFFUSE",
                        "SUNSS_IMAGING",
                        "UNASSIGNED",
                        "DCB",
                        "HOME",
                        "BLACKSPOT",
                        "AFL",
                    )]
                    noBarycentricFibers -= set(pfsConfig.select(targetType=noTarget).fiberId)
                if noBarycentricFibers:
                    self.log.warn(
                        "Unable to apply barycentric correction to fibers: %s", sorted(noBarycentricFibers)
                    )
            else:
                self.log.warn("Skipping barycentric correction for lamp exposure")

        spectrographs = [self.mergeSpectra(ss, wavelength) for ss in spectra]  # Merge in wavelength
        metadata = allSpectra[0].metadata.copy()
        metadata.update(getPfsVersions())
        merged = PfsMerged.fromMerge(spectrographs, metadata=metadata)  # Merge across spectrographs

        lsfList = [self.mergeLsfs(ll, ss, wavelength) for ll, ss in zip(lsfList, spectra)]
        mergedLsf = self.combineLsfs(lsfList)

        return Struct(pfsMerged=merged, pfsMergedLsf=mergedLsf, sky1d=sky1d)

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
        pfsArmList = defaultdict(list)
        lsfList = defaultdict(list)
        sky1dRefs = defaultdict(list)
        haveMedRes = False
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
            if dataId["arm"] == "m":
                haveMedRes = True

        pfsConfig = butler.get(inputRefs.pfsConfig)
        outputs = self.run(list(pfsArmList.values()), pfsConfig, list(lsfList.values()), haveMedRes)

        butler.put(outputs.pfsMerged, outputRefs.pfsMerged)
        butler.put(outputs.pfsMergedLsf, outputRefs.pfsMergedLsf)
        for sky1d, ref in zip(outputs.sky1d, sum(sky1dRefs.values(), [])):
            if sky1d is not None:
                butler.put(sky1d, ref)

    def normalizeSpectra(self, spectra, wavelength: np.ndarray):
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

        This process converts the units of the spectra from electrons (used for
        pfsArm) to electrons/nm (used for pfsMerged), because the new
        normalisation has been divided by the dispersion.

        Parameters
        ----------
        spectra : iterable of `pfs.datamodel.PsfArm`
            Extracted spectra from the different arms, for a single
            spectrograph. The spectra will be modified in-place.
        wavelength : `np.ndarray`
            Target wavelength array after resampling.

        Returns
        -------
        norm : `numpy.ndarray` of `float`
            Adopted normalisation.
        """
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
            continuum = self.fitContinuum.fitContinuum(spectrum)

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

    def mergeSpectra(self, spectraList, wavelength: np.ndarray):
        """Combine all spectra from the same exposure

        All spectra should have the same fibers, so we simply iterate over the
        fibers, combining each spectrum from that fiber.

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsFiberArraySet`
            List of spectra to coadd.
        wavelength : `np.ndarray`
            Wavelength array to which to resample.

        Returns
        -------
        result : `pfs.datamodel.PfsMerged`
            Merged spectra.
        """
        archetype = spectraList[0]
        identity = Identity.fromMerge([ss.identity for ss in spectraList])
        metadata = archetype.metadata.copy()
        fiberId = archetype.fiberId
        if any(np.any(ss.fiberId != fiberId) for ss in spectraList):
            raise RuntimeError("Selection of fibers differs")
        resampled = [ss.resample(wavelength) for ss in spectraList]
        flags = MaskHelper.fromMerge([ss.flags for ss in spectraList])
        combination = combineSpectraSets(resampled, flags, self.config.mask, self.config.suspect)

        notes = PfsMerged.NotesClass.empty(len(archetype))
        for name in self.config.notesCopyFirst:
            getattr(notes, name)[:] = getattr(archetype.notes, name)

        fiberProfilesHashes = {
            (ss.identity.spectrograph, ss.identity.arm): ss.metadata["PFS.HASH.FIBERPROFILES"]
            for ss in spectraList
        }  # Hashes, indexed by spectrograph,arm to define the order
        metadata["PFS.HASH.FIBERPROFILES"] = createHash(fiberProfilesHashes.values())  # Hash of hashes

        return PfsMerged(identity, fiberId, combination.wavelength, combination.flux, combination.mask,
                         combination.sky, combination.norm, combination.covar, flags, metadata,
                         notes)

    def mergeLsfs(self, lsfList, spectraList, wavelength: np.ndarray):
        """Merge LSFs for different arms within a spectrograph

        Parameters
        ----------
        lsfList : iterable of `dict` (`int`: `pfs.drp.stella.Lsf`)
            Line-spread functions indexed by fiberId, for each arm.
        spectraList : iterable of `pfs.datamodel.PfsFiberArraySet`
            Spectra for each arm.
        wavelength : `np.ndarray`
            Wavelength array to which to resample.

        Returns
        -------
        lsf : `dict` (`int`: `pfs.drp.stella.Lsf`)
            Merged line-spread functions indexed by fiberId.
        """
        fiberId = set(lsfList[0].keys())
        for lsf in lsfList:
            assert set(lsf.keys()) == fiberId
        warpedLsfList = []
        minIndexList = []
        maxIndexList = []
        last = wavelength.size - 1
        for lsf, spectra in zip(lsfList, spectraList):
            warpedLsf = {}
            minIndex = {}
            maxIndex = {}
            for ii in range(len(spectra)):
                ff = spectra.fiberId[ii]
                warpedLsf[ff] = lsf.get(ff).warp(spectra.wavelength[ii], wavelength)
                minIndex[ff] = max(0, np.searchsorted(wavelength, spectra.wavelength[ii][0], "left"))
                maxIndex[ff] = min(last, np.searchsorted(wavelength, spectra.wavelength[ii][-1], "right"))

            warpedLsfList.append(warpedLsf)
            minIndexList.append(minIndex)
            maxIndexList.append(maxIndex)

        return {ff: CoaddLsf(
            [ww.get(ff, None) for ww in warpedLsfList],
            [minIndex[ff] for minIndex in minIndexList],
            [maxIndex[ff] for maxIndex in maxIndexList],
        ) for ff in fiberId}

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
        skyConfig = self.selectSky.run(pfsConfig.select(fiberId=spectra.fiberId))
        skySpectra = spectra.select(pfsConfig, fiberId=skyConfig.fiberId)
        if len(skySpectra) == 0:
            raise RuntimeError("No sky spectra to use for sky subtraction")

        sky1d = self.fitSkyModel.run(skySpectra, skyConfig)
        subtractSky1d(spectra, pfsConfig, sky1d)
        return sky1d

    def _getMetadataName(self):
        return None
