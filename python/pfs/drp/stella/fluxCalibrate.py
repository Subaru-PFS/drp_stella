from typing import Iterable, List, Union, Dict, Optional, Set
from collections import defaultdict

import numpy as np
from numpy.typing import ArrayLike

from lsst.pex.config import Field, ConfigurableField
from lsst.pipe.base import ArgumentParser, Struct
from lsst.daf.persistence import Butler

from lsst.pipe.base import CmdLineTask, PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from pfs.datamodel.pfsConfig import PfsConfig, TargetType
from pfs.datamodel import FiberStatus, Target
from pfs.datamodel.pfsFiberNorms import PfsFiberNorms
from pfs.drp.stella.lsf import Lsf, LsfDict

from .focalPlaneFunction import FocalPlaneFunction
from .datamodel import PfsArm, PfsSingle, PfsMerged, PfsReference, PfsFiberArray, PfsFiberArraySet
from .fitFocalPlane import FitFocalPlaneTask
from .interpolate import calculateDispersion
from .subtractSky1d import subtractSky1d
from .FluxTableTask import FluxTableTask
from .utils import getPfsVersions
from .gen3 import readDatasetRefs
from .datamodel.pfsTargetSpectra import PfsCalibratedSpectra
from .barycentricCorrection import applyBarycentricCorrection


__all__ = (
    "PfsReferenceSet",
    "applyFiberNorms",
    "fluxCalibrate",
    "FluxCalibrateConfig",
    "FluxCalibrateTask",
)

PfsReferenceSet = Dict[int, PfsReference]
"""Mapping of fiberId to the appropriate PfsReference"""


def applyFiberNorms(
    pfsArm: PfsArm,
    fiberNorms: PfsFiberNorms,
    doCheckHashes: bool = True,
) -> Set[int]:
    """Apply fiber normalisation to a PfsArm

    Parameters
    ----------
    pfsArm : `PfsArm`
        Arm spectra to which to apply normalisation.
    fiberNorms : `PfsFiberNorms`
        Fiber normalisations.
    doCheckHashes : `bool`, optional
        Check hashes for consistency?

    Returns
    -------
    missingFiberId : `set` of `int`
        FiberIds for which we don't have a normalisation.
    """
    if doCheckHashes:
        spectrograph = pfsArm.identity.spectrograph
        expectHash = fiberNorms.fiberProfilesHash[spectrograph]
        gotHash = pfsArm.metadata["PFS.HASH.FIBERPROFILES"]
        if gotHash != expectHash:
            raise RuntimeError(
                f"Hash mismatch for fiberProfiles: {gotHash} != {expectHash}"
            )

    # Apply normalisation to each fiber
    missingFiberId = set()
    badFiberNorms = pfsArm.flags.add("BAD_FIBERNORMS")
    for ii, fiberId in enumerate(pfsArm.fiberId):
        if fiberId in fiberNorms:
            nn = fiberNorms[fiberId]

            bad = (nn == 0.0) | ~np.isfinite(nn)
            if np.any(bad):
                nn[bad] = 1.0
                pfsArm.mask[ii][bad] |= badFiberNorms

            pfsArm.norm[ii] *= nn
        else:
            missingFiberId.add(fiberId)
            pfsArm.mask[ii] |= badFiberNorms

    return missingFiberId


def fluxCalibrate(spectra: Union[PfsFiberArray, PfsFiberArraySet], pfsConfig: PfsConfig,
                  fluxCal: FocalPlaneFunction) -> None:
    """Apply flux calibration to spectra

    Parameters
    ----------
    spectra : subclass of `PfsFiberArray` or `PfsFiberArraySet`
        Spectra (or spectrum) to flux-calibrate.
    pfsConfig : `PfsConfig`
        Top-end configuration.
    fluxCal : subclass of `FocalPlaneFunction`
        Flux calibration model.
    """
    cal = fluxCal(spectra.wavelength, pfsConfig.select(fiberId=spectra.fiberId))
    with np.errstate(divide="ignore", invalid="ignore"):
        spectra /= spectra.norm
        spectra /= cal.values  # includes spectrum.variance /= cal.values**2
        spectra.variance[:] += cal.variances*spectra.flux**2/cal.values**2
    spectra.norm[:] = 1.0  # We've deliberately changed the normalisation
    bad = np.array(cal.masks) | (np.array(cal.values) == 0.0)
    bad |= ~np.isfinite(cal.values) | ~np.isfinite(cal.variances)
    spectra.mask[bad] |= spectra.flags.add("BAD_FLUXCAL")


def calibratePfsArm(
    spectra: PfsArm,
    pfsConfig: PfsConfig,
    sky1d: FocalPlaneFunction,
    fluxCal: FocalPlaneFunction,
    fiberNorms: Optional[PfsFiberNorms] = None,
    doCheckFiberNormsHashes: bool = True,
    wavelength: Optional[ArrayLike] = None,
) -> PfsArm:
    """Calibrate a PfsArm

    Parameters
    ----------
    spectra : `PfsArm`
        PfsArm spectra, modified.
    sky1d : `FocalPlaneFunction`
        1d sky model.
    fluxCal : `FocalPlaneFunction`
        Flux calibration model.
    fiberNorms : `PfsFiberNorms`, optional
        Fiber normalisations.
    doCheckFiberNormsHashes : `bool`, optional
        Check hashes in the fiberNorms for consistency?
    wavelength : `numpy.ndarray` of `float`, optional
        Wavelength array for optional resampling.

    Returns
    -------
    spectra : `PfsArm`
        Calibrated PfsArm spectra.
    """
    pfsConfig = pfsConfig.select(fiberId=spectra.fiberId)
    spectra /= calculateDispersion(spectra.wavelength)  # Convert to electrons/nm
    if fiberNorms is not None:
        applyFiberNorms(spectra, fiberNorms, doCheckFiberNormsHashes)
    subtractSky1d(spectra, pfsConfig, sky1d)
    applyBarycentricCorrection(spectra)
    fluxCalibrate(spectra, pfsConfig, fluxCal)
    if wavelength is not None:
        spectra = spectra.resample(wavelength)  # sampling of pfsArm related to the flux values
    return spectra


class FluxCalibrateConnections(PipelineTaskConnections, dimensions=("instrument", "exposure")):
    """Connections for FluxCalibrateTask"""

    pfsMerged = InputConnection(
        name="pfsMerged",
        doc="Merged spectra from exposure",
        storageClass="PfsMerged",
        dimensions=("instrument", "exposure"),
    )
    pfsMergedLsf = InputConnection(
        name="pfsMergedLsf",
        doc="Line-spread function of merged spectra",
        storageClass="LsfDict",
        dimensions=("instrument", "exposure"),
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )
    references = InputConnection(
        name="pfsFluxReference",
        doc="Fit reference spectrum of flux standards",
        storageClass="PfsFluxReference",
        dimensions=("instrument", "exposure"),
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra",
        storageClass="PfsArm",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    sky1d = InputConnection(
        name="sky1d",
        doc="1d sky model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )

    fluxCal = OutputConnection(
        name="fluxCal",
        doc="Flux calibration model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "exposure"),
    )
    pfsCalibrated = OutputConnection(
        name="pfsCalibrated",
        doc="Flux-calibrated object spectrum",
        storageClass="PfsCalibratedSpectra",
        dimensions=("instrument", "exposure"),
    )
    pfsCalibratedLsf = OutputConnection(
        name="pfsCalibratedLsf",
        doc="Line-spread function for pfsCalibrated",
        storageClass="LsfDict",
        dimensions=("instrument", "exposure"),
    )


class FluxCalibrateConfig(PipelineTaskConfig, pipelineConnections=FluxCalibrateConnections):
    """Configuration for FluxCalibrateTask"""
    sysErr = Field(dtype=float, default=1.0e-4,
                   doc=("Fraction of value to add to variance before fitting. This attempts to offset the "
                        "loss of variance as covariance when we resample, the result of which is "
                        "underestimated errors and excess rejection."))
    fitFluxCal = ConfigurableField(target=FitFocalPlaneTask, doc="Fit flux calibration model")
    fluxTable = ConfigurableField(target=FluxTableTask, doc="Flux table")
    doWrite = Field(dtype=bool, default=True, doc="Write outputs?")


class FluxCalibrateTask(CmdLineTask, PipelineTask):
    """Measure and apply the flux calibration"""
    ConfigClass = FluxCalibrateConfig
    _DefaultName = "fluxCalibrate"

    fitFluxCal: FitFocalPlaneTask
    fluxTable: FluxTableTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitFluxCal")
        self.makeSubtask("fluxTable")

    def run(
        self,
        pfsMerged: PfsMerged,
        pfsMergedLsf: LsfDict,
        references: Dict[int, PfsReference],
        pfsConfig: PfsConfig,
        pfsArmList: List[PfsArm],
        sky1dList: Iterable[FocalPlaneFunction],
    ) -> Struct:
        """Measure and apply the flux calibration

        Parameters
        ----------
        pfsMerged : `PfsMerged`
            Merged spectra, containing observations of ``FLUXSTD`` sources.
        pfsMergedLsf : `LsfDict`
            Line-spread functions for merged spectra.
        references : `dict` mapping `int` to `PfsReferenceSet`
            Reference spectra, indexed by fiberId.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        pfsArmList : iterable of `PfsArm`
            List of extracted spectra, for constructing the flux table.
        sky1d : iterable of `FocalPlaneFunction`
            Corresponding list of 1d sky subtraction models.

        Returns
        -------
        fluxCal : `FocalPlaneFunction`
            Flux calibration solution.
        pfsCalibrated : `PfsCalibratedSpectra`
            Calibrated spectra.
        pfsCalibratedLsf : `LsfDict`
            Line-spread functions for calibrated spectra.
        """
        mergedFluxCal = pfsMerged.select(pfsConfig, targetType=TargetType.FLUXSTD)
        pfsConfigFluxCal = pfsConfig.select(fiberId=mergedFluxCal.fiberId)

        self.calculateCalibrations(mergedFluxCal, references)
        fluxCal = self.fitFluxCal.run(mergedFluxCal, pfsConfigFluxCal)
        fluxCalibrate(pfsMerged, pfsConfig, fluxCal)

        calibrated = []
        fiberToArm = defaultdict(list)
        for ii, (pfsArm, sky1d) in enumerate(zip(pfsArmList, sky1dList)):
            subtractSky1d(pfsArm, pfsConfig, sky1d)
            fluxCalibrate(pfsArm, pfsConfig, fluxCal)
            for ff in pfsArm.fiberId:
                fiberToArm[ff].append(ii)
            calibrated.append(pfsArm)

        pfsMerged = pfsMerged.select(pfsConfig, fiberStatus=FiberStatus.GOOD)
        pfsCalibrated: Dict[Target, PfsSingle] = {}
        pfsCalibratedLsf: Dict[Target, Lsf] = {}
        for ff in pfsMerged.fiberId:
            extracted = pfsMerged.extractFiber(PfsSingle, pfsConfig, ff)
            extracted.fluxTable = self.fluxTable.run(
                [calibrated[ii].identity.getDict() for ii in fiberToArm[ff]],
                [pfsArmList[ii].extractFiber(PfsSingle, pfsConfig, ff) for ii in fiberToArm[ff]],
            )
            extracted.metadata = getPfsVersions()

            target = extracted.target
            pfsCalibrated[target] = extracted
            pfsCalibratedLsf[target] = pfsMergedLsf[ff]

        return Struct(
            fluxCal=fluxCal,
            pfsCalibrated=PfsCalibratedSpectra(pfsCalibrated.values()),
            pfsCalibratedLsf=LsfDict(pfsCalibratedLsf),
        )

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
        armInputs = readDatasetRefs(butler, inputRefs, "pfsArm", "sky1d")
        inputs = butler.get(inputRefs)

        pfsConfig = inputs["pfsConfig"].select(targetType=TargetType.FLUXSTD)
        pfsFluxReference = inputs.pop("references")
        references = {fiberId: pfsFluxReference.extractFiber(PfsReference, pfsConfig, fiberId)
                      for fiberId in pfsFluxReference.fiberId}

        outputs = self.run(
            **inputs, pfsArmList=armInputs.pfsArm, sky1dList=armInputs.sky1d, references=references
        )
        butler.put(outputs, outputRefs)

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsMerged", level="Visit",
                               help="data IDs, e.g. --id exp=12345")
        return parser

    def runDataRef(self, dataRef):
        """Measure and apply the flux calibration

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for merged spectrum.

        Returns
        -------
        fluxCal : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.
        spectra : `list` of `pfs.datamodel.PfsSingle`
            Calibrated spectra for each fiber.
        """
        pfsMerged = dataRef.get("pfsMerged")
        pfsMergedLsf = dataRef.get("pfsMergedLsf")
        pfsConfig = dataRef.get("pfsConfig")

        butler = dataRef.getButler()
        pfsMergedFluxCal = pfsMerged.select(pfsConfig, targetType=TargetType.FLUXSTD)
        pfsConfigFluxCal = pfsConfig.select(fiberId=pfsMergedFluxCal.fiberId)
        references = self.readReferences(butler, pfsConfigFluxCal)

        armRefList = list(butler.subset("raw", dataId=dataRef.dataId))
        pfsArmList = [armRef.get("pfsArm") for armRef in armRefList]
        sky1dList = [armRef.get("sky1d") for armRef in armRefList]

        outputs = self.run(pfsMerged, pfsMergedLsf, references, pfsConfig, pfsArmList, sky1dList)

        if self.config.doWrite:
            dataRef.put(outputs.fluxCal, "fluxCal")

            # Gen2 writes the pfsCalibrated spectra individually
            for target in outputs.pfsCalibrated:
                pfsSingle = outputs.pfsCalibrated[target]
                dataId = pfsSingle.getIdentity().copy()
                dataId.update(dataRef.dataId)
                butler.put(pfsSingle, "pfsSingle", dataId)
                butler.put(outputs.pfsCalibratedLsf[target], "pfsSingleLsf", dataId)

        return outputs

    def readReferences(self, butler: Butler, pfsConfig: PfsConfig) -> PfsReferenceSet:
        """Read the physical reference fluxes

        If you get a read error here, it's likely because you haven't got a
        physical reference flux; try running ``calibrateReferenceFlux``.

        Parameters
        ----------
        butler : `lsst.daf.persistence.Butler`
            Data butler.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for identifying flux standards. This should
            contain only the fibers of interest.

        Returns
        -------
        references : `dict` mapping `int` to `pfs.datamodel.PfsSimpleSpectrum`
            Reference spectra, indexed by fiber identifier.
        """
        return {ff: butler.get("pfsReference", pfsConfig.getIdentity(ff)) for ff in pfsConfig.fiberId}

    def calculateCalibrations(self, merged: PfsMerged, references: PfsReferenceSet) -> None:
        """Calculate the flux calibration vector for each fluxCal fiber

        Parameters
        ----------
        merged : `PfsMerged`
            Merged spectra. Contains only fluxCal fibers. These will be
            modified.
        references : `dict` mapping `int` to `PfsReference`
            Reference spectra, indexed by fiber identifier.
        """
        ref = np.zeros_like(merged.flux)
        for ii, fiberId in enumerate(merged.fiberId):
            ref[ii] = references[fiberId].flux

        merged.covar[:, 0] += self.config.sysErr*merged.flux  # add systematic error
        merged /= merged.norm
        merged /= ref
        merged.norm[:] = 1.0  # We're deliberately changing the normalisation

    def _getMetadataName(self):
        return None
