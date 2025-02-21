from typing import Iterable, List, Union, Dict, Optional, Set
from collections import defaultdict

import numpy as np
from numpy.typing import ArrayLike

from lsst.pex.config import Field, ConfigurableField
from lsst.pipe.base import Struct

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from pfs.datamodel.pfsConfig import PfsConfig, TargetType
from pfs.datamodel import FiberStatus, Target
from pfs.datamodel.pfsFiberNorms import PfsFiberNorms
from pfs.datamodel.drp import PfsCalibrated
from pfs.drp.stella.lsf import Lsf, LsfDict

from .focalPlaneFunction import FocalPlaneFunction
from .datamodel import PfsArm, PfsSingle, PfsMerged, PfsReference, PfsFiberArray, PfsFiberArraySet
from .fitFocalPlane import FitFocalPlaneTask
from .interpolate import calculateDispersion
from .subtractSky1d import subtractSky1d
from .FluxTableTask import FluxTableTask
from .utils import getPfsVersions
from .gen3 import readDatasetRefs
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
        if spectrograph in fiberNorms.fiberProfilesHash:
            expectHash = fiberNorms.fiberProfilesHash[spectrograph]
            gotHash = pfsArm.metadata["PFS.HASH.FIBERPROFILES"]
            if gotHash != expectHash:
                raise RuntimeError(
                    f"Hash mismatch for fiberProfiles: {gotHash} != {expectHash}"
                )

    badFiberNorms = pfsArm.flags.add("BAD_FIBERNORMS")
    fiberNorms = fiberNorms.select(fiberId=pfsArm.fiberId)

    # Catch fibers without a fiberNorms entry
    missingFiberId = set(pfsArm.fiberId) - set(fiberNorms.fiberId)
    if missingFiberId:
        missing = pfsArm.select(fiberId=list(missingFiberId))
        missing.mask |= badFiberNorms
        pfsArm = pfsArm.select(fiberId=fiberNorms.fiberId)

    # Apply the fiberNorms
    assert np.array_equal(pfsArm.fiberId, fiberNorms.fiberId)
    bad = (fiberNorms.values == 0.0) | ~np.isfinite(fiberNorms.values)
    if np.any(bad):
        fiberNorms.values[bad] = 1.0
        pfsArm.mask[bad] |= badFiberNorms
    pfsArm.norm *= fiberNorms.values

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


class FluxCalibrateConnections(PipelineTaskConnections, dimensions=("instrument", "visit")):
    """Connections for FluxCalibrateTask"""

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
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )
    references = InputConnection(
        name="pfsFluxReference",
        doc="Fit reference spectrum of flux standards",
        storageClass="PfsFluxReference",
        dimensions=("instrument", "visit"),
    )
    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted spectra",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    sky1d = InputConnection(
        name="sky1d",
        doc="1d sky model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    fluxCal = OutputConnection(
        name="fluxCal",
        doc="Flux calibration model",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit"),
    )
    pfsCalibrated = OutputConnection(
        name="pfsCalibrated",
        doc="Flux-calibrated object spectrum",
        storageClass="PfsCalibratedSpectra",  # deprecated in favor of PfsCalibrated
        dimensions=("instrument", "visit"),
    )
    pfsCalibratedLsf = OutputConnection(
        name="pfsCalibratedLsf",
        doc="Line-spread function for pfsCalibrated",
        storageClass="LsfDict",
        dimensions=("instrument", "visit"),
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


class FluxCalibrateTask(PipelineTask):
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
        pfsCalibrated : `PfsCalibrated`
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
            pfsCalibrated=PfsCalibrated(pfsCalibrated.values()),
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
