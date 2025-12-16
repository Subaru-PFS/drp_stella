from collections import defaultdict

import numpy as np
from numpy.typing import ArrayLike

from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from lsst.pex.config import ConfigurableField

from pfs.datamodel import FiberStatus, PfsConfig, PfsFiberNorms, Target, TargetType
from pfs.datamodel.drp import PfsCalibrated

from .barycentricCorrection import applyBarycentricCorrection
from .datamodel import PfsArm, PfsFiberArray, PfsFiberArraySet, PfsMerged, PfsSingle
from .focalPlaneFunction import FluxCalib, FocalPlaneFunction
from .gen3 import readDatasetRefs
from .interpolate import calculateDispersion
from .lsf import Lsf, LsfDict
from .subtractSky1d import subtractSky1d
from .utils import getPfsVersions
from .FluxTableTask import FluxTableTask

from collections.abc import Iterable

__all__ = ("applyFiberNorms", "calibratePfsArm", "fluxCalibrate", "ApplyFluxCalConfig", "ApplyFluxCalTask")


def applyFiberNorms(
    pfsArm: PfsArm,
    fiberNorms: PfsFiberNorms,
    doCheckHashes: bool = True,
) -> set[int]:
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
                raise RuntimeError(f"Hash mismatch for fiberProfiles: {gotHash} != {expectHash}")

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


def fluxCalibrate(
    spectra: PfsFiberArray | PfsFiberArraySet,
    pfsConfig: PfsConfig,
    fluxCal: FocalPlaneFunction,
    useFluxCalVariance: bool = False,
) -> None:
    """Apply flux calibration to spectra

    Parameters
    ----------
    spectra : subclass of `PfsFiberArray` or `PfsFiberArraySet`
        Spectra (or spectrum) to flux-calibrate.
    pfsConfig : `PfsConfig`
        Top-end configuration.
    fluxCal : subclass of `FocalPlaneFunction`
        Flux calibration model.
    useFluxCalVariance : `bool`, optional
        Include the variance in the fluxCal when computing the output variance?
        This term can dwarf the other variance contributions, leading to
        erroneous conclusions about the noise in the calibrated spectra.
    """
    cal = fluxCal(spectra.wavelength, pfsConfig.select(fiberId=spectra.fiberId))
    with np.errstate(divide="ignore", invalid="ignore"):
        spectra /= spectra.norm
        spectra /= cal.values  # includes spectrum.variance /= cal.values**2
        if useFluxCalVariance:
            spectra.variance[:] += cal.variances * spectra.flux**2 / cal.values**2
    spectra.norm[:] = 1.0  # We've deliberately changed the normalisation
    bad = np.array(cal.masks) | (np.array(cal.values) == 0.0)
    bad |= ~np.isfinite(cal.values) | ~np.isfinite(cal.variances)
    spectra.mask[bad] |= spectra.flags.add("BAD_FLUXCAL")


def calibratePfsArm(
    spectra: PfsArm,
    pfsConfig: PfsConfig,
    sky1d: FocalPlaneFunction,
    fluxCal: FocalPlaneFunction,
    fiberNorms: PfsFiberNorms | None = None,
    doCheckFiberNormsHashes: bool = True,
    wavelength: ArrayLike | None = None,
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


class ApplyFluxCalConnections(PipelineTaskConnections, dimensions=("instrument", "visit")):
    """Connections for ApplyFluxCalTask"""

    fluxCal = InputConnection(
        name="fluxCal",
        doc="Flux calibration model",
        storageClass="FocalPlaneFunction",
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
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
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
    pfsCalibrated = OutputConnection(
        name="pfsCalibrated",
        doc="Flux-calibrated object spectrum",
        storageClass="PfsCalibrated",
        dimensions=("instrument", "visit"),
    )
    pfsCalibratedLsf = OutputConnection(
        name="pfsCalibratedLsf",
        doc="Line-spread function for pfsCalibrated",
        storageClass="LsfDict",
        dimensions=("instrument", "visit"),
    )


class ApplyFluxCalConfig(PipelineTaskConfig, pipelineConnections=ApplyFluxCalConnections):
    """Configuration for ApplyFluxCalTask"""

    fluxTable = ConfigurableField(target=FluxTableTask, doc="Flux table")


class ApplyFluxCalTask(PipelineTask):
    """Apply the flux calibration"""

    ConfigClass = ApplyFluxCalConfig
    _DefaultName = "applyFluxCal"

    fluxTable: FluxTableTask

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.makeSubtask("fluxTable")

    def run(
        self,
        fluxCal: FluxCalib,
        pfsMerged: PfsMerged,
        pfsMergedLsf: LsfDict,
        pfsConfig: PfsConfig,
        pfsArmList: list[PfsArm],
        sky1dList: Iterable[FluxCalib],
    ) -> Struct:
        """Measure and apply the flux calibration

        Parameters
        ----------
        fluxCal : `FluxCalib`
            Flux calibration solution.
        pfsMerged : `PfsMerged`
            Merged spectra, containing observations of ``FLUXSTD`` sources.
        pfsMergedLsf : `LsfDict`
            Line-spread functions for merged spectra.
        references : `PfsFluxReference`
            Reference spectra.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        pfsArmList : iterable of `PfsArm`
            List of extracted spectra, for constructing the flux table.
        sky1dList : iterable of `FluxCalib`
            Corresponding list of 1d sky subtraction models.

        Returns
        -------
        pfsCalibrated : `PfsCalibrated`
            Calibrated spectra.
        pfsCalibratedLsf : `LsfDict`
            Line-spread functions for calibrated spectra.
        """
        fluxCalibrate(pfsMerged, pfsConfig, fluxCal)

        calibrated = []
        fiberToArm = defaultdict(list)
        for ii, (pfsArm, sky1d) in enumerate(zip(pfsArmList, sky1dList)):
            calibratePfsArm(pfsArm, pfsConfig, sky1d, fluxCal)
            for ff in pfsArm.fiberId:
                fiberToArm[ff].append(ii)
            calibrated.append(pfsArm)

        selection = pfsConfig.getSelection(fiberStatus=FiberStatus.GOOD)
        selection &= ~pfsConfig.getSelection(targetType=TargetType.ENGINEERING)
        fiberId = pfsMerged.fiberId[np.isin(pfsMerged.fiberId, pfsConfig.fiberId[selection])]

        pfsCalibrated: dict[Target, PfsSingle] = {}
        pfsCalibratedLsf: dict[Target, Lsf] = {}
        for ff in fiberId:
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
            pfsCalibrated=PfsCalibrated(pfsCalibrated.values(), getPfsVersions()),
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

        outputs = self.run(**inputs, pfsArmList=armInputs.pfsArm, sky1dList=armInputs.sky1d)
        butler.put(outputs, outputRefs)
