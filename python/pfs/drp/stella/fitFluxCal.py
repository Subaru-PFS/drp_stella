from collections import defaultdict
import math

from astropy import constants as const
import numpy as np

import lsstDebug
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base import (
    ArgumentParser,
    CmdLineTask,
    PipelineTask,
    PipelineTaskConfig,
    Struct,
)
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

import lsst.daf.persistence
from lsst.pex.config import Field, ListField, ConfigurableField

from pfs.datamodel import FiberStatus, PfsConfig, Target, TargetType
from pfs.datamodel.pfsFluxReference import PfsFluxReference

from .datamodel import PfsArm, PfsFiberArray, PfsMerged, PfsSimpleSpectrum, PfsSingle
from .datamodel.pfsTargetSpectra import PfsCalibratedSpectra
from .fitFocalPlane import FitFocalPlaneTask
from .fluxCalibrate import fluxCalibrate, FluxCalibrateConnections
from .focalPlaneFunction import FocalPlaneFunction
from .gen3 import readDatasetRefs
from .lsf import warpLsf, Lsf, LsfDict
from .subtractSky1d import subtractSky1d
from .utils import getPfsVersions
from .utils import debugging
from .FluxTableTask import FluxTableTask

from typing import Iterable
from typing import Dict, List

__all__ = ["FitFluxCalConfig", "FitFluxCalTask"]


class FitFluxCalConfig(PipelineTaskConfig, pipelineConnections=FluxCalibrateConnections):
    """Configuration for FitFluxCalTask"""

    sysErr = Field(
        dtype=float,
        default=1.0e-4,
        doc=(
            "Fraction of value to add to variance before fitting. This attempts to offset the "
            "loss of variance as covariance when we resample, the result of which is "
            "underestimated errors and excess rejection."
        ),
    )
    fitFocalPlane = ConfigurableField(target=FitFocalPlaneTask, doc="Fit flux calibration model")
    fluxTable = ConfigurableField(target=FluxTableTask, doc="Flux table")
    doWrite = Field(dtype=bool, default=True, doc="Write outputs?")
    doAdjustCalibVectors = Field(
        dtype=bool,
        default=True,
        doc="Adjust heights of calib vectors to each other before fitting a function to them.",
    )
    adjustCalibVectorsRangeStart = Field(
        dtype=float, default=600, doc="Start of wavelength range to use for height adjustment [nm]."
    )
    adjustCalibVectorsRangeStop = Field(
        dtype=float, default=700, doc="Stop of wavelength range to use for height adjustment [nm]."
    )
    badMask = ListField(dtype=str, default=["BAD", "SAT", "CR", "NO_DATA"], doc="Mask planes for bad pixels")


class FitFluxCalTask(CmdLineTask, PipelineTask):
    """Measure and apply the flux calibration"""

    ConfigClass = FitFluxCalConfig
    _DefaultName = "fitFluxCal"

    fitFocalPlane: FitFocalPlaneTask
    fluxTable: FluxTableTask

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitFocalPlane")
        self.makeSubtask("fluxTable")

        self.debugInfo = lsstDebug.Info(__name__)

    def run(
        self,
        pfsMerged: PfsMerged,
        pfsMergedLsf: LsfDict,
        references: PfsFluxReference,
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
        references : `PfsFluxReference`
            Reference spectra.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        pfsArmList : iterable of `PfsArm`
            List of extracted spectra, for constructing the flux table.
        sky1dList : iterable of `FocalPlaneFunction`
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
        fluxCal = self.calculateCalibrations(pfsConfig, pfsMerged, pfsMergedLsf, references)
        fluxCalibrate(pfsMerged, pfsConfig, fluxCal)

        calibrated = []
        fiberToArm = defaultdict(list)
        for ii, (pfsArm, sky1d) in enumerate(zip(pfsArmList, sky1dList)):
            subtractSky1d(pfsArm, pfsConfig, sky1d)
            fluxCalibrate(pfsArm, pfsConfig, fluxCal)
            for ff in pfsArm.fiberId:
                fiberToArm[ff].append(ii)
            calibrated.append(pfsArm)

        selection = pfsConfig.getSelection(fiberStatus=FiberStatus.GOOD)
        selection &= ~pfsConfig.getSelection(targetType=TargetType.ENGINEERING)
        fiberId = pfsMerged.fiberId[np.isin(pfsMerged.fiberId, pfsConfig.fiberId[selection])]

        pfsCalibrated: Dict[Target, PfsSingle] = {}
        pfsCalibratedLsf: Dict[Target, Lsf] = {}
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
            fluxCal=fluxCal,
            pfsCalibrated=PfsCalibratedSpectra(pfsCalibrated.values()),
            pfsCalibratedLsf=LsfDict(pfsCalibratedLsf),
        )

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
        armInputs = readDatasetRefs(butler, inputRefs, "pfsArm", "sky1d")
        inputs = butler.get(inputRefs)

        outputs = self.run(**inputs, pfsArmList=armInputs.pfsArm, sky1dList=armInputs.sky1d)
        butler.put(outputs, outputRefs)

    @classmethod
    def _makeArgumentParser(cls) -> ArgumentParser:
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(
            name="--id", datasetType="pfsMerged", level="Visit", help="data IDs, e.g. --id exp=12345"
        )
        return parser

    def runDataRef(self, dataRef: lsst.daf.persistence.ButlerDataRef) -> Struct:
        """Measure and apply the flux calibration

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for merged spectrum.

        Returns
        -------
        fluxCal : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.
        pfsCalibrated : `PfsCalibratedSpectra`
            Calibrated spectra.
        pfsCalibratedLsf : `LsfDict`
            Line-spread functions for calibrated spectra.
        """
        pfsMerged = dataRef.get("pfsMerged")
        pfsMergedLsf = dataRef.get("pfsMergedLsf")
        pfsConfig = dataRef.get("pfsConfig")
        references = dataRef.get("pfsFluxReference")

        butler = dataRef.getButler()
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
                self.forceSpectrumToBePersistable(pfsSingle)
                butler.put(pfsSingle, "pfsSingle", dataId)
                butler.put(outputs.pfsCalibratedLsf[target], "pfsSingleLsf", dataId)

        return outputs

    def calculateCalibrations(
        self,
        pfsConfig: PfsConfig,
        pfsMerged: PfsMerged,
        pfsMergedLsf: LsfDict,
        pfsFluxReference: PfsFluxReference,
    ) -> FocalPlaneFunction:
        """Model flux calibration over the focal plane

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            PFS fiber configuration.
        pfsMerged : `PfsMerged`
            Merged spectra, containing observations of ``FLUXSTD`` sources.
        pfsMergedLsf : `LsfDict`
            Line-spread functions for merged spectra.
        pfsFluxReference: `pfs.datamodel.pfsFluxReference.PfsFluxReference`
            Model reference template set for flux calibration.

        Returns
        -------
        fluxCal: `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.
        """
        c = const.c.to("km/s").value

        # We don't need any flux references with any failure flags
        pfsFluxReference = pfsFluxReference[pfsFluxReference.fitFlag == 0]
        if len(pfsFluxReference) == 0:
            raise RuntimeError("No available flux reference (maybe every fitting procecss has failed)")

        # This is going to be (observed spectra) / (reference spectra)
        calibVectors = pfsMerged[np.isin(pfsMerged.fiberId, pfsFluxReference.fiberId)]

        ref = np.empty_like(calibVectors.flux)
        for i, fiberId in enumerate(calibVectors.fiberId):
            refSpec = pfsFluxReference.extractFiber(PfsSimpleSpectrum, pfsConfig, fiberId)

            # We convolve `refSpec` with LSF before resampling
            # because the resampling interval is not short enough
            # compared to `refSpec`'s inherent LSF.
            refLsf = warpLsf(pfsMergedLsf[fiberId], calibVectors.wavelength[i, :], refSpec.wavelength)
            refSpec.flux = refLsf.computeKernel((len(refSpec) - 1) / 2.0).convolve(refSpec.flux)

            # Then we stretch `refSpec` according to its radial velocity.
            # (Resampling takes place in so doing.)
            # The LSF gets slightly wider or narrower by this operation,
            # but we hope it negligible.
            beta = pfsFluxReference.fitParams["radial_velocity"][i].astype(float) / c
            # `refSpec.wavelength[...]` is not mutable. We replace this member.
            refSpec.wavelength = refSpec.wavelength * np.sqrt((1.0 + beta) / (1.0 - beta))
            refSpec = refSpec.resample(calibVectors.wavelength[i, :])

            ref[i, :] = refSpec.flux

        calibVectors.covar[:, 0] += self.config.sysErr * calibVectors.flux  # add systematic error
        calibVectors /= calibVectors.norm
        calibVectors /= ref
        calibVectors.norm[...] = 1.0  # We're deliberately changing the normalisation

        if self.config.doAdjustCalibVectors:
            self.adjustCalibVectors(calibVectors)

        # TODO: Smooth the flux calibration vectors.

        if self.debugInfo.doWriteCalibVector:
            debugging.writeExtraData(
                f"fitFluxCal-output/calibVector-{pfsMerged.filename}.pickle",
                fiberId=calibVectors.fiberId,
                calibVector=calibVectors.flux,
            )

        # Before the call to `fitFocalPlane`, we have to ensure
        # that all the bad flags in `config.mask` are contained in `flags`.
        # This operation modifies `pfsMerged`, but we hope it won't be harmful.
        for name in self.fitFocalPlane.config.mask:
            calibVectors.flags.add(name)

        fluxStdConfig = pfsConfig[np.isin(pfsConfig.fiberId, pfsFluxReference.fiberId)]
        return self.fitFocalPlane.run(calibVectors, fluxStdConfig)

    def adjustCalibVectors(self, calibVectors: PfsMerged) -> None:
        """Adjust heights of calib vectors (observed spectra) / (reference spectra)
        to each other.

        Parameters
        ----------
        calibVectors : `PfsMerged`
            Calib vectors. ``calibVectors.norm`` must be 1 (constant).
        """
        badMask = calibVectors.flags.get(*(m for m in self.config.badMask if m in calibVectors.flags))
        n = len(calibVectors)
        heights = np.empty(shape=(n,), dtype=float)
        for i in range(n):
            selection = (
                (self.config.adjustCalibVectorsRangeStart <= calibVectors.wavelength[i, :])
                & (calibVectors.wavelength[i, :] <= self.config.adjustCalibVectorsRangeStop)
                & (0 == (calibVectors.mask[i, :] & badMask))
            )
            heights[i] = np.nanmedian(calibVectors.flux[i, selection])

        # We use the highest calib vector as the reference to which the other
        # calib vectors are adjusted, for we assume that the fiber of the
        # highest calib vector was positioned better than any other fiber.
        reference = np.nanmax(heights)

        multiplier = (reference / heights).reshape(-1, 1)
        calibVectors *= multiplier
        calibVectors.norm[...] = 1.0

    def forceSpectrumToBePersistable(self, spectrum: PfsFiberArray) -> None:
        """Force ``spectrum`` to be able to be written to file.

        Parameters
        ----------
        spectrum : `PfsFiberArray`
            An observed spectrum.
        """
        if not (math.isfinite(spectrum.target.ra) and math.isfinite(spectrum.target.dec)):
            # Because target's (ra, dec) is written in the FITS header,
            # these values must be finite.
            self.log.warning(
                "Target's (ra, dec) is not finite. Replaced by 0 in the FITS header (%s)",
                spectrum.getIdentity(),
            )
            # Even if ra or dec is finite, we replace both with zero, for
            # (0, 0) looks more alarming than, say, (9.87654321, 0) to users.
            spectrum.target.ra = 0
            spectrum.target.dec = 0

    def _getMetadataName(self) -> None:
        return None
