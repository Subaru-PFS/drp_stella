from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.signal as signal

from lsst.pex.config import Config, ConfigurableField, DictField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.daf.butler import DataCoordinate
from lsst.afw.image import Exposure, Image, ImageF, Mask, MaskedImage

from .pipelines.lookups import lookupFiberNorms
from .FiberTraceContinued import FiberTrace
from .FiberTraceSetContinued import FiberTraceSet
from .SpectrumSetContinued import SpectrumSet
from .fiberProfileSet import FiberProfileSet
from .datamodel.pfsConfig import PfsConfig
from pfs.datamodel import FiberStatus, TargetType, Identity
from .DetectorMapContinued import DetectorMap
from .reduceExposure import ReduceExposureConfig, ReduceExposureTask
from .scatteredLight import ScatteredLightTask

if TYPE_CHECKING:
    from .datamodel import PfsArm
    from .DetectorMapContinued import DetectorMap


__all__ = ("ScatteredLightTask", "ScatteredLightConfig", "ScatteredLightModel")


class ScatteredLightRemovalConnections(
    PipelineTaskConnections, dimensions=("instrument", "visit", "arm", "spectrograph")
):
    exposure = InputConnection(
        name="postISRCCD",
        doc="Exposure to reduce",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    crMask = InputConnection(
        name="crMask",
        doc="Cosmic-ray mask",
        storageClass="Mask",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Profile of fibers",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    detectorMap = PrerequisiteConnection(
        name="detectorMap_calib",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )

    outputExposure = OutputConnection(
        name="calexp",
        doc="Calibrated exposure",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config:
            return
        if not self.config.doApplyCrMask:
            self.prerequisiteInputs.remove("crMask")
        if self.config.doBoxcarExtraction:
            self.prerequisiteInputs.remove("fiberProfiles")


class ScatteredLightRemovalConfig(ReduceExposureConfig,
                                  pipelineConnections=ScatteredLightRemovalConnections):
    doApplyCrMask = Field(dtype=bool, default=True, doc="Apply cosmic-ray mask to input exposure?")
    doBoxcarExtraction = Field(dtype=bool, default=False, doc="Extract with a boxcar of width boxcarWidth")
    boxcarWidth = Field(dtype=float, default=5,
                        doc="Extract with a boxcar of width boxcarWidth if doBoxcarExtraction is True")
    doScatteredLight = Field(dtype=bool, default=True, doc="Apply scattered light correction?")
    scatteredLight = ConfigurableField(target=ScatteredLightTask, doc="Scattered light correction")


class ScatteredLightRemovalTask(ReduceExposureTask):
    ConfigClass = ScatteredLightRemovalConfig
    _DefaultName = "scatteredLightRemoval"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("scatteredLight")
    
    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection
    ) -> None:
        inputs = butler.get(inputRefs)
        dataId = inputRefs.exposure.dataId
        if self.config.doBoxcarExtraction:
            inputs["fiberProfiles"] = None
            inputs["fiberNorms"] = None
        outputs = self.run(**inputs, dataId=dataId)
        butler.put(outputs, outputRefs)
        return outputs

    def run(
        self,
        exposure: Exposure,
        pfsConfig: PfsConfig,
        fiberProfiles: FiberProfileSet | None,
        detectorMap: DetectorMap,
        dataId: dict[str, str] | DataCoordinate,
        crMask: Mask | None = None,
    ) -> Struct:
        """Process an arm exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure data to reduce.
        pfsConfig : `pfs.datamodel.PfsConfig`
            PFS fiber configuration.
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of fibers.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        dataId : `dict` [`str`, `str`] or `DataCoordinate`
            Data identifier.
        crMask : `lsst.afw.image.Mask`, optional
            Cosmic-ray mask.

        Returns
        -------
        pfsArm : `pfs.drp.stella.datamodel.PfsArm`
            Extracted spectra.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        """

        if self.config.doApplyCrMask:
            if crMask is None:
                raise RuntimeError("crMask required but not provided")
            exposure.mask |= crMask

        if not config.doScatteredLight:
            return Struct(
                outputExposure=exposure,
            )

        arm = dataId["arm"]
        spectrograph = dataId["spectrograph"]
        visitInfo = exposure.visitInfo
        identity = Identity(
            visit=dataId["visit"],
            arm=arm,
            spectrograph=spectrograph,
            pfsDesignId=dataId["pfs_design_id"],
            obsTime=visitInfo.date.toString(visitInfo.date.TAI),
            expTime=visitInfo.exposureTime,
        )

        spatialOffset = self.config.spatialOffset
        spectralOffset = self.config.spectralOffset
        if spatialOffset != 0.0 or spectralOffset != 0.0:
            self.log.info("Adjusting detectorMap slit offset by %f,%f", spatialOffset, spectralOffset)
            detectorMap.applySlitOffset(spatialOffset, spectralOffset)

        check = self.checkPfsConfig(pfsConfig, detectorMap, spectrograph)
        pfsConfig = check.pfsConfig
        boxcarWidth = check.boxcarWidth

        if boxcarWidth > 0:
            fiberProfiles = FiberProfileSet.makeEmpty(None)
            for fid in pfsConfig.fiberId:
                # the Gaussian will be replaced by a boxcar, so params don't matter
                fiberProfiles[fid] = FiberProfile.makeGaussian(1, exposure.getHeight(), 5, 1)

        fiberId = np.array(sorted(set(pfsConfig.fiberId) & set(detectorMap.fiberId)))
        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap, boxcarWidth)

        spectra = self.extractSpectra.run(
            exposure.maskedImage,
            fiberTraces,
            detectorMap,
            fiberId,
            True if boxcarWidth > 0 else False,
        ).spectra

        pfsArm = spectra.toPfsArm(identity)
        self.scatteredLight.run(exposure.maskedImage, pfsArm, detectorMap)

        return Struct(
            outputExposure=exposure,
        )
