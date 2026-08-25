import numpy as np

import lsstDebug
from lsst.afw.display import Display
from lsst.afw.image import ExposureF, Mask
from lsst.pex.config import ConfigurableField, DictField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.obs.pfs.utils import getLamps
from pfs.datamodel import Identity

from ..adjustDetectorMap import AdjustDetectorMapTask
from ..centroidLines import CentroidLinesTask
from ..centroidSolar import CentroidSolarTask, defaultLsf
from ..centroidTraces import CentroidTracesTask
from ..datamodel import PfsConfig
from ..DetectorMapContinued import DetectorMap
from ..extractSpectraTask import ExtractSpectraTask
from ..fiberProfileSet import FiberProfileSet
from ..fitDetectorMap import FittingError
from ..readLineList import ReadLineListTask
from ..referenceLine import ReferenceLineSet

__all__ = ("MeasureCentroidsTask", "MeasureDetectorMapTask", "MeasureExposureCentroidsTask")


class MeasureCentroidsConnections(
    PipelineTaskConnections, dimensions=("instrument", "visit", "arm", "spectrograph")
):
    """Connections for MeasureCentroidsTask"""

    exposure = InputConnection(
        name="postISRCCD",
        doc="Input ISR-corrected exposure",
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

    calibDetectorMap = PrerequisiteConnection(
        name="detectorMap_calib",
        doc="Mapping from fiberId,wavelength to x,y: measured from real data",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )

    centroids = OutputConnection(
        name="centroids",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config:
            return
        if not self.config.doApplyCrMask:
            self.prerequisiteInputs.remove("crMask")


class MeasureCentroidsConfig(PipelineTaskConfig, pipelineConnections=MeasureCentroidsConnections):
    """Configuration for MeasureCentroidsTask"""

    doApplyCrMask = Field(dtype=bool, default=True, doc="Apply cosmic-ray mask to input exposure?")
    readLineList = ConfigurableField(
        target=ReadLineListTask, doc="Read line lists for detectorMap adjustment"
    )
    doForceTraces = Field(dtype=bool, default=True, doc="Force use of traces for non-continuum data?")
    centroidLines = ConfigurableField(target=CentroidLinesTask, doc="Centroid lines")
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")


class MeasureCentroidsTask(PipelineTask):
    """Measure centroids on an exposure"""

    ConfigClass = MeasureCentroidsConfig
    _DefaultName = "measureCentroids"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)
        self.makeSubtask("readLineList")
        self.makeSubtask("centroidLines")
        self.makeSubtask("centroidTraces")

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        inputs = butler.get(inputRefs)
        inputs["detectorMap"] = inputs.pop("calibDetectorMap")

        outputs = self.run(**inputs)
        butler.put(outputs, outputRefs)
        return outputs

    def run(
        self,
        exposure: ExposureF,
        pfsConfig: PfsConfig,
        detectorMap: DetectorMap,
        crMask: Mask | None = None,
    ):
        """Measure (both line and trace) centroids on an exposure

        Parameters
        ----------
        exposure : `ExposureF`
            Exposure from which to measure centroids.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        crMask : `Mask`, optional
            Cosmic-ray mask.

        Returns
        -------
        refLines : `pfs.drp.stella.referenceLine.ReferenceLineSet`
            Reference lines.
        centroids : `ArcLineSet`
            Measured centroids.
        """
        if self.config.doApplyCrMask:
            if not crMask:
                raise ValueError("Cosmic-ray mask required but not provided")
            exposure.mask |= crMask
        refLines = self.readLineList.run(detectorMap, exposure.getMetadata())
        lines = self.centroidLines.run(exposure, refLines, detectorMap, pfsConfig, seed=exposure.visitInfo.id)
        if self.config.doForceTraces or not lines:
            traces = self.centroidTraces.run(exposure, detectorMap, pfsConfig)
            lines.extend(traces)
        return Struct(exposure=exposure, refLines=refLines, centroids=lines)


class MeasureDetectorMapConnections(MeasureCentroidsConnections):
    """Connections for MeasureDetectorMapTask"""

    outputDetectorMap = OutputConnection(
        name="detectorMap",
        doc="Corrected mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class MeasureDetectorMapConfig(MeasureCentroidsConfig, pipelineConnections=MeasureDetectorMapConnections):
    """Configuration for MeasureDetectorMapTask"""

    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Measure slit offsets")
    requireAdjustDetectorMap = Field(
        dtype=bool, default=False, doc="Require detectorMap adjustment to succeed?"
    )


class MeasureDetectorMapTask(MeasureCentroidsTask):
    """Measure centroids on a single exposure and adjust the detectorMap"""

    ConfigClass = MeasureDetectorMapConfig
    _DefaultName = "measureDetectorMap"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("adjustDetectorMap")

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        inputs = butler.get(inputRefs)
        inputs["detectorMap"] = inputs.pop("calibDetectorMap")

        arm = inputRefs.exposure.dataId.arm.name
        assert arm in "brnm"

        outputs = self.run(**inputs, arm=arm)
        butler.put(outputs.centroids, outputRefs.centroids)
        butler.put(outputs.detectorMap, outputRefs.outputDetectorMap)
        return outputs

    def run(
        self,
        exposure: ExposureF,
        pfsConfig: PfsConfig,
        detectorMap: DetectorMap,
        arm: str,
        crMask: Mask | None = None,
    ):
        """Measure centroids on a single exposure and adjust the detectorMap

        Parameters
        ----------
        exposure : `ExposureF`
            Exposure from which to measure centroids.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        arm : `str`
            Spectrograph arm in use (``b``, ``r``, ``n``, ``m``).
        crMask : `Mask`, optional
            Cosmic-ray mask.

        Returns
        -------
        refLines : `pfs.drp.stella.referenceLine.ReferenceLineSet`
            Reference lines.
        centroids : `ArcLineSet`
            Measured centroids.
        detectorMap : `DetectorMap`
            Adjusted mapping of fiberId,wavelength to x,y.
        """
        data = super().run(exposure, pfsConfig, detectorMap, crMask=crMask)

        if self.debugInfo.detectorMap:
            display = Display(frame=1)
            display.mtv(exposure)
            detectorMap.display(display, wavelengths=data.refLines.wavelength, ctype="red", plotTraces=False)

        try:
            detectorMap = self.adjustDetectorMap.run(
                detectorMap,
                data.centroids,
                arm,
                exposure.visitInfo,
                exposure.metadata,
                exposure.visitInfo.id,
            ).detectorMap
        except FittingError as exc:
            if self.config.requireAdjustDetectorMap:
                raise
            self.log.warn("DetectorMap adjustment failed: %s", exc)

        if self.debugInfo.detectorMap:
            detectorMap.display(
                display, wavelengths=data.refLines.wavelength, ctype="green", plotTraces=False
            )

        data.detectorMap = detectorMap
        return data


class MeasureExposureCentroidsConnections(MeasureCentroidsConnections):
    """Connections for MeasureExposureCentroidsTask"""

    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Profile of fibers",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )


class MeasureExposureCentroidsConfig(
    MeasureCentroidsConfig, pipelineConnections=MeasureExposureCentroidsConnections
):
    """Configuration for MeasureExposureCentroidsTask"""

    extractSpectra = ConfigurableField(
        target=ExtractSpectraTask, doc="Extract spectra for twilight centroiding"
    )
    centroidSolar = ConfigurableField(
        target=CentroidSolarTask, doc="Centroid twilight (solar-absorption) lines"
    )
    adjustDetectorMap = ConfigurableField(
        target=AdjustDetectorMapTask,
        doc="Adjust detectorMap to this exposure's traces before extracting twilight spectra",
    )
    requireAdjustDetectorMap = Field(
        dtype=bool, default=False, doc="Require detectorMap adjustment to succeed?"
    )
    gaussianLsfWidth = DictField(
        keytype=str,
        itemtype=float,
        doc="Gaussian sigma (nm) for LSF as a function of the spectrograph arm",
        default=dict(b=0.081, r=0.109, m=0.059, n=0.109),
    )


class MeasureExposureCentroidsTask(MeasureCentroidsTask):
    """Measure centroids on an exposure, dispatching on lamp state

    This differs from `MeasureCentroidsTask` in that it distinguishes three
    kinds of exposure by the lamps that were lit (`lsst.obs.pfs.utils.getLamps`):
    quartz (trace-only), arc (line list + centroiding), and twilight (solar
    absorption centroiding, `CentroidSolarTask`).
    """

    ConfigClass = MeasureExposureCentroidsConfig
    _DefaultName = "measureExposureCentroids"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("extractSpectra")
        self.makeSubtask("centroidSolar")
        self.makeSubtask("adjustDetectorMap")

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        inputs = butler.get(inputRefs)
        inputs["detectorMap"] = inputs.pop("calibDetectorMap")

        arm = inputRefs.exposure.dataId.arm.name
        spectrograph = inputRefs.exposure.dataId.spectrograph.num
        assert arm in "brnm"

        outputs = self.run(**inputs, arm=arm, spectrograph=spectrograph)
        butler.put(outputs, outputRefs)
        return outputs

    def run(
        self,
        exposure: ExposureF,
        pfsConfig: PfsConfig,
        detectorMap: DetectorMap,
        arm: str,
        spectrograph: int,
        fiberProfiles: FiberProfileSet | None = None,
        crMask: Mask | None = None,
    ):
        """Measure centroids on an exposure, dispatching on lamp state

        Parameters
        ----------
        exposure : `ExposureF`
            Exposure from which to measure centroids.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        arm : `str`
            Spectrograph arm in use (``b``, ``r``, ``n``, ``m``).
        spectrograph : `int`
            Spectrograph module number.
        fiberProfiles : `FiberProfileSet`, optional
            Profiles of fibers, needed for the twilight branch.
        crMask : `Mask`, optional
            Cosmic-ray mask.

        Returns
        -------
        refLines : `pfs.drp.stella.referenceLine.ReferenceLineSet`
            Reference lines.
        centroids : `ArcLineSet`
            Measured centroids.
        """
        if self.config.doApplyCrMask:
            if not crMask:
                raise ValueError("Cosmic-ray mask required but not provided")
            exposure.mask |= crMask

        lamps = getLamps(exposure.getMetadata())
        if lamps == {"Quartz"}:
            lines = self.centroidTraces.run(exposure, detectorMap, pfsConfig)
            return Struct(exposure=exposure, refLines=ReferenceLineSet.empty(), centroids=lines)

        if lamps:
            refLines = self.readLineList.run(detectorMap, exposure.getMetadata())
            lines = self.centroidLines.run(
                exposure, refLines, detectorMap, pfsConfig, seed=exposure.visitInfo.id
            )
            if self.config.doForceTraces or not lines:
                traces = self.centroidTraces.run(exposure, detectorMap, pfsConfig)
                lines.extend(traces)
            return Struct(exposure=exposure, refLines=refLines, centroids=lines)

        return self.runTwilight(exposure, pfsConfig, detectorMap, arm, spectrograph, fiberProfiles)

    def runTwilight(
        self,
        exposure: ExposureF,
        pfsConfig: PfsConfig,
        detectorMap: DetectorMap,
        arm: str,
        spectrograph: int,
        fiberProfiles: FiberProfileSet,
    ):
        """Measure solar-absorption centroids on a twilight exposure

        Parameters
        ----------
        exposure : `ExposureF`
            Twilight exposure from which to measure centroids.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        detectorMap : `DetectorMap`
            Calibration mapping of fiberId,wavelength to x,y.
        arm : `str`
            Spectrograph arm in use (``b``, ``r``, ``n``, ``m``).
        spectrograph : `int`
            Spectrograph module number.
        fiberProfiles : `FiberProfileSet`
            Profiles of fibers.

        Returns
        -------
        refLines : `pfs.drp.stella.referenceLine.ReferenceLineSet`
            Reference lines (empty; twilight centroiding doesn't use a line list).
        centroids : `ArcLineSet`
            Measured (solar-absorption) centroids.
        """
        visitInfo = exposure.visitInfo
        seed = visitInfo.id

        traces = self.centroidTraces.run(exposure, detectorMap, pfsConfig)
        try:
            detectorMap = self.adjustDetectorMap.run(
                detectorMap, traces, arm, visitInfo, exposure.metadata, seed=seed
            ).detectorMap
        except FittingError as exc:
            if self.config.requireAdjustDetectorMap:
                raise
            self.log.warn("DetectorMap adjustment failed: %s", exc)

        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)

        identity = Identity(
            visit=visitInfo.id,
            arm=arm,
            spectrograph=spectrograph,
            pfsDesignId=pfsConfig.pfsDesignId,
            obsTime=visitInfo.date.toString(visitInfo.date.TAI),
            expTime=visitInfo.exposureTime,
        )
        fiberId = np.array(sorted(set(pfsConfig.fiberId) & set(detectorMap.fiberId)))
        spectra = self.extractSpectra.run(exposure.maskedImage, fiberTraces, detectorMap, fiberId).spectra
        pfsArm = spectra.toPfsArm(identity)

        lsf = defaultLsf(arm, pfsArm.fiberId, detectorMap, self.config.gaussianLsfWidth)
        lines = self.centroidSolar.run(pfsArm, pfsConfig, detectorMap, lsf, visitInfo).lines
        return Struct(exposure=exposure, refLines=ReferenceLineSet.empty(), centroids=lines)
