import lsstDebug
from lsst.afw.display import Display
from lsst.afw.image import ExposureF, Mask
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from ..adjustDetectorMap import AdjustDetectorMapTask
from ..centroidLines import CentroidLinesTask
from ..centroidTraces import CentroidTracesTask, tracesToLines
from ..datamodel import PfsConfig
from ..DetectorMapContinued import DetectorMap
from ..fitDistortedDetectorMap import FittingError
from ..readLineList import ReadLineListTask

__all__ = ("MeasureCentroidsTask", "MeasureDetectorMapTask")


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
    traceSpectralError = Field(
        dtype=float, default=1.0, doc="Error in the spectral dimension to give trace centroids (pixels)"
    )


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
            lines.extend(tracesToLines(detectorMap, traces, self.config.traceSpectralError))
        return Struct(refLines=refLines, centroids=lines)


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
                detectorMap, data.centroids, arm, exposure.visitInfo.id
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
