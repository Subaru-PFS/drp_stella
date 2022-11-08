import lsstDebug
from lsst.afw.display import Display
from lsst.afw.image import ExposureF
from lsst.obs.pfs.utils import getLampElements
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
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


class MeasureCentroidsConnections(PipelineTaskConnections, dimensions=("instrument", "exposure", "detector")):
    """Connections for MeasureCentroidsTask"""

    exposure = InputConnection(
        name="postISRCCD",
        doc="Input ISR-corrected exposure",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "detector"),
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )

    # We'll choose one based on the config parameter 'useBootstrapDetectorMap'
    bootstrapDetectorMap = PrerequisiteConnection(
        name="detectorMap_bootstrap",
        doc="Mapping from fiberId,wavelength to x,y: derived from instrument model",
        storageClass="DetectorMap",
        dimensions=("instrument", "detector"),
    )
    calibDetectorMap = PrerequisiteConnection(
        name="detectorMap",
        doc="Mapping from fiberId,wavelength to x,y: measured from real data",
        storageClass="DetectorMap",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )

    centroids = OutputConnection(
        name="centroids",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "exposure", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config:
            return
        if config.useBootstrapDetectorMap:
            self.prerequisiteInputs.remove("calibDetectorMap")
        else:
            self.prerequisiteInputs.remove("bootstrapDetectorMap")


class MeasureCentroidsConfig(PipelineTaskConfig, pipelineConnections=MeasureCentroidsConnections):
    """Configuration for MeasureCentroidsTask"""

    useBootstrapDetectorMap = Field(dtype=bool, default=False, doc="Use bootstrap detectorMap?")
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
        butler: ButlerQuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
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
        inputs = butler.get(inputRefs)
        inputs["detectorMap"] = inputs.pop(
            ("bootstrap" if self.config.useBootstrapDetectorMap else "calib") + "DetectorMap"
        )
        outputs = self.run(**inputs)
        butler.put(outputs.centroids, outputRefs.centroids)
        return outputs

    def run(self, exposure: ExposureF, pfsConfig: PfsConfig, detectorMap: DetectorMap):
        """Measure (both line and trace) centroids on an exposure

        Parameters
        ----------
        exposure : `ExposureF`
            Exposure from which to measure centroids.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.

        Returns
        -------
        refLines : `pfs.drp.stella.referenceLine.ReferenceLineSet`
            Reference lines.
        centroids : `ArcLineSet`
            Measured centroids.
        """
        refLines = self.readLineList.run(detectorMap, exposure.getMetadata())
        lines = self.centroidLines.run(exposure, refLines, detectorMap, pfsConfig)
        if self.config.doForceTraces or not lines or "Continuum" in getLampElements(exposure.getMetadata()):
            traces = self.centroidTraces.run(exposure, detectorMap, pfsConfig)
            lines.extend(tracesToLines(detectorMap, traces, self.config.traceSpectralError))
        return Struct(refLines=refLines, centroids=lines)


class MeasureDetectorMapConnections(MeasureCentroidsConnections):
    """Connections for MeasureDetectorMapTask"""

    outputDetectorMap = OutputConnection(
        name="detectorMap_used",
        doc="Corrected mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "exposure", "detector"),
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
        butler: ButlerQuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        outputs = super().runQuantum(butler, inputRefs, outputRefs)
        butler.put(outputs.detectorMap, outputRefs.outputDetectorMap)

    def run(self, exposure: ExposureF, pfsConfig: PfsConfig, detectorMap: DetectorMap):
        """Measure centroids on a single exposure and adjust the detectorMap

        Parameters
        ----------
        exposure : `ExposureF`
            Exposure from which to measure centroids.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.

        Returns
        -------
        refLines : `pfs.drp.stella.referenceLine.ReferenceLineSet`
            Reference lines.
        centroids : `ArcLineSet`
            Measured centroids.
        detectorMap : `DetectorMap`
            Adjusted mapping of fiberId,wavelength to x,y.
        """
        data = super().run(exposure, pfsConfig, detectorMap)

        if self.debugInfo.detectorMap:
            display = Display(frame=1)
            display.mtv(exposure)
            detectorMap.display(display, wavelengths=data.refLines.wavelength, ctype="red", plotTraces=False)

        try:
            detectorMap = self.adjustDetectorMap.run(
                detectorMap, data.centroids, exposure.visitInfo.id
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
