from typing import Any, Dict, List

import numpy as np

from lsst.pex.config import Config, ConfigurableField, Field, ListField

from lsst.afw.image import Exposure, VisitInfo
from lsst.cp.pipe.cpCombine import CalibCombineConfig, CalibCombineConnections, CalibCombineTask
from lsst.daf.base import PropertyList
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import Task, PipelineTask, PipelineTaskConfig, PipelineTaskConnections, QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from pfs.datamodel import CalibIdentity, PfsConfig

from .adjustDetectorMap import AdjustDetectorMapTask
from .calibs import setCalibHeader
from .centroidTraces import CentroidTracesTask
from .DetectorMapContinued import DetectorMap
from .FiberKernel import fitFiberKernel
from .FiberKernelContinued import FiberKernel
from .fiberProfileSet import FiberProfileSet
from .struct import Struct

__all__ = ("MeasureFiberKernelTask", "ConvolveFiberProfilesTask", "MeasureExposureFiberKernelTask")


class MeasureFiberKernelConfig(Config):
    mask = ListField(
        dtype=str,
        default=["BAD_FLAT", "CR", "SAT", "NO_DATA", "SUSPECT"],
        doc="Mask planes to exclude when fitting the kernel",
    )
    kernelHalfWidth = Field(dtype=int, default=3, doc="Half-width of the fiber kernel in pixels")
    xKernelNum = Field(dtype=int, default=9, doc="Number of kernel blocks in the x-direction")
    yKernelNum = Field(dtype=int, default=9, doc="Number of kernel blocks in the y-direction")
    numRows = Field(
        dtype=int,
        default=0,
        doc="Number of rows to use when fitting the kernel; if 0, use all rows.",
    )
    maxIter = Field(dtype=int, default=20, doc="Maximum number of iterations to run")
    andersonDepth = Field(dtype=int, default=5, doc="Anderson acceleration depth")
    andersonDamping = Field(dtype=float, default=0.25, doc="Anderson acceleration damping parameter")
    fluxTol = Field(
        dtype=float,
        default=1.0e-2,
        doc="Tolerance for change in flux between iterations for convergence",
    )
    lsqThreshold = Field(
        dtype=float,
        default=1.0e-16,
        doc="Threshold for least-squares solution; regularisation is applied to singular values below this",
    )


class MeasureFiberKernelTask(Task):
    ConfigClass = MeasureFiberKernelConfig
    _DefaultName = "measureFiberKernel"

    def run(self, exposure: Exposure, detectorMap: DetectorMap, profiles: FiberProfileSet) -> FiberKernel:
        """Measure the fiber kernel for a single exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            The input exposure to measure the kernel from.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        profiles : `FiberProfileSet`
            Fiber profiles to use for measuring the kernel.

        Returns
        -------
        kernel : `FiberKernel`
            The measured fiber kernel.
        """
        fiberTraces = profiles.makeFiberTracesFromDetectorMap(detectorMap)

        rows = None
        if self.config.numRows > 0:
            rows = np.linspace(0, exposure.getHeight() - 1, self.config.numRows, dtype=np.int32)

        kernel, background = fitFiberKernel(
            exposure.maskedImage,
            fiberTraces,
            exposure.mask.getPlaneBitMask(self.config.mask),
            self.config.kernelHalfWidth,
            self.config.xKernelNum,
            self.config.yKernelNum,
            rows,
            self.config.maxIter,
            self.config.andersonDepth,
            self.config.andersonDamping,
            self.config.fluxTol,
            self.config.lsqThreshold,
        )
        self.log.info("Measured background:\n%s", background.array)
        return FiberKernel(kernel)  # Convert from pybind class to "Continued" class

    def convolveProfiles(
        self,
        kernel: FiberKernel,
        profiles: FiberProfileSet,
        detectorMap: DetectorMap,
        identity: str,
        visitInfo: VisitInfo,
        metadata: PropertyList,
    ) -> FiberProfileSet:
        """Convolve fiber profiles with the kernel

        Parameters
        ----------
        kernel : `FiberKernel`
            The fiber kernel to convolve with the profiles.
        profiles : `FiberProfileSet`
            The fiber profiles to convolve.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        identity : `str`
            Identity of the resultant fiber profiles.
        visitInfo : `VisitInfo`
            VisitInfo to attach to the resultant fiber profiles.
        metadata : `PropertyList`
            Metadata to attach to the resultant fiber profiles.

        Returns
        -------
        convolved : `FiberProfileSet`
            The convolved fiber profiles. The normalization is not preserved:
            you should re-measure the normalization with the convolved profiles.
        """
        convolved = {
            fiberId: kernel.convolveProfile(
                profiles[fiberId], detectorMap.getXCenter(fiberId, profiles[fiberId].rows)
            ) for fiberId in profiles
        }
        return FiberProfileSet(convolved, identity, visitInfo, metadata)


class ConvolveFiberProfilesConnections(
    CalibCombineConnections, dimensions=("instrument", "arm", "spectrograph")
):
    """Pipeline connections for ConvolveFiberProfileTask

    Gen3 middleware pipeline input/output definitions.
    """
    inputExpHandles = InputConnection(
        name="postISRCCD",
        doc="Input exposures",
        storageClass="Exposure",
        dimensions=("visit", "arm", "spectrograph"),
        multiple=True,
        deferLoad=True,
    )
    detectorMap = PrerequisiteConnection(
        name="detectorMap_calib",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        multiple=True,
        isCalibration=True,
    )
    pfsConfig = InputConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit", "pfs_design_id"),
        multiple=True,
    )
    profiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Input fiber profiles for convolution",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        multiple=True,
        isCalibration=True,
    )
    kernel = OutputConnection(
        name="profileKernel",  # Kernel applied to the profile
        doc="Measured convolution kernel",
        storageClass="FiberKernel",
        dimensions=("instrument", "arm", "spectrograph"),
    )
    convolved = OutputConnection(
        name="fiberProfiles_convolved",
        doc="Convolved fiber profile",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )


class ConvolveFiberProfilesConfig(
    CalibCombineConfig, pipelineConnections=ConvolveFiberProfilesConnections
):
    """Configuration for ConvolveFiberProfilesTask"""
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")
    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Adjust the detector map")
    measureFiberKernel = ConfigurableField(target=MeasureFiberKernelTask, doc="Measure the fiber kernel")

    def setDefaults(self):
        super().setDefaults()
        self.calibrationType = "fiberProfiles"


class ConvolveFiberProfilesTask(CalibCombineTask):
    ConfigClass = ConvolveFiberProfilesConfig
    _DefaultName = "convolveFiberProfiles"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("centroidTraces")
        self.makeSubtask("adjustDetectorMap")
        self.makeSubtask("measureFiberKernel")

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
        dataIdList = [handle.dataId for handle in inputRefs.inputExpHandles]
        detector = set((dataId.arm.name, dataId.spectrograph.num) for dataId in dataIdList)
        assert len(detector) == 1
        arm, spectrograph = detector.pop()

        obsDate = min(dataId.timespan.begin.to_datetime().date().isoformat() for dataId in dataIdList)
        visit = min(dataId.visit.id for dataId in dataIdList)
        identity = CalibIdentity(obsDate=obsDate, spectrograph=spectrograph, arm=arm, visit0=visit)

        inputExpHandles = butler.get(inputRefs.inputExpHandles)
        detectorMap = butler.get(inputRefs.detectorMap[0])
        fiberProfiles = butler.get(inputRefs.profiles[0])
        pfsConfig = butler.get(inputRefs.pfsConfig[0])
        outputs = self.run(inputExpHandles, detectorMap, fiberProfiles, pfsConfig, identity, dataIdList)
        butler.put(outputs, outputRefs)

    def run(
        self,
        inputExpHandles: List[DeferredDatasetHandle],
        detectorMap: DetectorMap,
        profiles: FiberProfileSet,
        pfsConfig: PfsConfig,
        identity: CalibIdentity,
        inputDims: List[Dict[str, Any]],
    ) -> Struct:
        """Combine exposures and measure fiber profiles for a single detector

        Parameters
        ----------
        inputExpHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input list of exposure handles to combine.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        profiles : `FiberProfileSet`
            Fiber profiles to convolve with the kernel.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        identity : `CalibIdentity`
            Identity of the resultant fiber profiles.
        inputDims : `list` [`dict`]
            List of dictionaries of input data dimensions/values.
            Each list entry should contain:

            ``"visit"``
                exposure id value (`int`)
            ``"arm"``
                spectrograph arm (`str`)
            ``"spectrograph"``
                spectrograph number (`int`)

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``kernel``
                The measured fiber kernel.
            ``fiberProfiles_convolved``
                Fiber profiles convolved with the kernel.
            ``combined``
                The combined input exposures.
        """
        combined = super().run(inputExpHandles, inputDims).outputData
        combined.getInfo().setVisitInfo(
            min((handle.get(component="visitInfo") for handle in inputExpHandles), key=lambda vi: vi.id)
        )

        traces = self.centroidTraces.run(combined, detectorMap, pfsConfig)
        detectorMap = self.adjustDetectorMap.run(
            detectorMap,
            traces,
            identity.arm,
            combined.visitInfo,
            combined.metadata,
            seed=combined.visitInfo.id,
        ).detectorMap

        kernel = self.measureFiberKernel.run(combined, detectorMap, profiles)
        convolved = self.measureFiberKernel.convolveProfiles(
            kernel, profiles, detectorMap, identity, combined.visitInfo, combined.metadata.deepCopy()
        )

        visitList = [dims["visit"] for dims in inputDims]
        outputId = dict(
            arm=identity.arm,
            spectrograph=identity.spectrograph,
            calibTime=combined.visitInfo.date.toPython().isoformat(),
            calibDate=combined.visitInfo.date.toPython().isoformat().split("T")[0],
            visit0=min(visitList),
        )
        setCalibHeader(convolved.metadata, "fiberProfiles", [dims["visit"] for dims in inputDims], outputId)

        return Struct(
            kernel=kernel,
            convolved=convolved,
            combined=combined,
        )


class MeasureExposureFiberKernelConnnections(
    PipelineTaskConnections, dimensions=("visit", "arm", "spectrograph")
):
    exposure = InputConnection(
        name="calexp",
        doc="Input exposure",
        storageClass="Exposure",
        dimensions=("visit", "spectrograph", "arm"),
    )
    detectorMap = InputConnection(
        name="detectorMap",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("visit", "arm", "spectrograph"),
    )
    profiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Input fiber profiles for convolution",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    kernel = OutputConnection(
        name="kernel",  # Kernel applied to the profile
        doc="Measured convolution kernel",
        storageClass="FiberKernel",
        dimensions=("visit", "arm", "spectrograph"),
    )


class MeasureExposureFiberKernelConfig(
    PipelineTaskConfig, pipelineConnections=MeasureExposureFiberKernelConnnections
):
    """Configuration for MeasureExposureFiberKernelTask"""
    measureFiberKernel = ConfigurableField(target=MeasureFiberKernelTask, doc="Measure the fiber kernel")


class MeasureExposureFiberKernelTask(PipelineTask):
    ConfigClass = MeasureExposureFiberKernelConfig
    _DefaultName = "measureExposureFiberKernel"

    def run(self, exposure, detectorMap, profiles) -> Struct:
        """Measure the fiber kernel for a single exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            The input exposure from which to measure the kernel.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        profiles : `FiberProfileSet`
            Fiber profiles to use for measuring the kernel.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            * ``kernel``: the measured fiber kernel.
        """
        kernel = self.measureFiberKernel.run(exposure, detectorMap, profiles).kernel
        return Struct(kernel=kernel)
