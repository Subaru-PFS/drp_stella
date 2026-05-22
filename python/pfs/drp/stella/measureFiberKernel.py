from typing import Any, Dict, List

import numpy as np
import astropy.io.fits

from lsst.pex.config import ConfigurableField, Field, ListField

from lsst.cp.pipe.cpCombine import CalibCombineConfig, CalibCombineConnections, CalibCombineTask
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
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

__all__ = ("MeasureFiberKernelTask", "ExposureFiberNormsTask")


class MeasureFiberKernelConnections(
    CalibCombineConnections, dimensions=("instrument", "arm", "spectrograph")
):
    """Pipeline connections for MeasureFiberKernelTask

    Gen3 middleware pipeline input/output definitions.
    """
    inputExpHandles = InputConnection(
        name="postISRCCD",
        doc="Input exposures",
        storageClass="Exposure",
        dimensions=("visit", "spectrograph", "arm", "spectrograph"),
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


class MeasureFiberKernelConfig(
    CalibCombineConfig, pipelineConnections=MeasureFiberKernelConnections
):
    """Configuration for MeasureFiberKernelTask"""
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")
    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Adjust the detector map")
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

    def setDefaults(self):
        super().setDefaults()
        self.calibrationType = "fiberProfiles"


class MeasureFiberKernelTask(CalibCombineTask):
    ConfigClass = MeasureFiberKernelConfig
    _DefaultName = "measureFiberKernel"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("centroidTraces")
        self.makeSubtask("adjustDetectorMap")

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
    ):
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

        fiberTraces = profiles.makeFiberTracesFromDetectorMap(detectorMap)

        rows = None
        if self.config.numRows > 0:
            rows = np.linspace(0, combined.getHeight() - 1, self.config.numRows, dtype=np.int32)

        kernel, background = fitFiberKernel(
            combined.maskedImage,
            fiberTraces,
            combined.mask.getPlaneBitMask(self.config.mask),
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
        kernel = FiberKernel(kernel)  # Convert from pybind class to "Continued" class
        self.log.info("Measured background:\n%s", background.array)

        convolved = {
            fiberId: kernel.convolveProfile(
                profiles[fiberId], detectorMap.getXCenter(fiberId, profiles[fiberId].rows)
            ) for fiberId in profiles
        }

        visitList = [dims["visit"] for dims in inputDims]
        outputId = dict(
            arm=identity.arm,
            spectrograph=identity.spectrograph,
            calibTime=combined.visitInfo.date.toPython().isoformat(),
            calibDate=combined.visitInfo.date.toPython().isoformat().split("T")[0],
            visit0=min(visitList),
        )
        header = combined.metadata.deepCopy()
        setCalibHeader(header, "fiberProfile", [dims["visit"] for dims in inputDims], outputId)

        return Struct(
            kernel=kernel,
            fiberProfiles_convolved=FiberProfileSet(convolved, identity, combined.visitInfo, header),
            combined=combined,
        )
