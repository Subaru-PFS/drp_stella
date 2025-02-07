from typing import Any, Dict, List

from lsst.cp.pipe.cpCombine import CalibCombineConfig, CalibCombineConnections, CalibCombineTask
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pex.config import ConfigurableField, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from pfs.datamodel import CalibIdentity, PfsConfig, FiberStatus, TargetType

from ..blackSpotCorrection import BlackSpotCorrectionTask
from ..buildFiberProfiles import BuildFiberProfilesTask
from ..DetectorMapContinued import DetectorMap
from ..fiberProfileSet import FiberProfileSet

__all__ = ("MeasureFiberProfilesTask", "CombineFiberProfilesTask")


class MeasureFiberProfilesConnections(
    CalibCombineConnections, dimensions=("instrument", "arm", "spectrograph", "pfs_design_id")
):
    """Connections for MeasureFiberProfilesTask"""

    inputExpHandles = InputConnection(
        name="calexp",
        doc="Input combined profiles.",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "spectrograph", "arm", "spectrograph"),
        multiple=True,
        deferLoad=True,
    )
    detectorMap = InputConnection(
        name="detectorMap",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit", "pfs_design_id"),
        multiple=True,
    )

    outputData = OutputConnection(
        name="fiberProfiles_subset",
        doc="Output subset of fiber profiles.",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph", "pfs_design_id"),
    )


class MeasureFiberProfilesConfig(CalibCombineConfig, pipelineConnections=MeasureFiberProfilesConnections):
    """Configuration for MeasureFiberProfilesTask"""

    profiles = ConfigurableField(target=BuildFiberProfilesTask, doc="Build fiber profiles")
    blackspots = ConfigurableField(target=BlackSpotCorrectionTask, doc="Black spot correction")
    fiberStatus = ListField(
        dtype=str,
        default=["GOOD", "BROKENFIBER"],
        doc="Fiber status for which to build profiles",
    )
    targetType = ListField(
        dtype=str,
        default=["SCIENCE", "SKY", "FLUXSTD", "UNASSIGNED", "SUNSS_IMAGING", "SUNSS_DIFFUSE", "HOME"],
        doc="Target type for which to build profiles",
    )

    def setDefaults(self):
        super().setDefaults()
        self.calibrationType = "fiberProfiles"
        self.profiles.profileRadius = 2  # Full fiber density, so can't go out very wide
        self.profiles.mask = ["BAD", "SAT", "CR", "INTRP", "BAD_FLAT", "NO_DATA", "SUSPECT"]
        self.profiles.doBlindFind = False
        self.mask = ["BAD", "SAT", "CR", "INTRP", "BAD_FLAT", "NO_DATA", "SUSPECT"]


class MeasureFiberProfilesTask(CalibCombineTask):
    ConfigClass = MeasureFiberProfilesConfig
    _DefaultName = "fiberProfilesCombine"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("profiles")
        self.makeSubtask("blackspots")

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
        detectorMap = butler.get(min(inputRefs.detectorMap, key=lambda ref: ref.dataId["visit"]))
        pfsConfig = butler.get(inputRefs.pfsConfig[0])
        outputs = self.run(inputExpHandles, detectorMap, pfsConfig, identity, dataIdList)
        butler.put(outputs, outputRefs)

    def run(
        self,
        inputExpHandles: List[DeferredDatasetHandle],
        detectorMap: DetectorMap,
        pfsConfig: PfsConfig,
        identity: CalibIdentity,
        inputDims: List[Dict[str, Any]] = None,
    ):
        """Combine exposures and measure fiber profiles for a single detector

        Parameters
        ----------
        inputExpHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Input list of exposure handles to combine.
        detectorMap : `DetectorMap`
            Mapping from fiberId,wavelength to x,y.
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

            ``outputData``
                Fiber profiles measured from the combined image.

        Raises
        ------
        RuntimeError
            Raised if no input data is found.  Also raised if
            ``config.exposureScaling == InputList``, and a necessary scale
            was not found.
        """
        combined = super().run(inputExpHandles, inputDims).outputData
        combined.getInfo().setVisitInfo(
            min((handle.get(component="visitInfo") for handle in inputExpHandles), key=lambda vi: vi.id)
        )

        fiberStatus = [FiberStatus.fromString(fs) for fs in self.config.fiberStatus]
        targetType = [TargetType.fromString(tt) for tt in self.config.targetType]
        pfsConfig = pfsConfig.select(
            fiberStatus=fiberStatus, targetType=targetType, fiberId=detectorMap.fiberId
        )

        profileData = self.profiles.run(
            combined, identity=identity, detectorMap=detectorMap, pfsConfig=pfsConfig
        )
        profiles = profileData.profiles
        if len(profiles) == 0:
            raise RuntimeError("No profiles found")
        self.log.info("%d fiber profiles found", len(profiles))

        # Set the normalisation of the FiberProfiles
        # The normalisation is the flat: we want extracted spectra to be relative to the flat.

        return Struct(outputData=profiles)


class CombineFiberProfilesConnections(
    PipelineTaskConnections, dimensions=("instrument", "arm", "spectrograph")
):
    """Connections for CombineFiberProfilesTask"""

    profiles = InputConnection(
        name="fiberProfiles_subset",
        doc="Input fiber profile subsets.",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph", "pfs_design_id"),
        multiple=True,
    )
    merged = OutputConnection(
        name="fiberProfiles",
        doc="Combined fiber profiles.",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )


class CombineFiberProfilesConfig(PipelineTaskConfig, pipelineConnections=CombineFiberProfilesConnections):
    """Configuration for CombineFiberProfilesTask"""

    pass


class CombineFiberProfilesTask(PipelineTask):
    """Combine fiber profiles

    We produce fiber profiles with different ``pfs_design_id`` dimensions,
    corresponding to different sets of fibers hiding behind black spots. These
    different profiles need to be merged to produce a single set of profiles for
    all the fibers.
    """

    ConfigClass = CombineFiberProfilesConfig
    _DefaultName = "mergeFiberProfilesGen3"

    def run(self, profiles):
        """Combine fiber profiles

        Parameters
        ----------
        profiles : iterable of `FiberProfileSet`
            List of fiber profiles.

        Returns
        -------
        merged : `FiberProfileSet`
            Combined set of fiber profiles.
        """
        merged = FiberProfileSet.fromCombination(*profiles)
        self.log.info("Combined %d profiles", len(merged))
        return Struct(merged=merged)
