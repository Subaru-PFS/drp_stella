from typing import Any, Dict, List

from lsst.cp.pipe.cpCombine import CalibCombineConfig, CalibCombineConnections, CalibCombineTask
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pex.config import ConfigurableField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.butlerQuantumContext import ButlerQuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from pfs.datamodel import CalibIdentity, PfsConfig

from ..buildFiberProfiles import BuildFiberProfilesTask
from ..DetectorMapContinued import DetectorMap
from ..fiberProfileSet import FiberProfileSet

__all__ = ("MeasureFiberProfilesTask", "MergeFiberProfilesTask")


class MeasureFiberProfilesConnections(
    CalibCombineConnections, dimensions=("instrument", "detector", "pfs_design_id")
):
    """Connections for MeasureFiberProfilesTask"""

    inputExpHandles = InputConnection(
        name="fiberProfilesProc",
        doc="Input combined profiles.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "spectrograph", "arm", "detector"),
        multiple=True,
        deferLoad=True,
    )
    detectorMap = InputConnection(
        name="detectorMap_used",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "exposure", "detector"),
        multiple=True,
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )

    outputData = OutputConnection(
        name="fiberProfiles_subset",
        doc="Output subset of fiber profiles.",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "detector", "pfs_design_id"),
    )


class MeasureFiberProfilesConfig(CalibCombineConfig, pipelineConnections=MeasureFiberProfilesConnections):
    """Configuration for MeasureFiberProfilesTask"""

    profiles = ConfigurableField(target=BuildFiberProfilesTask, doc="Build fiber profiles")

    def setDefaults(self):
        super().setDefaults()
        self.profiles.profileRadius = 2  # Full fiber density, so can't go out very wide
        self.profiles.mask = ["BAD", "SAT", "CR", "INTRP", "NO_DATA"]
        self.mask = ["BAD", "SAT", "CR", "INTRP", "NO_DATA"]


class MeasureFiberProfilesTask(CalibCombineTask):
    ConfigClass = MeasureFiberProfilesConfig
    _DefaultName = "fiberProfilesCombine"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("profiles")

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
        dataIdList = [handle.dataId for handle in inputRefs.inputExpHandles]
        expRecords = [
            next(iter(butler.registry.queryDimensionRecords("exposure", dataId=dataId)))
            for dataId in dataIdList
        ]
        detRecords = (
            next(iter(butler.registry.queryDimensionRecords("detector", dataId=dataId)))
            for dataId in dataIdList
        )
        detector = set((rr.arm, rr.spectrograph) for rr in detRecords)
        assert len(detector) == 1
        arm, spectrograph = detector.pop()
        inputDims = [dataId.byName() for dataId in dataIdList]

        obsDate = min(rr.timespan.begin.to_datetime().date().isoformat() for rr in expRecords)
        visit = min(rr.id for rr in expRecords)
        identity = CalibIdentity(obsDate=obsDate, spectrograph=spectrograph, arm=arm, visit0=visit)

        inputExpHandles = butler.get(inputRefs.inputExpHandles)
        detectorMap = butler.get(min(inputRefs.detectorMap, key=lambda ref: ref.dataId["exposure"]))
        pfsConfig = butler.get(inputRefs.pfsConfig)
        outputs = self.run(inputExpHandles, detectorMap, pfsConfig, identity, inputDims)
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

            ``"exposure"``
                exposure id value (`int`)
            ``"detector"``
                detector id value (`int`)

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

        profileData = self.profiles.run(
            combined, identity=identity, detectorMap=detectorMap, pfsConfig=pfsConfig
        )
        profiles = profileData.profiles
        if len(profiles) == 0:
            raise RuntimeError("No profiles found")
        self.log.info("%d fiber profiles found", len(profiles))

        return Struct(outputData=profiles)


class MergeFiberProfilesConnections(PipelineTaskConnections, dimensions=("instrument", "detector")):
    """Connections for MergeFiberProfilesTask"""

    profiles = InputConnection(
        name="fiberProfiles_subset",
        doc="Input fiber profile subsets.",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "detector", "pfs_design_id"),
        multiple=True,
    )
    merged = OutputConnection(
        name="fiberProfiles",
        doc="Merged fiber profiles.",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )


class MergeFiberProfilesConfig(PipelineTaskConfig, pipelineConnections=MergeFiberProfilesConnections):
    """Configuration for MergeFiberProfilesTask"""

    pass


class MergeFiberProfilesTask(PipelineTask):
    """Merge fiber profiles

    We produce fiber profiles with different ``pfs_design_id`` dimensions,
    corresponding to different sets of fibers hiding behind black spots. These
    different profiles need to be merged to produce a single set of profiles for
    all the fibers.
    """

    ConfigClass = MergeFiberProfilesConfig
    _DefaultName = "mergeFiberProfiles"

    def run(self, profiles):
        """Merge fiber profiles

        Parameters
        ----------
        profiles : iterable of `FiberProfileSet`
            List of fiber profiles.

        Returns
        -------
        merged : `FiberProfileSet`
            Merged set of fiber profiles.
        """
        merged = FiberProfileSet.fromCombination(*profiles)
        self.log.info("Merged %d profiles", len(merged))
        return Struct(merged=merged)
