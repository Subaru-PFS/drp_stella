from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

__all__ = ("CalibsForVisitConfig", "CalibsForVisitTask")


class CalibsForVisitConnections(
    PipelineTaskConnections, dimensions=("instrument", "visit", "arm", "spectrograph")
):
    """Connections for CalibsForVisitTask

    The calibs are prerequisites, so they are looked up for the visit in the
    usual way. The ``fiberProfiles`` and ``imagePsf`` are optional (e.g., the
    target of `PsfMatchCalibsTask` needs only the ``detectorMap``).
    """

    detectorMap = PrerequisiteConnection(
        name="detectorMap_calib",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Profiles of fibers",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
        multiple=True,  # So it can be optional
        minimum=0,
    )
    psf = PrerequisiteConnection(
        name="imagePsf",
        doc="PSF model",
        storageClass="Psf",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
        multiple=True,  # So it can be optional
        minimum=0,
    )

    outputDetectorMap = OutputConnection(
        name="detectorMap_forVisit",
        doc="Calib mapping from fiberId,wavelength to x,y valid for the visit",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    outputFiberProfiles = OutputConnection(
        name="fiberProfiles_forVisit",
        doc="Calib profiles of fibers valid for the visit",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    outputPsf = OutputConnection(
        name="imagePsf_forVisit",
        doc="Calib PSF model valid for the visit",
        storageClass="Psf",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class CalibsForVisitConfig(PipelineTaskConfig, pipelineConnections=CalibsForVisitConnections):
    """Configuration for CalibsForVisitTask"""

    pass


class CalibsForVisitTask(PipelineTask):
    """Copy the calibs valid for a visit to datasets with a visit dimension

    A task whose quantum has no temporal dimensions (e.g., one combining
    multiple visits, such as `PsfMatchCalibsTask`) can't look up the calibs
    valid for each of its visits. This task does the lookup for each visit,
    and writes copies of the calibs with the visit dimension, which such a
    task can read as regular inputs.

    The ``fiberProfiles`` and ``imagePsf`` are optional: if they aren't
    available for the visit, they aren't written.
    """

    ConfigClass = CalibsForVisitConfig
    _DefaultName = "calibsForVisit"

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Copy the calibs

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
        butler.put(butler.get(inputRefs.detectorMap), outputRefs.outputDetectorMap)
        for name, outputName in (("fiberProfiles", "outputFiberProfiles"), ("psf", "outputPsf")):
            refs = getattr(inputRefs, name, [])
            if len(refs) > 1:
                dataId = butler.quantum.dataId
                raise RuntimeError(f"Expected at most one {name} for {dataId}, found {len(refs)}")
            if refs:
                butler.put(butler.get(refs[0]), getattr(outputRefs, outputName))
            else:
                self.log.info("No %s for %s; not writing it", name, butler.quantum.dataId)
