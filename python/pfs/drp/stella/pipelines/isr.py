from lsst.pex.config import ConfigurableField, Field
from lsst.afw.image import Exposure
from lsst.pipe.base import Struct
from lsst.pipe.tasks.repair import RepairTask
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from lsst.obs.pfs.isrTask import PfsIsrTask, PfsIsrTaskConfig
from lsst.ip.isr.isrTask import IsrTaskConnections

__all__ = ("IsrTask",)


class IsrConnections(IsrTaskConnections):
    """Connections for IsrTask"""

    flat = PrerequisiteConnection(
        name="fiberFlat",
        doc="Combined flat",
        storageClass="ExposureF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )


class IsrConfig(PfsIsrTaskConfig, pipelineConnections=IsrConnections):
    """Configuration for IsrTask"""

    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    repair = ConfigurableField(target=RepairTask, doc="Task to repair artifacts")

    def setDefaults(self):
        super().setDefaults()
        self.doWrite = True  # Ensure postISRCCD is written


class IsrTask(PfsIsrTask):
    """Perform instrumental signature removal and repair artifacts"""

    ConfigClass = IsrConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("repair")

    def run(self, *args, **kwargs) -> Struct:
        """Perform instrumental signature removal and repair artifacts

        This subclass supplements the standard PFS ISR with additional artifact
        repairs (which are not normally a part of the 2d imaging ISR because
        it uses the PSF, which needs to be measured from the image).
        """
        result = super().run(*args, **kwargs)
        if self.config.doRepair:
            self.repairExposure(result.exposure)
        return result

    def repairExposure(self, exposure: Exposure):
        """Repair CCD defects in the exposure

        Uses the PSF specified in the config.
        """
        modelPsfConfig = self.config.repair.interp.modelPsf
        psf = modelPsfConfig.apply()
        exposure.setPsf(psf)
        self.repair.run(exposure)
