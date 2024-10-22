from typing import ClassVar, Type

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pex.config import Config, ConfigurableField, Field

from .repair import PfsRepairTask

__all__ = ("CosmicRayTask",)


class CosmicRayConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "exposure", "arm", "spectrograph"),
):
    inputExposure = InputConnection(
        name="postISRCCD",
        doc="Exposure to repair",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
    )
    outputExposure = OutputConnection(
        name="calexp",
        doc="Repaired exposure",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
    )


class CosmicRayConfig(PipelineTaskConfig, pipelineConnections=CosmicRayConnections):
    """Configuration for CosmicRayTask"""
    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    repair = ConfigurableField(target=PfsRepairTask, doc="Task to repair artifacts")

    def setDefaults(self):
        super().setDefaults()
        self.repair.interp.modelPsf.defaultFwhm = 1.5  # FWHM of the PSF in pixels


class CosmicRayTask(PipelineTask):
    """Perform cosmic-ray removal

    We use the standard single-exposure cosmic-ray removal task, but in the
    future we intend to upgrade this to use a more sophisticated algorithm using
    multiple exposures.
    """
    ConfigClass: ClassVar[Type[Config]] = CosmicRayConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("repair")

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection
    ) -> None:
        """Entry point for running the task under the Gen3 middleware"""
        exposure = butler.get(inputRefs.inputExposure)
        outputs = self.run(exposure)
        butler.put(outputs.exposure, outputRefs.outputExposure)

    def run(self, exposure) -> Struct:
        """Perform cosmic-ray removal"""
        modelPsfConfig = self.config.repair.interp.modelPsf
        psf = modelPsfConfig.apply()
        exposure.setPsf(psf)

        if self.config.doRepair:
            self.repair.run(exposure)

        return Struct(exposure=exposure)
