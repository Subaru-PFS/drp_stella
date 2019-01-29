from lsst.pex.config import Config, ConfigurableField, Field
from lsst.pipe.base import CmdLineTask, ArgumentParser

from pfs.datamodel.pfsConfig import TargetType
from pfs.datamodel.drp import PfsObject
from .fitReference import FitReferenceTask


class CalculateReferenceFluxConfig(Config):
    """Configuration for CalculateReferenceFluxTask"""
    fitReference = ConfigurableField(target=FitReferenceTask, doc="Fit reference spectrum")
    doOverwrite = Field(dtype=bool, default=False, doc="Overwrite existing reference spectrum?")


class CalculateReferenceFluxTask(CmdLineTask):
    """Calculate the physical reference flux for flux standards

    The heavy lifting is done by the ``fitReference`` sub-task.
    """
    ConfigClass = CalculateReferenceFluxConfig
    _DefaultName = "calculateReferenceFlux"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitReference")

    @classmethod
    def _makeArgumentParser(cls):
        """Make ArgumentParser"""
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="pfsMerged",
                               help="data IDs, e.g. --id exp=12345")
        return parser

    def run(self, dataRef):
        """Run on an exposure

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
        """
        merged = dataRef.get("pfsMerged")
        pfsConfig = dataRef.get("pfsConfig")
        butler = dataRef.getButler()
        select = pfsConfig.targetType == int(TargetType.FLUXSTD)
        for fiberId in pfsConfig.fiberId[select]:
            spectrum = merged.extractFiber(PfsObject, pfsConfig, fiberId)
            dataId = spectrum.getIdentity()
            if not self.config.doOverwrite and butler.datasetExists("pfsReference", dataId):
                self.log.info("Skipping calculation of new reference for %s" % (dataId,))
                continue
            reference = self.fitReference.run(spectrum)
            dataRef.getButler().put(reference, "pfsReference", dataId)

    def _getMetadataName(self):
        return None
