from lsst.pex.config import Config, ListField, ChoiceField
from lsst.pipe.base import Task

from pfs.datamodel import PfsConfig, FiberStatus, TargetType

__all__ = ("SelectFibersConfig", "SelectFibersTask")


class SelectFibersConfig(Config):
    """Configuration for SelectFibersConfig"""
    fiberStatus = ListField(dtype=str, default=("GOOD",), doc="Fiber status to require")
    targetType = ListField(dtype=str, default=(), doc="Target types to select")
    fiberFilter = ChoiceField(dtype=str, default="ALL",
                              allowed={"ALL": "Use all fibers selected by targetType",
                                       "ODD": "Use only odd fiberIds after selecting by targetType",
                                       "EVEN": "Use only even fiberIds after selecting by targetType",
                                       },
                              doc="Additional filters to provide to input fiberId list")


class SelectFibersTask(Task):
    """Select fibers"""
    ConfigClass = SelectFibersConfig
    _DefaultName = "selectFibers"

    def run(self, pfsConfig: PfsConfig) -> PfsConfig:
        """Select fibers

        Fibers are selected based on their ``fiberStatus`` and ``targetType``,
        and then odd or even fibers may be selected.

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for identifying sky fibers.

        Returns
        -------
        pfsConfig : `pfs.datamodel.PfsConfig`
            Subset of input ``pfsConfig``, containing only the selected fibers.
        """
        fiberStatus = [FiberStatus.fromString(fs) for fs in self.config.fiberStatus]
        targetType = [TargetType.fromString(fs) for fs in self.config.targetType]
        pfsConfig = pfsConfig.select(fiberStatus=fiberStatus, targetType=targetType)

        if self.config.fiberFilter == "ALL":
            return pfsConfig

        # Apply filter
        fiberId = pfsConfig.fiberId
        if self.config.fiberFilter == "ODD":
            fiberId = fiberId[fiberId % 2 == 1]
        elif self.config.fiberFilter == "EVEN":
            fiberId = fiberId[fiberId % 2 == 0]

        return pfsConfig.select(fiberId=fiberId)
