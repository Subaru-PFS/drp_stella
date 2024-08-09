from typing import Iterable

from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from pfs.datamodel import PfsConfig

from ..arcLine import ArcLineSet
from ..subtractSky2d import SubtractSky2dTask

__all__ = ("FitSky2dTask",)


class FitSky2dConnections(PipelineTaskConnections, dimensions=("instrument", "exposure", "arm")):
    """Connections for FitSky2dTask"""

    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
    )
    photometry = InputConnection(
        name="photometry",
        doc="Line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
        multiple=True,
    )

    sky2d = OutputConnection(
        name="sky2d",
        doc="2D sky subtraction model",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "arm"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doSubtractSky2d:
            self.outputs.remove("sky2d")


class FitSky2dConfig(PipelineTaskConfig, pipelineConnections=FitSky2dConnections):
    """Configuration for FitSky2dTask"""

    doSubtractSky2d = Field(dtype=bool, default=False, doc="Subtract sky on 2D image?")
    subtractSky2d = ConfigurableField(target=SubtractSky2dTask, doc="2D sky subtraction")


class FitSky2dTask(PipelineTask):
    """Fit a model to the sky line fluxes"""

    ConfigClass = FitSky2dConfig
    _DefaultName = "fitSky2d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("subtractSky2d")

    def run(
        self,
        pfsConfig: PfsConfig,
        photometry: Iterable[ArcLineSet],
    ) -> Struct:
        """Fit a model to the sky line fluxes

        This may be a no-op if ``doSubtractSky2d`` is ``False``.

        Parameters
        ----------
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        photometry : iterable of `ArcLineSet`
            List of line measurements.

        Returns
        -------
        sky2d : `pfs.drp.stella.subtractSky2d.SkyModel`
            2D sky model.
        """
        sky2d = None
        if self.config.doSubtractSky2d:
            sky2d = self.subtractSky2d.measureSky(pfsConfig, photometry)
        return Struct(sky2d=sky2d)
