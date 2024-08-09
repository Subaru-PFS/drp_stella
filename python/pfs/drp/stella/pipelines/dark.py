from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.cp.pipe.cpDarkTask import CpDarkConnections, CpDarkTaskConfig, CpDarkTask


class PfsDarkConnections(CpDarkConnections, dimensions=("instrument", "exposure", "arm", "spectrograph")):
    inputExp = InputConnection(
        name="cpDarkISR",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
    )

    outputExp = OutputConnection(
        name="cpDarkProc",
        doc="Output combined proposed calibration.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
    )


class PfsDarkConfig(CpDarkTaskConfig, pipelineConnections=PfsDarkConnections):
    """Configuration for PfsDarkTask"""
    pass


class PfsDarkTask(CpDarkTask):
    """Combine pre-processed exposures to create a proposed dark"""
    ConfigClass = PfsDarkConfig
