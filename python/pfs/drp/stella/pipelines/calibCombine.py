from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.cp.pipe.cpCombine import CalibCombineConnections, CalibCombineConfig, CalibCombineTask


class PfsCalibCombineConnections(CalibCombineConnections):
    inputExpHandles = InputConnection(
        name="cpInputs",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
        multiple=True,
        deferLoad=True,
    )

    outputData = OutputConnection(
        name="cpProposal",
        doc="Output combined proposed calibration to be validated and certified..",
        storageClass="ExposureF",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )


class PfsCalibCombineConfig(CalibCombineConfig, pipelineConnections=PfsCalibCombineConnections):
    """Configuration for PfsCalibCombineTask"""
    pass


class PfsCalibCombineTask(CalibCombineTask):
    """Combine pre-processed exposures to create a proposed calibration"""
    ConfigClass = PfsCalibCombineConfig
