import numpy as np

import lsst.afw.image as afwImage

from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.cp.pipe.cpCombine import CalibCombineConnections, CalibCombineConfig, CalibCombineTask


class PfsCalibCombineConnections(CalibCombineConnections):
    inputExpHandles = InputConnection(
        name="cpInputs",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
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

    def calibStats(self, exp, calibrationType):
        """Measure bulk statistics for the calibration.

        Override to work around DM-47920 (present in LSST 28). HIERARCH keys
        with "." are triggering the bug, so we replace "." with "_".

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            Exposure to calculate statistics for.
        calibrationType : `str`
            Type of calibration to record in header.
        """
        metadata = exp.getMetadata()

        noGoodPixelsBit = afwImage.Mask.getPlaneBitMask(self.config.noGoodPixelsMask)

        # percentiles
        for amp in exp.getDetector():
            ampImage = exp[amp.getBBox()]
            percentileValues = np.nanpercentile(ampImage.image.array,
                                                self.config.distributionPercentiles)
            for level, value in zip(self.config.distributionPercentiles, percentileValues):
                key = f"LSST CALIB {calibrationType.upper()} {amp.getName()} DISTRIBUTION {level}-PCT"
                key = key.replace(".", "_")
                metadata[key] = value

            bad = ((ampImage.mask.array & noGoodPixelsBit) > 0)
            key = f"LSST CALIB {calibrationType.upper()} {amp.getName()} BADPIX-NUM"
            metadata[key] = bad.sum()
