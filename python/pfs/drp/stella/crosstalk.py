from typing import List

import numpy as np

from lsst.pex.config import Config
from lsst.pipe.base import Task
from lsst.afw.image import ExposureF, ImageF
from lsst.afw.cameraGeom import Detector, Amplifier
from lsst.ip.isr.crosstalk import CrosstalkCalib


class PfsCrosstalkConfig(Config):
    """Configuration for PfsCrosstalkTask"""
    useConfigCoefficients = True
    crosstalkValues = None


class PfsCrosstalkTask(Task):
    ConfigClass = PfsCrosstalkConfig
    _DefaultName = "crosstalk"

    def prepCrosstalk(self, dataRef, crosstalk=None):
        pass

    def run(self, exposure: ExposureF, *args, **kwargs) -> None:
        """Remove crosstalk from an exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to process; modified.
        *args, **kwargs
            Additional arguments passed to the task for compatibility with
            ``lsst.ip.isr.crosstalk.CrosstalkTask``; ignored.
        """
        detector = exposure.getDetector()
        if not detector.hasCrosstalk():
            self.log.info("No crosstalk coefficients provided")
            return

        coeffs = detector.getCrosstalk()
        numAmps = len(detector)
        if coeffs.size != numAmps**2:
            raise RuntimeError("Expected %d coefficients, got %d" % (numAmps**2, coeffs.size))
        coeffs = coeffs.reshape(numAmps, numAmps)
        self.log.debug("Crosstalk coefficients: %s", coeffs)

        source = exposure.image
        crosstalk = ImageF(source.getBBox())
        crosstalk.set(0)

        for index, amp in enumerate(detector):
            sourceAmps = self.makeSources(source, detector, amp)
            target = crosstalk[amp.getBBox()]
            for ss, vv in zip(sourceAmps, coeffs[index]):
                target.array += vv*ss

        exposure.maskedImage -= crosstalk

    def makeSources(
        self, image: ImageF, detector: Detector, refAmp: Amplifier
    ) -> List[np.ndarray]:
        """Make a list of source amplifiers

        We extract each of the amplifiers and align them.

        Parameters
        ----------
        image : `lsst.afw.image.Image`
            Image to extract sources from.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector geometry.
        refAmp : `lsst.afw.cameraGeom.Amplifier`
            Reference amplifier.

        Returns
        -------
        sources : `list` of `numpy.ndarray`
            List of source images.
        """
        return [
            CrosstalkCalib.extractAmp(
                image, amp, refAmp, True
            ).array.astype(float) for amp in detector
        ]
