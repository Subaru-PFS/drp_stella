import math
import numpy as np

from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.cp.pipe.cpDark import CpDarkConnections, CpDarkTaskConfig, CpDarkTask

from lsst.afw.geom import SpanSet
from lsst.pipe.base import Struct
from lsst.meas.algorithms import SingleGaussianPsf


class PfsDarkConnections(CpDarkConnections, dimensions=("instrument", "visit", "arm", "spectrograph")):
    inputExp = InputConnection(
        name="cpDarkISR",
        doc="Input pre-processed exposures to combine.",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    outputExp = OutputConnection(
        name="cpDarkProc",
        doc="Output combined proposed calibration.",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class PfsDarkConfig(CpDarkTaskConfig, pipelineConnections=PfsDarkConnections):
    """Configuration for PfsDarkTask"""
    pass


class PfsDarkTask(CpDarkTask):
    """Combine pre-processed exposures to create a proposed dark"""
    ConfigClass = PfsDarkConfig

    def run(self, inputExp):
        """Preprocess input exposures prior to DARK combination.

        This task detects and repairs cosmic rays strikes.

        This override fixes a bug in the LSST code, where the CR spans can be
        grown larger than the image.

        Parameters
        ----------
        inputExp : `lsst.afw.image.Exposure`
            Pre-processed dark frame data to combine.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            The results struct containing:

            ``outputExp``
                CR rejected, ISR processed Dark Frame
                (`lsst.afw.image.Exposure`).
        """
        psf = SingleGaussianPsf(
            self.config.psfSize,
            self.config.psfSize,
            self.config.psfFwhm/(2*math.sqrt(2*math.log(2))),
        )
        inputExp.setPsf(psf)
        scaleExp = inputExp.clone()
        mi = scaleExp.getMaskedImage()

        # DM-23680:
        # Darktime scaling necessary for repair.run() to ID CRs correctly.
        scale = inputExp.getInfo().getVisitInfo().getDarkTime()
        if np.isfinite(scale) and scale != 0.0:
            mi /= scale

        self.repair.run(scaleExp, keepCRs=False)
        if self.config.crGrow > 0:
            crMask = inputExp.getMaskedImage().getMask().getPlaneBitMask("CR")
            spans = SpanSet.fromMask(inputExp.mask, crMask)
            spans = spans.dilated(self.config.crGrow).clippedTo(inputExp.getBBox())
            spans.setMask(inputExp.mask, crMask)

        # Undo scaling; as above, DM-23680.
        if np.isfinite(scale) and scale != 0.0:
            mi *= scale

        return Struct(
            outputExp=inputExp,
        )
