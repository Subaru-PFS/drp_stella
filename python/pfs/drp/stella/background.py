from typing import ClassVar, Type

import numpy as np

from lsst.pex.config import Config, Field, RangeField, ListField
from lsst.pipe.base import Task
from lsst.afw.image import MaskedImage

from .DetectorMap import DetectorMap
from .math import calculateMedian
from .spline import SplineD

__all__ = ("BackgroundConfig", "BackgroundTask")


class BackgroundConfig(Config):
    """Configuration for background measurement"""

    maskHalfWidth = Field(dtype=int, default=10, doc="Half-width of masking around fibers")
    binSize = RangeField(dtype=int, default=128, min=1, doc="Size of bins (pixels)")
    mask = ListField(
        dtype=str,
        default=["SAT", "BAD", "NO_DATA", "FIBERTRACE"],
        doc="Names of mask planes to ignore while estimating the background",
    )


class BackgroundTask(Task):
    ConfigClass: ClassVar[Type[Config]] = BackgroundConfig
    _DefaultName: ClassVar[str] = "background"

    def run(self, maskedImage: MaskedImage, detectorMap: DetectorMap) -> SplineD:
        xCenter = detectorMap.getXCenter()
        xLow = int(np.min(xCenter)) - self.config.maskHalfWidth
        xHigh = int(np.max(xCenter) + 0.5) + self.config.maskHalfWidth
        bad = (maskedImage.mask.array & maskedImage.mask.getPlaneBitMask(self.config.mask)) != 0
        bad[:, xLow:xHigh] = True

        height = maskedImage.getHeight()
        numSwaths = max(5, int(np.ceil(2*height/self.config.binSize)))
        bounds = np.linspace(0, height - 1, numSwaths, dtype=int)

        numKnots = numSwaths - 2
        knots = np.zeros(numKnots, dtype=float)
        values = np.zeros(numKnots, dtype=float)
        array = maskedImage.image.array
        for ii, (yLow, yHigh) in enumerate(zip(bounds[:-2], bounds[2:])):
            knots[ii] = 0.5*(yLow + yHigh)
            values[ii] = calculateMedian(array[yLow:yHigh].flatten(), bad[yLow:yHigh].flatten())

        spline = SplineD(knots, values)
        for yy in range(height):
            array[yy] -= spline(yy)
        return spline
