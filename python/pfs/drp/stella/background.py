from typing import ClassVar, Type

import numpy as np

from lsst.pex.config import Config, Field, DictField, RangeField, ListField
from lsst.pipe.base import Task
from lsst.afw.image import MaskedImage, Exposure

from pfs.datamodel import PfsConfig, FiberStatus
from .DetectorMap import DetectorMap
from .math import calculateMedian
from .spline import SplineD

__all__ = ("BackgroundConfig", "BackgroundTask", "DichroicBackgroundConfig", "DichroicBackgroundTask")


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


class DichroicBackgroundConfig(Config):
    """Configuration for background measurement in the dichroic region"""
    fiberHalfWidth = Field(dtype=float, default=15, doc="Radius around fiber to mask (pixels)")
    fiberStatus = ListField(
        dtype=str,
        default=["GOOD", "BROKENFIBER", "BLOCKED"],
        doc="FiberStatus values to use for dichroic background subtraction",
    )
    mask = ListField(
        dtype=str,
        default=["SAT", "BAD", "NO_DATA"],
        doc="Mask planes to ignore for dichroic background subtraction",
    )
    top = DictField(
        keytype=str,
        itemtype=int,
        default=dict(
            b=30,
            r=30,
        ),
        doc="Number of rows to use for dichroic background subtraction at top of detector, indexed by arm",
    )
    bottom = DictField(
        keytype=str,
        itemtype=int,
        default=dict(
            r=50,
            n=50,
        ),
        doc="Number of rows to use for dichroic background subtraction at bottom of detector, indexed by arm",
    )


class DichroicBackgroundTask(Task):
    ConfigClass: ClassVar[Type[Config]] = DichroicBackgroundConfig
    _DefaultName: ClassVar[str] = "background"

    def run(self, image: MaskedImage, arm: str, detectorMap: DetectorMap, pfsConfig: PfsConfig) -> np.ndarray:
        """Remove any diffuse background in the dichroic region

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image to process.
        arm : `str`
            Spectrograph arm name (``b``, ``r``, ``n``, ``m``).
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId to x,y on the detector.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.
        """
        height = image.getHeight()
        topRows = self.config.top.get(arm, None)
        bottomRows = self.config.bottom.get(arm, None)
        if topRows is None and bottomRows is None:
            return np.zeros(height)

        fiberStatus = [FiberStatus.fromString(fs) for fs in self.config.fiberStatus]
        fiberId = pfsConfig.select(fiberId=detectorMap.fiberId, fiberStatus=fiberStatus).fiberId
        bitmask = image.mask.getPlaneBitMask(self.config.mask)

        def measureBackground(rows: slice) -> float:
            """Measure the background in the given rows

            Parameters
            ----------
            rows : `slice`
                Rows to measure.

            Returns
            -------
            background `float`
                Median background in the given rows.
            """
            array = image.image.array[rows, :]
            mask = image.mask.array[rows, :]
            good = (mask & bitmask) == 0

            for ff in fiberId:
                for yy, xCenter in enumerate(detectorMap.getXCenter(ff)[rows]):
                    xLow = int(xCenter - self.config.fiberHalfWidth)
                    xHigh = int(xCenter + self.config.fiberHalfWidth + 0.5)
                    good[yy, xLow:xHigh] = False
            return np.median(array[good])

        top = measureBackground(slice(-topRows, height)) if topRows else None
        bottom = measureBackground(slice(bottomRows)) if bottomRows else None

        if top is not None and bottom is not None:
            # Subtract a ramp across the image
            yBottom = 0.5*bottomRows
            yTop = height - 0.5*topRows
            yy = np.arange(height)
            slope = (top - bottom)/(yTop - yBottom)
            background = (yy - yBottom)*slope + bottom
            self.log.info("Subtracting dichroic: bottom=%f top=%f", bottom, top)
        elif top is not None:
            background = np.full(height, top)
            self.log.info("Subtracting dichroic: top=%f", top)
        elif bottom is not None:
            background = np.full(height, bottom)
            self.log.info("Subtracting dichroic: bottom=%f", bottom)
        else:
            raise AssertionError("Should never get here")

        image.image.array -= background[:, np.newaxis]

        return background
