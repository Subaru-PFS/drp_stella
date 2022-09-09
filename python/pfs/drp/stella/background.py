from typing import ClassVar, Type

import numpy as np

from lsst.pex.config import Config, Field, ChoiceField, RangeField, ListField
from lsst.pipe.base import Task
import lsst.afw.math as afwMath
from lsst.afw.image import Mask, MaskedImage

from pfs.datamodel import PfsConfig, FiberStatus
from .DetectorMap import DetectorMap

__all__ = ("BackgroundConfig", "BackgroundTask")


class BackgroundConfig(Config):
    """Configuration for background measurement"""

    maskHalfWidth = Field(dtype=float, default=3.5, doc="Half-width of masking around fibers")
    statistic = ChoiceField(
        dtype=str,
        default="MEANCLIP",
        doc="type of statistic to use for grid points",
        allowed={"MEANCLIP": "clipped mean", "MEAN": "unclipped mean", "MEDIAN": "median"},
    )
    xBinSize = RangeField(dtype=int, default=256, min=1, doc="Superpixel size in x")
    yBinSize = RangeField(dtype=int, default=256, min=1, doc="Superpixel size in y")
    algorithm = ChoiceField(
        dtype=str,
        default="NATURAL_SPLINE",
        optional=True,
        doc="How to interpolate the background values. " "This maps to an enum; see afw::math::Background",
        allowed={
            "CONSTANT": "Use a single constant value",
            "LINEAR": "Use linear interpolation",
            "NATURAL_SPLINE": "cubic spline with zero second derivative at endpoints",
            "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust" " to outliers",
            "NONE": "No background estimation is to be attempted",
        },
    )
    mask = ListField(
        dtype=str,
        default=["SAT", "BAD", "NO_DATA", "FIBERTRACE"],
        doc="Names of mask planes to ignore while estimating the background",
    )


class BackgroundTask(Task):
    ConfigClass: ClassVar[Type[Config]] = BackgroundConfig
    _DefaultName: ClassVar[str] = "background"

    def run(
        self, maskedImage: MaskedImage, detectorMap: DetectorMap, pfsConfig: PfsConfig
    ) -> afwMath.BackgroundList:
        self.maskFibers(maskedImage.mask, detectorMap, pfsConfig)
        bg = self.measureBackground(maskedImage)
        maskedImage.image -= bg.getImage()
        return bg

    def maskFibers(self, mask: Mask, detectorMap: DetectorMap, pfsConfig: PfsConfig):
        height = mask.getHeight()
        bitmask = mask.getPlaneBitMask("FIBERTRACE")
        for fiberId in pfsConfig.select(
            fiberId=detectorMap.fiberId, fiberStatus=(FiberStatus.GOOD, FiberStatus.BROKENFIBER)
        ).fiberId:
            # Could push this down to C++ for efficiency boost
            xCenter = detectorMap.getXCenter(fiberId)
            xLow = np.clip(np.floor(xCenter - self.config.maskHalfWidth).astype(int), 0, None)
            xHigh = np.clip(
                np.ceil(xCenter + self.config.maskHalfWidth).astype(int) + 1, None, height
            )  # exclusive
            for yy in range(mask.getHeight()):
                mask.array[yy, xLow[yy] : xHigh[yy]] |= bitmask

    def measureBackground(self, image: MaskedImage) -> afwMath.BackgroundList:
        """Measure a background model for image

        This doesn't use a full-featured background model (e.g., no Chebyshev
        approximation) because we just want the binning behaviour.  This will
        allow us to average the bins later (`averageBackgrounds`).

        The `BackgroundMI` is wrapped in a `BackgroundList` so it can be
        pickled and persisted.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image for which to measure background.

        Returns
        -------
        bgModel : `lsst.afw.math.BackgroundList`
            Background model.
        """
        stats = afwMath.StatisticsControl()
        stats.setAndMask(image.getMask().getPlaneBitMask(self.config.background.mask))
        stats.setNanSafe(True)
        ctrl = afwMath.BackgroundControl(
            self.config.background.algorithm,
            max(int(image.getWidth() / self.config.background.xBinSize + 0.5), 1),
            max(int(image.getHeight() / self.config.background.yBinSize + 0.5), 1),
            "REDUCE_INTERP_ORDER",
            stats,
            self.config.background.statistic,
        )

        bg = afwMath.makeBackground(image, ctrl)

        return afwMath.BackgroundList(
            (
                bg,
                afwMath.stringToInterpStyle(self.config.background.algorithm),
                afwMath.stringToUndersampleStyle("REDUCE_INTERP_ORDER"),
                afwMath.ApproximateControl.UNKNOWN,
                0,
                0,
                False,
            )
        )
