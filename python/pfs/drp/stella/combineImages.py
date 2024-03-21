from typing import Iterable

from lsst.pex.config import Config, Field, ChoiceField, ListField
from lsst.pipe.base import Task
from lsst.afw.math import stringToStatisticsProperty, StatisticsControl, statisticsStack
from lsst.afw.image import Exposure, makeExposure

__all__ = ("CombineImagesConfig", "CombineImagesTask")


class CombineImagesConfig(Config):
    """Configuration for combining images"""
    mask = ListField(dtype=str, default=["BAD", "SAT", "INTRP", "CR", "NO_DATA"],
                     doc="Mask planes to reject from combination")
    combine = ChoiceField(dtype=str, default="MEANCLIP",
                          allowed=dict(MEAN="Sample mean", MEANCLIP="Clipped mean", MEDIAN="Sample median"),
                          doc="Statistic to use for combination (from lsst.afw.math)")
    rejThresh = Field(dtype=float, default=3.0, doc="Clipping threshold for combination")
    rejIter = Field(dtype=int, default=3, doc="Clipping iterations for combination")
    maxVisitsToCalcErrorFromInputVariance = Field(
        dtype=int, default=2,
        doc="Maximum number of visits to estimate variance from input variance, not per-pixel spread"
    )


class CombineImagesTask(Task):
    """Task to combine images"""
    ConfigClass = CombineImagesConfig
    _DefaultName = "combineImages"

    def run(self, exposureList: Iterable[Exposure]) -> Exposure:
        """Combine multiple exposures.

        Parameters
        ----------
        exposureList : iterable of `lsst.afw.image.Exposure`
            List of exposures to combine.

        Returns
        -------
        combined : `lsst.afw.image.Exposure`
            Combined exposure.
        """
        firstExp = exposureList[0]
        if len(exposureList) == 1:
            return firstExp  # That was easy!
        dimensions = firstExp.getDimensions()
        for exp in exposureList[1:]:
            if exp.getDimensions() != dimensions:
                raise RuntimeError(f"Dimension difference: {exp.getDimensions()} vs {dimensions}")

        combineStat = stringToStatisticsProperty(self.config.combine)
        ctrl = StatisticsControl(
            self.config.rejThresh, self.config.rejIter, firstExp.mask.getPlaneBitMask(self.config.mask)
        )
        numImages = len(exposureList)
        if numImages < 1:
            raise RuntimeError("No valid input data")
        if numImages < self.config.maxVisitsToCalcErrorFromInputVariance:
            ctrl.setCalcErrorFromInputVariance(True)

        # Combine images
        combined = makeExposure(statisticsStack([exp.maskedImage for exp in exposureList], combineStat, ctrl))
        combined.setMetadata(firstExp.getMetadata())
        combined.getInfo().setVisitInfo(firstExp.getInfo().getVisitInfo())
        return combined
