from lsst.afw.image import Exposure
from .centroidLines import CentroidLinesTask

__all__ = ("CentroidAbsorptionConfig", "CentroidAbsorptionTask")


class CentroidAbsorptionConfig(CentroidLinesTask.ConfigClass):
    """Configuration for `CentroidAbsorptionTask`"""

    def setDefaults(self):
        super().setDefaults()
        self.doSubtractTraces = True  # Very important for absorption lines!
        self.selectFibers.targetType = ["SKY"]  # Don't want contamination from bright science targets


class CentroidAbsorptionTask(CentroidLinesTask):
    """Centroid absorption lines in an exposure

    This task is a subclass of `CentroidLinesTask` that is specialized for
    centroiding absorption lines. It uses the same algorithm as the base class,
    but it inverts the image before centroiding to treat absorption lines as
    emission lines.
    """

    ConfigClass = CentroidAbsorptionConfig

    def run(self, exposure: Exposure, *args, **kwargs):
        """Centroid absorption lines in an exposure

        Parameters
        ----------
        exposure : `Exposure`
            The exposure to centroid.
        halfHeight : `int`
            The half-height of the region to use for centroiding.
        maskPlanes : `list[str]`
            The mask planes to use for masking bad pixels.

        Returns
        -------
        centroids : `list[Centroid]`
            The list of centroids found in the exposure.
        """
        # Invert the image to treat absorption lines as emission lines
        invertedExposure = exposure.clone()
        invertedExposure.image *= -1

        return super().run(invertedExposure, *args, **kwargs)
