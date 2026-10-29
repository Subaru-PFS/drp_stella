from lsst.utils import continueClass
from lsst.geom import Box2D, Point2D

import pfs.datamodel.pfsDetectorMap

from .PolynomialDistortion import PolynomialDistortion
from .Distortion import Distortion

__all__ = ("PolynomialDistortion",)


@continueClass  # noqa: F811 (redefinition)
class PolynomialDistortion:  # noqa: F811 (redefinition)
    @classmethod
    def fromDatamodel(cls, distortion):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        distortion : `pfs.datamodel.RotScaleDistortion`
            datamodel representation of RotScaleDistortion.

        Returns
        -------
        self : `pfs.drp.stella.RotScaleDistortion`
            drp_stella representation of RotScaleDistortion.
        """
        if not isinstance(distortion, pfs.datamodel.PolynomialDistortion):
            raise RuntimeError(f"Wrong type: {distortion}")
        box = distortion.box
        return cls(
            distortion.order,
            Box2D(Point2D(box.xMin, box.yMin), Point2D(box.xMax, box.yMax)),
            distortion.xCoefficients,
            distortion.yCoefficients,
        )

    def toDatamodel(self):
        """Convert to the pfs.datamodel representation

        Returns
        -------
        distortion : `pfs.datamodel.RotScaleDistortion`
            Datamodel representation of RotScaleDistortion.
        """
        return pfs.datamodel.PolynomialDistortion(
            self.getOrder(),
            pfs.datamodel.Box.fromLsst(self.getRange()),
            self.getXCoefficients(),
            self.getYCoefficients(),
        )


Distortion.register(PolynomialDistortion)
