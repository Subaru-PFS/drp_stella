from lsst.utils import continueClass
from lsst.geom import Box2D

import pfs.datamodel.pfsDetectorMap

from .DoubleDistortion import DoubleDistortion
from .Distortion import Distortion

__all__ = ("DoubleDistortion",)


@continueClass  # noqa: F811 (redefinition)
class DoubleDistortion:  # noqa: F811 (redefinition)
    @classmethod
    def fromDatamodel(cls, distortion):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        distortion : `pfs.datamodel.DoubleDistortion`
            datamodel representation of DoubleDistortion.

        Returns
        -------
        self : `pfs.drp.stella.DoubleDistortion`
            drp_stella representation of DoubleDistortion.
        """
        if not isinstance(distortion, pfs.datamodel.DoubleDistortion):
            raise RuntimeError(f"Wrong type: {distortion}")
        return cls(
            distortion.order,
            Box2D(distortion.box.toLsst()),
            distortion.xLeft,
            distortion.yLeft,
            distortion.xRight,
            distortion.yRight,
        )

    def toDatamodel(self):
        """Convert to the pfs.datamodel representation

        Returns
        -------
        distortion : `pfs.datamodel.DoubleDistortion`
            Datamodel representation of DoubleDistortion.
        """
        return pfs.datamodel.DoubleDistortion(
            self.getOrder(),
            pfs.datamodel.Box.fromLsst(self.getRange()),
            self.getXLeftCoefficients(),
            self.getYLeftCoefficients(),
            self.getXRightCoefficients(),
            self.getYRightCoefficients(),
        )


Distortion.register(DoubleDistortion)
