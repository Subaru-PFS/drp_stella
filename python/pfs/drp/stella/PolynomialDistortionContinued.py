from lsst.utils import continueClass
from lsst.geom import Box2D

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
        return cls(
            distortion.xOrder,
            distortion.yOrder,
            Box2D(distortion.box.toLsst()),
            distortion.xCoefficients,
            distortion.yCoefficients,
        )

    def toDatamodel(self):
        """Convert to the pfs.datamodel representation

        Returns
        -------
        distortion : `pfs.datamodel.PolynomialDistortion`
            Datamodel representation of PolynomialDistortion.
        """
        return pfs.datamodel.PolynomialDistortion(
            self.getXOrder(),
            self.getYOrder(),
            pfs.datamodel.Box.fromLsst(self.getRange()),
            self.getXCoefficients(),
            self.getYCoefficients(),
        )


Distortion.register(PolynomialDistortion)
