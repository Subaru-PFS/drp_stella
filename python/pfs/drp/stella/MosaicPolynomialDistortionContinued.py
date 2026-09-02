from lsst.utils import continueClass
from lsst.geom import Box2D

import pfs.datamodel.pfsDetectorMap

from .MosaicPolynomialDistortion import MosaicPolynomialDistortion
from .Distortion import Distortion

__all__ = ("MosaicPolynomialDistortion",)


@continueClass  # noqa: F811 (redefinition)
class MosaicPolynomialDistortion:  # noqa: F811 (redefinition)
    @classmethod
    def fromDatamodel(cls, distortion):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        distortion : `pfs.datamodel.MosaicPolynomialDistortion`
            datamodel representation of MosaicPolynomialDistortion.

        Returns
        -------
        self : `pfs.drp.stella.MosaicPolynomialDistortion`
            drp_stella representation of MosaicPolynomialDistortion.
        """
        if not isinstance(distortion, pfs.datamodel.MosaicPolynomialDistortion):
            raise RuntimeError(f"Wrong type: {distortion}")
        return cls(distortion.order, Box2D(distortion.box.toLsst()), distortion.coefficients)

    def toDatamodel(self):
        """Convert to the pfs.datamodel representation

        Returns
        -------
        distortion : `pfs.datamodel.MosaicPolynomialDistortion`
            Datamodel representation of MosaicPolynomialDistortion.
        """
        return pfs.datamodel.MosaicPolynomialDistortion(
            self.getOrder(),
            pfs.datamodel.Box.fromLsst(self.getRange()),
            self.getCoefficients(),
        )


Distortion.register(MosaicPolynomialDistortion)
