from lsst.utils import continueClass
from lsst.geom import Box2D

import pfs.datamodel.pfsDetectorMap

from .RotScaleDistortion import RotScaleDistortion
from .Distortion import Distortion

__all__ = ("RotScaleDistortion",)


@continueClass  # noqa: F811 (redefinition)
class RotScaleDistortion:  # noqa: F811 (redefinition)
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
        if not isinstance(distortion, pfs.datamodel.RotScaleDistortion):
            raise RuntimeError(f"Wrong type: {distortion}")
        return cls(Box2D(distortion.box.toLsst()), distortion.parameters)

    def toDatamodel(self):
        """Convert to the pfs.datamodel representation

        Returns
        -------
        distortion : `pfs.datamodel.RotScaleDistortion`
            Datamodel representation of RotScaleDistortion.
        """
        return pfs.datamodel.RotScaleDistortion(
            pfs.datamodel.Box.fromLsst(self.getRange()), self.getParameters()
        )


Distortion.register(RotScaleDistortion)
