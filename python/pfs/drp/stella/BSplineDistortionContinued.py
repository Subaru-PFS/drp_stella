import numpy as np

from lsst.utils import continueClass

import pfs.datamodel.pfsDetectorMap

from .BSplineDistortion import BSpline2D, BSplineDistortion
from .Distortion import Distortion

__all__ = ("BSpline2D", "BSplineDistortion")


@continueClass  # noqa: F811 (redefinition)
class BSpline2D:  # noqa: F811 (redefinition)
    @classmethod
    def fromDatamodel(cls, spline: pfs.datamodel.BSpline2D) -> "BSpline2D":
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        spline : `pfs.datamodel.BSpline2D`
            datamodel representation of BSpline2D.

        Returns
        -------
        self : `pfs.drp.stella.BSpline2D`
            drp_stella representation of BSpline2D.
        """
        if spline.isEmpty():
            return cls(spline.order)
        return cls(
            spline.order,
            np.ascontiguousarray(spline.xKnots, dtype=float),
            np.ascontiguousarray(spline.yKnots, dtype=float),
            np.ascontiguousarray(spline.coefficients, dtype=float),
        )

    def toDatamodel(self) -> pfs.datamodel.BSpline2D:
        """Convert to the pfs.datamodel representation

        Returns
        -------
        spline : `pfs.datamodel.BSpline2D`
            datamodel representation of BSpline2D.
        """
        if self.isEmpty():
            return pfs.datamodel.BSpline2D.empty(self.getOrder())
        return pfs.datamodel.BSpline2D(
            self.getOrder(), self.getXKnots(), self.getYKnots(), self.getCoefficients()
        )

    def __reduce__(self):
        """How to pickle"""
        if self.isEmpty():
            return self.__class__, (self.getOrder(),)
        return self.__class__, (self.getOrder(), self.getXKnots(), self.getYKnots(), self.getCoefficients())


@continueClass  # noqa: F811 (redefinition)
class BSplineDistortion:  # noqa: F811 (redefinition)
    @classmethod
    def fromDatamodel(cls, distortion: pfs.datamodel.BSplineDistortion) -> "BSplineDistortion":
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        distortion : `pfs.datamodel.BSplineDistortion`
            datamodel representation of BSplineDistortion.

        Returns
        -------
        self : `pfs.drp.stella.BSplineDistortion`
            drp_stella representation of BSplineDistortion.
        """
        if not isinstance(distortion, pfs.datamodel.BSplineDistortion):
            raise RuntimeError(f"Wrong type: {distortion}")
        return cls(
            np.ascontiguousarray(distortion.xBoundaries, dtype=float),
            [BSpline2D.fromDatamodel(ss) for ss in distortion.xSplines],
            [BSpline2D.fromDatamodel(ss) for ss in distortion.ySplines],
        )

    def toDatamodel(self) -> pfs.datamodel.BSplineDistortion:
        """Convert to the pfs.datamodel representation

        Returns
        -------
        distortion : `pfs.datamodel.BSplineDistortion`
            Datamodel representation of BSplineDistortion.
        """
        return pfs.datamodel.BSplineDistortion(
            self.getXBoundaries(),
            [ss.toDatamodel() for ss in self.getXSplines()],
            [ss.toDatamodel() for ss in self.getYSplines()],
        )


Distortion.register(BSplineDistortion)
