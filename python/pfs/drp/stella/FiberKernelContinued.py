from typing import Any

from lsst.utils import continueClass
from lsst.geom import Extent2I

from .FiberKernel import FiberKernel
from pfs.datamodel.pfsFiberKernel import PfsFiberKernel

__all__ = ["FiberKernel"]


@continueClass  # noqa: F811 redefinition
class FiberKernel:  # noqa: F811 (redefinition)
    def toDatamodel(self, metadata: dict[str, Any] | None = None) -> PfsFiberKernel:
        """Convert to a PfsFiberKernel datamodel instance"""
        return PfsFiberKernel(
            imageWidth=self.dims.getX(),
            imageHeight=self.dims.getY(),
            halfWidth=self.halfWidth,
            xNumBlocks=self.xNumBlocks,
            yNumBlocks=self.yNumBlocks,
            values=self.values,  # Coefficients and values are the same for now
            metadata=metadata or {},
        )

    @classmethod
    def fromDatamodel(cls, kernel: PfsFiberKernel) -> "FiberKernel":
        """Construct from a PfsFiberKernel datamodel instance"""
        return cls(
            dims=Extent2I(kernel.imageWidth, kernel.imageHeight),
            halfWidth=kernel.halfWidth,
            xNumBlocks=kernel.xNumBlocks,
            yNumBlocks=kernel.yNumBlocks,
            values=kernel.values,
        )

    def writeFits(self, filename):
        """Write to a FITS file"""
        self.toDatamodel().writeFits(filename)

    @classmethod
    def readFits(cls, filename) -> "FiberKernel":
        """Read from a FITS file"""
        return cls.fromDatamodel(PfsFiberKernel.readFits(filename))
