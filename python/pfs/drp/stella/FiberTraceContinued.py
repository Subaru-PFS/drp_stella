from lsst.utils import continueClass
from lsst.afw.image import PARENT

from pfs.datamodel.pfsFiberTrace import PfsFiberTrace

from .FiberTrace import FiberTrace

__all__ = ["FiberTrace"]


@continueClass
class FiberTrace:
    def applyToMask(self, mask):
        """Apply the trace mask to the provided mask

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask to which to apply the trace mask.
        """
        traceMask = self.getTrace().mask
        mask.Factory(mask, traceMask.getBBox(), PARENT)[:] |= traceMask

