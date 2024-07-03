from lsst.utils import continueClass
from lsst.afw.image import PARENT

from .FiberTrace import FiberTrace

__all__ = ["FiberTrace"]


@continueClass  # noqa: F811 redefinition
class FiberTrace:  # noqa: F811 (redefinition)
    def applyToMask(self, mask):
        """Apply the trace mask to the provided mask

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask to which to apply the trace mask.
        """
        traceMask = self.getTrace().mask
        mask.Factory(mask, traceMask.getBBox(), PARENT)[:] |= traceMask

    def extractSpectrum(self, maskedImage, badBitmask, minFracMask=0.3):
        """Extract a spectrum using the trace

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to extract spectrum.
        badBitmask : `lsst.afw.image.MaskPixel`
            Bitmask for bad pixels.
        minFracMask : `float`
            Minimum fractional contribution of pixel for mask to be accumulated.

        Returns
        -------
        spectrum : `pfs.drp.stella.Spectrum`
            Extracted spectrum.
        """
        from .FiberTraceSetContinued import FiberTraceSet
        traces = FiberTraceSet(1)
        traces.add(self)
        spectra = traces.extractSpectra(maskedImage, badBitmask, minFracMask)
        return spectra[0]
