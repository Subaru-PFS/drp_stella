from lsst.utils import continueClass
from lsst.afw.image import PARENT

from .FiberTrace import FiberTrace

__all__ = ["FiberTrace"]


@continueClass  # noqa: F811 redefinition
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

    def extractSpectrum(self, maskedImage, badBitmask):
        """Extract a spectrum using the trace

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to extract spectrum.
        badBitmask : `lsst.afw.image.MaskPixel`
            Bitmask for bad pixels.

        Returns
        -------
        spectrum : `pfs.drp.stella.Spectrum`
            Extracted spectrum.
        """
        from .FiberTraceSetContinued import FiberTraceSet
        traces = FiberTraceSet(1)
        traces.add(self)
        spectra = traces.extractSpectra(maskedImage, badBitmask)
        return spectra[0]
