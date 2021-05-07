import math

from lsst.afw.detection import GaussianPsf

from pfs.drp.stella.SpectralPsf import SpectralPsf
from pfs.drp.stella.SpectralPsfContinued import ImagingSpectralPsf

__all__ = ("checkPsf", "fwhmToSigma", "sigmaToFwhm")


def fwhmToSigma(fwhm):
    """Convert FWHM to sigma for a Gaussian"""
    return fwhm/(2*math.sqrt(2*math.log(2)))


def sigmaToFwhm(sigma):
    """Convert sigma to FWHM for a Gaussian"""
    return sigma*(2*math.sqrt(2*math.log(2)))


def checkPsf(exposure, detectorMap=None, fwhm=1.5, kernelSize=4.0):
    """Ensure that the exposure has a `SpectralPsf`

    Creates the PSF in the exposure if it doesn't have one.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Exposure of interest.
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping of fiberId,wavelength to x,y.
    fwhm : `float`
        Full-width at half-maximum for PSF to create, if required.
    kernelSize : `float`
        Multiples of sigma for kernel half-size.

    Returns
    -------
    psf : `pfs.drp.stella.SpectralPsf`
        Point-spread function.
    """
    psf = exposure.getPsf()
    if psf is None:
        sigma = fwhmToSigma(fwhm)
        size = 2*int(sigma*kernelSize + 0.5) + 1
        psf = GaussianPsf(size, size, sigma)
    if detectorMap is not None and not isinstance(psf, SpectralPsf):
        psf = ImagingSpectralPsf(psf, detectorMap)
    exposure.setPsf(psf)
    return psf
