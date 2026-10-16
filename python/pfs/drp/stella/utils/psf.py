import math

from lsst.afw.detection import GaussianPsf

__all__ = ("checkPsf", "fwhmToSigma", "sigmaToFwhm")


def fwhmToSigma(fwhm):
    """Convert FWHM to sigma for a Gaussian"""
    return fwhm/(2*math.sqrt(2*math.log(2)))


def sigmaToFwhm(sigma):
    """Convert sigma to FWHM for a Gaussian"""
    return sigma*(2*math.sqrt(2*math.log(2)))


def checkPsf(exposure, fwhm=1.5, kernelSize=4.0):
    """Ensure that the exposure has a Psf

    Creates the PSF in the exposure if it doesn't have one.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Exposure of interest.
    fwhm : `float`
        Full-width at half-maximum for PSF to create, if required.
    kernelSize : `float`
        Multiples of sigma for kernel half-size.

    Returns
    -------
    psf : `lsst.afw.detection.Psf`
        Point-spread function.
    """
    psf = exposure.getPsf()
    if psf is None:
        sigma = fwhmToSigma(fwhm)
        size = 2*int(sigma*kernelSize + 0.5) + 1
        psf = GaussianPsf(size, size, sigma)
        exposure.setPsf(psf)
    return psf
