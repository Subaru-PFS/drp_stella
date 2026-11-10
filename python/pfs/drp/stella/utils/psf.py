import math

from lsst.afw.detection import GaussianPsf

__all__ = ("CENTROID_FWHM", "checkPsf", "fwhmToSigma", "sigmaToFwhm", "makeGaussianPsf")


CENTROID_FWHM = 2.9
"""FWHM (pixels) of the Gaussian smoothing kernel used for centroiding

This is shared by the line (`pfs.drp.stella.centroidLines.CentroidLinesTask`)
and trace (`pfs.drp.stella.centroidTraces.CentroidTracesTask`) centroiders so
that their measurements are consistent. The value is the measured width of
the PSF (sigma ~ 1.24 pixels in the spatial dimension, from the adaptive
second moments of arc lines in r1), which is close to that of the other arms.
"""


def fwhmToSigma(fwhm):
    """Convert FWHM to sigma for a Gaussian"""
    return fwhm/(2*math.sqrt(2*math.log(2)))


def sigmaToFwhm(sigma):
    """Convert sigma to FWHM for a Gaussian"""
    return sigma*(2*math.sqrt(2*math.log(2)))


def makeGaussianPsf(fwhm, kernelSize=4.0):
    """Make a Gaussian PSF

    Parameters
    ----------
    fwhm : `float`
        Full-width at half-maximum of the PSF (pixels).
    kernelSize : `float`
        Multiples of sigma for kernel half-size.

    Returns
    -------
    psf : `lsst.afw.detection.GaussianPsf`
        Gaussian point-spread function.
    """
    sigma = fwhmToSigma(fwhm)
    size = 2*int(sigma*kernelSize + 0.5) + 1
    return GaussianPsf(size, size, sigma)


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
        psf = makeGaussianPsf(fwhm, kernelSize)
        exposure.setPsf(psf)
    return psf
