from types import SimpleNamespace
import numpy as np

from lsst.afw.geom import SpanSet
from lsst.afw.math import GaussianFunction1D, IntegerDeltaFunction1D
from lsst.afw.math import SeparableKernel, convolve, ConvolutionControl

from .utils.psf import fwhmToSigma

__all__ = ["getIndices", "calculateCentroid", "calculateSecondMoments", "convolveImage"]


def getIndices(bbox, dtype=float):
    """Return x and y indices for an image, given the bounding box

    Parameters
    ----------
    bbox : `lsst.geom.Box2I`
        Bounding box for image.
    dtype : numerical type
        Data type for indices.

    Returns
    -------
    xx, yy : `numpy.ndarray` of ``dtype``
        Indices for the pixels in the image in x and y.
    """
    return (np.arange(bbox.getMinX(), bbox.getMaxX() + 1, dtype=dtype)[np.newaxis, :],
            np.arange(bbox.getMinY(), bbox.getMaxY() + 1, dtype=dtype)[:, np.newaxis])


def calculateCentroid(image):
    """Calculate centroid for an image

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image on which to calculate centroid.

    Returns
    -------
    centroid : `types.SimpleNamespace`
        Centroid coordinates in the ``x`` and ``y`` attributes.
    """
    xx, yy = getIndices(image.getBBox())
    norm = np.sum(image.array.astype(float))
    xCen = np.sum(np.sum(image.array.astype(float), axis=0)*xx)/norm
    yCen = np.sum(np.sum(image.array.astype(float), axis=1)*yy.T)/norm
    return SimpleNamespace(x=xCen, y=yCen)


def calculateSecondMoments(image, centroid=None):
    """Calculate second moments for an image

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image on which to calculate centroid.
    centroid : `types.SimpleNamespace`, optional
        Object with ``x`` and ``y`` attributes with the centroids.

    Returns
    -------
    moments : `types.SimpleNamespace`
        Second moments in the ``xx``, ``yy`` and ``xy`` attributes.
    """
    if centroid is None:
        centroid = calculateCentroid(image)
    norm = np.sum(image.array.astype(float))
    xx, yy = getIndices(image.getBBox())
    xWidth = np.sum(np.sum(image.array.astype(float), axis=0)*(xx - centroid.x)**2)/norm
    yWidth = np.sum(np.sum(image.array.astype(float), axis=1)*(yy.T - centroid.y)**2)/norm
    xyWidth = np.sum(image.array.astype(float)*(xx - centroid.x)*(yy - centroid.y))/norm
    return SimpleNamespace(xx=xWidth, yy=yWidth, xy=xyWidth)


def convolveImage(maskedImage, colFwhm, rowFwhm=None, growMask=1, kernelSize=4.0, sigmaNotFwhm=False):
    """Convolve image by Gaussian kernels in x and y

    The convolution kernel size can be specified separately for the columns
    and rows in the config.

    The boundary is unconvolved, and is set to ``NaN``.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        Image to convolve.
    colFwhm : `float`
        FWHM (pixels) of kernel across columns (spatial dimension).
    rowFwhm : `float`, optional
        FWHM (pixels) of kernel across rows (spectral dimension). If not
        provided, will be identical to ``colFwhm``.
    growMask : `int`, optional
        Number of pixels to grow mask.
    kernelSize : `float`, optional
        Half-size of convolution kernel, relative to sigma.
    sigmaNotFwhm : `bool`
        ``colFwhm`` and ``rowFwhm`` are actually Gaussian sigma, not FWHM.

    Returns
    -------
    convolved : `lsst.afw.image.MaskedImage`
        Convolved image.
    """
    if rowFwhm is None:
        rowFwhm = colFwhm

    def sigmaToSize(sigma):
        """Determine kernel size from Gaussian sigma"""
        return 2*int(kernelSize*sigma) + 1

    xSigma = colFwhm if sigmaNotFwhm else fwhmToSigma(colFwhm)
    ySigma = rowFwhm if sigmaNotFwhm else fwhmToSigma(rowFwhm)

    xKernel = GaussianFunction1D(xSigma) if xSigma > 0 else IntegerDeltaFunction1D(0.0)
    yKernel = GaussianFunction1D(ySigma) if ySigma > 0 else IntegerDeltaFunction1D(0.0)

    kernel = SeparableKernel(sigmaToSize(xSigma), sigmaToSize(ySigma), xKernel, yKernel)

    convolvedImage = maskedImage.Factory(maskedImage.getBBox())
    convolve(convolvedImage, maskedImage, kernel, ConvolutionControl())

    # Redo the convolution of the mask plane, using a smaller kernel
    mask = convolvedImage.mask
    mask.array[:] = maskedImage.mask.array
    if growMask > 0:
        for name in convolvedImage.mask.getMaskPlaneDict():
            bitmask = convolvedImage.mask.getPlaneBitMask(name)
            SpanSet.fromMask(mask, bitmask).dilated(growMask).clippedTo(mask.getBBox()).setMask(mask, bitmask)

    return convolvedImage
