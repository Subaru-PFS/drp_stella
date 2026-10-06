import functools
from typing import Callable, Tuple

import numpy as np
import scipy.sparse

__all__ = [
    "binningMatrix1D",
    "binningMatrix2D",
    "makeOversampledGrid",
    "integrateOverPixels",
    "analyticGaussianPixelIntegral",
]


def binningMatrix1D(numPixels: int, oversampling: int) -> scipy.sparse.csr_matrix:
    """Construct a 1D sparse box-integration (binning) operator

    Maps a 1D oversampled grid (length ``numPixels*oversampling``, with the
    ``oversampling`` sub-samples of each pixel contiguous) down to the pixel
    grid (length ``numPixels``), by averaging each pixel's sub-samples. This
    is an exact area-weighted integral (not an interpolation), appropriate
    for a regular oversampled grid with evenly-spaced sub-pixel samples.

    Parameters
    ----------
    numPixels : `int`
        Number of pixels.
    oversampling : `int`
        Number of oversampled sub-pixels per pixel.

    Returns
    -------
    operator : `scipy.sparse.csr_matrix`
        Sparse operator of shape ``(numPixels, numPixels*oversampling)``.
    """
    if oversampling < 1:
        raise ValueError(f"oversampling must be >= 1 (got {oversampling})")
    numOversampled = numPixels * oversampling
    rows = np.repeat(np.arange(numPixels), oversampling)
    cols = np.arange(numOversampled)
    data = np.full(numOversampled, 1.0 / oversampling)
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(numPixels, numOversampled))


@functools.lru_cache(maxsize=None)
def binningMatrix2D(numX: int, numY: int, oversampling: int) -> scipy.sparse.csr_matrix:
    """Construct a 2D sparse box-integration (binning) operator

    Maps a row-major (C-order) flattened oversampled grid of shape
    ``(numY*oversampling, numX*oversampling)`` down to a row-major flattened
    pixel grid of shape ``(numY, numX)``, by averaging each pixel's
    ``oversampling x oversampling`` block of sub-samples. Built as the
    Kronecker product of the two 1D operators (`binningMatrix1D`), which is
    exact for row-major flattening.

    Depends only on its (small integer) arguments, never on data, and is
    called once per line per fit iteration with the same stamp size almost
    every time, so results are memoized.

    Parameters
    ----------
    numX, numY : `int`
        Number of pixels along x and y.
    oversampling : `int`
        Number of oversampled sub-pixels per pixel, per axis.

    Returns
    -------
    operator : `scipy.sparse.csr_matrix`
        Sparse operator of shape
        ``(numX*numY, numX*numY*oversampling**2)``. Callers must not modify
        the returned matrix in place, since it is shared across calls.
    """
    yOperator = binningMatrix1D(numY, oversampling)
    xOperator = binningMatrix1D(numX, oversampling)
    return scipy.sparse.kron(yOperator, xOperator, format="csr")


def makeOversampledGrid(
    xIndices: np.ndarray, yIndices: np.ndarray, oversampling: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an oversampled coordinate grid for a rectangular stamp

    Parameters
    ----------
    xIndices, yIndices : `numpy.ndarray`
        Pixel-center coordinates along x and y (e.g., offsets from a PSF
        center); need not be integers, but are assumed to be evenly spaced
        by 1 (as for detector pixels).
    oversampling : `int`
        Number of oversampled sub-pixels per pixel, per axis.

    Returns
    -------
    xGrid, yGrid : `numpy.ndarray`
        Oversampled coordinate grids, each of shape
        ``(len(yIndices)*oversampling, len(xIndices)*oversampling)``,
        ordered to match `binningMatrix2D`.
    """
    subOffsets = (np.arange(oversampling, dtype=float) + 0.5) / oversampling - 0.5
    xFull = (np.asarray(xIndices, dtype=float)[:, np.newaxis] + subOffsets[np.newaxis, :]).ravel()
    yFull = (np.asarray(yIndices, dtype=float)[:, np.newaxis] + subOffsets[np.newaxis, :]).ravel()
    return np.meshgrid(xFull, yFull, indexing="xy")


def integrateOverPixels(
    evaluate: Callable[[np.ndarray, np.ndarray], np.ndarray],
    xIndices: np.ndarray,
    yIndices: np.ndarray,
    oversampling: int = 4,
) -> np.ndarray:
    """Evaluate a continuous profile and integrate it exactly over pixels

    The profile is undersampled (the PSF core FWHM is under 2 pixels), so it
    must be integrated over each pixel rather than merely sampled at pixel
    centers. This evaluates ``evaluate`` on an oversampled grid and bins the
    result down to pixels with an exact (non-interpolating) sparse operator;
    it does not use `pfs.drp.stella.SpectralPsf.OversampledPsf`'s Lanczos
    resampling, which serves a different purpose (realizing a final `Psf`
    image for the public interface, not integrating a model during a fit).

    Parameters
    ----------
    evaluate : callable
        Function of ``(xGrid, yGrid) -> values``, evaluating the continuous
        profile at the given coordinates. Both ``xGrid.shape ==
        yGrid.shape`` and the returned array shape match; coordinates are
        typically offsets from a PSF center.
    xIndices, yIndices : `numpy.ndarray`
        Pixel-center coordinates (e.g., offsets from a PSF center) along x
        and y, evenly spaced by 1.
    oversampling : `int`, optional
        Number of oversampled sub-pixels per pixel, per axis.

    Returns
    -------
    pixels : `numpy.ndarray`, shape ``(len(yIndices), len(xIndices))``
        The profile integrated over each pixel.
    """
    xGrid, yGrid = makeOversampledGrid(xIndices, yIndices, oversampling)
    values = evaluate(xGrid, yGrid)
    operator = binningMatrix2D(len(xIndices), len(yIndices), oversampling)
    pixels = operator @ values.ravel()
    return pixels.reshape(len(yIndices), len(xIndices))


def analyticGaussianPixelIntegral(
    xIndices: np.ndarray, yIndices: np.ndarray, sigmaX: float, sigmaY: float
) -> np.ndarray:
    """Compute the exact analytic pixel integral of a 2D Gaussian

    Integrating a Gaussian exactly over a unit-width pixel is equivalent to
    evaluating a Gaussian convolved with a unit-width top-hat at the pixel
    center (see `pfs.drp.stella.psfProfiles.tophatConvGaussian1D`). This
    provides a cross-check for `integrateOverPixels` (with `oversampling`
    high enough to be accurate) when the profile being integrated is purely
    a 2D Gaussian.

    Parameters
    ----------
    xIndices, yIndices : `numpy.ndarray`
        Pixel-center coordinates (e.g., offsets from a PSF center) along x
        and y.
    sigmaX, sigmaY : `float`
        Gaussian sigma along x and y.

    Returns
    -------
    pixels : `numpy.ndarray`, shape ``(len(yIndices), len(xIndices))``
        The Gaussian integrated exactly over each pixel.
    """
    from .psfProfiles import tophatConvGaussian1D

    xx = np.asarray(xIndices, dtype=float)
    yy = np.asarray(yIndices, dtype=float)
    integralX = tophatConvGaussian1D(xx, 1.0, sigmaX)
    integralY = tophatConvGaussian1D(yy, 1.0, sigmaY)
    return np.outer(integralY, integralX)
