import numpy as np
from scipy.interpolate import interp1d

__all__ = ["interpolateFlux", "interpolateMask"]


def calculateDispersion(wavelength):
    """Calculate dispersion (nm/pixel) as a function of wavelength

    This might be the inverse of what you consider "dispersion": this is
    wavelength increment per pixel, not pixels per unit wavelength.

    Parameters
    ----------
    wavelength : array_like of `float`
        Wavelength value for each pixel.

    Returns
    -------
    dispersion : `numpy.ndarray` of `float`
        Wavelength increment value for each pixel.
    """
    dispersion = np.empty_like(wavelength)
    dispersion[0] = wavelength[1] - wavelength[0]
    dispersion[-1] = wavelength[-1] - wavelength[-2]
    dispersion[1:-1] = 0.5*(wavelength[2:] - wavelength[0:-2])
    return np.abs(dispersion)


def interpolateFlux(fromWavelength, fromFlux, toWavelength, fill=0.0, jacobian=True, variance=False):
    """Interpolate a flux-like spectrum

    Basic linear interpolation, suitable for fluxes and flux-like (e.g., maybe
    variances) quantities.

    Parameters
    ----------
    fromWavelength : array-like of `float`
        Source wavelength array.
    fromFlux : array-like of `float`
        Source flux(-like) array.
    toWavelength : array-like of `float`
        Target wavelength array.
    fill : `float`, optional
        Fill value.
    jacobian : `bool`, optional
        Correct for the Jacobian of the transformation?
    variance : `bool`, optional
        Calculate Jacobian correction values for variance? The usual Jacobian
        correction values are squared. If ``jacobian=False``, then ``variance``
        doesn't matter.

    Returns
    -------
    toFlux : `numpy.ndarray` of `float`
        Target flux-(like) array.
    """
    if jacobian:
        correction = 1.0/calculateDispersion(fromWavelength)
        if variance:
            correction *= correction
        fromFlux = fromFlux*correction
    with np.errstate(invalid="ignore"):
        toFlux = interp1d(fromWavelength, fromFlux, kind="linear", bounds_error=False,
                          fill_value=fill, copy=True, assume_sorted=True)(toWavelength)
    if jacobian:
        correction = calculateDispersion(toWavelength)
        if variance:
            correction *= correction
        toFlux *= correction
    return toFlux


def interpolateMask(fromWavelength, fromMask, toWavelength, fill=0):
    """Interpolate a mask spectrum

    Linear interpolation for masks.

    Parameters
    ----------
    fromWavelength : array-like of `float`
        Source wavelength array.
    fromMask : array-like of `int`
        Source mask array.
    toWavelength : array-like of `float`
        Target wavelength array.
    fill : `float`, optional
        Fill value.

    Returns
    -------
    toMask : `numpy.ndarray` of `int`
        Target mask array.
    """
    length = len(fromWavelength)
    with np.errstate(invalid="ignore"):
        index = interp1d(fromWavelength, np.arange(length), kind="linear", bounds_error=False,
                         fill_value=-1, copy=True, assume_sorted=True)(toWavelength)
    intIndex = index.astype(int)
    result = np.empty(toWavelength.shape, dtype=fromMask.dtype)
    intIndex[(intIndex == index) & (index > 0)] -= 1  # Linear interpolation takes the index before
    select = (intIndex >= 0) & (intIndex < length - 1)
    result[select] = fromMask[intIndex[select]] | fromMask[intIndex[select] + 1]
    result[~select] = fill
    return result
