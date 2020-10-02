import numpy as np
from scipy.interpolate import interp1d

__all__ = ["interpolateFlux", "interpolateMask"]


def interpolateFlux(fromWavelength, fromFlux, toWavelength, fill=0.0):
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

    Returns
    -------
    toFlux : `numpy.ndarray` of `float`
        Target flux-(like) array.
    """
    with np.errstate(invalid="ignore"):
        return interp1d(fromWavelength, fromFlux, kind="linear", bounds_error=False,
                        fill_value=fill, copy=True, assume_sorted=True)(toWavelength)


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
