import dataclasses
from types import SimpleNamespace
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import convolve1d
from scipy.signal import savgol_coeffs

__all__ = ("robustRms", "fitStraightLine", "ChisqList", "savgol_filter")


def robustRms(array: ArrayLike, nanSafe=False) -> float:
    """Calculate a robust RMS of the array using the inter-quartile range

    Uses the standard conversion of IQR to RMS for a Gaussian.

    Parameters
    ----------
    array : `numpy.ndarray`
        Array for which to calculate RMS.
    nanSafe : `bool`, optional
        If `True`, ignore NaN values in the array.

    Returns
    -------
    rms : `float`
        Robust RMS.
    """
    if nanSafe:
        array = np.asarray(array)
        array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    lq, uq = np.percentile(array, (25.0, 75.0))
    return 0.741 * (uq - lq)


def fitStraightLine(xx: np.ndarray, yy: np.ndarray) -> SimpleNamespace:
    """Fit a straight line, y = slope*x + intercept

    Parameters
    ----------
    xx : `numpy.ndarray` of `float`, size ``N``
        Ordinate.
    yy : `numpy.ndarray` of `float`, size ``N``
        Co-ordinate.

    Returns
    -------
    slope : `float`
        Slope of line.
    intercept : `float`
        Intercept of line.
    xMean : `float`
        Mean of x values.
    yMean : `float`
        Mean of y values.
    """
    xMean = xx.mean()
    yMean = yy.mean()
    dx = xx - xMean
    dy = yy - yMean
    xySum = np.sum(dx * dy)
    xxSum = np.sum(dx**2)
    slope = xySum / xxSum
    intercept = yMean - slope * xMean
    return SimpleNamespace(slope=slope, intercept=intercept, xMean=xMean, yMean=yMean)


@dataclasses.dataclass
class ChisqList:
    """List of chi^2 for various models compared to a common observed data.

    Parameters
    ----------
    chisq : `np.ndarray` of `float`
        chi^2.
    dof : int
        degree of freedom.
    """

    chisq: np.ndarray
    dof: int

    def toProbability(self, *, prior: Union[ArrayLike, None] = None) -> np.ndarray:
        """Convert ``self.chisq`` to probability distribution

        Parameters
        ----------
        prior : `np.ndarray` of `float`, optional
            Prior probability distribution. If this is not given,
            the uniform distribution is assumed.

        Returns
        -------
        pdf : `np.ndarray` of `float`
            ``pdf[i]`` is the probability that the i-th model is the truth.
        """
        minchisq = np.min(self.chisq)
        if minchisq == np.inf:
            # The condition implies that all elements of `chisq[]`
            # are equal to each other, being `inf`.
            # We assume the likelihood uniformly distributed in this case.
            prob = np.ones(shape=self.chisq.shape, dtype=float)
        else:
            # We compute not `exp(-chisq/2)` but `exp((minchisq - chisq)/2)`
            # to avoid underflow.
            delta_chisq = self.chisq - minchisq
            prob = np.exp(delta_chisq / (-2.0))

        if prior is not None:
            prob *= prior
        return prob / np.sum(prob)


def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0,
                  axis=-1, mode='interp', cval=0.0, xerr=None):
    """ Apply a Savitzky-Golay filter to an array, including an error estimate
    Options as for scipy.signal.savgol_filter, with the extension:

    xerr: `np.ndarray` of `float`
       Errors in x, or `None`.  Taken to be independent

    Returns
    -------
       `np.ndarray` of `float` (if xerr is None)
          the filtered values of x
       (`np.ndarray` of `float`, `np.ndarray` of `float`) (if xerr is not None)
         the filtered values of x and the errors in those values
    """

    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' "
                         "'wrap' or 'interp'.")

    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

    if mode == "interp":
        # Do not pad.  Instead, for the elements within `window_length // 2`
        # of the ends of the sequence, use the polynomial that is fitted to
        # the last `window_length` elements.
        from scipy.signal._savitzky_golay import _fit_edges_polyfit  # N.b. importing internal function

        y = convolve1d(x, coeffs, axis=axis, mode="constant")
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)

        if xerr is not None:
            yerr = np.sqrt(convolve1d(xerr**2, coeffs**2, axis=axis, mode="constant"))
            _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, yerr)
    else:
        # Any mode other than 'interp' is passed on to ndimage.convolve1d.
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)
        if xerr is not None:
            yerr = np.sqrt(convolve1d(xerr**2, coeffs**2, axis=axis, mode=mode, cval=cval))

    return y if xerr is None else (y, yerr)
