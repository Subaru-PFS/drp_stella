from types import SimpleNamespace
import numpy as np
from numpy.typing import ArrayLike

__all__ = ("robustRms", "fitStraightLine")


def robustRms(array: ArrayLike) -> float:
    """Calculate a robust RMS of the array using the inter-quartile range

    Uses the standard conversion of IQR to RMS for a Gaussian.

    Parameters
    ----------
    array : `numpy.ndarray`
        Array for which to calculate RMS.

    Returns
    -------
    rms : `float`
        Robust RMS.
    """
    lq, uq = np.percentile(array, (25.0, 75.0))
    return 0.741*(uq - lq)


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
    xySum = np.sum(dx*dy)
    xxSum = np.sum(dx**2)
    slope = xySum/xxSum
    intercept = yMean - slope*xMean
    return SimpleNamespace(slope=slope, intercept=intercept, xMean=xMean, yMean=yMean)
