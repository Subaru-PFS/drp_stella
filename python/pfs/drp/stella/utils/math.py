import dataclasses
from types import SimpleNamespace
from typing import Union
import numpy as np
from numpy.typing import ArrayLike

__all__ = ("robustRms", "fitStraightLine", "ChisqList")


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
