import numpy as np

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task, Struct
import lsstDebug

__all__ = ("FitPolynomialConfig", "FitPolynomialTask")


class FitPolynomialConfig(Config):
    """Configuration for FitPolynomialTask"""
    order = Field(dtype=int, default=5, doc="Polynomial order")
    rejIter = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejThresh = Field(dtype=float, default=3.0, doc="Rejection threshold (sigma)")
    easeOff = Field(dtype=int, default=5,
                    doc="Number of factors of two we should try when easing off rejection")


class FitPolynomialTask(Task):
    """Fit a 1D polynomial

    We fit a polynomial with optional rejection.

    Different functional forms can be implemented by replacing the ``fit``
    method.
    """
    ConfigClass = FitPolynomialConfig
    _DefaultName = "fitPolynomial"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, xx, yy, good=None, xMin=None, xMax=None):
        """Perform iterative fitting of a polynomial

        Parameters
        ----------
        xx : array_like, shape ``(N,)``
            Ordinate values.
        yy : array_like, shape ``(N,)``
            Coordinate values.
        good : array_like of `bool`, shape ``(N,)``
            Boolean array indicating if values are believed to be good.
        xMin, xMax : `int`, optional
            Minimum and maximum ordinate values.

        Returns
        -------
        func : callable
            Function that returns the fit value.
        fit : `numpy.ndarray`, shape ``(N,)``
            Values of the function at each of the input ``xx`` values.
        residuals : `numpy.ndarray`, shape ``(N,)``
            Fit residuals (``yy - fit``).
        rms : `float`
            Robust RMS of the residuals (from quartiles).
        good : `numpy.ndarray` of `bool`, shape ``(N,)``
            Boolean array indicating whether each point was used in the fit.
        """
        if good is None:
            good = np.ones_like(xx, dtype=bool)
        numGood = good.sum()
        for ii in range(self.config.rejIter):
            func = self.fit(xx, yy, good)
            fit = self.characterize(xx, yy, good, func)
            if self.debugInfo.plot:
                self.plot(xx, yy, good, fit)
            median = np.median(fit.residuals)
            for easeOff in range(self.config.easeOff + 1):  # Ease off rejection
                newGood = np.abs(fit.residuals - median) < fit.rms*self.config.rejThresh*2**easeOff
                newNumGood = newGood.sum()
                if newNumGood > self.config.order:
                    break
            else:
                raise RuntimeError("Rejection removed too many points")
            if newNumGood == numGood:
                break
            good = newGood
            numGood = newNumGood
        else:
            func = self.fit(xx, yy, good)
            fit = self.characterize(xx, yy, good, func)
            if self.debugInfo.plot:
                self.plot(xx, yy, good, fit)
        fit.good = good
        return fit

    def fit(self, xx, yy, good, xMin=None, xMax=None):
        """Fit a polynomial

        The rejection already determined is applied; no further rejection or
        iteration is performed here.

        Parameters
        ----------
        xx : array_like, shape ``(N,)``
            Ordinate values.
        yy : array_like, shape ``(N,)``
            Coordinate values.
        good : array_like of `bool`, shape ``(N,)``
            Boolean array indicating whether each point should be used in the
            fit.
        xMin, xMax : `int`, optional
            Minimum and maximum ordinate values.

        Returns
        -------
        func : callable
            Function that returns the fit value.
        """
        domain = (xMin, xMax) if xMin is not None and xMax is not None else None
        return np.polynomial.Chebyshev.fit(xx[good], yy[good], self.config.order, domain=domain)

    def characterize(self, xx, yy, good, func):
        """Characterize the polynomial fit

        Parameters
        ----------
        xx : array_like, shape ``(N,)``
            Ordinate values.
        yy : array_like, shape ``(N,)``
            Coordinate values.
        good : array_like of `bool`, shape ``(N,)``
            Boolean array indicating whether each point should be used in the
            fit.
        func : callable
            Function that returns the fit value.

        Returns
        -------
        func : callable
            Function that returns the fit value.
        fit : `numpy.ndarray`, shape ``(N,)``
            Values of the function at each of the input ``xx`` values.
        residuals : `numpy.ndarray`, shape ``(N,)``
            Fit residuals (``yy - fit``).
        rms : `float`
            Robust RMS of the residuals (from quartiles).
        """
        fit = func(xx)
        residuals = yy - fit
        lq, uq = np.percentile(residuals[good], [25.0, 75.0])
        rms = 0.741*(uq - lq)
        return Struct(func=func, fit=fit, residuals=residuals, rms=rms)

    def plot(self, xx, yy, good, fit):
        """Plot the polynomial fit

        Parameters
        ----------
        xx : array_like, shape ``(N,)``
            Ordinate values.
        yy : array_like, shape ``(N,)``
            Coordinate values.
        good : array_like of `bool`, shape ``(N,)``
            Boolean array indicating whether each point should be used in the
            fit.
        fit : `lsst.pipe.base.Struct`
            Fit results from the ``fit`` method.
        """
        import matplotlib.pyplot as plt

        plt.figure().subplots_adjust(hspace=0)
        axes = []
        axes.append(plt.subplot2grid((4, 1), (0, 0), rowspan=3))
        axes.append(plt.subplot2grid((4, 1), (3, 0), sharex=axes[0]))

        axes[0].plot(xx[good], yy[good], 'bo')
        axes[1].plot(xx[good], fit.residuals[good], 'bo')

        if not np.all(good):
            axes[0].plot(xx[~good], yy[~good], 'ro')
            axes[1].plot(xx[~good], fit.residuals[~good], 'ro')

        xMin, xMax = axes[0].get_xlim()
        sample = np.linspace(xMin, xMax, 1000)
        axes[0].plot(sample, fit.func(sample), 'k-')
        axes[1].axhline(0, ls=':', color='black')

        axes[0].set_title("Polynomial fit")
        axes[0].set_ylabel("y")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("Residuals")

        plt.show()
