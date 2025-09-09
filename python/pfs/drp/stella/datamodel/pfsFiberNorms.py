from typing import Optional, Tuple, TYPE_CHECKING
import warnings

import numpy as np

import pfs.datamodel
from pfs.utils.fibers import spectrographFromFiberId, fiberHoleFromFiberId

from ..utils.math import robustRms

if TYPE_CHECKING:
    import matplotlib

__all__ = ("PfsFiberNorms",)


class PfsFiberNorms(pfs.datamodel.PfsFiberNorms):
    @property
    def spectrograph(self):
        """Return spectrograph number"""
        return spectrographFromFiberId(self.fiberId)

    @property
    def fiberHole(self):
        """Return fiber hole number"""
        return fiberHoleFromFiberId(self.fiberId)

    def __imul__(self, rhs):
        """In-place multiplication"""
        if isinstance(rhs, pfs.datamodel.PfsFiberNorms):
            rhs = rhs.values
        rhs = np.array(rhs).copy()  # Ensure rhs does not share memory with an element of self
        with np.errstate(invalid="ignore"):
            self.values *= rhs
        return self

    def __itruediv__(self, rhs):
        """In-place division"""
        if isinstance(rhs, pfs.datamodel.PfsFiberNorms):
            rhs = rhs.values
        rhs = np.array(rhs).copy()  # Ensure rhs does not share memory with an element of self
        with np.errstate(divide="ignore"):
            self.values /= rhs
        return self

    def plot(
        self,
        pfsConfig: pfs.datamodel.PfsConfig,
        axes: Optional["matplotlib.axes.Axes"] = None,
        lower: float = 2.5,
        upper: float = 2.5,
        size: float = 10,
    ) -> Tuple["matplotlib.figure.Figure", "matplotlib.axes.Axes"]:
        """Plot fiber normalization values

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration for the PFS system
        axes : `matplotlib.axes.Axes`, optional
            Axes to plot on; if None, create a new figure and axes.
        lower, upper : `float`
            Lower and upper bounds for plot. This is in units of standard
            deviations if positive; otherwise it is in units of the data. In
            both cases, the bounds are relative to the median value.
        size : `float`
            Size of the markers.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot.
        axes : `matplotlib.axes.Axes`
            Axes containing the plot.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from matplotlib.colors import Normalize
        from ..fitDistortedDetectorMap import addColorbar

        cmap = matplotlib.cm.coolwarm
        if axes is None:
            fig, axes = plt.subplots()
        else:
            fig = axes.figure

        pfsConfig = pfsConfig.select(fiberId=self.fiberId)
        indices = np.argsort(pfsConfig.fiberId)
        assert np.array_equal(pfsConfig.fiberId[indices], self.fiberId)
        xx = pfsConfig.pfiCenter[indices, 0]
        yy = pfsConfig.pfiCenter[indices, 1]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            values = np.nanmedian(self.values, axis=1)
        good = np.isfinite(values)
        median = np.median(values[good])
        rms = robustRms(values[good])
        if lower > 0:
            vmin = max(median - lower*rms, np.nanmin(values))
        else:
            vmin = median - abs(lower)
        if upper > 0:
            vmax = min(median + upper*rms, np.nanmax(values))
        else:
            vmax = median + abs(upper)
        norm = Normalize(vmin=vmin, vmax=vmax)

        axes.scatter(xx, yy, marker="o", c=values, cmap=cmap, norm=norm, s=size)
        axes.set_aspect("equal")
        addColorbar(fig, axes, cmap, norm, "Fiber normalization")

        return fig, axes
