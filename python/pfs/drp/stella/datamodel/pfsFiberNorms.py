from typing import TYPE_CHECKING, Optional

import pfs.datamodel

if TYPE_CHECKING:
    from matplotlib import Figure, Axes

__all__ = ("PfsFiberNorms",)


class PfsFiberNorms(pfs.datamodel.PfsFiberNorms):
    # Add the plot function as a class method.
    def plot(self,
             pfsConfig: pfs.datamodel.PfsConfig,
             axes: Optional["Axes"] = None,
             lower: float = 2.5,
             upper: float = 2.5,
             size: float = 10,
             title: Optional[str] = None
             ) -> "Figure":
        """Plot fiber normalization values

        Parameters
        ----------
        fiberNorms : `PfsFiberNorms`
            Fiber normalization values.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration for the PFS system
        axes : `matplotlib.axes.Axes`, optional
            Axes to plot on; if None, create a new figure and axes.
        lower, upper : `float`
            Lower and upper bounds for plot, in units of standard deviations. Default is 2.5.
        size : `float`
            Size of the markers. Default is 10.
        title : `str`, optional
            Title for the plot.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot.
        """
        from pfs.drp.qa.utils.plotting import plotFiberNorms
        return plotFiberNorms(self, pfsConfig, axes=axes, lower=lower, upper=upper, size=size, title=title)
