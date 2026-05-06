from __future__ import annotations

from typing import Any

import numpy as np

from lsst.utils import continueClass
from lsst.geom import Extent2I

from .FiberKernel import FiberKernel
from pfs.datamodel.pfsFiberKernel import PfsFiberKernel

__all__ = ["FiberKernel"]


@continueClass  # noqa: F811 redefinition
class FiberKernel:  # noqa: F811 (redefinition)
    def toDatamodel(self, metadata: dict[str, Any] | None = None) -> PfsFiberKernel:
        """Convert to a PfsFiberKernel datamodel instance"""
        return PfsFiberKernel(
            imageWidth=self.dims.getX(),
            imageHeight=self.dims.getY(),
            halfWidth=self.halfWidth,
            xNumBlocks=self.xNumBlocks,
            yNumBlocks=self.yNumBlocks,
            values=self.values,  # Coefficients and values are the same for now
            metadata=metadata or {},
        )

    @classmethod
    def fromDatamodel(cls, kernel: PfsFiberKernel) -> "FiberKernel":
        """Construct from a PfsFiberKernel datamodel instance"""
        return cls(
            dims=Extent2I(kernel.imageWidth, kernel.imageHeight),
            halfWidth=kernel.halfWidth,
            xNumBlocks=kernel.xNumBlocks,
            yNumBlocks=kernel.yNumBlocks,
            values=kernel.values,
        )

    def writeFits(self, filename):
        """Write to a FITS file"""
        self.toDatamodel().writeFits(filename)

    @classmethod
    def readFits(cls, filename) -> "FiberKernel":
        """Read from a FITS file"""
        return cls.fromDatamodel(PfsFiberKernel.readFits(filename))

    def plot(
        self,
        figArgs: dict | None = None,
        plotArgs: dict | None = None,
        textArgs: dict | None = None,
        suppressCenter: bool = False,
    ) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Plot the kernel values as a function of offset for each block.

        Parameters
        ----------
        figArgs : `dict`, optional
            Arguments to pass to `matplotlib.pyplot.figure` when creating the
            figure.
        plotArgs : `dict`, optional
            Arguments to pass to `matplotlib.axes.Axes.plot` when plotting the
            kernel values.
        textArgs : `dict`, optional
            Arguments to pass to `matplotlib.axes.Axes.text` when plotting the
            centroid and RMS values.
        suppressCenter : `bool`, optional
            If `True`, subtract the center pixel value from each block; allows
            the shape of the kernel to be more easily seen when the center value
            is much larger than the others.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The created figure.
        axes : `numpy.ndarray` of `matplotlib.axes.Axes`
            The axes containing the plots for each block.
        """
        import matplotlib.pyplot as plt

        if figArgs is None:
            figArgs = dict(figsize=(10, 10))
        if plotArgs is None:
            plotArgs = dict(color="k", marker="o", linestyle="-")
        if textArgs is None:
            textArgs = dict(x=0.05, y=0.95, va="top", fontsize=6)

        images = self.makeOffsetImages(self.xNumBlocks, self.yNumBlocks)

        fig, axes = plt.subplots(nrows=self.yNumBlocks, ncols=self.xNumBlocks, sharex=True, sharey=True)

        indices = np.arange(-self.halfWidth, self.halfWidth + 1)
        for ii in range(self.xNumBlocks):
            for jj in range(self.yNumBlocks):
                array = images[:, jj, ii]

                centroid = (indices*array).sum()/array.sum()
                rms = np.sqrt((indices**2*array).sum()/array.sum())

                if suppressCenter:
                    array[self.halfWidth] -= 1.0
                ax = axes[jj, ii]
                ax.plot(indices, array, **plotArgs)
                if not suppressCenter:
                    string = rf"$\mu$={centroid:.2f}" + "\n" + rf"$\sigma$={rms:.2f}"
                    ax.text(s=string, transform=ax.transAxes, **textArgs)
                    ax.axvline(0.0, color="grey", linestyle=":")

                ax.set_xlabel("dx")
                ax.set_ylabel("Value")
                ax.label_outer()

        fig.subplots_adjust(wspace=0, hspace=0)

        return fig, axes
