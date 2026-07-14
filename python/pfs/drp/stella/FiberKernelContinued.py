from __future__ import annotations

from typing import Any

import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

from lsst.utils import continueClass
from lsst.geom import Extent2I

from .FiberKernel import FiberKernel
from .fiberProfile import FiberProfile
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

    def convolveProfile(
        self, profile: FiberProfile, xCenter: np.ndarray, order: int = 1
    ) -> FiberProfile:
        """Convolve a fiber profile with the kernel

        We oversample the kernel by the profile's oversampling factor (with
        spline interpolation of the specified order) and then convolve with
        the profile.

        Parameters
        ----------
        profile : `FiberProfile`
            The fiber profile to convolve.
        xCenter : `numpy.ndarray`
            The x center of the fiber profile for each profile sample:
            ``detectorMap.getXCenter(fiberId, profile.rows)``.
        order : `int`, optional
            The interpolation order to use when oversampling the kernel.

        Returns
        -------
        convolved : `FiberProfile`
            The convolved fiber profile. The normalization is not preserved:
            you should re-measure the normalization with the convolved profile.
        """
        convolved = profile.profiles.copy()
        oversample = profile.oversample
        for ii, (xx, yy) in enumerate(zip(xCenter, profile.rows)):
            kernel = self.evaluate(xx, yy)
            for jj in range(oversample):
                convolved[ii, jj::oversample] = scipy.signal.convolve(
                    profile.profiles[ii, jj::oversample], kernel, mode="same", method="direct"
                )
        return FiberProfile(
            profile.radius, profile.oversample, profile.rows.copy(), convolved, norm=None
        )

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
        if figArgs is None:
            figArgs = dict(figsize=(10, 10))
        if plotArgs is None:
            plotArgs = dict(color="k", marker="o", linestyle="-")
        if textArgs is None:
            textArgs = dict(x=0.05, y=0.95, va="top", fontsize=6)

        images = self.makeOffsetImages(self.xNumBlocks, self.yNumBlocks)

        fig, axes = plt.subplots(
            nrows=self.yNumBlocks, ncols=self.xNumBlocks, sharex=True, sharey=True, squeeze=False
        )

        indices = np.arange(-self.halfWidth, self.halfWidth + 1)
        for ii in range(self.xNumBlocks):
            for jj in range(self.yNumBlocks):
                array = images[:, jj, ii]

                centroid = (indices*array).sum()/array.sum()
                rms = np.sqrt(((indices - centroid)**2*array).sum()/array.sum())

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
