from typing import Any, Callable, Dict, Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
from scipy import interpolate

from lsst.utils import continueClass
from lsst.afw.image import Mask

from pfs.datamodel import PfsFiberArray, MaskHelper, Observations

from .Spectrum import Spectrum


if TYPE_CHECKING:
    from matplotlib.pyplot import Figure, Axes

__all__ = ["Spectrum"]


@continueClass  # noqa: F811 (redefinition)
class Spectrum:  # type: ignore  # noqa: F811 (redefinition)
    """Flux as a function of wavelength"""

    # Types for interfaces defined in C++ pybind layer; useful for typing in this file
    # Better types provided in stub file.
    flux: np.ndarray
    background: np.ndarray
    norm: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray
    wavelength: np.ndarray
    mask: Mask
    normFlux: np.ndarray
    fiberId: int
    getNumPixels: Callable[["Spectrum"], int]
    __len__: Callable[["Spectrum"], int]
    isWavelengthSet: Callable[["Spectrum"], bool]

    def plot(
        self, numRows: int = 3, doBackground: bool = False, filename: str = None
    ) -> Tuple["Figure", "Axes"]:
        """Plot spectrum

        Parameters
        ----------
        numRows : `int`
            Number of row panels over which to plot the spectrum.
        doBackground : `bool`, optional
            Plot the background values in addition to the flux values?
        filename : `str`, optional
            Name of file to which to write the plot. If a ``filename`` is
            specified, the matplotlib `figure` will be closed.

        Returns
        -------
        figure : `matplotlib.figure`
            Figure on which we plotted.
        axes : `list` of `matplotlib.Axes`
            Axes on which we plotted.
        """
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(numRows)

        division = np.linspace(0, len(self), numRows + 1, dtype=int)[1:-1]
        self.plotDivided(axes, division, doBackground=doBackground)

        if filename is not None:
            figure.savefig(filename, bbox_inches='tight')
            plt.close(figure)
        return figure, axes

    def plotDivided(
        self,
        axes: "Axes",
        division: np.ndarray,
        doBackground: bool = False,
        fluxStyle: Dict[str, Any] = None,
        backgroundStyle: Dict[str, Any] = None,
    ):
        """Plot spectrum that has been divided into parts

        This is intended as the implementation of the guts of the ``plot``
        method, but it may be of use elsewhere.

        Parameters
        ----------
        axes : `list` of `matplotlib.Axes`
            Axes on which to plot.
        division : `numpy.ndarray`
            Array of indices at which to divide the spectrum. This should be one
            element shorter than the list of ``axes``.
        doBackground : `bool`, optional
            Plot the background values in addition to the flux values?
        fluxStyle : `dict`
            Arguments for the ``matplotlib.Axes.plot`` method when plotting the
            flux vector.
        backgroundStyle : `dict`
            Arguments for the ``matplotlib.Axes.plot`` method when plotting the
            background vector.
        """
        # subplots(1) returns an Axes not embedded in a list
        try:
            axes[0]
        except TypeError:
            axes = [axes]

        assert len(axes) == len(division) + 1, ("Axes/division length mismatch: %d vs %d" %
                                                (len(axes), len(division)))
        if fluxStyle is None:
            fluxStyle = dict(ls="solid", color="black")
        if backgroundStyle is None:
            backgroundStyle = dict(ls="solid", color="blue")

        useWavelength = self.wavelength
        if np.all(useWavelength == 0.0):
            useWavelength = np.arange(len(self), dtype=float)
        wavelength = np.split(useWavelength, division)
        flux = np.split(self.flux, division)
        if doBackground:
            background = np.split(self.background, division)

        for ii, ax in enumerate(axes):
            ax.plot(wavelength[ii], flux[ii], **fluxStyle)
            if doBackground:
                ax.plot(wavelength[ii], background[ii], **backgroundStyle)

    def wavelengthToPixels(self, wavelength: ArrayLike) -> ArrayLike:
        """Convert wavelength to pixels

        Parameters
        ----------
        wavelength : array_like of `float`
            Wavelength value(s) to convert to pixels.

        Returns
        -------
        pixels : array_like of `float`
            Corresponding pixel value(s).
        """
        return interpolate.interp1d(self.wavelength, np.arange(len(self)))(wavelength)

    def pixelsToWavelength(self, pixels: ArrayLike) -> ArrayLike:
        """Convert pixels to wavelength

        Parameters
        ----------
        pixels : array_like of `float`
            Pixel value(s) to convert to wavelength.

        Returns
        -------
        wavelength : array_like of `float`
            Corresponding wavelength value(s).
        """
        return interpolate.interp1d(np.arange(len(self)), self.wavelength, assume_sorted=True)(pixels)
