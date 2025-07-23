import numpy as np
from scipy import interpolate
from typing import TYPE_CHECKING

from lsst.afw.image import Mask
from lsst.utils import continueClass

from .Spectrum import Spectrum

if TYPE_CHECKING:
    from pfs.datamodel import PfsSimpleSpectrum


__all__ = ["Spectrum"]


@continueClass  # noqa: F811 (redefinition)
class Spectrum:  # noqa: F811 (redefinition)
    """Flux as a function of wavelength"""
    def plot(self, numRows=3, filename=None):
        """Plot spectrum

        Parameters
        ----------
        numRows : `int`
            Number of row panels over which to plot the spectrum.
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
        self.plotDivided(axes, division)

        if filename is not None:
            figure.savefig(filename, bbox_inches='tight')
            plt.close(figure)
        return figure, axes

    def plotDivided(self, axes, division, fluxStyle=None):
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
        fluxStyle : `dict`
            Arguments for the ``matplotlib.Axes.plot`` method when plotting the
            flux vector.
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

        useWavelength = self.getWavelength()
        if np.all(useWavelength == 0.0):
            useWavelength = np.arange(len(self), dtype=float)
        wavelength = np.split(useWavelength, division)
        flux = np.split(self.getFlux(), division)

        for ii, ax in enumerate(axes):
            ax.plot(wavelength[ii], flux[ii], **fluxStyle)

    def wavelengthToPixels(self, wavelength):
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

    def pixelsToWavelength(self, pixels):
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

    @classmethod
    def fromPfsSpectrum(cls, spectrum: "PfsSimpleSpectrum", fiberId: int = 0) -> "Spectrum":
        """Create a `Spectrum` from a PFS spectrum

        Like a `pfs.datamodel.PfsSimpleSpectrum`, `pfs.datamodel.PfsFiberArray`,
        `pfs.datamodel.PfsCalibrated` or `pfs.datamodel.PfsObject`.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsSimpleSpectrum`
            PFS spectrum to convert.
        fiberId : `int`, optional
            Fiber ID to use for the spectrum.

        Returns
        -------
        spectrum : `Spectrum`
            Converted spectrum.
        """
        length = len(spectrum.flux)
        norm = np.ones_like(spectrum.flux, dtype=np.float32)
        mask = Mask(length, 1)
        mask.array[:] = spectrum.mask
        if hasattr(spectrum, "variance"):
            variance = spectrum.variance
        else:
            variance = np.ones_like(spectrum.flux, dtype=np.float32)
        wavelength = np.array(spectrum.wavelength, dtype=np.float64)  # Ensure this is a regular array
        result = cls(
            spectrum.flux.astype(np.float32), mask, norm, variance.astype(np.float32), wavelength
        )
        result.setFiberId(fiberId)
        return result
