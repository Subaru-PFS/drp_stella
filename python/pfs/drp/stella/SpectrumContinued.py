import numpy as np
from scipy import interpolate

from lsst.utils import continueClass

from .Spectrum import Spectrum

__all__ = ["Spectrum"]


@continueClass  # noqa: F811 (redefinition)
class Spectrum:  # noqa: F811 (redefinition)
    """Flux as a function of wavelength"""
    def plot(self, numRows=3, doBackground=False, filename=None):
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

    def plotDivided(self, axes, division, doBackground=False, fluxStyle=None, backgroundStyle=None):
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

        useWavelength = self.getWavelength()
        if np.all(useWavelength == 0.0):
            useWavelength = np.arange(len(self), dtype=float)
        wavelength = np.split(useWavelength, division)
        flux = np.split(self.getSpectrum(), division)
        if doBackground:
            background = np.split(self.getBackground(), division)

        for ii, ax in enumerate(axes):
            ax.plot(wavelength[ii], flux[ii], **fluxStyle)
            if doBackground:
                ax.plot(wavelength[ii], background[ii], **backgroundStyle)

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
