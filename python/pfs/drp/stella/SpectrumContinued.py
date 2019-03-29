import numpy as np

from lsst.utils import continueClass

from .ReferenceLine import ReferenceLine
from .Spectrum import Spectrum

import math
from scipy import interpolate
import astropy.modeling

__all__ = ["Spectrum"]

BAD_REFERENCE = (ReferenceLine.MISIDENTIFIED | ReferenceLine.CLIPPED | ReferenceLine.SATURATED |
                 ReferenceLine.INTERPOLATED | ReferenceLine.CR)


def plotReferenceLines(ax, referenceLines, flux, wavelength, badReference=BAD_REFERENCE):
    """Plot reference lines on a spectrum

    Parameters
    ----------
    ax : `matplotlib.Axes`
        Axes on which to plot.
    referenceLines : `list` of `pfs.drp.stella.ReferenceLine`
        Reference lines to plot.
    flux : `numpy.ndarray`
        Flux vector, for setting height of reference line indicators.
    wavelength : `numpy.ndarray`
        Wavelength vector, for setting height of reference line indicators.
    badReference : `pfs.drp.stella.ReferenceLine.Status`
        Bit mask for identifying bad reference lines. Bad reference lines
        are plotted in red instead of black.
    """
    minWavelength = wavelength[0]
    maxWavelength = wavelength[-1]
    isGood = np.isfinite(flux)
    ff = flux[isGood]
    wl = wavelength[isGood]
    vertical = np.max(ff)
    for rl in referenceLines:
        xx = rl.wavelength
        if xx < minWavelength or xx > maxWavelength:
            continue
        style = "dotted" if rl.status & ReferenceLine.RESERVED > 0 else "solid"
        color = "red" if rl.status & badReference > 0 else "black"

        index = int(np.searchsorted(wl, xx))
        yy = np.max(ff[max(0, index - 2):min(len(ff) - 1, index + 2 + 1)])
        ax.plot((xx, xx), (yy + 0.10*vertical, yy + 0.20*vertical), ls=style, color=color)
        ax.text(xx, yy + 0.25*vertical, rl.description, color=color, ha='center')


@continueClass  # noqa: F811 (redefinition)
class Spectrum:
    """Flux as a function of wavelength"""
    def plot(self, numRows=3, doBackground=False, doReferenceLines=False, badReference=BAD_REFERENCE,
             filename=None):
        """Plot spectrum

        Parameters
        ----------
        numRows : `int`
            Number of row panels over which to plot the spectrum.
        doBackground : `bool`, optional
            Plot the background values in addition to the flux values?
        doReferenceLines : `bool`, optional
            Plot the reference lines?
        badReference : `pfs.drp.stella.ReferenceLine.Status`
            Bit mask for identifying bad reference lines. Bad reference lines
            are plotted in red instead of black.
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
        self.plotDivided(axes, division, doBackground=doBackground, doReferenceLines=doReferenceLines,
                         badReference=badReference)

        if filename is not None:
            figure.savefig(filename, bbox_inches='tight')
            plt.close(figure)
        return figure, axes

    def plotDivided(self, axes, division, doBackground=False, doReferenceLines=False,
                    badReference=BAD_REFERENCE, fluxStyle=None, backgroundStyle=None):
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
        doReferenceLines : `bool`, optional
            Plot the reference lines?
        badReference : `pfs.drp.stella.ReferenceLine.Status`
            Bit mask for identifying bad reference lines. Bad reference lines
            are plotted in red instead of black.
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
            useWavelength = np.arange(len(self), dtype=np.float32)
        wavelength = np.split(useWavelength, division)
        flux = np.split(self.getSpectrum(), division)
        if doBackground:
            background = np.split(self.getBackground(), division)

        for ii, ax in enumerate(axes):
            ax.plot(wavelength[ii], flux[ii], **fluxStyle)
            if doBackground:
                ax.plot(wavelength[ii], background[ii], **backgroundStyle)
            if doReferenceLines:
                plotReferenceLines(ax, self.getReferenceLines(), flux[ii], wavelength[ii], badReference)

    def findPeaks(self, minPeakFlux=500):
        """Find positive peaks in the flux

        Peak flux must exceed ``minPeakFlux``.

        Parameters
        ----------
        flux : `numpy.ndarray` of `float`
            Array of fluxes.

        Returns
        -------
        indices : `numpy.ndarray` of `int`
            Indices of peaks.
        """
        flux = self.spectrum
        diff = flux[1:] - flux[:-1]  # flux[i + 1] - flux[i]
        select = (diff[:-1] > 0) & (diff[1:] < 0) & (flux[1:-1] > minPeakFlux)
        indices = np.nonzero(select)[0] + 1  # +1 to account for the definition of diff

        return indices

    def getLineIntensities(self, wavelength, fittingRadius=10):
        """Estimate line intensities at specified wavelength

        Peak flux must exceed ``minPeakFlux``.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            Array of wavelength.

        Returns
        -------
        intensities : `numpy.ndarray` of `float`
            Estimated intensities.
            None is set for outside of spectrum's wavelengh range.
        """
        # Find peaks to identify interlopers
        peaks = self.findPeaks()

        # Interpolation for the conversion from wavelength to pixel number
        f = interpolate.interp1d(self.wavelength, range(self.getNumPixels()))

        # Initial guess of line width
        width = 1.0

        # interloper search range
        interloperWidth = int(3*width+0.5)

        intensities = list()

        for wl in wavelength:
            if wl < self.wavelength[0] or wl > self.wavelength[-1]:
                intensities.append(None)
                continue

            # Pixel position for a specific wavelength in integer
            estimatedPos = int(f(wl) + 0.5)

            # Initial guess of amplitude
            amplitude = self.spectrum[estimatedPos]

            # Define fitting pixel range
            lowIndex = max(estimatedPos-fittingRadius, 0)
            highIndex = min(estimatedPos+fittingRadius, self.getNumPixels()-1)
            indices = np.arange(lowIndex, highIndex+1)
            good = np.ones_like(indices, dtype=bool)

            interlopers = np.nonzero((peaks >= lowIndex - interloperWidth) &
                                     (peaks < highIndex + interloperWidth) &
                                     ((peaks < estimatedPos - width) |
                                      (peaks > estimatedPos + width)))[0]
            for ii in peaks[interlopers]:
                lowBound = max(lowIndex, ii - interloperWidth) - lowIndex
                highBound = min(highIndex, ii + interloperWidth) - lowIndex
                good[lowBound:highBound+1] = False
            if good.sum() < 5:
                intensities.append(None)
                continue

            lineModel = astropy.modeling.models.Gaussian1D(amplitude,
                                                           estimatedPos,
                                                           width,
                                                           bounds={"mean": (lowIndex, highIndex)},
                                                           name="line")
            fitter = astropy.modeling.fitting.LevMarLSQFitter()
            fit = fitter(lineModel, indices[good], self.spectrum[lowIndex:highIndex+1][good])

            intensities.append(fit.amplitude * math.sqrt(2*math.pi) * fit.stddev)

        return np.array(intensities)
