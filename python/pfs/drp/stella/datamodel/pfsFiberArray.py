import numpy as np
import pfs.datamodel

from ..interpolate import interpolateFlux, interpolateMask

__all__ = ("PfsSimpleSpectrum", "PfsFiberArray",)


class PfsSimpleSpectrum(pfs.datamodel.PfsSimpleSpectrum):
    def __imul__(self, rhs):
        """Flux multiplication, in-place"""
        self.flux *= rhs
        return self

    def __itruediv__(self, rhs):
        """Flux division, in-place"""
        self.flux /= rhs
        return self

    def plot(self, ignorePixelMask=0x0, show=True):
        """Plot the object spectrum

        Parameters
        ----------
        ignorePixelMask : `int`
            Mask to apply to flux pixels.
        show : `bool`, optional
            Show the plot?

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        axes : `matplotlib.Axes`
            Axes containing the plot.
        """
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots()
        good = (self.mask & ignorePixelMask) == 0
        axes.plot(self.wavelength[good], self.flux[good], 'k-', label="Flux")
        axes.set_xlabel("Wavelength (nm)")
        axes.set_ylabel("Flux (nJy)")
        axes.set_title(str(self.getIdentity()))
        if show:
            figure.show()
        return figure, axes

    def resample(self, wavelength):
        """Resampled the spectrum in wavelength

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            Desired wavelength sampling.

        Returns
        -------
        resampled : `PfsSimpleSpectrum`
            Resampled spectrum.
        """
        flux = interpolateFlux(self.wavelength, self.flux, wavelength)
        mask = interpolateMask(self.wavelength, self.mask, wavelength)
        return type(self)(self.target, wavelength, flux, mask, self.flags)


class PfsFiberArray(pfs.datamodel.PfsFiberArray, PfsSimpleSpectrum):
    def __imul__(self, rhs):
        """Flux multiplication, in-place"""
        rhs = np.array(rhs).copy()  # Ensure rhs does not share memory with an element of self
        with np.errstate(invalid="ignore"):
            self.flux *= rhs
            self.sky *= rhs
            rhsSquared = rhs**2
            for ii in range(3):
                self.covar[:, ii, :]*rhsSquared
        return self

    def __itruediv__(self, rhs):
        """Flux division, in-place"""
        return self.__imul__(1.0/rhs)

    def plot(self, plotSky=True, plotErrors=True, ignorePixelMask=0x0, show=True):
        """Plot the object spectrum

        Parameters
        ----------
        plotSky : `bool`
            Plot sky measurements?
        plotErrors : `bool`
            Plot flux errors?
        ignorePixelMask : `int`
            Mask to apply to flux pixels.
        show : `bool`, optional
            Show the plot?

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        axes : `matplotlib.Axes`
            Axes containing the plot.
        """
        figure, axes = super().plot(ignorePixelMask=ignorePixelMask, show=False)
        good = (self.mask & ignorePixelMask) == 0
        if plotSky:
            axes.plot(self.wavelength[good], self.sky[good], 'r-', label="Sky")
        if plotErrors:
            axes.plot(self.wavelength[good], np.sqrt(self.variance[good]), 'b-', label="Flux errors")
        if show:
            figure.show()
        return figure, axes

    def resample(self, wavelength):
        """Resampled the spectrum in wavelength

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            Desired wavelength sampling.

        Returns
        -------
        resampled : `PfsFiberArray`
            Resampled spectrum.
        """
        # Remove NANs: they get everywhere
        badFlux = ~np.isfinite(self.flux)
        badVariance = ~np.isfinite(self.variance)
        badSky = ~np.isfinite(self.sky)
        bad = badFlux | badVariance | badSky
        mask = self.mask.copy()
        mask[bad] |= self.flags.get("NO_DATA")

        flux = interpolateFlux(self.wavelength, np.where(badFlux, 0.0, self.flux), wavelength)
        mask = interpolateMask(self.wavelength, mask, wavelength, fill=self.flags.get("NO_DATA"))
        sky = interpolateFlux(self.wavelength, np.where(badSky, 0.0, self.sky), wavelength)
        covar = np.array([interpolateFlux(self.wavelength, np.where(badVariance, 0.0, cc), wavelength,
                                          variance=True) for cc in self.covar])
        covar2 = np.array([[0]])  # Not sure what to put here
        return type(self)(self.target, self.observations, wavelength, flux, mask, sky, covar, covar2,
                          self.flags, self.fluxTable)
