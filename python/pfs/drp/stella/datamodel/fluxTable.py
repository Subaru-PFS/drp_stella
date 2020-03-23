import numpy as np
import pfs.datamodel

from .interpolate import interpolateFlux, interpolateMask

__all__ = ("FluxTable",)


class FluxTable(pfs.datamodel.FluxTable):
    def plot(self, ignoreFlags=None, show=True):
        """Plot the object spectrum

        Parameters
        ----------
        ignorePixelMask : `int`
            Mask to apply to flux pixels.
        show : `bool`, optional
            Show the plot and block on the window?

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        axes : `matplotlib.Axes`
            Axes containing the plot.
        """
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots()
        good = (((self.mask & self.flags.get(*ignoreFlags)) == 0) if ignoreFlags is not None else
                np.ones_like(self.mask, dtype=bool))
        axes.plot(self.wavelength[good], self.flux[good], 'k-', label="Flux")
        axes.set_xlabel("Wavelength (nm)")
        axes.set_ylabel("Flux (nJy)")
        if show:
            plt.show()
        return figure, axes

    def resample(self, wavelength):
        """Resample to a common wavelength vector

        This is provided as a possible convenience to the user and a means to
        facilitate testing.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            New wavelength values (nm).

        Returns
        -------
        resampled : `FluxTable`
            Resampled flux table.
        """
        flags = self.flags.copy()
        flags.add("NO_DATA")

        flux = interpolateFlux(self.wavelength, self.flux, wavelength)
        error = interpolateFlux(self.wavelength, self.error, wavelength)
        mask = interpolateMask(self.wavelength, self.mask, wavelength,
                               fill=flags["NO_DATA"]).astype(self.mask.dtype)
        return type(self)(wavelength, flux, error, mask, flags)
