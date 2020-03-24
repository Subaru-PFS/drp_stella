import numpy as np
import pfs.datamodel

from .interpolate import interpolateFlux, interpolateMask

__all__ = ("PfsFiberArraySet",)


class PfsFiberArraySet(pfs.datamodel.PfsFiberArraySet):
    def __imul__(self, rhs):
        """In-place multiplication"""
        with np.errstate(invalid="ignore"):
            self.flux *= rhs
            self.sky *= rhs
            for ii in range(3):
                self.covar[:, ii, :] *= np.array(rhs)**2
        return self

    def __itruediv__(self, rhs):
        """In-place division"""
        with np.errstate(invalid="ignore", divide="ignore"):
            self.flux /= rhs
            self.sky /= rhs
            for ii in range(3):
                self.covar[:, ii, :] /= np.array(rhs)**2
        return self

    def plot(self, fiberId=None, usePixels=False, ignorePixelMask=0x0, show=True):
        """Plot the spectra

        Parameters
        ----------
        fiberId : iterable of `int`, optional
            Fibers to plot, or ``None`` to plot all.
        usePixels : `bool`, optional
            Plot as a function of pixel index, rather than wavelength?
        ignorePixelMask : `int`, optional
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
        import matplotlib.cm

        if fiberId is None:
            fiberId = self.fiberId
        if usePixels:
            wavelength = np.arange(self.length)
            xLabel = "Pixel"
        else:
            wavelength = self.wavelength
            xLabel = "Wavelength (nm)"

        figure, axes = plt.subplots()

        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(fiberId)))
        for ff, cc in zip(fiberId, colors):
            index = np.where(self.fiberId == ff)[0]
            good = (self.mask[index] & ignorePixelMask) == 0
            axes.plot(wavelength[index][good], self.flux[index][good], ls="solid", color=cc)

        axes.set_xlabel(xLabel)
        axes.set_ylabel("Flux")

        if show:
            figure.show()
        return figure, axes

    def resample(self, wavelength, fiberId=None):
        """Construct a new PfsFiberArraySet resampled to a common wavelength vector

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            New wavelength values (nm).
        fiberId : `numpy.ndarray` of int, optional
            Fibers to resample. If ``None``, resample all fibers.

        Returns
        -------
        result : `PfsFiberArraySet`
            Resampled spectra.
        """
        if fiberId is None:
            fiberId = self.fiberId

        numSpectra = len(fiberId)
        numSamples = len(wavelength)
        flux = np.empty((numSpectra, numSamples), dtype=self.flux.dtype)
        mask = np.empty((numSpectra, numSamples), dtype=self.mask.dtype)
        sky = np.empty((numSpectra, numSamples), dtype=self.sky.dtype)
        covar = np.zeros((numSpectra, 3, numSamples), dtype=self.covar.dtype)

        for ii, ff in enumerate(fiberId):
            jj = np.argwhere(self.fiberId == ff)[0][0]
            flux[ii] = interpolateFlux(self.wavelength[jj], self.flux[jj], wavelength)
            sky[ii] = interpolateFlux(self.wavelength[jj], self.sky[jj], wavelength)
            # XXX dropping covariance on the floor: just doing the variance for now
            covar[ii][0] = interpolateFlux(self.wavelength[jj], self.covar[jj][0], wavelength, fill=np.inf)
            mask[ii] = interpolateMask(self.wavelength[jj], self.mask[jj], wavelength,
                                       fill=self.flags["NO_DATA"]).astype(self.mask.dtype)

        return type(self)(self.identity, fiberId, np.concatenate([[wavelength]]*numSpectra),
                          flux, mask, sky, covar, self.flags, self.metadata)
