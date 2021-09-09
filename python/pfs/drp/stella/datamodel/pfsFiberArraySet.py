import numpy as np
import pfs.datamodel

from pfs.utils.fibers import spectrographFromFiberId, fiberHoleFromFiberId

from .interpolate import interpolateFlux, interpolateMask

__all__ = ("PfsFiberArraySet",)


class PfsFiberArraySet(pfs.datamodel.PfsFiberArraySet):
    @property
    def spectrograph(self):
        """Return spectrograph number"""
        return spectrographFromFiberId(self.fiberId)

    @property
    def fiberHole(self):
        """Return fiber hole number"""
        return fiberHoleFromFiberId(self.fiberId)

    def __imul__(self, rhs):
        """In-place multiplication"""
        with np.errstate(invalid="ignore"):
            self.flux *= rhs
            self.sky *= rhs
            self.norm *= rhs
            for ii in range(3):
                self.covar[:, ii, :] *= np.array(rhs)**2
        return self

    def __itruediv__(self, rhs):
        """In-place division"""
        with np.errstate(invalid="ignore", divide="ignore"):
            self.flux /= rhs
            self.sky /= rhs
            self.norm /= rhs
            for ii in range(3):
                self.covar[:, ii, :] /= np.array(rhs)**2
        return self

    def plot(self, fiberId=None, usePixels=False, ignorePixelMask=0x0, normalized=False, show=True):
        """Plot the spectra

        Parameters
        ----------
        fiberId : iterable of `int`, optional
            Fibers to plot, or ``None`` to plot all.
        usePixels : `bool`, optional
            Plot as a function of pixel index, rather than wavelength?
        ignorePixelMask : `int`, optional
            Mask to apply to flux pixels.
        normalized : `bool`, optional
            Plot normalised flux?
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
            index = np.where(self.fiberId == ff)[0][0]
            good = ((self.mask[index] & ignorePixelMask) == 0)
            lam = wavelength if usePixels else wavelength[index]
            flux = self.flux[index][good]
            if normalized:
                flux /= self.norm[index][good]
            axes.plot(lam[good], flux, ls="solid", color=cc, label=str(ff))

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
        norm = np.empty((numSpectra, numSamples), dtype=self.norm.dtype)
        covar = np.zeros((numSpectra, 3, numSamples), dtype=self.covar.dtype)

        for ii, ff in enumerate(fiberId):
            jj = np.argwhere(self.fiberId == ff)[0][0]
            norm[ii] = interpolateFlux(self.wavelength[jj], self.norm[jj], wavelength)
            badNorm = (self.norm[jj] == 0) | ~np.isfinite(self.norm[jj])
            badFlux = badNorm | ~np.isfinite(self.flux[jj])
            badVariance = badNorm | ~np.isfinite(self.variance[jj])
            badSky = badNorm | ~np.isfinite(self.sky[jj])
            bad = badFlux | badVariance | badSky
            mm = self.mask[jj].copy()
            mm[bad] |= self.flags["NO_DATA"]
            with np.errstate(invalid="ignore"):
                ff = self.flux[jj]/self.norm[jj]
                ss = self.sky[jj]/self.norm[jj]
                vv = self.covar[jj][0]/self.norm[jj]**2
                flux[ii] = interpolateFlux(self.wavelength[jj], np.where(badFlux, 0.0, ff),
                                           wavelength)*norm[ii]
                sky[ii] = interpolateFlux(self.wavelength[jj], np.where(badSky, 0.0, ss), wavelength)*norm[ii]
                # XXX dropping covariance on the floor: just doing the variance for now
                covar[ii][0] = interpolateFlux(self.wavelength[jj], np.where(badVariance, 0.0, vv),
                                               wavelength, fill=np.inf)*norm[ii]**2
            mask[ii] = interpolateMask(self.wavelength[jj], mm, wavelength,
                                       fill=self.flags["NO_DATA"]).astype(self.mask.dtype)

        return type(self)(self.identity, fiberId, np.concatenate([[wavelength]]*numSpectra),
                          flux, mask, sky, norm, covar, self.flags, self.metadata)
