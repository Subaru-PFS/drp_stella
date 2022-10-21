import numpy as np
import pfs.datamodel

from pfs.utils.fibers import spectrographFromFiberId, fiberHoleFromFiberId

from ..interpolate import interpolateFlux, interpolateVariance, interpolateMask

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
        rhs = np.array(rhs).copy()  # Ensure rhs does not share memory with an element of self
        with np.errstate(invalid="ignore"):
            self.flux *= rhs
            self.sky *= rhs
            self.norm *= rhs
            rhsSquared = rhs**2
            for ii in range(3):
                self.covar[:, ii, :] *= rhsSquared
        return self

    def __itruediv__(self, rhs):
        """In-place division"""
        with np.errstate(divide="ignore"):
            return self.__imul__(1.0/rhs)

    def plot(self, fiberId=None, usePixels=False, ignorePixelMask=0x0, normalized=False, show=True,
             figure=None, axes=None, **kwargs):
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
        figure : `matplotlib.Figure` or ``None``
            The figure to use
        axes : `matplotlib.Axes` or ``None``
            The axes to use.
        **kwargs
            Keyword arguments for ``axes.plot``.

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        axes : `matplotlib.Axes`
            Axes containing the plot.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from matplotlib.colors import Normalize
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if fiberId is None:
            fiberId = self.fiberId
        else:
            fiberId = np.array(fiberId)
        if usePixels:
            wavelength = np.arange(self.length)
            xLabel = "Pixel"
        else:
            wavelength = self.wavelength
            xLabel = "Wavelength (nm)"

        if figure is None:
            if axes is None:
                figure, axes = plt.subplots()
            else:
                figure = axes.get_figure()
        elif axes is None:
            axes = figure.gca()

        cmap = matplotlib.cm.rainbow
        colors = cmap(np.linspace(0, 1, len(fiberId)))
        for ff, cc in zip(fiberId, colors):
            index = np.where(self.fiberId == ff)[0][0]
            good = ((self.mask[index] & ignorePixelMask) == 0)
            lam = wavelength if usePixels else wavelength[index]
            flux = self.flux[index][good]
            if normalized:
                with np.errstate(invalid="ignore", divide="ignore"):
                    flux /= self.norm[index][good]
            axes.plot(lam[good], flux, ls="solid", color=cc, label=str(ff), **kwargs)

        # Colorbar
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size='5%', pad=0.05)
        norm = Normalize(fiberId.min(), fiberId.max())
        colors = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        colors.set_array([])
        figure.colorbar(colors, cax=cax, orientation="vertical", label="fiberId")

        axes.set_xlabel(xLabel)
        axes.set_ylabel("Flux")

        if show:
            figure.show()
        return figure, axes

    def resample(self, wavelength, fiberId=None, jacobian=True):
        """Construct a new PfsFiberArraySet resampled to a common wavelength vector

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            New wavelength values (nm).
        fiberId : `numpy.ndarray` of int, optional
            Fibers to resample. If ``None``, resample all fibers.
        jacobian : `bool`
            Apply Jacobian, so that flux density is preserved?

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
            # We want the 'flux' array to remain in units close to counts on the detector. Calibrated fluxes
            # are measured relative to the 'norm': flux/norm. That means that when interpolating, we should
            # only apply the Jacobian correction to the 'norm', and not to the fluxes. This way, the 'flux'
            # remains in units of counts, and in the calculation of 'flux/norm', the Jacobian is applied
            # exactly once.
            norm[ii] = interpolateFlux(self.wavelength[jj], self.norm[jj], wavelength, jacobian=jacobian)
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
                                           wavelength, jacobian=False)*norm[ii]
                sky[ii] = interpolateFlux(self.wavelength[jj], np.where(badSky, 0.0, ss), wavelength,
                                          jacobian=False)*norm[ii]
                # XXX dropping covariance on the floor: just doing the variance for now
                covar[ii][0] = interpolateVariance(self.wavelength[jj], np.where(badVariance, 0.0, vv),
                                                   wavelength, fill=np.inf, jacobian=False)*norm[ii]**2
            mask[ii] = interpolateMask(self.wavelength[jj], mm, wavelength,
                                       fill=self.flags["NO_DATA"]).astype(self.mask.dtype)

        return type(self)(self.identity, fiberId, np.concatenate([[wavelength]]*numSpectra),
                          flux, mask, sky, norm, covar, self.flags, self.metadata)
