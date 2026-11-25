import numpy as np
import pfs.datamodel
from lsst.pipe.base import Struct

from pfs.utils.fibers import spectrographFromFiberId, fiberHoleFromFiberId

from ..interpolate import Resampler, maskCovariance, scaleCovariance

__all__ = ("PfsFiberArraySet",)


class PfsFiberArraySet(pfs.datamodel.PfsFiberArraySet):
    _ylabel = "flux"                    # y-axis label for plots, overridden in derived classes such as pfsArm

    @property
    def lsf(self):
        """Line-spread function of the spectra (`pfs.drp.stella.SpectraLsf`), or `None`

        The LSF read from a file is converted when first used.
        """
        lsf = getattr(self, "_lsf", None)
        if isinstance(lsf, pfs.datamodel.PfsLsf):
            from ..spectraLsf import spectraLsfFromDatamodel
            keys = [int(ff) for ff in self.fiberId]
            wavelength = {ff: wl for ff, wl in zip(keys, self.wavelength)}
            lsf = self._lsf = spectraLsfFromDatamodel(lsf, keys, wavelength, self.length)
        return lsf

    @lsf.setter
    def lsf(self, value):
        self._lsf = value

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
        if isinstance(rhs, pfs.datamodel.PfsFiberNorms):
            rhs = rhs.values
        rhs = np.array(rhs).copy()  # Ensure rhs does not share memory with an element of self
        with np.errstate(invalid="ignore"):
            self.flux *= rhs
            self.sky *= rhs
            self.norm *= rhs
            scaleCovariance(self.covar, rhs)
        return self

    def __itruediv__(self, rhs):
        """In-place division"""
        if isinstance(rhs, pfs.datamodel.PfsFiberNorms):
            rhs = rhs.values
        rhs = np.array(rhs).copy()  # Ensure rhs does not share memory with an element of self
        with np.errstate(divide="ignore", invalid="ignore"):
            self.flux /= rhs
            self.sky /= rhs
            self.norm /= rhs
            scaleCovariance(self.covar, rhs, divide=True)
        return self

    def plot(self, fiberId=None, usePixels=False, ignorePixelMask=0x0, normalized=False, show=True,
             figure=None, ax=None):
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
        ax : `matplotlib.Axes` or ``None``
            The x/y axes to use.

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        ax : `matplotlib.Axes`
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

        if figure is None:
            if ax is None:
                figure, ax = plt.subplots(squeeze=True)
            else:
                figure = ax.get_figure()
        elif ax is None:
            ax = figure.gca()

        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(fiberId)))
        for ff, cc in zip(fiberId, colors):
            index = np.where(self.fiberId == ff)[0]
            if len(index) == 0:
                print(f"Unable to find fiberId {ff}")
                continue

            index = index[0]
            good = ((self.mask[index] & ignorePixelMask) == 0)
            lam = wavelength if usePixels else wavelength[index]
            flux = self.flux[index][good]
            if normalized:
                with np.errstate(invalid="ignore", divide="ignore"):
                    flux /= self.norm[index][good]
            ax.plot(lam[good], flux, ls="solid", color=cc, label=str(ff))

        ax.set_xlabel(xLabel)
        ax.set_ylabel(self._ylabel)

        if show:
            figure.show()
        return figure, ax

    def _numCorrelatedDiagonals(self) -> int:
        """Number of diagonals of the covariance up to the last non-zero one

        1 means that the pixels are uncorrelated.
        """
        offDiagonal = np.isfinite(self.covar[:, 1:]) & (self.covar[:, 1:] != 0)  # inf marks NO_DATA
        nonZero = np.flatnonzero(np.any(offDiagonal, axis=(0, 2)))
        return 1 + (nonZero[-1] + 1 if len(nonZero) > 0 else 0)

    def _prepareMask(self, index: int, variance: np.ndarray, bad: list[str] | None) -> Struct:
        """Prepare the mask of a single fiber for interpolation

        Pixels with non-finite flux, sky or variance, or a bad normalisation,
        are marked as ``NO_DATA``. This ensures consistent treatment of bad
        pixels by ``resample`` and ``resampleVariance``.

        Parameters
        ----------
        index : `int`
            Index of the fiber.
        variance : `numpy.ndarray` of `float`
            Variance to interpolate for this fiber. Pixels for which this or
            the variance of the spectrum is non-finite are treated as bad.
        bad : `list` of `str`, optional
            List of mask names to consider bad.

        Returns
        -------
        mask : `numpy.ndarray` of `int`
            Mask to use for interpolation.
        badFlux : `numpy.ndarray` of `bool`
            Pixels with bad flux values (to be replaced before interpolation).
        badVariance : `numpy.ndarray` of `bool`
            Pixels with bad variance values (to be replaced before
            interpolation).
        badMask : `int`
            Bitmask for bad pixels, which are excluded from interpolation.
        fillMask : `int`
            Bitmask for pixels in the output that failed interpolation.
        """
        badNorm = (self.norm[index] == 0) | ~np.isfinite(self.norm[index])
        badFlux = ~np.isfinite(self.flux[index])
        badVariance = badNorm | ~np.isfinite(self.variance[index]) | ~np.isfinite(variance)
        badSky = badNorm | ~np.isfinite(self.sky[index])
        noData = self.flags.get("NO_DATA")
        mask = self.mask[index].copy()
        mask[badFlux | badVariance | badSky] |= noData
        return Struct(
            mask=mask,
            badFlux=badFlux,
            badVariance=badVariance,
            badMask=self.flags.get(*bad) if bad else 0,
            fillMask=noData,
        )

    def resample(
        self,
        wavelength,
        fiberId=None,
        order: int = 3,
        bad: list[str] | None = None,
        minWeight: float = 0.1,
        numCovar: int = 0,
    ) -> "PfsFiberArraySet":
        """Construct a new PfsFiberArraySet resampled to a common wavelength vector

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            New wavelength values (nm).
        fiberId : `numpy.ndarray` of int, optional
            Fibers to resample. If ``None``, resample all fibers.
        order : `int`, optional
            Interpolation order to use. Less than or equal to 1 means linear
            interpolation; higher orders use Lanczos interpolation of the given
            order.
        bad : `list` of `str`, optional
            List of mask names to consider bad.
        minWeight : `float`, optional
            Minimum weight for Lanczos interpolation.
        numCovar : `int`, optional
            Number of covariance diagonals to interpolate. If 0, as many as
            needed to hold all of the covariance, which depends on the kernel,
            the change in sampling and any correlations already present (see
            `pfs.drp.stella.interpolate.Resampler.numCovarNeeded`).

        Returns
        -------
        result : `PfsFiberArraySet`
            Resampled spectra.
        """
        if fiberId is None:
            fiberId = self.fiberId
        indices = [np.argwhere(self.fiberId == ff)[0][0] for ff in fiberId]
        resamplers = [Resampler(self.wavelength[jj], wavelength, order=order, fill=0.0) for jj in indices]
        if numCovar == 0:
            numCovar = max(rr.numCovarNeeded(self._numCorrelatedDiagonals()) for rr in resamplers)

        numSpectra = len(fiberId)
        numSamples = len(wavelength)
        flux = np.empty((numSpectra, numSamples), dtype=self.flux.dtype)
        mask = np.empty((numSpectra, numSamples), dtype=self.mask.dtype)
        sky = np.empty((numSpectra, numSamples), dtype=self.sky.dtype)
        norm = np.empty((numSpectra, numSamples), dtype=self.norm.dtype)
        covar = np.zeros((numSpectra, numCovar, numSamples), dtype=self.covar.dtype)

        for ii, (jj, resampler) in enumerate(zip(indices, resamplers)):
            norm[ii] = resampler.resampleFlux(self.norm[jj])
            prepared = self._prepareMask(jj, self.variance[jj], bad)
            result = resampler(
                np.where(prepared.badFlux, 0.0, self.flux[jj]).astype(np.float32),
                prepared.mask,
                np.where(prepared.badVariance, 0.0, self.variance[jj]).astype(np.float32),
                minWeight=minWeight,
                badMask=prepared.badMask,
                fillMask=prepared.fillMask,
                numCovar=numCovar,
                covariance=maskCovariance(self.covar[jj], prepared.badVariance),
            )
            badNorm = (self.norm[jj] == 0) | ~np.isfinite(self.norm[jj])
            badSky = badNorm | ~np.isfinite(self.sky[jj])
            flux[ii] = result.flux
            mask[ii] = result.mask
            covar[ii] = result.covariance

            sky[ii] = resampler.resampleFlux(np.where(badSky, 0.0, self.sky[jj]))

        return type(self)(self.identity, fiberId, np.concatenate([[wavelength]]*numSpectra),
                          flux, mask, sky, norm, covar, self.flags, self.metadata)

    def resampleVariance(
        self,
        wavelength: np.ndarray,
        variance: np.ndarray,
        order: int = 3,
        bad: list[str] | None = None,
        minWeight: float = 0.1,
        numCovar: int = 0,
    ) -> Struct:
        """Resample a variance array in the same manner as the spectra

        This provides the mask and covariance that ``resample`` would produce
        if the spectra had the provided ``variance``, without resampling the
        flux, sky or norm.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            New wavelength values (nm).
        variance : `numpy.ndarray` of `float`
            Variance to resample, with the same shape as the ``flux``.
            Non-finite values are treated as bad pixels.
        order : `int`, optional
            Interpolation order to use. Less than or equal to 1 means linear
            interpolation; higher orders use Lanczos interpolation of the given
            order.
        bad : `list` of `str`, optional
            List of mask names to consider bad.
        minWeight : `float`, optional
            Minimum weight for Lanczos interpolation.
        numCovar : `int`, optional
            Number of covariance diagonals to interpolate. If 0, as many as
            needed (as for ``resample``).

        Returns
        -------
        mask : `numpy.ndarray` of `int`, shape ``(N, M)``
            Resampled mask for each fiber. Pixels affected by bad pixels
            (including those with a non-finite ``variance``) have the
            ``NO_DATA`` bit set.
        covariance : `numpy.ndarray` of `float`, shape ``(N, numCovar, M)``
            Resampled covariance for each fiber.

        Raises
        ------
        NotImplementedError
            If the spectra's pixels are correlated (non-zero off-diagonal
            covariance): the ``variance`` would be resampled as if its pixels
            were independent, unlike the spectra's.
        """
        if variance.shape != self.flux.shape:
            raise RuntimeError(f"Shape mismatch between variance {variance.shape} and flux {self.flux.shape}")
        if self._numCorrelatedDiagonals() > 1:
            raise NotImplementedError(
                "Resampling a variance for spectra with correlated pixels (non-zero off-diagonal covariance)"
            )
        resamplers = [
            Resampler(self.wavelength[ii], wavelength, order=order, fill=0.0) for ii in range(len(self))
        ]
        if numCovar == 0:
            numCovar = max(rr.numCovarNeeded() for rr in resamplers)
        mask = np.empty((len(self), len(wavelength)), dtype=self.mask.dtype)
        covar = np.zeros((len(self), numCovar, len(wavelength)), dtype=self.covar.dtype)
        for ii, resampler in enumerate(resamplers):
            prepared = self._prepareMask(ii, variance[ii], bad)
            result = resampler.resampleVariance(
                prepared.mask,
                np.where(prepared.badVariance, 0.0, variance[ii]).astype(np.float32),
                minWeight=minWeight,
                badMask=prepared.badMask,
                fillMask=prepared.fillMask,
                numCovar=numCovar,
            )
            mask[ii] = result.mask
            covar[ii] = result.covariance
        return Struct(mask=mask, covariance=covar)
