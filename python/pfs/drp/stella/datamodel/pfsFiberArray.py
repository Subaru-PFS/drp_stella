from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
import pfs.datamodel
from lsst.pipe.base import Struct

from ..interpolate import Resampler, interpolateFlux, interpolateMask

if TYPE_CHECKING:
    from matplotlib import Figure, Axes

__all__ = ("PfsSimpleSpectrum", "PfsFiberArray",)


class PfsSimpleSpectrum(pfs.datamodel.PfsSimpleSpectrum):
    def __imul__(self, rhs: ArrayLike) -> "PfsSimpleSpectrum":
        """Flux multiplication, in-place"""
        with np.errstate(invalid="ignore"):
            self.flux *= rhs
        return self

    def __itruediv__(self, rhs: ArrayLike) -> "PfsSimpleSpectrum":
        """Flux division, in-place"""
        with np.errstate(invalid="ignore", divide="ignore"):
            self.flux /= rhs
        return self

    def plot(self, ignorePixelMask: Optional[int] = None,
             figure: Optional[Figure] if TYPE_CHECKING else [] = None,
             ax: Optional[Axes] if TYPE_CHECKING else [] = None,
             trimToUsable: Optional[bool] = False,
             title: Optional[str] = None,
             show: bool = True) -> Tuple["Figure", "Axes"]:
        """Plot the object spectrum

        Parameters
        ----------
        ignorePixelMask : `int`
            Mask to apply to flux pixels. Defaults to the ``NO_DATA`` bitmask.
        show : `bool`, optional
            Show the plot?

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        ax : `matplotlib.Axes`
            Axes containing the plot.
        """
        import matplotlib.pyplot as plt
        from matplotlib.cbook import contiguous_regions

        if ignorePixelMask is None:
            ignorePixelMask = self.flags.get("NO_DATA")

        if figure is None:
            if ax is None:
                figure, ax = plt.subplots(squeeze=True)
            else:
                figure = ax.get_figure()
        elif ax is None:
            ax = figure.gca()

        good = (self.mask & ignorePixelMask) == 0

        for start, stop in contiguous_regions(~good):
            if stop >= self.wavelength.size:
                stop = self.wavelength.size - 1
            if trimToUsable:
                good &= ~((self.wavelength[start] < self.wavelength) &
                          (self.wavelength < self.wavelength[stop]))
            else:
                ax.axvspan(self.wavelength[start], self.wavelength[stop], color="grey", alpha=0.1)

        ax.plot(self.wavelength[good], self.flux[good], 'k-', label="Flux")

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Flux (nJy)")
        if title is None:
            title = str(self.getIdentity())
        ax.set_title(title)
        if show:
            figure.show()
        return figure, ax

    def resample(self, wavelength: np.ndarray, order: int = 3) -> "PfsSimpleSpectrum":
        """Resampled the spectrum in wavelength

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            Desired wavelength sampling.
        order : `int`, optional
            Interpolation order to use. Less than or equal to 1 means linear
            interpolation; higher orders use Lanczos interpolation of the given
            order.

        Returns
        -------
        resampled : `PfsSimpleSpectrum`
            Resampled spectrum.
        """
        flux = interpolateFlux(self.wavelength, self.flux, wavelength, order=order, fill=0.0)
        mask = interpolateMask(self.wavelength, self.mask, wavelength)
        return type(self)(self.target, wavelength, flux, mask, self.flags)


class PfsFiberArray(pfs.datamodel.PfsFiberArray, PfsSimpleSpectrum):
    def __imul__(self, rhs: ArrayLike) -> "PfsFiberArray":
        """Flux multiplication, in-place"""
        rhs = np.array(rhs).copy()  # Ensure rhs does not share memory with an element of self
        with np.errstate(invalid="ignore"):
            self.flux *= rhs
            self.sky *= rhs
            self.covar *= (rhs**2).reshape(1, -1)
        return self

    def __itruediv__(self, rhs: Union[float, np.ndarray]) -> "PfsFiberArray":
        """Flux division, in-place"""
        return self.__imul__(1.0/rhs)

    def plot(
        self,
        plotSky: bool = True,
        plotErrors: bool = True,
        ignorePixelMask: Optional[int] = None,
        figure: Optional[Figure] if TYPE_CHECKING else [] = None,
        ax: Optional[Axes] if TYPE_CHECKING else [] = None,
        trimToUsable: Optional[bool] = False,
        title: Optional[str] = None,
        show: bool = True,
    ) -> Tuple["Figure", "Axes"]:
        """Plot the object spectrum

        Parameters
        ----------
        plotSky : `bool`
            Plot sky measurements?
        plotErrors : `bool`
            Plot flux errors?
        ignorePixelMask : `int`, optional
            Mask to apply to flux pixels. Defaults to the ``NO_DATA`` bitmask.
        show : `bool`, optional
            Show the plot?

        Returns
        -------
        figure : `matplotlib.Figure`
            Figure containing the plot.
        axes : `matplotlib.Axes`
            Axes containing the plot.
        """
        if ignorePixelMask is None:
            ignorePixelMask = self.flags.get("NO_DATA")
        figure, axes = super().plot(figure=figure, ax=ax, ignorePixelMask=ignorePixelMask,
                                    trimToUsable=trimToUsable, title=title,
                                    show=False)
        good = (self.mask & ignorePixelMask) == 0

        if trimToUsable:
            from matplotlib.cbook import contiguous_regions
            for start, stop in contiguous_regions(~good):
                if stop >= self.wavelength.size:
                    stop = self.wavelength.size - 1
                good &= ~((self.wavelength[start] < self.wavelength) &
                          (self.wavelength < self.wavelength[stop]))

        if plotSky:
            axes.plot(self.wavelength[good], self.sky[good], 'b-', alpha=0.5,
                      label="Sky")
        if plotErrors:
            axes.plot(self.wavelength[good], np.sqrt(self.variance[good]), 'r-', alpha=0.5,
                      label="Flux errors")
        if show:
            figure.show()
        return figure, axes

    def _prepareMask(self, variance: np.ndarray, bad: list[str] | None) -> Struct:
        """Prepare the mask for interpolation

        Pixels with non-finite flux, sky or variance are marked as ``NO_DATA``.
        This ensures consistent treatment of bad pixels by ``resample`` and
        ``resampleVariance``.

        Parameters
        ----------
        variance : `numpy.ndarray` of `float`
            Variance to interpolate. Pixels for which this or the variance of
            the spectrum is non-finite are treated as bad.
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
        # Remove NANs: they get everywhere
        badFlux = ~np.isfinite(self.flux)
        badVariance = ~np.isfinite(self.variance) | ~np.isfinite(variance)
        badSky = ~np.isfinite(self.sky)
        noData = self.flags.get("NO_DATA")
        mask = self.mask.copy()
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
        wavelength: np.ndarray,
        order: int = 3,
        bad: list[str] | None = None,
        minWeight: float = 0.1,
        numCovar: int = 0,
    ) -> "PfsFiberArray":
        """Resampled the spectrum in wavelength

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            Desired wavelength sampling.
        order : `int`, optional
            Interpolation order to use. Less than or equal to 1 means linear
            interpolation; higher orders use Lanczos interpolation of the given
            order.
        bad : `list` of `str`, optional
            List of mask names to consider bad.
        minWeight : `float`, optional
            Minimum weight for Lanczos interpolation.
        numCovar : `int`, optional
            Number of covariance diagonals to interpolate. If 0, use
            ``order + 1``, which is suitable for interpolation on a similar
            wavelength grid to the original.

        Returns
        -------
        resampled : `PfsFiberArray`
            Resampled spectrum.
        """
        if numCovar == 0:
            numCovar = order + 1
        prepared = self._prepareMask(self.variance, bad)
        resampler = Resampler(self.wavelength, wavelength, order=order, fill=0.0)
        result = resampler(
            np.where(prepared.badFlux, 0.0, self.flux),
            prepared.mask,
            np.where(prepared.badVariance, 0.0, self.variance),
            minWeight=minWeight,
            badMask=prepared.badMask,
            fillMask=prepared.fillMask,
            numCovar=numCovar,
        )
        badSky = ~np.isfinite(self.sky)
        sky = resampler.resampleFlux(np.where(badSky, 0.0, self.sky))

        covar2 = np.array([[0]])  # Not sure what to put here
        return type(self)(
            self.target,
            self.observations,
            wavelength,
            result.flux,
            result.mask,
            sky,
            result.covariance,
            covar2,
            self.flags,
            self.fluxTable,
        )

    def resampleVariance(
        self,
        wavelength: np.ndarray,
        variance: np.ndarray,
        order: int = 3,
        bad: list[str] | None = None,
        minWeight: float = 0.1,
        numCovar: int = 0,
    ) -> "Struct":
        """Resample a variance array in the same manner as the spectrum

        This provides the mask and covariance that ``resample`` would produce
        if the spectrum had the provided ``variance``, without resampling the
        flux or sky.

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`
            Desired wavelength sampling.
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
            Number of covariance diagonals to interpolate. If 0, use
            ``order + 1``.

        Returns
        -------
        mask : `numpy.ndarray` of `int`, shape ``(N,)``
            Resampled mask. Pixels affected by bad pixels (including those
            with a non-finite ``variance``) have the ``NO_DATA`` bit set.
        covariance : `numpy.ndarray` of `float`, shape ``(numCovar, N)``
            Resampled covariance.
        """
        if numCovar == 0:
            numCovar = order + 1
        prepared = self._prepareMask(variance, bad)
        resampler = Resampler(self.wavelength, wavelength, order=order, fill=0.0)
        result = resampler.resampleVariance(
            prepared.mask,
            np.where(prepared.badVariance, 0.0, variance),
            minWeight=minWeight,
            badMask=prepared.badMask,
            fillMask=prepared.fillMask,
            numCovar=numCovar,
        )
        return Struct(mask=result.mask, covariance=result.covariance)
