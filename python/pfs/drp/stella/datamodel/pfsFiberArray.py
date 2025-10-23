from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
import pfs.datamodel

from ..interpolate import interpolate, interpolateFlux, interpolateMask

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

    def resample(
        self,
        wavelength: np.ndarray,
        order: int = 3,
        bad: list[str] | None = None,
        minWeight: float = 0.1,
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

        Returns
        -------
        resampled : `PfsFiberArray`
            Resampled spectrum.
        """
        # Remove NANs: they get everywhere
        badFlux = ~np.isfinite(self.flux)
        badVariance = ~np.isfinite(self.variance)
        badSky = ~np.isfinite(self.sky)
        isBad = badFlux | badVariance | badSky
        mask = self.mask.copy()
        mask[isBad] |= self.flags.get("NO_DATA")

        badMask = self.flags.get(*bad) if bad else 0
        kwargs = dict(fill=0.0, order=order)

        result = interpolate(
            self.wavelength,
            np.where(badFlux, 0.0, self.flux),
            mask,
            np.where(badVariance, 0.0, self.variance),
            wavelength,
            minWeight=minWeight,
            badMask=badMask,
            fillMask=self.flags.get("NO_DATA"),
            **kwargs,
        )
        sky = interpolateFlux(self.wavelength, np.where(badSky, 0.0, self.sky), wavelength, **kwargs)
        covar = np.array([
            result.variance, np.zeros_like(result.variance), np.zeros_like(result.variance)
        ])

        covar2 = np.array([[0]])  # Not sure what to put here
        return type(self)(
            self.target,
            self.observations,
            wavelength,
            result.flux,
            result.mask,
            sky,
            covar,
            covar2,
            self.flags,
            self.fluxTable,
        )
