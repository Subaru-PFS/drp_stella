import itertools
from functools import lru_cache
from typing import Dict, Iterable, Tuple

import astropy.io.fits
import numpy as np
from numpy.typing import ArrayLike

from .interpolate import interpolateFlux

__all__ = ("AtmosphericTransmission",)


class AtmosphericTransmissionInterpolator:
    """Object for interpolating the atmospheric transmission in PWV

    When fitting a spectrum, we know the zenith distance and wavelength
    sampling, and we're fitting for the precipitable water vapor (PWV). This
    object provides an efficient interpolation of the atmospheric model for
    a requested PWV.

    Parameters
    ----------
    pwv : `np.ndarray`
        Precipitable water vapor (mm) values.
    transmission: `np.ndarray`
        Corresponding transmission (as a function of wavelength) values. Any
        interpolation in wavelength should already have been done.
    """

    def __init__(self, pwv: np.ndarray, transmission: np.ndarray):
        if transmission.shape[0] != pwv.size:
            raise RuntimeError(
                f"Size mismatch between pwv ({pwv.size}) and transmission ({transmission.shape[0]})"
            )
        indices = np.argsort(pwv)
        self.pwv: np.ndarray = pwv[indices]
        self.transmission: np.ndarray = transmission[indices]

    @lru_cache
    def __call__(self, pwv: float) -> np.ndarray:
        """Interpolate the transmission spectra for the PWV value

        Parameters
        ----------
        pwv : `float`
            Precipitable water vapor (mm) value at which to interpolate.

        Returns
        -------
        transmission : `np.ndarray`
            Transmission spectrum for the input PWV.
        """
        if pwv < self.pwv[0]:
            return np.full_like(self.transmission[0], np.nan)
        index = min(int(np.searchsorted(self.pwv, pwv)), self.pwv.size - 2)
        if self.pwv[index] == pwv:
            return self.transmission[index]
        pwvHigh: float = self.pwv[index]
        pwvLow: float = self.pwv[index - 1]
        highWeight = (pwv - pwvLow) / (pwvHigh - pwvLow)
        lowWeight = 1.0 - highWeight
        return lowWeight * self.transmission[index - 1] + highWeight * self.transmission[index]


class AtmosphericTransmission:
    """Model of atmospheric transmission

    The model includes both zenith distance (ZD) and precipitable water vapor
    (PWV).

    Parameters
    ----------
    wavelength : `np.ndarray` of `float`, shape ``(W,)``
        Wavelength sampling.
    zd : `np.ndarray` of `float`, shape ``(Z,)``
        Zenith distance values; repeated values allowed.
    pwv : `np.ndarray` of `float`, shape ``(P,)``
        Precipitable water vapor (mm) values; repeated values allowed.
    transmission : `dict` mapping (`float`,`float`) to `np.ndarray`
        Transmission spectra, indexed by a tuple of zenith distance value and
        PWV value. Each transmission spectrum should have the same length as
        the ``wavelength`` array.
    """

    def __init__(
        self,
        wavelength: np.ndarray,
        zd: Iterable[float],
        pwv: Iterable[float],
        transmission: Dict[Tuple[float, float], np.ndarray],
    ):
        self.wavelength = wavelength
        self.zd = np.array(sorted(set(zd)))
        self.pwv = np.array(sorted(set(pwv)))
        self.transmission = transmission
        for key in itertools.product(self.zd, self.pwv):
            if key not in self.transmission:
                raise RuntimeError(f"Grid point ZD,PWV={key} not present in data")
            if self.transmission[key].shape != self.wavelength.shape:
                raise RuntimeError(f"Shape of transmission for ZD,PWV={key} doesn't match")

    @classmethod
    def fromFits(cls, filename: str) -> "AtmosphericTransmission":
        """Construct from FITS file

        Parameters
        ----------
        filename : `str`
            Filename of atmospheric model FITS file.

        Returns
        -------
        self : `AtmosphericTransmission`
            Model constructed from FITS file.
        """
        with astropy.io.fits.open(filename) as fits:
            # FITS data is big-endian; cast to native byte order float64, since the numerical
            # routines used for interpolation require it.
            wavelength = fits["WAVELENGTH"].data.astype(np.float64)
            zd = fits["TRANSMISSION"].data["zd"].astype(np.float64)
            pwv = fits["TRANSMISSION"].data["pwv"].astype(np.float64)
            transmission = np.array(
                [tt.astype(np.float64) for tt in fits["TRANSMISSION"].data["transmission"]]
            )
            assert zd.size == pwv.size, "Grid size mismatch"
            assert transmission.shape[0] == zd.size, "Transmission size mismatch"
            assert transmission.shape[1] == wavelength.size, "Wavelength size mismatch"
        return cls(
            wavelength=wavelength,
            zd=zd,
            pwv=pwv,
            transmission={(zz, pp): tt for zz, pp, tt in zip(zd, pwv, transmission)},
        )

    def makeInterpolator(self, zd: float, wavelength: ArrayLike) -> AtmosphericTransmissionInterpolator:
        """Construct an interpolator

        When fitting an atmospheric model to data, we know the zenith distance
        at which the data was obtained and the wavelength sampling, and we're
        fitting for the precipitable water vapor (PWV). The fitting process will
        evaluate the model for many different PWV values, so that needs to be as
        efficient as possible. This method does the interpolation in zenith
        distance and wavelength up front, and provides an interpolator that
        operates solely on the PWV value.

        Note that no LSF convolution is applied here: the returned transmission
        spectra are the raw model resampled onto ``wavelength``. Callers that
        need to convolve the transmission with an LSF should do so jointly with
        whatever spectrum they multiply the transmission into (convolution does
        not distribute over multiplication), not on the transmission alone.

        Parameters
        ----------
        zd : `float`
            Zenith distance (degrees) at which to interpolate.
        wavelength : array_like
            Wavelength array for interpolation.

        Returns
        -------
        interpolator : `AtmosphericTransmissionInterpolator`
            Object that will perform interpolation in PWV.
        """
        if zd < self.zd[0]:
            raise RuntimeError(
                f"Zenith distance {zd} doesn't fall in range spanned by model: {self.zd[0]} to {self.zd[-1]}"
            )
        wavelength = np.asarray(wavelength)
        index = min(int(np.searchsorted(self.zd, zd)), self.zd.size - 2)
        if self.zd[index] == zd:
            transmission = np.array([
                interpolateFlux(self.wavelength, self.transmission[(zd, pwv)], wavelength, fill=np.nan)
                for pwv in self.pwv
            ])
            return AtmosphericTransmissionInterpolator(self.pwv, transmission)
        zdLow = self.zd[index - 1]
        zdHigh = self.zd[index]
        highWeight = (zd - zdLow) / (zdHigh - zdLow)
        lowWeight = 1.0 - highWeight
        transmission = np.full((self.pwv.size, wavelength.size), np.nan, dtype=float)
        for ii, pwv in enumerate(self.pwv):
            low = self.transmission[zdLow, pwv]
            high = self.transmission[zdHigh, pwv]
            pwvTransmission = low * lowWeight + high * highWeight
            transmission[ii] = interpolateFlux(self.wavelength, pwvTransmission, wavelength, fill=np.nan)
        return AtmosphericTransmissionInterpolator(self.pwv, transmission)

    def __call__(self, zd: float, pwv: float, wavelength: ArrayLike) -> np.ndarray:
        """Evaluate the atmospheric transmission

        This method provides a full-featured interpolation of the model. For
        faster individual interpolations when varying only PWV (e.g., when
        fitting a spectrum of known zenith distance and wavelength sampling),
        use the interpolator provided by the ``makeInterpolator`` method.

        Parameters
        ----------
        zd: `float`
            Zenith distance, in degrees.
        pwv : `float`
            Precipitable water vapour, in mm.
        wavelength : array_like
            Wavelength array for which to provide corresponding transmission.

        Returns
        -------
        result : `np.ndarray` of `float`
            Transmission for the provided wavelengths.
        """
        return self.makeInterpolator(zd, wavelength)(pwv)
