from __future__ import annotations

from functools import lru_cache
import os
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from pfs.datamodel import PfsFiberArray, PfsSimpleSpectrum

__all__ = ("TransmissionCurve", "FilterCurve", "trapezoidal")


class TransmissionCurve:
    """A transmission curve

    Used for synthetic photometry as part of photometric calibration.

    The transmission curve is represented as a lookup table, with arryas of
    wavelength and the corresponding transmission.

    Parameters
    ----------
    wavelength : array_like
        Array of wavelength values, nm.
    transmission : array_like
        Array of corresponding transmission values.
    """

    def __init__(self, wavelength: np.ndarray, transmission: np.ndarray):
        if len(wavelength) != len(transmission):
            raise RuntimeError(
                "Mismatched lengths for wavelength and transmission: "
                f"{len(wavelength)} vs {len(transmission)}"
            )
        self.wavelength = wavelength
        self.transmission = transmission
        self._interpolator = interp1d(self.wavelength, self.transmission, bounds_error=False, fill_value=0.0)
        self.normalization = trapezoidal(
            self.wavelength, np.ones_like(self.wavelength), weight=self.transmission / self.wavelength
        )

    def interpolate(self, wavelength: np.ndarray) -> np.ndarray:
        """Interpolate the filter transmission curve at the provided wavelength

        Parameters
        ----------
        wavelength : array_like
            Wavelength at which to interpolate the transmission curve.

        Returns
        -------
        transmission : array_like
            Transmission at the provided wavelength.
        """
        return self._interpolator(wavelength)

    def integrateArrays(
        self,
        specWavelength: np.ndarray | None = None,
        specFlux: np.ndarray | None = None,
        power: float = 1.0,
    ) -> float:
        r"""Integrate the filter transmission curve for synthetic photometry

        The integral is:

        .. math:: \int S(\lambda) (F(\lambda) / \lambda d\lambda)^p

        where :math:`F(\lambda)` is the filter transmission curve,
        :math:`S(\lambda)` is the spectrum, and the extra `\lambda` term is due
        to using photon-counting detectors. :math:`p` (``power``) is usually 1,
        but it can be, say, 2 in the numerator of a variance formula.

        Parameters
        ----------
        specWavelength : `numpy.ndarray`, optional
            Wavelength array for ``flux``. This is ignored if ``flux`` is ``None``.
        specFlux : `numpy.ndarray`, optional
            Spectrum to integrate. If not provided, use unity.
        power : `float`
            :math:`p` in the integral (default: ``1``). This should usually be ``1``.
        """
        if specFlux is None:
            if specWavelength is not None:
                raise RuntimeError("specWavelength must be None if specFlux is None")
            if power == 1:
                return self.normalization
            specFlux = np.ones_like(specWavelength)
            specWavelength = self.wavelength
            weight = self.transmission / self.wavelength
        else:
            if specWavelength is None:
                raise RuntimeError("specWavelength must be provided if specFlux is provided")
            weight = self.interpolate(specWavelength) / specWavelength

        return trapezoidal(specWavelength, specFlux, weight, power=power)

    def integrate(self, spectrum=None, *, power=1, mask: list[str] | None = None) -> float:
        r"""Integrate the filter transmission curve for synthetic photometry

        The integral is:

        .. math:: \int S(\lambda) (F(\lambda) / \lambda d\lambda)^p

        where :math:`F(\lambda)` is the filter transmission curve,
        :math:`S(\lambda)` is the spectrum, and the extra `\lambda` term is due
        to using photon-counting detectors. :math:`p` (``power``) is usually 1,
        but it can be, say, 2 in the numerator of a variance formula.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsFiberArray`, optional
            Spectrum to integrate. If not provided, use unity.
        power : `int`
            :math:`p` in the integral (default: ``1``). This should usually be ``1``.
        mask : `list[str]`, optional
            List of mask planes to ignore when integrating.
        """
        if spectrum is None:
            if power == 1:
                return self.normalization
            return self.integrateArrays(None, None, power=power)

        wavelength = spectrum.wavelength
        flux = spectrum.flux
        if mask is not None:
            isGood = (spectrum.mask & spectrum.flags.get(*mask)) == 0
            wavelength = wavelength[isGood]
            flux = flux[isGood]

        return self.integrateArrays(wavelength, flux, power=power)

    @overload
    def photometer(
        self, spectrum: PfsSimpleSpectrum, doComputeError: Literal[False], mask: list[str] | None = None
    ) -> float:
        ...

    @overload
    def photometer(
        self, spectrum: PfsFiberArray, doComputeError: Literal[True], mask: list[str] | None
    ) -> tuple[float, float]:
        ...

    def photometer(
        self,
        spectrum: PfsSimpleSpectrum | PfsFiberArray,
        doComputeError: bool = False,
        mask: list[str] | None = None,
    ) -> float | tuple[float, float]:
        """Measure flux with this filter.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsSimpleSpectrum`
            Spectrum to integrate.
        doComputeError : `bool`
            Whether to compute an error bar (standard deviation).
            If ``doComputeError=True``, ``spectrum`` must be of
            `pfs.datamodel.PfsFiberArray` type.
        mask : `list[str]`, optional
            List of mask planes to ignore when integrating.

        Returns
        -------
        flux : `float`
            Integrated flux.
        error : `float`
            Standard deviation of ``flux``. (Returned only if ``doComputeError=True``).
        """
        wavelength = spectrum.wavelength
        flux = spectrum.flux
        isGood = None
        if mask is not None:
            isGood = (spectrum.mask & spectrum.flags.get(*mask)) == 0
            wavelength = wavelength[isGood]
            flux = flux[isGood]

        fluxNumer = self.integrateArrays(wavelength, flux)
        fluxDenom = self.normalization
        flux = fluxNumer / fluxDenom

        if not doComputeError:
            return flux

        variance = spectrum.covar[0, :]
        if mask is not None:
            variance = variance[isGood]
        varianceNumer = self.integrateArrays(wavelength, variance, power=2)
        error = np.sqrt(varianceNumer) / fluxDenom
        return flux, error


class FilterCurve(TransmissionCurve):
    """A filter transmission curve

    The bandpass is read from disk, and represents the transmission curve of
    the filter.

    Parameters
    ----------
    filterName : `str`
        Name of the filter. Must be one that is known.
    """

    filenames = {
        "g_hsc": "HSC/hsc_g_v2018.dat",
        # This is HSC-R (as opposed to HSC-R2)
        "r_old_hsc": "HSC/hsc_r_v2018.dat",
        "r2_hsc": "HSC/hsc_r2_v2018.dat",
        # This is HSC-I (as opposed to HSC-I2)
        "i_old_hsc": "HSC/hsc_i_v2018.dat",
        "i2_hsc": "HSC/hsc_i2_v2018.dat",
        "z_hsc": "HSC/hsc_z_v2018.dat",
        "y_hsc": "HSC/hsc_y_v2018.dat",
        "g_ps1": "PS1/PS1_g.dat",
        "r_ps1": "PS1/PS1_r.dat",
        "i_ps1": "PS1/PS1_i.dat",
        "z_ps1": "PS1/PS1_z.dat",
        "y_ps1": "PS1/PS1_y.dat",
        "bp_gaia": "Gaia/Gaia_Bp.txt",
        "rp_gaia": "Gaia/Gaia_Rp.txt",
        "g_gaia": "Gaia/Gaia_G.txt",
        "u_sdss": "SDSS/u_all_tel_atm13.dat",
        "g_sdss": "SDSS/g_all_tel_atm13.dat",
        "r_sdss": "SDSS/r_all_tel_atm13.dat",
        "i_sdss": "SDSS/i_all_tel_atm13.dat",
        "z_sdss": "SDSS/z_all_tel_atm13.dat",
        "nj_fake": "fake/fake_narrow_J.dat",
    }
    """Mapping of filter name to filename"""

    def __init__(self, filterName):
        wavelength, transmission = self._read(filterName)
        super().__init__(wavelength, transmission)
        self.filterName = filterName

    @lru_cache(maxsize=128)
    def _read(self, filterName: str) -> tuple[np.ndarray, np.ndarray]:
        """Read the filter curve data from disk.

        This is cached to avoid reading the same file multiple times.
        """
        if filterName not in self.filenames:
            raise RuntimeError(f"Unrecognised filter: {filterName}")
        filename = os.path.join(
            os.environ["OBS_PFS_DIR"], "pfs", "fluxCal", "bandpass", self.filenames[filterName]
        )
        data = np.genfromtxt(filename, dtype=[("wavelength", "f4"), ("flux", "f4")])
        return data["wavelength"], data["flux"]


def trapezoidal(x: np.ndarray, y: np.ndarray, weight: np.ndarray, power: float = 1) -> float:
    r"""Compute :math:`\int y(x) (w(x) dx)^p` with trapezoidal rule.

    :math:`p` is usually 1.

    :math:`p` can also be 2. If `y(x)` for each x is a stochastic variable,
    you can compute the statistical error of the integral,
    substituting 2 for :math:`p` and :math:`(\Delta y)^2` for :math:`y`.

    Other values of :math:`p` are accepted, but the return value will be
    nonsense mathematically.

    Parameters
    ----------
    x : `np.ndarray`
        Sampling points of :math:`x`. ``len(x)`` must be >= 3.
    y : `np.ndarray`
        Sampling points of :math:`y`.
    weight : `np.ndarray`
        Weight. This will be raised to the power of :math:`p` (``power``)
        unlike ``y``.
    power : `float`
        :math:`p`.

    Returns
    -------
    integral : float
        :math:`\int y(x) (w(x) dx)^p`
    """
    if power == 1:
        return 0.5 * (
            y[0] * weight[0] * (x[1] - x[0])
            + y[-1] * weight[-1] * (x[-1] - x[-2])
            + np.sum(y[1:-1] * weight[1:-1] * (x[2:] - x[:-2]))
        )
    else:
        return 0.5**power * (
            y[0] * (weight[0] * (x[1] - x[0])) ** power
            + y[-1] * (weight[-1] * (x[-1] - x[-2])) ** power
            + np.sum(y[1:-1] * (weight[1:-1] * (x[2:] - x[:-2])) ** power)
        )
