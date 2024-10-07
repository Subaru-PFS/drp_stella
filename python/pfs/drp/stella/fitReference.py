import os
from typing import Literal, overload
from typing import Tuple

import numpy as np
import scipy.integrate
import scipy.interpolate
import astropy.io.fits
import astropy.wcs

from lsst.pex.config import Config
from lsst.pipe.base import Task

from pfs.datamodel import MaskHelper
from .datamodel import PfsFiberArray, PfsReference, PfsSimpleSpectrum
from .utils import getPfsVersions


def _trapezoidal(x: np.ndarray, y: np.ndarray, weight: np.ndarray, power: float = 1) -> float:
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

    def __init__(self, wavelength, transmission):
        if len(wavelength) != len(transmission):
            raise RuntimeError(
                "Mismatched lengths for wavelength and transmission: "
                f"{len(wavelength)} vs {len(transmission)}"
            )
        self.wavelength = wavelength
        self.transmission = transmission

    def interpolate(self, wavelength):
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
        return np.interp(wavelength, self.wavelength, self.transmission, 0.0, 0.0)

    def _integrate(self, specFlux=None, specWavelength=None, power=1):
        r"""Integrate the filter transmission curve for synthetic photometry

        The integral is:

        .. math:: \int S(\lambda) (F(\lambda) / \lambda d\lambda)^p

        where :math:`F(\lambda)` is the filter transmission curve,
        :math:`S(\lambda)` is the spectrum, and the extra `\lambda` term is due
        to using photon-counting detectors. :math:`p` (``power``) is usually 1,
        but it can be, say, 2 in the numerator of a variance formula.

        Parameters
        ----------
        specFlux : `numpy.ndarray`, optional
            Spectrum to integrate. If not provided, use unity.
        specWavelength : `numpy.ndarray`, optional
            Wavelength array for ``flux``. This is ignored if ``flux`` is ``None``.
        power : `int`
            :math:`p` in the integral (default: ``1``). This should usually be ``1``.
        """
        if specFlux is not None:
            x = specWavelength
            y = specFlux
            weight = self.interpolate(specWavelength) / specWavelength
        else:
            x = self.wavelength
            y = np.ones(shape=len(self.wavelength))
            weight = self.transmission / self.wavelength
        return _trapezoidal(x, y, weight, power=power)

    def integrate(self, spectrum=None, power=1):
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
        """
        if spectrum is not None:
            return self._integrate(spectrum.flux, spectrum.wavelength, power=power)
        else:
            return self._integrate(None, None, power=power)

    @overload
    def photometer(self, spectrum: PfsSimpleSpectrum, doComputeError: Literal[False]) -> float:
        ...

    @overload
    def photometer(self, spectrum: PfsFiberArray, doComputeError: Literal[True]) -> Tuple[float, float]:
        ...

    def photometer(self, spectrum, doComputeError=False):
        """Measure flux with this filter.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsSimpleSpectrum`
            Spectrum to integrate.
        doComputeError : `bool`
            Whether to compute an error bar (standard deviation).
            If ``doComputeError=True``, ``spectrum`` must be of
            `pfs.datamodel.PfsFiberArray` type.

        Returns
        -------
        flux : `float`
            Integrated flux.
        error : `float`
            Standard deviation of ``flux``. (Returned only if ``doComputeError=True``).
        """
        fluxNumer = self._integrate(spectrum.flux, spectrum.wavelength)
        fluxDenom = self._integrate(None, None)
        flux = fluxNumer / fluxDenom

        if doComputeError:
            varianceNumer = self._integrate(spectrum.covar[0, :], spectrum.wavelength, power=2)
            error = np.sqrt(varianceNumer) / fluxDenom
            return flux, error
        else:
            return flux


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
    }
    """Mapping of filter name to filename"""

    def __init__(self, filterName):
        if filterName not in self.filenames:
            raise RuntimeError(f"Unrecognised filter: {filterName}")

        filename = os.path.join(
            os.environ["OBS_PFS_DIR"], "pfs", "fluxCal", "bandpass", self.filenames[filterName]
        )
        data = np.genfromtxt(filename, dtype=[("wavelength", "f4"), ("flux", "f4")])
        wavelength = data["wavelength"]
        transmission = data["flux"]  # Relative transmission
        super().__init__(wavelength, transmission)


AMBRE_FILES = [
    "p6500_g+4.0_m0.0_t01_z+0.00_a+0.00.AMBRE_Extp.fits",
    "p6500_g+4.0_m0.0_t01_z-1.00_a+0.00.AMBRE_Extp.fits",
    "p7000_g+4.0_m0.0_t01_z+0.00_a+0.00.AMBRE_Extp.fits",
    "p7000_g+4.0_m0.0_t01_z-1.00_a+0.00.AMBRE_Extp.fits",
    "p7500_g+4.0_m0.0_t01_z+0.00_a+0.00.AMBRE_Extp.fits",
    "p7500_g+4.0_m0.0_t01_z-1.00_a+0.00.AMBRE_Extp.fits",
]


def readAmbre(target, wavelength):
    """Read the appropriate AMBRE spectrum

    The identification of the spectrum here matches what is used in the 2D
    simulator.

    Parameters
    ----------
    target : `pfs.datamodel.TargetData`
        Fiber target. The `objId`` identifies which spectrum was used.
    wavelength : `numpy.ndarray` of `float`
        Wavelength vector for result.

    Returns
    -------
    ambre : `pfs.datamodel.drp.PfsReference`
        AMBRE spectrum.
    """
    filename = os.path.join(
        os.environ["DRP_PFS_DATA_DIR"], "fluxCalSim", AMBRE_FILES[target.objId % len(AMBRE_FILES)]
    )
    with astropy.io.fits.open(filename) as fits:
        wcs = astropy.wcs.WCS(fits[1].header)
        refFlux = fits[1].data["Flux"]  # nJy

    # astropy treats the WCS as having two axes (because it's in a table with NAXIS=2).
    # We just ignore the other axis.
    refWavelength = wcs.pixel_to_world(np.arange(len(refFlux)), 0)[0].to("nm").value

    flux = scipy.interpolate.interp1d(
        refWavelength,
        refFlux,
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        copy=True,
        assume_sorted=True,
    )(wavelength)

    flags = MaskHelper(NO_DATA=1)
    mask = np.where(np.isfinite(flux), 0, flags.get("NO_DATA"))

    return PfsReference(target, wavelength, flux, mask, flags, metadata=getPfsVersions())


class FitReferenceConfig(Config):
    """Configuration for FitReferenceTask"""

    pass


class FitReferenceTask(Task):
    """Fit a physical reference spectrum to an observed spectrum

    This implementation is a placeholder, appropriate only for processing
    simulated spectra with a constant flux density (per unit frequency)
    """

    ConfigClass = FitReferenceConfig
    _DefaultName = "fitReference"

    def run(self, spectrum):
        """Fit a physical spectrum to an observed spectrum

        This implementation is a placeholder, as the algorithm is overly
        simplistic: we provide a spectrum with a constant flux density. This
        might be appropriate for processing simulated spectra.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsFiberArray`
            Spectrum to fit.

        Returns
        -------
        spectrum : `pfs.datamodel.PfsReference`
            Reference spectrum.
        """
        fiberFlux = set(spectrum.target.fiberFlux.values())
        if len(fiberFlux) != 1:
            raise RuntimeError(
                "This implementation requires a single fiber flux, but was provided: %s"
                % (spectrum.target.fiberFlux,)
            )
        fiberFlux = fiberFlux.pop()

        bandpass = None
        for ff in spectrum.target.fiberFlux.keys():
            try:
                bandpass = FilterCurve(ff)
            except RuntimeError:
                pass
            else:
                break
        else:
            raise RuntimeError(f"Unable to find bandpass for any of {spectrum.target.fiberFlux.keys()}")

        ref = readAmbre(spectrum.target, spectrum.wavelength)
        # For now, only use the filter curve in calculating the expected flux.
        # More precise work in the future may fold in the CCD QE, the mirror and the lens barrel.
        ref *= fiberFlux * bandpass.integrate() / bandpass.integrate(ref)  # nJy
        return ref
