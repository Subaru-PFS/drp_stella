import os
import numpy as np
import scipy.integrate
import scipy.interpolate
import astropy.io.fits
import astropy.wcs

from lsst.pex.config import Config
from lsst.pipe.base import Task

from pfs.datamodel import MaskHelper
from .datamodel import PfsReference
from .utils import getPfsVersions


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
            raise RuntimeError("Mismatched lengths for wavelength and transmission: "
                               f"{len(wavelength)} vs {len(transmission)}")
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

    def integrate(self, spectrum=None, quadpack=True):
        r"""Integrate the filter transmission curve for synthetic photometry

        The integral is:

        .. math:: \int F(\lambda) S(\lambda) \lambda d\lambda

        where :math:`F(\lambda)` is the filter transmission curve,
        :math:`S(\lambda)` is the spectrum, and the extra `\lambda` term is due
        to using photon-counting detectors.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsFiberArray`, optional
            Spectrum to integrate. If not provided, use unity.
        quadpack : `bool`, optional
            Whether to use QUADPACK (default: True).
            If False, the integral is computed simply with the trapezoidal rule.
            The return value is more predictable when this parameter is False.
        """
        if quadpack:
            def function(wavelength):
                """The function we're integrating

                The function is the product of the spectrum, the bandpass and an
                extra wavelength term to account for photon counting.
                """
                ss = (
                    np.interp(wavelength, spectrum.wavelength, spectrum.flux, 0.0, 0.0)
                    if spectrum is not None else 1.0
                )
                ff = self.interpolate(wavelength)
                return ss*ff/wavelength

            return scipy.integrate.quad(function, self.wavelength[0], self.wavelength[-1],
                                        epsabs=0.0, epsrel=2.0e-3, limit=100)[0]
        else:
            if spectrum is not None:
                y = spectrum.flux * self.interpolate(spectrum.wavelength) / spectrum.wavelength
                x = spectrum.wavelength
            else:
                y = self.transmission / self.wavelength
                x = self.wavelength
            return 0.5 * np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]))

    def photometer(self, spectrum, quadpack=True):
        """Measure flux with this filter.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsSimpleSpectrum`
            Spectrum to integrate.
        quadpack : `bool`
            Whether to use QUADPACK (default: True).
            If False, integrals are computed simply with the trapezoidal rule.
            The return value is more predictable when this parameter is False.

        Returns
        -------
        flux : `float`
            Integrated flux.
        """
        return self.integrate(spectrum, quadpack=quadpack) / self.integrate(quadpack=quadpack)


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

        filename = os.path.join(os.environ["OBS_PFS_DIR"], "pfs", "fluxCal", "bandpass",
                                self.filenames[filterName])
        data = np.genfromtxt(filename, dtype=[('wavelength', 'f4'), ('flux', 'f4')])
        wavelength = data["wavelength"]
        transmission = data["flux"]  # Relative transmission
        super().__init__(wavelength, transmission)


AMBRE_FILES = ["p6500_g+4.0_m0.0_t01_z+0.00_a+0.00.AMBRE_Extp.fits",
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
    filename = os.path.join(os.environ["DRP_PFS_DATA_DIR"], "fluxCalSim",
                            AMBRE_FILES[target.objId % len(AMBRE_FILES)])
    with astropy.io.fits.open(filename) as fits:
        wcs = astropy.wcs.WCS(fits[1].header)
        refFlux = fits[1].data["Flux"]  # nJy

    # astropy treats the WCS as having two axes (because it's in a table with NAXIS=2).
    # We just ignore the other axis.
    refWavelength = wcs.pixel_to_world(np.arange(len(refFlux)), 0)[0].to("nm").value

    flux = scipy.interpolate.interp1d(refWavelength, refFlux, kind="linear", bounds_error=False,
                                      fill_value=np.nan, copy=True, assume_sorted=True)(wavelength)

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
            raise RuntimeError("This implementation requires a single fiber flux, but was provided: %s" %
                               (spectrum.target.fiberFlux,))
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
        ref *= fiberFlux*bandpass.integrate()/bandpass.integrate(ref)  # nJy
        return ref
