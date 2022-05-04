from pfs.datamodel.masks import MaskHelper
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.datamodel.target import Target
from pfs.datamodel.wavelengthArray import WavelengthArray

import numpy as np
from numpy.lib import recfunctions
import astropy.io.fits

import functools
import os


class FluxModelSet:
    """A set of flux model spectra.

    This class does not load all models into memory but read them from files
    every time they are requested. This behavior is due to so vast a number
    of the models that they cannot be contained in DRAM.

    Parameters
    ----------
    dirname : `str`
        Path to a directory that contains spectrum files.
        This is typically ``lsst.utils.getPackageDir("fluxmodeldata")``.
    """

    def __init__(self, dirname):
        self.dirname = dirname

    @property  # type: ignore [misc]
    @functools.lru_cache()
    def parameters(self):
        """List of available parameters.

        This is a numpy structured array, among whose columns are at least
        the SED parameters ("teff", "logg", "m", and "alpha") that can be
        passed to ``self.readSpectrum()``.

        This array also includes broadband photometries that are integrations
        of the synthetic spectra with various filters applied to them.

        Returns
        -------
        parameters : `numpy.array`
            A structured array,
            whose columns are SED parameters and various broadband fluxes.
        """
        filename = "broadband/photometries.fits"
        with astropy.io.fits.open(os.path.join(self.dirname, filename)) as fits:
            # We convert FITS_rec back to numpy's structured array
            # because FITS_rec cannot be indexed with multiple column names.
            parameters = np.asarray(fits[1].data)

        # Field names of "photometries.fits" start with capitals.
        # We make them lower for consistency with parameter names
        # of other methods of this class.
        parameters = recfunctions.rename_fields(parameters, {
            key: key.lower() for key in ["Teff", "Logg", "M", "Alpha"]
        })
        return parameters

    def getFileName(self, *, teff, logg, m, alpha):
        """Get the name of the file that corresponds to the given arguments.

        Parameters
        ----------
        teff : `float`
            Effective temperature in K.
        logg : `float`
            Surface gravity in Log(g/cm/s^2).
        m : `float`
            Metalicity in M/H.
        alpha : `float`
            Alpha-elements abundance in alpha/Fe.

        Returns
        -------
        path : `str`
            File path starting with ``self.dirname``.
        """
        args = {
            "teff": int(round(teff)),
            "logg": logg + 0.0,  # "+ 0.0" turns -0 to +0.
            "m": m + 0.0,
            "alpha": alpha + 0.0,
        }
        filename = "spectra/fluxmodel_%(teff)d_g_%(logg).2f_z_%(m).2f_a_%(alpha).1f.fits" % args
        return os.path.join(self.dirname, filename)

    def getSpectrum(self, teff, logg, m, alpha):
        """Get the spectrum that corresponds to the given arguments.

        Parameters
        ----------
        teff : `float`
            Effective temperature in K.
        logg : `float`
            Surface gravity in Log(g/cm/s^2).
        m : `float`
            Metalicity in M/H.
        alpha : `float`
            Alpha-elements abundance in alpha/Fe.

        Returns
        -------
        spectrum : `pfs.datamodel.pfsSimpleSpectrum.PfsSimpleSpectrum`
            The spectrum.
        """
        path = self.getFileName(teff=teff, logg=logg, m=m, alpha=alpha)
        return self.readSpectrum(path)

    def readSpectrum(self, path):
        """Read the file at ``path``.

        Parameters
        ----------
        path : `str`
            Spectrum's file name.

        Returns
        -------
        spectrum : `pfs.datamodel.pfsSimpleSpectrum.PfsSimpleSpectrum`
            The spectrum.
        """
        with astropy.io.fits.open(path) as hdus:
            header = hdus[0].header
            start = header["CRVAL1"] + header["CDELT1"] * (1 - header["CRPIX1"])
            stop = header["CRVAL1"] + header["CDELT1"] * (header["NAXIS1"] - header["CRPIX1"])
            wavelength = WavelengthArray(start, stop, header["NAXIS1"], dtype=float)

            flux = hdus[0].data.astype(float)

            target = Target(0, 0, "0,0", 0)
            mask = np.zeros(shape=wavelength.shape, dtype=int)
            flags = MaskHelper()
            mask[:] = np.where(np.isfinite(flux), 0, flags.add("BAD"))
            return PfsSimpleSpectrum(target, wavelength, flux, mask, flags)
