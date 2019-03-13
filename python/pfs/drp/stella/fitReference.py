import numpy as np

from lsst.pex.config import Config
from lsst.pipe.base import Task

from pfs.datamodel.drp import PfsReference


class FitReferenceConfig(Config):
    """Configuration for FitReferenceTask"""
    pass


class FitReferenceTask(Task):
    """Fit a physical reference spectrum to an observed spectrum

    This implementation is a placeholder, appropriate only for processing
    simulated spectra with a constant flux density (per unit frequency)
    """
    ConfigClass = FitReferenceConfig

    def run(self, spectrum):
        """Fit a physical spectrum to an observed spectrum

        This implementation is a placeholder, as the algorithm is overly
        simplistic: we provide a spectrum with a constant flux density. This
        might be appropriate for processing simulated spectra.

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsSpectrum`
            Spectrum to fit.

        Returns
        -------
        spectrum : `pfs.datamodel.PfsReference`
            Reference spectrum.
        """
        wavelength = spectrum.wavelength
        mag = set(spectrum.target.fiberMags.values())
        if len(mag) != 1:
            raise RuntimeError("This implementation requires a single ABmag, but was provided: %s" %
                               (spectrum.target.fiberMags,))
        mag = mag.pop()

        flux = 3631e9*10**(-0.4*mag)*np.ones_like(wavelength)  # nJy
        mask = np.zeros_like(wavelength, dtype=int)
        return PfsReference(spectrum.target, wavelength, flux, mask, spectrum.flags)
