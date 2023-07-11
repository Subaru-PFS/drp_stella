from typing import Iterable

import numpy as np

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task

from pfs.datamodel.wavelengthArray import WavelengthArray

from .interpolate import calculateDispersion


class WavelengthConfig(Config):
    """Configuration for wavelength determination"""
    minWavelength = Field(dtype=float, default=350, doc="Minimum wavelength (nm)")
    maxWavelength = Field(dtype=float, default=1270, doc="Maximum wavelength (nm)")


class WavelengthTask(Task):
    ConfigClass = WavelengthConfig
    _DefaultName = "wavelength"

    def run(self, wavelengthArrays: Iterable[np.ndarray]) -> np.ndarray:
        """Calculate a suitable wavelength sampling for the merged spectra

        We choose a sampling that is the lowest (in nm/pixel) of the input
        samplings (so that we don't lose any information) and adopt that for
        the entire wavelength range.

        Parameters
        ----------
        spectra : iterable of `pfs.datamodel.PsfArm`
            Extracted spectra from the different arms, for a single exposure.

        Returns
        -------
        wavelength : `WavelengthArray`
            Wavelength array for the merged spectra.
        """
        dispersion = min(calculateDispersion(wl).min() for wl in wavelengthArrays)  # nm/pix
        dWavelength = self.config.maxWavelength - self.config.minWavelength  # nm
        length = int(round(dWavelength/dispersion)) + 1
        return WavelengthArray(self.config.minWavelength, self.config.maxWavelength, length)
