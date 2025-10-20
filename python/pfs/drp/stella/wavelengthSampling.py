import numpy as np

from lsst.pex.config import Config, Field
from lsst.pipe.base import Task

from pfs.datamodel.wavelengthArray import WavelengthArray

__all__ = ("WavelengthSamplingConfig", "WavelengthSamplingTask")


class WavelengthSamplingConfig(Config):
    """Configuration for wavelength sampling"""
    minWavelength = Field(dtype=float, default=370, doc="Minimum wavelength (nm)")
    maxWavelength = Field(dtype=float, default=1270, doc="Maximum wavelength (nm)")
    resolution = Field(dtype=float, default=0.08, doc="Resolution (nm/pixel)")

    medResMinWavelength = Field(dtype=float, default=695, doc="Minimum wavelength for medium resolution (nm)")
    medResMaxWavelength = Field(dtype=float, default=905, doc="Maximum wavelength for medium resolution (nm)")
    medResResolution = Field(dtype=float, default=0.045, doc="Resolution for medium resolution (nm/pixel)")


class WavelengthSamplingTask(Task):
    """Task to create a wavelength sampling"""
    ConfigClass = WavelengthSamplingConfig
    _DefaultName = "wavelengthSampling"

    def run(self, haveMediumResolution: bool = False) -> np.ndarray:
        """Create the wavelength sampling

        Parameters
        ----------
        haveMediumResolution : `bool`
            Do we have medium-resolution data?

        Returns
        -------
        wavelength : `numpy.ndarray`
            Wavelength sampling
        """
        if haveMediumResolution:
            return self.mediumResolution()
        return self.lowResolution()

    def lowResolution(self) -> np.ndarray:
        """Create a low-resolution wavelength sampling

        Returns
        -------
        wavelength : `numpy.ndarray`
            Wavelength sampling
        """
        length = int((self.config.maxWavelength - self.config.minWavelength)/self.config.resolution + 1)
        return WavelengthArray(self.config.minWavelength, self.config.maxWavelength, length)

    def mediumResolution(self) -> np.ndarray:
        """Create a medium-resolution wavelength sampling

        Returns
        -------
        wavelength : `numpy.ndarray`
            Wavelength sampling
        """
        lowRes = self.lowResolution()
        blue = lowRes < self.config.medResMinWavelength
        red = lowRes > self.config.medResMaxWavelength

        medResRange = self.config.medResMaxWavelength - self.config.medResMinWavelength
        length = int(medResRange/self.config.medResResolution + 1)
        medRes = WavelengthArray(self.config.medResMinWavelength, self.config.medResMaxWavelength, length)

        return np.concatenate([lowRes[blue], medRes, lowRes[red]])
