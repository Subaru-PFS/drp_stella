import types
import numpy as np
import astropy.io.fits
import scipy.interpolate

from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.pipe.base import Task

from pfs.datamodel.pfsConfig import TargetType
from .fitFocalPlane import FitFocalPlaneTask


class SubtractSky1DSolution(types.SimpleNamespace):
    """Spectral function on the focal plane

    This implementation is a placeholder, as it simply returns a constant
    spectrum.

    Parameters
    ----------
    wavelength : `numpy.ndarray`
        Wavelength for spectrum, nm.
    flux : `numpy.ndarray`
        Flux for spectrum, nJy.
    """
    def __init__(self, wavelength, flux):
        interpolator = scipy.interpolate.interp1d(wavelength, flux, kind='linear', bounds_error=False,
                                                  fill_value=0, copy=True, assume_sorted=True)
        super().__init__(wavelength=wavelength, flux=flux, interpolate=interpolator)

    def __call__(self, wavelengths, positions):
        """Evaluate the spectrum at the provided wavelengths+positions

        Parameters
        ----------
        wavelengths : `numpy.ndarray` of shape ``(N, M)``
            Wavelengths at which to evaluate.
        positions : `numpy.ndarray` of shape ``(N, 2)``
            Positions at which to evaluate.

        Returns
        -------
        result : `numpy.ndarray` of shape ``(N, M)``
            Vector function evaluated at each position.
        """
        result = np.empty(wavelengths.shape, dtype=self.flux.dtype)
        for ii, (wl, pos) in enumerate(zip(wavelengths, positions)):
            # Ignoring pos
            result[ii] = self.interpolate(wl)
        return result

    @classmethod
    def readFits(cls, filename):
        """Read from FITS file

        Parameters
        ----------
        filename : `str`
            Filename to read.

        Returns
        -------
        self : `FocalPlaneFunction`
            Function read from FITS file.
        """
        with astropy.io.fits.open(filename) as fits:
            wavelength = fits[0].data
            flux = fits[1].data
        return cls(wavelength, flux)

    def writeFits(self, filename):
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        fits = astropy.io.fits.HDUList()
        fits.append(astropy.io.fits.ImageHDU(self.wavelength))
        fits.append(astropy.io.fits.ImageHDU(self.flux))
        with open(filename, "wb") as fd:
            fits.writeto(fd)


class SubtractSky1dConfig(Config):
    """Configuration for SubtractSky1dTask"""
    fit = ConfigurableField(target=FitFocalPlaneTask, doc="Fit over the focal plane")
    mask = ListField(dtype=str, default=["NO_DATA", "SAT"], doc="Mask flags for rejection of observed")
    minWavelength = Field(dtype=float, default=300, doc="Minimum wavelength for resampled spectra (nm)")
    maxWavelength = Field(dtype=float, default=1300, doc="Maximum wavelength for resampled spectra (nm)")
    deltaWavelength = Field(dtype=float, default=0.03, doc="Wavelength spacing for resampled spectra (nm)")


class SubtractSky1dTask(Task):
    """Subtraction of sky in the 1D spectra

    This is a placeholder implementation that simply fits the sky spectra over
    the focal plane. No attempt is made to deal with the (bright) sky lines
    separately from the continuum.
    """
    ConfigClass = SubtractSky1dConfig
    _DefaultName = "subtractSky1d"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fit")

    def run(self, spectraList, pfsConfig, lsfList):
        """Measure and subtract the sky from the 1D spectra

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsSpectra`
            List of spectra from which to subtract the sky.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration of the top-end, for identifying sky fibers.
        lsfList : iterable of LSF (type TBD)
            List of line-spread functions.

        Returns
        -------
        sky1d : `pfs.drp.stella.FocalPlaneFunction`
            1D sky model.
        """
        resampledList = self.resampleSpectra(spectraList)
        sky1d = self.measureSky(resampledList, pfsConfig, lsfList)
        for spectra, lsf in zip(spectraList, lsfList):
            self.subtractSky(spectra, lsf, pfsConfig, sky1d)
        return sky1d

    def resampleSpectra(self, spectraList):
        """Resample the spectra to a common wavelength scale

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsSpectra`
            Spectra to resample

        Returns
        -------
        resampled : `list` of `pfs.datamodel.PfsSpectra`
            Resampled spectra.
        """
        minWl = self.config.minWavelength
        maxWl = self.config.maxWavelength
        dWl = self.config.deltaWavelength
        wavelength = minWl + dWl*np.arange(int((maxWl - minWl)/dWl), dtype=float)
        return [spectra.resample(wavelength) for spectra in spectraList]

    def measureSky(self, spectraList, pfsConfig, lsfList):
        """Measure the 1D sky model

        Parameters
        ----------
        resampledList : iterable of `pfs.datamodel.PfsSpectra`
            List of spectra (with common wavelengths) from which to measure
            the sky.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration of the top-end, for identifying sky fibers.
        lsfList : iterable of LSF (type TBD)
            List of line-spread functions.

        Returns
        -------
        sky1d : `pfs.drp.stella.FocalPlaneFunction`
            1D sky model.
        """
        select = pfsConfig.targetType == int(TargetType.SKY)
        vectors = []
        errors = []
        masks = []
        fiberId = []
        skyFibers = set(pfsConfig.fiberId[select])
        for spectra, lsf in zip(spectraList, lsfList):
            maskVal = spectra.flags.get(*self.config.mask)
            for ii, ff in enumerate(spectra.fiberId):
                if ff not in skyFibers:
                    continue
                vectors.append(spectra.flux[ii])
                errors.append(np.sqrt(spectra.covar[ii][0]))
                masks.append((spectra.mask[ii] & maskVal) > 0)
                fiberId.append(ff)
        fit = self.fit.run(vectors, errors, masks, fiberId, pfsConfig)
        return SubtractSky1DSolution(spectraList[0].wavelength[0], fit.vector)

    def subtractSky(self, spectra, lsf, pfsConfig, sky1d):
        """Subtract the 1D sky model from the spectra

        Parameters
        ----------
        spectra : `pfs.datamodel.PfsSpectra`
            Spectra to have sky subtracted.
        lsf : LSF (type TBD)
            Line-spread function.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting location of fibers.
        sky1d : `pfs.drp.stella.FocalPlaneFunction`
            1D sky model.
        """
        centers = pfsConfig.extractCenters(pfsConfig.fiberId)
        fluxes = sky1d(spectra.wavelength, centers)
        spectra.flux -= fluxes
