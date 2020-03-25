import numpy as np

from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.pipe.base import Task

from pfs.datamodel.pfsConfig import TargetType
from .fitFocalPlane import FitFocalPlaneTask


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
        spectraList : iterable of `pfs.datamodel.PfsFiberArraySet`
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
            self.subtractSkySpectra(spectra, lsf, pfsConfig, sky1d)
        return sky1d

    def resampleSpectra(self, spectraList):
        """Resample the spectra to a common wavelength scale

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsFiberArraySet`
            Spectra to resample

        Returns
        -------
        resampled : `list` of `pfs.datamodel.PfsFiberArraySet`
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
        resampledList : iterable of `pfs.datamodel.PfsFiberArraySet`
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
        wavelength = spectraList[0].wavelength[0]
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
        return self.fit.run(wavelength, vectors, errors, masks, fiberId, pfsConfig)

    def subtractSkySpectra(self, spectra, lsf, pfsConfig, sky1d):
        """Subtract the 1D sky model from the spectra, in-place

        Parameters
        ----------
        spectra : `pfs.datamodel.PfsFiberArraySet`
            Spectra to have sky subtracted.
        lsf : LSF (type TBD)
            Line-spread function.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting location of fibers.
        sky1d : `pfs.drp.stella.FocalPlaneFunction`
            1D sky model.
        """
        spectra.flux -= self.fit.apply(sky1d, spectra.wavelength, pfsConfig.fiberId, pfsConfig)

    def subtractSkySpectrum(self, spectrum, lsf, fiberId, pfsConfig, sky1d):
        """Subtract the 1D sky model from the spectrum, in-place

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsFiberArray`
            Spectrum to have sky subtracted.
        lsf : LSF (type TBD)
            Line-spread function.
        fiberId : `int`
            Fiber identifier.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting location of fibers.
        sky1d : `pfs.drp.stella.FocalPlaneFunction`
            1D sky model.
        """
        spectrum.flux -= self.fit.apply(sky1d, spectrum.wavelength, [fiberId], pfsConfig)
