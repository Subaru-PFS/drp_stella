import numpy as np

from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.pipe.base import Task

from pfs.datamodel.pfsConfig import TargetType
from .fitFocalPlane import FitFocalPlaneTask

import lsstDebug


class SubtractSky1dConfig(Config):
    """Configuration for SubtractSky1dTask"""
    fit = ConfigurableField(target=FitFocalPlaneTask, doc="Fit over the focal plane")
    mask = ListField(dtype=str, default=["NO_DATA", "SAT", "BAD_FLAT"],
                     doc="Mask flags for rejection of observed")
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
        self.debugInfo = lsstDebug.Info(__name__)

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
        if not np.any(pfsConfig.targetType == TargetType.SKY):
            raise RuntimeError("No sky fibers found")
        if self.debugInfo.plotSkyFluxes:
            self.plotSkyFibers(spectraList, pfsConfig, "Sky flux")
        resampledList = self.resampleSpectra(spectraList, pfsConfig)
        sky1d = self.measureSky(resampledList, pfsConfig, lsfList)
        for spectra, lsf in zip(spectraList, lsfList):
            self.subtractSkySpectra(spectra, lsf, pfsConfig, sky1d)
        if self.debugInfo.plotSkyResiduals:
            self.plotSkyFibers(spectraList, pfsConfig, "Sky residuals")
        return sky1d

    def resampleSpectra(self, spectraList, pfsConfig):
        """Resample the sky spectra to a common wavelength scale

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsFiberArraySet`
            Spectra to resample
        pfsConfig : `pfs.datamodel.PfsConfig`
            Configuration of the top-end, for identifying sky fibers.

        Returns
        -------
        resampled : `list` of `pfs.datamodel.PfsFiberArraySet`
            Resampled sky fiber spectra.
        """
        minWl = self.config.minWavelength
        maxWl = self.config.maxWavelength
        dWl = self.config.deltaWavelength
        wavelength = minWl + dWl*np.arange(int((maxWl - minWl)/dWl), dtype=float)
        index = pfsConfig.selectByTargetType(TargetType.SKY)
        return [spectra.resample(wavelength, pfsConfig.fiberId[index]) for spectra in spectraList]

    def measureSky(self, spectraList, pfsConfig, lsfList):
        """Measure the 1D sky model

        Parameters
        ----------
        spectraList : iterable of `pfs.datamodel.PfsFiberArraySet`
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
        wavelength = spectraList[0].wavelength[0]  # They're all resampled to a common wavelength scale
        vectors = []
        errors = []
        masks = []
        fiberId = []
        for spectra, lsf in zip(spectraList, lsfList):
            maskVal = spectra.flags.get(*self.config.mask)
            for ii, ff in enumerate(spectra.fiberId):
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
        skyFlux = self.fit.apply(sky1d, spectra.wavelength, pfsConfig.fiberId, pfsConfig)
        spectra.flux -= skyFlux
        spectra.sky += skyFlux

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
        skyFlux = self.fit.apply(sky1d, spectrum.wavelength, [fiberId], pfsConfig)
        spectrum.flux -= skyFlux
        spectrum.sky += skyFlux

    def plotSkyFibers(self, spectraList, pfsConfig, title):
        """Plot spectra from sky fibers

        The spectra from different fibers are shown as different colors.
        Masked points are drawn as dots.

        Parameters
        ----------
        spectraList : iterable of `PfsArm`
            Extracted spectra from spectrograph arms for a single exposure.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for identifying sky fibers.
        title : `str`
            Title for plot.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm

        indices = pfsConfig.selectByTargetType(TargetType.SKY)
        fiberId = dict(zip(pfsConfig.fiberId[indices],
                           matplotlib.cm.rainbow(np.linspace(0, 1, len(indices)))))

        figure, axes = plt.subplots()
        for spectra in spectraList:
            for ii, ff in enumerate(spectra.fiberId):
                if ff not in fiberId:
                    continue
                axes.plot(spectra.wavelength[ii], spectra.flux[ii], ls="solid", color=fiberId[ff])
                bad = (spectra.mask[ii] & spectra.flags.get(*self.config.mask)) != 0
                if np.any(bad):
                    axes.plot(spectra.wavelength[ii][bad], spectra.flux[ii][bad], ".", color=fiberId[ff])

        axes.set_xlabel("Wavelength (nm)")
        axes.set_ylabel("Flux")
        axes.set_title(title)
        figure.show()
        input("Hit ENTER to continue... ")
