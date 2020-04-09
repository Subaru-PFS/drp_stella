import numpy as np

from lsst.pex.config import Config, ConfigurableField, ListField
from lsst.pipe.base import Task

from .datamodel import PfsSingle
from .fitFocalPlane import FitFocalPlaneTask


class MeasureFluxCalibrationConfig(Config):
    """Configuration for MeasureFluxCalibrationTask"""
    fit = ConfigurableField(target=FitFocalPlaneTask, doc="Fit over the focal plane")
    refMask = ListField(dtype=str, default=[], doc="Mask flags for rejection of reference")
    obsMask = ListField(dtype=str, default=["NO_DATA", "SAT"], doc="Mask flags for rejection of observed")


class MeasureFluxCalibrationTask(Task):
    """Measure the flux calibration"""
    ConfigClass = MeasureFluxCalibrationConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fit")

    def run(self, merged, references, pfsConfig):
        """Measure the flux calibration

        This is a placeholder implementation that simply calculates the
        flux calibration vector for each of the flux standards, and fits
        them. No attempt is made to fit a functional form to these flux
        calibration vectors.

        Parameters
        ----------
        merged : `pfs.datamodel.drp.PfsMerged`
            Arm-merged spectra.
        references : `dict` mapping `int` to `pfs.datamodel.PfsSimpleSpectrum`
            Reference spectra, indexed by fiber identifier.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting fiber positions.

        Returns
        -------
        calib : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.
        """
        vectors = []
        errors = []
        masks = []
        for fiberId, ref in references.items():
            spectrum = merged.extractFiber(PfsSingle, pfsConfig, fiberId)
            vectors.append(spectrum.flux/ref.flux)
            errors.append(np.sqrt(spectrum.covar[0])/ref.flux)
            bad = (ref.mask & ref.flags.get(*self.config.refMask)) > 0
            bad |= (spectrum.mask & spectrum.flags.get(*self.config.obsMask)) > 0
            masks.append(bad)
        wavelength = merged.wavelength[0]
        for wl in merged.wavelength[1:, :]:
            assert np.all(wl == wavelength)
        return self.fit.run(wavelength, vectors, errors, masks, list(references.keys()), pfsConfig)

    def applySpectra(self, spectra, pfsConfig, calib):
        """Apply the flux calibration to spectra, in-place

        Parameters
        ----------
        spectra : `pfs.datamodel.PfsFiberArraySet`
            Spectra to correct.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting fiber positions.
        calib : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.

        Returns
        -------
        results : `list` of `pfs.datamodel.PfsSingle`
            Flux-calibrated object spectra.
        """
        cal = self.fit.apply(calib, spectra.wavelength, pfsConfig.fiberId, pfsConfig)
        with np.errstate(divide="ignore", invalid="ignore"):
            spectra /= cal
        badMask = spectra.flags.add("BAD_FLUXCAL")
        for ii in range(len(spectra)):
            bad = (~np.isfinite(cal[ii])) | (cal[ii] == 0.0)
            if np.any(bad):
                spectra.mask[ii][bad] |= badMask

    def applySpectrum(self, spectrum, fiberId, pfsConfig, calib):
        """Apply the flux calibration to a single spectrum, in-place

        Parameters
        ----------
        spectrum : `pfs.datamodel.PfsFiberArray`
            Spectrum to correct.
        fiberId : `int`
            Fiber identifier.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting fiber positions.
        calib : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.
        """
        cal = self.fit.apply(calib, spectrum.wavelength, [fiberId], pfsConfig)
        with np.errstate(divide="ignore", invalid="ignore"):
            spectrum /= cal
        badMask = spectrum.flags.add("BAD_FLUXCAL")
        bad = (~np.isfinite(cal)) | (cal == 0.0)
        if np.any(bad):
            spectrum.mask[bad] |= badMask
