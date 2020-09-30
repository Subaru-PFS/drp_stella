import numpy as np

from lsst.pex.config import Config, ConfigurableField, ListField
from lsst.pipe.base import Task

from .datamodel import PfsSingle
from .fitFocalPlane import FitFocalPlaneTask


class MeasureFluxCalibrationConfig(Config):
    """Configuration for MeasureFluxCalibrationTask"""
    fit = ConfigurableField(target=FitFocalPlaneTask, doc="Fit over the focal plane")
    refMask = ListField(dtype=str, default=[], doc="Mask flags for rejection of reference")
    obsMask = ListField(dtype=str, default=["NO_DATA", "SAT", "BAD_FLAT"],
                        doc="Mask flags for rejection of observed")


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
        values = []
        variances = []
        masks = []
        for fiberId, ref in references.items():
            spectrum = merged.extractFiber(PfsSingle, pfsConfig, fiberId)
            values.append(spectrum.flux/ref.flux)
            variances.append(spectrum.covar[0]/ref.flux**2)
            bad = (ref.mask & ref.flags.get(*self.config.refMask)) > 0
            bad |= (spectrum.mask & spectrum.flags.get(*self.config.obsMask)) > 0
            masks.append(bad)
        wavelength = merged.wavelength[0]
        for wl in merged.wavelength[1:, :]:
            assert np.all(wl == wavelength)
        return self.fit.run(wavelength, values, masks, variances, list(references.keys()), pfsConfig)

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
            spectra /= cal.values  # includes spectra.variance /= cal.values**2
            spectra.covar[:, 0, :] += cal.variances*spectra.flux**2/np.array(cal.values)**2
        bad = np.array(cal.masks) | ~np.isfinite(cal.values) | (np.array(cal.values) == 0.0)
        spectra.mask[bad] |= spectra.flags.add("BAD_FLUXCAL")

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
            spectrum /= cal.values  # includes spectrum.variance /= cal.values**2
            spectrum.covar[0] += cal.variances*spectrum.flux**2/cal.values**2
        bad = np.array(cal.masks) | ~np.isfinite(cal.values) | (np.array(cal.values) == 0.0)
        spectrum.mask[bad] |= spectrum.flags.add("BAD_FLUXCAL")
