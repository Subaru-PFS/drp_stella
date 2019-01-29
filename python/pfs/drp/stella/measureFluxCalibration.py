import numpy as np

from lsst.pex.config import Config, ConfigurableField, ListField
from lsst.pipe.base import Task

from pfs.datamodel.drp import PfsObject
from .fitFocalPlane import FitFocalPlaneTask


class MeasureFluxCalibrationConfig(Config):
    """Configuration for MeasureFluxCalibrationTask"""
    fit = ConfigurableField(target=FitFocalPlaneTask, doc="Fit over the focal plane")
    refMask = ListField(dtype=str, default=["NO_DATA"], doc="Mask flags for rejection of reference")
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
            spectrum = merged.extractFiber(PfsObject, pfsConfig, fiberId)
            vectors.append(spectrum.flux/ref.flux)
            errors.append(np.sqrt(spectrum.covar[0])/ref.flux)
            bad = (ref.mask & ref.flags.get(*self.config.refMask)) > 0
            bad |= (spectrum.mask & spectrum.flags.get(*self.config.obsMask)) > 0
            masks.append(bad)
        return self.fit.run(vectors, errors, masks, list(references.keys()), pfsConfig)

    def apply(self, merged, pfsConfig, calib):
        """Apply the flux calibration

        Parameters
        ----------
        merged : `pfs.datamodel.drp.PfsMerged`
            Arm-merged spectra.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration, for getting fiber positions.
        calib : `pfs.drp.stella.FocalPlaneFunction`
            Flux calibration.

        Returns
        -------
        results : `list` of `pfs.datamodel.PfsObject`
            Flux-calibrated object spectra.
        """
        vectors = self.fit.apply(calib, pfsConfig.fiberId, pfsConfig)
        results = []
        for fiberId, vv in zip(pfsConfig.fiberId, vectors):
            spectrum = merged.extractFiber(PfsObject, pfsConfig, fiberId)
            spectrum /= vv
            results.append(spectrum)
        return results
