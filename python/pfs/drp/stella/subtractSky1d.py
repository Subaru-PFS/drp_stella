import numpy as np

from lsst.pex.config import Config, ConfigurableField
from lsst.pipe.base import Task, Struct

from pfs.datamodel import PfsConfig
from pfs.datamodel import PfsFiberArraySet
from .fitFocalPlane import FitFocalPlanePolynomialTask
from .focalPlaneFunction import FocalPlaneFunction, SkyModel
from .skyNorms import MeasureSkyNormsTask


__all__ = ("subtractSky1d", "FitSky1dConfig", "FitSky1dTask")


def subtractSky1d(spectra: PfsFiberArraySet, pfsConfig: PfsConfig, sky1d: FocalPlaneFunction) -> None:
    """Subtract sky model from spectra

    Parameters
    ----------
    spectra : `PfsFiberArraySet`
        Spectra from which to subtract sky model. The spectra are modified.
    pfsConfig : `PfsConfig`
        Top-end configuration.
    sky1d : `FocalPlaneFunction`
        Sky model.
    """
    sky = sky1d(spectra.wavelength, pfsConfig.select(fiberId=spectra.fiberId))
    skyValues = sky.values*spectra.norm
    skyVariances = sky.variances*spectra.norm**2
    spectra.flux -= skyValues
    spectra.sky += skyValues
    bitmask = spectra.flags.add("BAD_SKY")
    spectra.mask[np.array(sky.masks)] |= bitmask
    spectra.covar[:, 0, :] += skyVariances


class FitSky1dConfig(Config):
    """Configuration for SubtractSky1dTask"""
    skyNorms = ConfigurableField(target=MeasureSkyNormsTask, doc="Measure sky normalizations")
    focalPlanePoly = ConfigurableField(target=FitFocalPlanePolynomialTask, doc="Fit focal plane polynomial")


class FitSky1dTask(Task):
    """Fit sky model from spectra

    Optionally measures scaling factors for each fiber (independently) using
    the sky lines. The scaling factors are applied to the sky model before
    subtraction.
    """

    ConfigClass = FitSky1dConfig
    _DefaultName = "subtractSky1d"

    skyNorms: MeasureSkyNormsTask
    focalPlanePoly: FitFocalPlanePolynomialTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("skyNorms")
        self.makeSubtask("focalPlanePoly")

    def run(
        self,
        pfsArm: PfsFiberArraySet,
        pfsConfig: PfsConfig,
        skyNorms: FocalPlaneFunction,
    ) -> Struct:
        """Fit 1D sky model

        Parameters
        ----------
        pfsArm : `PfsFiberArraySet`
            Spectra from which to subtract sky model. The spectra are modified.
        pfsConfig : `PfsConfig`
            Top-end configuration.
        skyNorms : `FocalPlaneFunction`
            Common-mode sky normalizations for each fiber.

        Returns
        -------
        sky1d : `SkyModel`
            Sky models subtracted.
        """
        pfsConfig = pfsConfig.select(fiberId=pfsArm.fiberId)
        skyConfig = self.skyNorms.selectSky.run(pfsConfig)
        skyArm = pfsArm.select(fiberId=skyConfig.fiberId)

        wavelength = np.median(skyArm.wavelength)

        sky = self.skyNorms.runSingle(skyArm, skyConfig, skyNorms)
        data = sky.skyNorms([wavelength], skyConfig)
        covar = np.zeros((len(skyConfig), 3, 1), dtype=float)
        covar[:, 0, :] = data.variances

        mask = np.zeros_like(data.masks, dtype=int)
        mask[data.masks] = pfsArm.flags.get("NO_DATA")

        dummy = PfsFiberArraySet(
            pfsArm.identity,
            skyArm.fiberId,
            np.full_like(skyArm.fiberId, wavelength, dtype=float),
            data.values,
            mask,
            np.zeros_like(skyArm.fiberId, dtype=float),
            np.ones_like(skyArm.fiberId, dtype=float),
            covar,
            pfsArm.flags,
        )

        focalPlanePoly = self.focalPlanePoly.run(dummy, skyConfig)
        return SkyModel(splines=sky.sky, fiberPoly=skyNorms, focalPlanePoly=focalPlanePoly)
