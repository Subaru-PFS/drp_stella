from collections.abc import Collection
import numpy as np

from lsst.pex.config import Config, ConfigurableField
from lsst.pipe.base import Task

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
        pfsArmList: Collection[PfsFiberArraySet],
        pfsConfig: PfsConfig,
        skyNormsList: Collection[FocalPlaneFunction],
    ) -> list[SkyModel]:
        """Fit 1D sky model

        Operates on all arms of the same kind from a single exposure.

        Parameters
        ----------
        pfsArmList : collection of `PfsFiberArraySet`
            Spectra from which to subtract sky model.
        pfsConfig : `PfsConfig`
            Top-end configuration.
        skyNormsList : collection of `FocalPlaneFunction`
            Common-mode sky normalizations.

        Returns
        -------
        sky1d : `list` of `SkyModel`
            Sky models subtracted.
        """
        pfsConfig = self.skyNorms.selectSky.run(pfsConfig)
        numFibers = len(pfsConfig)
        values = np.full(numFibers, np.nan, dtype=float)
        variances = np.full(numFibers, np.nan, dtype=float)
        masks = np.full(numFibers, True, dtype=bool)
        wavelength = np.median(pfsArmList[0].wavelength)
        splinesList = []

        for pfsArm, skyNorms in zip(pfsArmList, skyNormsList):
            skyConfig = pfsConfig.select(fiberId=pfsArm.fiberId)
            skyArm = pfsArm.select(fiberId=skyConfig.fiberId)

            sky = self.skyNorms.runSingle(skyArm, skyConfig, skyNorms)
            splinesList.append(sky.sky)
            data = sky.skyNorms([wavelength], skyConfig)

            indices = np.searchsorted(pfsConfig.fiberId, skyConfig.fiberId)
            values[indices] = data.values
            variances[indices] = data.variances
            masks[indices] = data.masks

        covar = np.zeros((numFibers, 3, 1), dtype=float)
        covar[:, 0, :] = variances[:, None]

        shape = (numFibers, 1)

        dummy = PfsFiberArraySet(
            pfsArmList[0].identity,
            pfsConfig.fiberId,
            np.full(shape, wavelength, dtype=float),
            values.reshape(shape),
            np.where(masks, pfsArm.flags.get("NO_DATA"), 0).reshape(shape),
            np.zeros(shape, dtype=float),
            np.ones(shape, dtype=float),
            covar,
            pfsArm.flags,
            {},
        )

        focalPlanePoly = self.focalPlanePoly.run(dummy, skyConfig)

        sky1d = [
            SkyModel(splines=splines, fiberPoly=skyNorms, focalPlanePoly=focalPlanePoly)
            for pfsArm, splines, skyNorms in zip(pfsArmList, splinesList, skyNormsList)
        ]

        return sky1d
