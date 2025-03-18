from deprecated import deprecated

import numpy as np

from lsst.pex.config import Config, ConfigurableField, Field, ListField
from lsst.pipe.base import Task, Struct

from pfs.datamodel import PfsConfig
from pfs.datamodel import PfsFiberArraySet
from .fitContinuum import FitContinuumTask
from .focalPlaneFunction import FocalPlaneFunction
from .referenceLine import ReferenceLineSet
from .utils.math import robustRms

__all__ = ("subtractSky1d", "SubtractSky1dConfig", "SubtractSky1dTask")


@deprecated("Use SubtractSky1dTask for better sky subtraction")
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
    config = SubtractSky1dConfig()
    config.doScaling = False
    sky = sky1d(spectra.wavelength, pfsConfig.select(fiberId=spectra.fiberId))
    SubtractSky1dTask(config=config).apply(spectra, sky)


class SubtractSky1dConfig(Config):
    """Configuration for SubtractSky1dTask"""
    doScaling = Field(dtype=bool, default=True, doc="Scale sky model to match data?")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum")
    minSignalToNoise = Field(dtype=float, default=5.0, doc="Minimum signal-to-noise for data to fit")
    mask = ListField(
        dtype=str, default=["BAD", "CR", "SAT", "BAD_FLAT", "NO_DATA"], doc="Mask bits to reject"
    )
    iterations = Field(dtype=int, default=2, doc="Number of iterations")
    rejection = Field(dtype=float, default=3.0, doc="Rejection threshold (stdev)")


class SubtractSky1dTask(Task):
    """Subtract sky model from spectra

    Optionally measures scaling factors for each fiber (independently) using
    the sky lines. The scaling factors are applied to the sky model before
    subtraction.
    """

    ConfigClass = SubtractSky1dConfig
    _DefaultName = "subtractSky1d"

    fitContinuum: FitContinuumTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitContinuum")

    def run(
        self,
        pfsArm: PfsFiberArraySet,
        pfsConfig: PfsConfig,
        sky1d: FocalPlaneFunction,
        refLines: ReferenceLineSet | None = None,
    ) -> Struct:
        """Subtract sky model from spectra

        Parameters
        ----------
        pfsArm : `PfsFiberArraySet`
            Spectra from which to subtract sky model. The spectra are modified.
        pfsConfig : `PfsConfig`
            Top-end configuration.
        sky1d : `FocalPlaneFunction`
            Sky model.
        refLines : `ReferenceLineSet`, optional
            Reference lines for fitting continuum.

        Returns
        -------
        scaling : `Struct`
            Scaling factors for sky model, or ``None`` if scaling is disabled.
        """
        pfsConfig = pfsConfig.select(fiberId=pfsArm.fiberId)
        skyModel = sky1d(pfsArm.wavelength, pfsConfig.select(fiberId=pfsArm.fiberId))
        scaling: Struct | None = None
        if self.config.doScaling:
            scaling = self.measureScaling(pfsArm, skyModel, refLines)
        self.apply(pfsArm, skyModel, scaling.factor if scaling is not None else None)
        return scaling

    def apply(
        self,
        pfsArm: PfsFiberArraySet,
        skyModel: Struct,
        factor: Struct | None = None,
    ) -> None:
        """Apply sky subtraction to spectra

        Parameters
        ----------
        pfsArm : `PfsFiberArraySet`
            Spectra from which to subtract sky model. The spectra are modified.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        skyModel : `Struct`
            Sky model, realized for the ``pfsArm``.
        factor : `np.ndarray`, optional
            Scaling factor for sky model.
        """
        if factor is not None:
            factor = np.ones(len(pfsArm), dtype=float)
        skyValues = skyModel.values*factor[:, None]*pfsArm.norm
        skyVariances = skyModel.variances*factor[:, None]**2*pfsArm.norm**2
        pfsArm.flux -= skyValues
        pfsArm.sky += skyValues
        bitmask = pfsArm.flags.add("BAD_SKY")
        pfsArm.mask[np.array(skyModel.masks)] |= bitmask
        pfsArm.covar[:, 0, :] += skyVariances

    def measureScaling(
        self,
        pfsArm: PfsFiberArraySet,
        skyModel: Struct,
        refLines: ReferenceLineSet | None = None,
    ) -> Struct:
        """Measure scaling factors for sky model

        Parameters
        ----------
        pfsArm : `PfsFiberArraySet`
            Spectra from which to subtract sky model.
        skyModel : `Struct`
            Sky model, realized for the ``pfsArm``.
        refLines : `ReferenceLineSet`, optional
            Reference lines for fitting continuum.

        Returns
        -------
        factor : `np.ndarray`
            Scaling factors of sky model for each fiber.
        armContinuum : `np.ndarray`
            Continuum fit to the ``pfsArm``.
        skyContinuum : `np.ndarray`
            Continuum fit to the sky model.
        """
        armContinuum = self.fitContinuum.run(pfsArm, refLines)

        zeros = np.zeros_like(pfsArm.flux)
        skySpectra = PfsFiberArraySet(
            pfsArm.identity,
            pfsArm.fiberId,
            pfsArm.wavelength,
            skyModel.values*pfsArm.norm,
            skyModel.masks,
            np.zeros_like(pfsArm.flux),
            np.ones_like(pfsArm.norm),
            np.vstack([skyModel.variances*pfsArm.norm**2, zeros, zeros]),
            pfsArm.flags,
        )
        skyContinuum = self.fitContinuum.run(skySpectra, refLines)

        norm = pfsArm.norm
        flux = pfsArm.flux - armContinuum*norm
        variance = pfsArm.variance
        sky = skyModel.values*norm - skyContinuum*norm
        reject = ((pfsArm.mask & pfsArm.flags.get(*self.config.mask)) != 0)
        reject |= (flux/np.sqrt(variance) < self.config.minSignalToNoise)

        def doFit(data: np.ndarray, model: np.ndarray, variance: np.ndarray) -> np.ndarray:
            """Fit model to data

            The model is very simple: a single scale factor for each spectrum.

            Parameters
            ----------
            data : `numpy.ndarray`, shape ``(numSpectra, numSamples)``
                Data to fit.
            model : `numpy.ndarray`, shape ``(numSpectra, numSamples)``
                Model to fit (evaluated at the same points as ``data``).
            variance : `numpy.ndarray`
                Variance of the data.

            Returns
            -------
            factor : `numpy.ndarray`, shape ``(numSpectra,)``
                Factor to multiply the model by to best fit the data.
            armContinuum : `numpy.ndarray`, shape ``(numSpectra, numSamples)``
                Continuum fit to the data.
            skyContinuum : `numpy.ndarray`, shape ``(numSpectra, numSamples)``
                Continuum fit to the model.
            """
            modelDotModel = np.ma.sum(model**2/variance, axis=1)
            dataDotModel = np.ma.sum(data*model/variance, axis=1)
            return dataDotModel/modelDotModel

        data = np.ma.masked_where(reject, flux)
        model = np.ma.masked_where(reject, sky)
        for iteration in range(self.config.iterations):
            factor = doFit(data, model, variance)
            residuals = data - factor[:, None]*model
            self.log.debug(
                "Iteration %d: factor = %.3f +/- %.3f, RMS = %.3f",
                iteration,
                np.nanmedian(factor),
                robustRms(factor[np.isfinite(factor)]),
                np.sqrt(np.mean(residuals**2)),
            )
            reject = np.abs(residuals) > 3*np.sqrt(variance)
            data.mask |= reject
            model.mask |= reject

        factor = doFit(data, model, variance)
        residuals = data - factor[:, None]*model
        self.log.info(
            "Sky subtraction scaling factor = %.3f +/- %.3f, RMS = %.3f",
            np.nanmedian(factor),
            robustRms(factor[np.isfinite(factor)]),
            np.sqrt(np.mean(residuals**2)),
        )

        return Struct(factor=factor, armContinuum=armContinuum, skyContinuum=skyContinuum)
