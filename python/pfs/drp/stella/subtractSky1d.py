from collections.abc import Collection
from functools import reduce

import numpy as np

from lsst.pex.config import Config, ConfigurableField, Field, ListField
from lsst.pipe.base import Task, Struct

from pfs.datamodel import PfsConfig
from pfs.datamodel import PfsFiberArraySet
from pfs.datamodel.pfsFocalPlaneFunction import PfsFocalPlaneFunction, PfsSkyModel
from .fitContinuum import FitContinuumTask
from .fitFocalPlane import FitBlockedOversampledSplineTask
from .focalPlaneFunction import FocalPlaneFunction
from .math import NormalizedPolynomial1D, calculateMedian
from .referenceLine import ReferenceLineSet
from .selectFibers import SelectFibersTask
from .utils.math import robustRms

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


class SkyModel(FocalPlaneFunction):
    """Model of the sky

    Consists of a spectral reproduction of the sky using
    `BlockedOversampledSpline`, with a polynomial correction to the
    normalization for each fiber.

    Parameters
    ----------
    spectra : `BlockedOversampledSpline`
        Sky spectra model.
    fiberId : `numpy.ndarray` of `int`
        Fiber identifiers.
    minWavelength : `float`
        Minimum wavelength of the model (used to normalize the polynomial).
    maxWavelength : `float`
        Maximum wavelength of the model (used to normalize the polynomial).
    polynomials : `list` of `numpy.ndarray`
        Polynomial coefficients for the normalization for each fiber.
    """
    DamdClass = PfsSkyModel

    def __init__(self, *args, datamodel: PfsFocalPlaneFunction | None = None, **kwargs):
        super().__init__(*args, datamodel=datamodel, **kwargs)
        minWavelength = self.minWavelength
        maxWavelength = self.maxWavelength
        self._polynomials = {
            ff: NormalizedPolynomial1D(poly, minWavelength, maxWavelength)
            for ff, poly in zip(self.fiberId, self.polynomials)
        }
        self._spectra = FocalPlaneFunction.fromDatamodel(self.spectra)

    @classmethod
    def fitArrays(cls, *args, **kwargs) -> FocalPlaneFunction:
        """Fit a sky model to arrays

        This is deliberately disabled. We don't fit sky models using this
        method, because it is built from multiple arms, whereas
        FocalPlaneFunction.fitArrays is really only intended for fitting a
        single arm.

        The proper fitting is done by SubtractSky1dTask.
        """
        raise NotImplementedError("SkyModel.fitArrays is not implemented; use SubtractSky1dTask")

    def evaluate(self, wavelengths: np.ndarray, fiberIds: np.ndarray, positions: np.ndarray) -> Struct:
        """Evaluate the function at the provided positions

        We interpolate the variance without doing any of the usual error
        propagation. Since there is likely some amount of unknown covariance
        (if only from resampling to a common wavelength scale), following the
        usual error propagation formulae as if there is no covariance would
        artificially suppress the noise estimates.

        Parameters
        ----------
        wavelengths : `numpy.ndarray` of shape ``(N, M)``
            Wavelength arrays.
        fiberIds : `numpy.ndarray` of `int` of shape ``(N,)``
            Fiber identifiers.
        positions : `numpy.ndarray` of shape ``(N, 2)``
            Focal-plane positions at which to evaluate.

        Returns
        -------
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Vector function evaluated at each position.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Indicates whether the value at each position is valid.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variances for each position.
        """
        values = np.empty_like(wavelengths)
        masks = np.empty_like(values, dtype=bool)
        variances = np.empty_like(values)
        skySpectra = self._spectra.evaluate(wavelengths, fiberIds, positions)
        for ii, (wl, ff) in enumerate(zip(wavelengths, fiberIds)):
            norm = self._polynomials[ff](wl)
            values[ii] = skySpectra.values[ii]*norm
            masks[ii] = skySpectra.masks[ii]
            variances[ii] = skySpectra.variances[ii]*norm**2
        return Struct(values=values, masks=masks, variances=variances)


class FitSky1dConfig(Config):
    """Configuration for SubtractSky1dTask"""
    selectSky = ConfigurableField(target=SelectFibersTask, doc="Select fibers for 1d sky subtraction")
    fitSkyModel = ConfigurableField(target=FitBlockedOversampledSplineTask,
                                    doc="Fit sky model over the focal plane")

    doNormalization = Field(dtype=bool, default=True, doc="Measure normalization for sky model?")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum")
    minSignalToNoise = Field(dtype=float, default=5.0, doc="Minimum signal-to-noise for data to fit")
    mask = ListField(
        dtype=str, default=["BAD", "CR", "SAT", "BAD_FLAT", "NO_DATA"], doc="Mask bits to reject"
    )
    minRatio = Field(dtype=float, default=0.5, doc="Minimum data/sky ratio for good data")
    maxRatio = Field(dtype=float, default=2.0, doc="Maximum data/sky ratio for good data")
    rejectRatio = Field(dtype=float, default=0.1, doc="Rejection limit of data/sky ratio")
    iterations = Field(dtype=int, default=2, doc="Number of iterations")
    rejection = Field(dtype=float, default=3.0, doc="Rejection threshold (stdev)")

    def setDefaults(self):
        super().setDefaults()
        self.selectSky.targetType = ("SKY", "SUNSS_DIFFUSE", "HOME")
        self.selectSky.targetType = ("SKY", "SUNSS_DIFFUSE", "HOME")
        # Scale back rejection because otherwise everything gets rejected
        self.fitSkyModel.rejIterations = 1
        self.fitSkyModel.rejThreshold = 4.0
        self.fitSkyModel.mask = ["NO_DATA", "BAD_FLAT", "BAD_FIBERNORMS", "SUSPECT"]


class FitSky1dTask(Task):
    """Fit sky model from spectra

    Optionally measures scaling factors for each fiber (independently) using
    the sky lines. The scaling factors are applied to the sky model before
    subtraction.
    """

    ConfigClass = FitSky1dConfig
    _DefaultName = "subtractSky1d"

    selectSky: SelectFibersTask
    fitSkyModel: FitBlockedOversampledSplineTask
    fitContinuum: FitContinuumTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("selectSky")
        self.makeSubtask("fitSkyModel")
        self.makeSubtask("fitContinuum")

    def runSingle(
        self,
        pfsArm: PfsFiberArraySet,
        pfsConfig: PfsConfig,
        refLines: ReferenceLineSet | None = None,
    ) -> Struct:
        """Fit 1D sky model from single pfsArm

        Parameters
        ----------
        pfsArm : iterable of `PfsFiberArraySet`
            Spectra from which to subtract sky model. The spectra are modified.
        pfsConfig : `PfsConfig`
            Top-end configuration.
        refLines : `ReferenceLineSet`, optional
            Reference lines for fitting continuum.

        Returns
        -------
        sky1d : `SkyModel`
            Sky model.
        """
        return self.run([pfsArm], pfsConfig, refLines)

    def run(
        self,
        pfsArmList: Collection[PfsFiberArraySet],
        pfsConfig: PfsConfig,
        refLines: ReferenceLineSet | None = None,
    ) -> Struct:
        """Fit 1D sky model from multiple spectra

        Parameters
        ----------
        pfsArmList : collection of `PfsFiberArraySet`
            Spectra from which to subtract sky model. The spectra are modified.
        pfsConfig : `PfsConfig`
            Top-end configuration.
        refLines : `ReferenceLineSet`, optional
            Reference lines for fitting continuum.

        Returns
        -------
        sky1dList : `list` of `SkyModel`
            Sky models for each arm.
        """
        fiberId: np.ndarray | None = None
        skyList: list[FocalPlaneFunction] = []  # Sky spectra model for each arm
        skySpectraList: list[Struct] = []  # Realised sky spectra for each arm
        minWavelength = np.array([pfsArm.wavelength.min() for pfsArm in pfsArmList], dtype=float)
        maxWavelength = np.array([pfsArm.wavelength.max() for pfsArm in pfsArmList], dtype=float)
        for pfsArm in pfsArmList:
            if fiberId is None:
                fiberId = pfsArm.fiberId
                pfsConfig = pfsConfig.select(fiberId=pfsArm.fiberId)
            elif not np.array_equal(fiberId, pfsArm.fiberId):
                raise ValueError("fiberId mismatch")
            sky = self.measureSky(pfsArm, pfsConfig)
            skyList.append(sky)
            skySpectraList.append(sky(pfsArm.wavelength, pfsConfig))

        norm: Struct
        if self.config.doNormalization:
            norm = self.measureNormalizations(pfsArmList, skySpectraList, refLines)
        else:
            norm = Struct(factor=np.ones(len(pfsArmList), dtype=float))

        skyModelList: list[SkyModel] = [
            SkyModel(
                spectra=sky.asDatamodel(),
                fiberId=fiberId,
                minWavelength=minWl,
                maxWavelength=maxWl,
                polynomials=norm.factor.reshape((fiberId.size, 1)),
            ) for sky, minWl, maxWl in zip(skyList, minWavelength, maxWavelength)
        ]
        return skyModelList

    def measureSky(self, pfsArm: PfsFiberArraySet, pfsConfig: PfsConfig) -> FocalPlaneFunction:
        """Measure sky spectra

        Parameters
        ----------
        pfsArm : `PfsFiberArraySet`
            Spectra from which to subtract sky model.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.

        Returns
        -------
        sky1d : `FocalBllaneFunction`
            Sky model.
        """
        skyConfig = self.selectSky.run(pfsConfig.select(fiberId=pfsArm.fiberId))
        skySpectra = pfsArm.select(pfsConfig, fiberId=skyConfig.fiberId)
        if len(skySpectra) == 0:
            raise RuntimeError("No sky spectra to use for sky subtraction")
        return self.fitSkyModel.run(skySpectra, skyConfig)

    def measureNormalizations(
        self,
        pfsArmList: Collection[PfsFiberArraySet],
        skyDataList: Collection[Struct],
        refLines: ReferenceLineSet | None = None,
    ) -> Struct:
        """Measure normalizations for sky model

        We currently measure a single normalization factor for each fiber, but
        this could be generalized to a more complex model.

        Parameters
        ----------
        pfsArmList : collection of `PfsFiberArraySet`
            Spectra from which to subtract sky model.
        skyDataList : collection of `Struct`
            Sky spectra, from model realized for each ``pfsArm``.
        refLines : `ReferenceLineSet`, optional
            Sky line list, used for fitting continuum.

        Returns
        -------
        factor : `np.ndarray`
            Scaling factors of sky data for each fiber.
        armContinuum : `np.ndarray`
            Continuum fit to the ``pfsArm``.
        skyContinuum : `np.ndarray`
            Continuum fit to the sky model.
        """
        if len(pfsArmList) != len(skyDataList):
            raise ValueError("pfsArm and skyModel must have the same length")

        armContinuumList: list[np.ndarray] = []
        skyContinuumList: list[np.ndarray] = []
        data: list[np.ma.MaskedArray] = []
        model: list[np.ma.MaskedArray] = []
        variance: list[np.ndarray] = []
        good: list[np.ndarray] = []
        for pfsArm, sky in zip(pfsArmList, skyDataList):
            armContinuum = self.fitContinuum.run(pfsArm, refLines)

            covar = np.zeros_like(pfsArm.covar)
            covar[:, 0, :] = sky.variances*pfsArm.norm**2
            skySpectra = PfsFiberArraySet(
                pfsArm.identity,
                pfsArm.fiberId,
                pfsArm.wavelength,
                sky.values*pfsArm.norm,
                sky.masks,
                np.zeros_like(pfsArm.flux),
                np.ones_like(pfsArm.norm),
                covar,
                pfsArm.flags,
                pfsArm.metadata,
            )
            skyContinuum = self.fitContinuum.run(skySpectra, refLines)

            norm = pfsArm.norm
            flux = pfsArm.flux - armContinuum*norm
            skyFlux = skySpectra.flux - skyContinuum  # norm is unity
            select = ((pfsArm.mask & pfsArm.flags.get(*self.config.mask)) == 0)
            good.append(select)
            with np.errstate(invalid="ignore", divide="ignore"):
                reject = ~select & (flux/np.sqrt(pfsArm.variance) < self.config.minSignalToNoise)
                ratio = flux/skyFlux
                reject |= (ratio < self.config.minRatio) | (ratio > self.config.maxRatio)
                factor = np.array([calculateMedian(rat, rej) for rat, rej in zip(ratio, reject)])
                reject |= np.abs(ratio - factor[:, None]) > self.config.rejectRatio

            armContinuumList.append(armContinuum)
            skyContinuumList.append(skyContinuum)
            data.append(np.ma.masked_where(reject, flux))
            model.append(np.ma.masked_where(reject, skyFlux))
            variance.append(pfsArm.variance)

        def doFit(data: list[np.ndarray], model: list[np.ndarray], variance: list[np.ndarray]) -> np.ndarray:
            """Fit model to data

            The model is very simple: a single scale factor for each spectrum.

            Parameters
            ----------
            data : `list` of `numpy.ndarray`, shape ``(numSpectra, numSamples)``
                Data to fit.
            model : `list` of `numpy.ndarray`, shape ``(numSpectra, numSamples)``
                Model to fit (evaluated at the same points as ``data``).
            variance : `list` of `numpy.ndarray`
                Variance of the data.

            Returns
            -------
            factor : `numpy.ndarray`, shape ``(numSpectra,)``
                Factor by which to multiply the model to best fit the data.
            """
            for dd, mm, vv in zip(data, model, variance):
                assert dd.shape == mm.shape and dd.shape == vv.shape
            modelDotModel = reduce(
                np.ma.add, (np.ma.sum(mm**2/vv, axis=1) for mm, vv in zip(model, variance))
            )
            dataDotModel = reduce(
                np.ma.add, (np.ma.sum(dd*mm/vv, axis=1) for dd, mm, vv in zip(data, model, variance))
            )
            return np.array(dataDotModel/modelDotModel)

        for iteration in range(self.config.iterations):
            factor = doFit(data, model, variance)
            self.log.debug(
                "Iteration %d: factor = %.3f +/- %.3f",
                np.nanmedian(np.array(factor)),
                robustRms(factor[np.isfinite(factor)]),
            )
            for dd, mm, vv, gg in zip(data, model, variance, good):
                residuals = dd - factor[:, None]*mm
                with np.errstate(invalid="ignore", divide="ignore"):
                    rr = np.array(residuals)/np.sqrt(vv)
                self.log.debug("RMS normalized residuals = %.1f", robustRms(rr[gg]))
                reject = np.abs(residuals) > self.config.rejection*np.sqrt(vv)
                dd.mask |= reject
                mm.mask |= reject

        factor = doFit(data, model, variance)
        self.log.info(
            "Sky subtraction scaling factor = %.3f +/- %.3f",
            np.nanmedian(np.array(factor)),
            robustRms(factor[np.isfinite(factor)]),
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = [
                robustRms((np.array(dd - factor[:, None]*mm)/np.sqrt(vv))[gg])
                for dd, mm, vv, gg in zip(data, model, variance, good)
            ]
        self.log.info("RMS normalized residuals = %s", rms)

        return Struct(
            factor=factor,
            armContinuum=armContinuumList,
            skyContinuum=skyContinuumList,
        )
