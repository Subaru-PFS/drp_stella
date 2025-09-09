from typing import Dict, Any
import numpy as np

from lsst.pex.config import Config, Field, ListField, RangeField
from lsst.pipe.base import Task

from pfs.datamodel import PfsConfig
from pfs.drp.stella.datamodel import PfsFiberArraySet
from .focalPlaneFunction import ConstantFocalPlaneFunction, OversampledSpline, BlockedOversampledSpline
from .focalPlaneFunction import FocalPlaneFunction, PolynomialPerFiber, FocalPlanePolynomial, ConstantPerFiber
from .focalPlaneFunction import FiberPolynomials

import lsstDebug

__all__ = ("FitFocalPlaneConfig", "FitFocalPlaneTask",
           "FitOversampledSplineConfig", "FitOversampledSplineTask",
           "FitBlockedOversampledSplineConfig", "FitBlockedOversampledSplineTask",
           "FitPolynomialPerFiberConfig", "FitPolynomialPerFiberTask",
           "FitFocalPlanePolynomialTask", "FitFocalPlanePolynomialTask",
           "FitConstantPerFiberConfig", "FitConstantPerFiberTask",
           "FitFiberPolynomialsConfig", "FitFiberPolynomialsTask",
           )


class FitFocalPlaneConfig(Config):
    """Configuration for FitFocalPlaneTask"""
    mask = ListField(dtype=str, default=["NO_DATA", "SUSPECT", "SAT", "BAD_FLAT", "CR"],
                     doc="Mask flags to ignore in fitting")
    rejIterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejThreshold = Field(dtype=float, default=3.0, doc="Rejection threshold (sigma)")
    sysErr = Field(dtype=float, default=1.0e-4,
                   doc=("Fraction of value to add to variance before fitting. This attempts to offset the "
                        "loss of variance as covariance when we resample, the result of which is "
                        "underestimated errors and excess rejection."))

    def getFitParameters(self) -> Dict[str, Any]:
        """Return fit parameters for ``FocalPlaneFunction.fit``

        Returns
        -------
        kwargs : `dict`
            Fit parameters.
        """
        result = self.toDict()
        for name in ("mask", "rejIterations", "rejThreshold", "sysErr"):
            del result[name]
        return result


class FitFocalPlaneTask(Task):
    """Fit a spectral function over the focal plane

    This implementation is a placeholder, as no attention is paid to the
    position of the fibers on the focal plane.
    """
    ConfigClass = FitFocalPlaneConfig
    _DefaultName = "fitFocalPlane"
    Function = ConstantFocalPlaneFunction  # function used

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)
        self._DefaultName = "fitFocalPlane"

    def run(self, spectra: PfsFiberArraySet, pfsConfig: PfsConfig, **kwargs):
        """Fit a vector function as a function of wavelength over the focal plane

        Note that this requires that all the input vectors have the same
        wavelength array.

        Parameters
        ----------
        spectra : `PfsFiberArraySet`
            Spectra to fit. This should contain only the fibers to be fit.
        pfsConfig : `PfsConfig`
            Top-end configuration. This should contain only the fibers to be
            fit.
        **kwargs
            Fitting parameters, overriding any provided in the configuration.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to the data.
        """
        fiberId = spectra.fiberId
        wavelength = spectra.wavelength
        with np.errstate(invalid="ignore", divide="ignore"):
            values = spectra.flux / spectra.norm
            variances = spectra.variance / spectra.norm**2
        mask = (spectra.mask & spectra.flags.get(*self.config.mask)) != 0
        positions = pfsConfig.pfiCenter
        return self.fitArrays(fiberId, wavelength, values, mask, variances, positions, **kwargs)

    def fitArrays(
        self,
        fiberId: np.ndarray,
        wavelength: np.ndarray,
        values: np.ndarray,
        mask: np.ndarray,
        variance: np.ndarray,
        positions: np.ndarray,
        **kwargs,
    ) -> "FocalPlaneFunction":
        """Fit a vector function as a function of wavelength over the focal plane

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber IDs for each fiber.
        wavelength : `numpy.ndarray` of `float`, shape ``(N, M)``
            Common wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Measured values for each wavelength.
        mask : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean flag indicating whether a point should be masked.
        variance : `numpy.ndarray` of `float`, shape ``(N, M)`
            Variance values for each wavelength.
        positions : `numpy.ndarray` of `float`, shape ``(N, 2)`
            Focal plane positions (x, y) for each fiber.
        **kwargs
            Fitting parameters, overriding any provided in the configuration.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to the data.
        """
        fitParams = self.config.getFitParameters()
        fitParams.update(kwargs)

        badFiber = np.all(mask, axis=1)
        if np.all(badFiber):
            raise RuntimeError("No good fibers")
        if np.any(badFiber):
            self.log.warn("Ignoring fibers with no good data: %s", fiberId[badFiber])
            goodFiber = ~badFiber
            fiberId = fiberId[goodFiber]
            wavelength = wavelength[goodFiber]
            values = values[goodFiber]
            variance = variance[goodFiber]
            mask = mask[goodFiber]
            positions = positions[goodFiber]

        with np.errstate(invalid="ignore"):
            variance = variance + self.config.sysErr * np.abs(values)

        # Robust fitting with rejection
        numSamples, length = values.shape
        rejected = np.zeros((numSamples, length), dtype=bool)
        for ii in range(self.config.rejIterations):
            func = self.Function.fitArrays(
                fiberId, wavelength, values, mask | rejected, variance, positions, robust=True, **fitParams
            )
            funcEval = func.evaluate(wavelength, fiberId, positions)
            with np.errstate(invalid="ignore", divide="ignore"):
                resid = (values - funcEval.values)/np.sqrt(variance + funcEval.variances)
                newRejected = ~rejected & ~mask & ~funcEval.masks & (np.abs(resid) > self.config.rejThreshold)
            good = ~(rejected | newRejected | mask | funcEval.masks)
            numGood = good.sum()
            chi2 = np.sum(resid[good]**2)
            self.log.debug("Fit focal plane function iteration %d: "
                           "chi^2=%f length=%d/%d numSamples=%d numGood=%d numBad=%d numRejected=%d",
                           ii, chi2, (~np.logical_and.reduce(funcEval.masks, axis=0)).sum(), length,
                           numSamples, numGood, (mask | funcEval.masks).sum(), rejected.sum())
            if numGood == 0:
                raise RuntimeError("No good points")
            if self.debugInfo.plot:
                self.plot(wavelength, values, mask | rejected, variance, funcEval, f"Iteration {ii}",
                          newRejected)
            if not np.any(newRejected):
                break
            rejected |= newRejected

        # A final fit with robust=False
        func = self.Function.fitArrays(
            fiberId, wavelength, values, mask | rejected, variance, positions, robust=False, **fitParams
        )
        funcEval = func.evaluate(wavelength, fiberId, positions)
        with np.errstate(invalid="ignore", divide="ignore"):
            resid = (values - funcEval.values)/np.sqrt(variance + funcEval.variances)

        good = ~(rejected | mask | funcEval.masks)
        chi2 = np.sum(resid[good]**2)
        numGood = good.sum()
        self.log.info("Fit focal plane function: "
                      "chi^2=%f length=%d/%d numSamples=%d numGood=%d numBad=%d numRejected=%d",
                      chi2, (~np.logical_and.reduce(funcEval.masks, axis=0)).sum(), length,
                      numSamples, numGood, (mask | funcEval.masks).sum(), rejected.sum())
        if numGood == 0:
            raise RuntimeError("No good points")

        if self.debugInfo.plot:
            self.plot(wavelength, values, mask | rejected, variance, funcEval, "Final")

        return func

    def runMultiple(
        self, spectraList: list[PfsFiberArraySet], pfsConfigList: list[PfsConfig], **kwargs
    ) -> "FocalPlaneFunction":
        """Fit a vector function as a function of wavelength over the focal plane
        to multiple sets of spectra

        Parameters
        ----------
        spectraList : list of `PfsFiberArraySet`
            Spectra to fit. This should contain only the fibers to be fit.
        pfsConfigList : list of `PfsConfig`
            Top-end configuration. This should contain only the fibers to be
            fit.
        **kwargs
            Fitting parameters, overriding any provided in the configuration.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to the data.
        """
        fiberIdList = [ss.fiberId for ss in spectraList]
        wavelengthList = [ss.wavelength for ss in spectraList]
        with np.errstate(invalid="ignore", divide="ignore"):
            valuesList = [ss.flux/ss.norm for ss in spectraList]
            variancesList = [ss.variance/ss.norm**2 for ss in spectraList]
        masksList = [ss.mask & ss.flags.get(*self.config.mask) for ss in spectraList]
        positionsList = [pfsConfig.pfiCenter for pfsConfig in pfsConfigList]
        return self.fitMultipleArrays(
            fiberIdList, wavelengthList, valuesList, masksList, variancesList, positionsList, **kwargs
        )

    def fitMultipleArrays(
        self,
        fiberIdList: list[np.ndarray],
        wavelengthList: list[np.ndarray],
        valuesList: list[np.ndarray],
        masksList: list[np.ndarray],
        variancesList: list[np.ndarray],
        positionsList: list[np.ndarray],
        **kwargs,
    ) -> "FocalPlaneFunction":
        """Fit a vector function as a function of wavelength over the focal plane

        Parameters
        ----------
        fiberIdList : list of `numpy.ndarray` of `int`, shape ``(N_i,)``
            Fiber IDs for each fiber.
        wavelengthList : list of `numpy.ndarray` of `float`, shape ``(N_i, M)``
            Common wavelength array.
        valuesList : list of `numpy.ndarray` of `float`, shape ``(N_i, M)``
            Measured values for each wavelength.
        masksList : list of `numpy.ndarray` of `bool`, shape ``(N_i, M)``
            Boolean flag indicating whether a point should be masked.
        variancesList : list of `numpy.ndarray` of `float`, shape ``(N_i, M)``
            Variance values for each wavelength.
        positionsList : list of `numpy.ndarray` of `float`, shape ``(N_i, 2)``
            Focal plane positions (x, y) for each fiber.
        **kwargs
            Fitting parameters, overriding any provided in the configuration.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to the data.
        """
        fitParams = self.config.getFitParameters()
        fitParams.update(kwargs)

        numList = set([len(lst) for lst in (fiberIdList, wavelengthList, valuesList,
                                            masksList, variancesList, positionsList)])
        if len(numList) != 1:
            raise RuntimeError("Input length mismatch")
        num = numList.pop()
        if num == 0:
            raise RuntimeError("No inputs")

        length = valuesList[0].shape[1]
        for ii in range(num):
            lengthList = set([wavelengthList[ii].shape[1], valuesList[ii].shape[1],
                              variancesList[ii].shape[1], masksList[ii].shape[1]])
            if len(lengthList) != 1:
                raise RuntimeError(f"Array {ii} length mismatch")
        numSamples = sum(values.shape[0] for values in valuesList)

        with np.errstate(invalid="ignore"):
            variancesList = [
                var + self.config.sysErr*np.abs(val) for val, var in zip(valuesList, variancesList)
            ]

        # Robust fitting with rejection
        rejected = [np.zeros_like(val, dtype=bool) for val in valuesList]
        for ii in range(self.config.rejIterations):
            func = self.Function.fitMultipleArrays(
                fiberIdList, wavelengthList, valuesList,
                [mm | rej for mm, rej in zip(masksList, rejected)],
                variancesList, positionsList, robust=True, **fitParams
            )
            funcEval = [
                func.evaluate(wl, ff, pos) for wl, ff, pos in zip(wavelengthList, fiberIdList, positionsList)
            ]
            with np.errstate(invalid="ignore", divide="ignore"):
                resid = [
                    (val - fe.values)/np.sqrt(var + fe.variances)
                    for val, var, fe in zip(valuesList, variancesList, funcEval)
                ]
                newRejected = [
                    ~rej & ~mm & ~fe.masks & (np.abs(rs) > self.config.rejThreshold)
                    for rej, mm, fe, rs in zip(rejected, masksList, funcEval, resid)
                ]
            good = [
                ~(rej | newRej | mm | fe.masks)
                for rej, newRej, mm, fe in zip(rejected, newRejected, masksList, funcEval)
            ]
            numGood = sum(gg.sum() for gg in good)
            chi2 = sum(np.sum(res[gg]**2) for res, gg in zip(resid, good))
            self.log.debug(
                ("Fit focal plane function iteration %d: "
                 "chi^2=%f length=%d/%d numSamples=%d numGood=%d numBad=%d numRejected=%d"),
                ii,
                chi2,
                (~np.logical_and.reduce([fe.masks for fe in funcEval], axis=0)).sum(),
                length,
                numSamples,
                numGood,
                sum(((mm | fe.masks).sum() for mm, fe in zip(masksList, funcEval))),
                sum(rej.sum() for rej in rejected),
            )
            if numGood == 0:
                raise RuntimeError("No good points")
            if not np.any([np.any(nr) for nr in newRejected]):
                break
            for rej, newRej in zip(rejected, newRejected):
                rej |= newRej

        # A final fit with robust=False
        func = self.Function.fitMultipleArrays(
            fiberIdList, wavelengthList, valuesList,
            [mm | rej for mm, rej in zip(masksList, rejected)],
            variancesList, positionsList, robust=False, **fitParams
        )
        funcEval = [
            func.evaluate(wl, ff, pos) for wl, ff, pos in zip(wavelengthList, fiberIdList, positionsList)
        ]
        with np.errstate(invalid="ignore", divide="ignore"):
            resid = [
                (val - fe.values)/np.sqrt(var + fe.variances)
                for val, var, fe in zip(valuesList, variancesList, funcEval)
            ]
            newRejected = [
                ~rej & ~mm & ~fe.masks & (np.abs(rs) > self.config.rejThreshold)
                for rej, mm, fe, rs in zip(rejected, masksList, funcEval, resid)
            ]
        good = [
            ~(rej | newRej | mm | fe.masks)
            for rej, newRej, mm, fe in zip(rejected, newRejected, masksList, funcEval)
        ]
        numGood = sum(gg.sum() for gg in good)
        chi2 = sum(np.sum(res[gg]**2) for res, gg in zip(resid, good))
        self.log.info(
            ("Fit focal plane function: "
             "chi^2=%f length=%d/%d numSamples=%d numGood=%d numBad=%d numRejected=%d"),
            chi2,
            (~np.logical_and.reduce([fe.masks for fe in funcEval], axis=0)).sum(),
            length,
            numSamples,
            numGood,
            sum(((mm | fe.masks).sum() for mm, fe in zip(masksList, funcEval))),
            sum(rej.sum() for rej in rejected),
        )
        if numGood == 0:
            raise RuntimeError("No good points")
        return func

    def plot(self, wavelength, values, masks, variances, funcEval, title, rejected=None):
        """Plot the input and fit values

        Parameters
        ----------
        wavelength : `numpy.ndarray` of `float`, shape ``(N, M)``
            Common wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Measured values for each sample+wavelength.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean flag indicating whether a point should be masked.
        variances : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Variance values for each sample+wavelength.
        funcEval : `Struct`
            Fit function evaluation, with at least ``values`` and ``masks``
            members.
        title : `str`
            Title to use for the plot.
        rejected : `numpy.ndarray` of `bool`, shape ``(N, M)``; optional
            Boolean flag indicating whether a point has been rejected.
        """
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=(3, 1)))
        num = len(values)
        for ii in range(num):
            resid = (values[ii] - funcEval.values[ii])/np.sqrt(variances[ii] + funcEval.variances[ii])
            ax1.plot(wavelength[ii], values[ii], "k-")
            ax2.plot(wavelength[ii], resid, "k-")
            mm = masks[ii]
            if np.any(mm):
                ax1.plot(wavelength[ii][mm], values[ii][mm], "kx")
                ax2.plot(wavelength[ii][mm], resid[mm], "kx")
            if rejected is not None:
                rej = rejected[ii]
                if np.any(rej):
                    ax1.plot(wavelength[ii][rej], values[ii][rej], "rx")
                    ax2.plot(wavelength[ii][rej], resid[rej], "rx")
            ax1.plot(wavelength[ii], funcEval.values[ii], "b-")
            if np.any(funcEval.masks[ii]):
                ax1.plot(wavelength[ii][funcEval.masks[ii]], funcEval.values[ii][funcEval.masks[ii]], "bx")

        ax2.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Value")
        ax2.set_ylabel("Residual")
        fig.suptitle(title)
        plt.show()


class FitOversampledSplineConfig(FitFocalPlaneConfig):
    """Configuration for fitting an `OversampledSpline`"""
    splineOrder = RangeField(dtype=int, default=3, min=1, doc="Spline order")
    oversample = RangeField(dtype=float, default=1.25, min=1.0, doc="Oversampling factor")
    defaultValue = Field(dtype=float, default=0.0, doc="Default value for out-of-range data")


class FitOversampledSplineTask(FitFocalPlaneTask):
    """Fit an `OversampledSpline`

    The `OversampledSpline` isn't a function of position on the focal plane.
    It's fine for representing spectra that do not vary over the focal plane.
    """
    ConfigClass = FitOversampledSplineConfig
    Function = OversampledSpline


class FitBlockedOversampledSplineConfig(FitOversampledSplineConfig):
    """Configuration for fitting a `BlockedOversampledSpline`"""
    blockSize = Field(dtype=int, default=20,
                      doc="Block size. Must be large enough to always get enough source fibers, "
                          "but small enough to follow changes")


class FitBlockedOversampledSplineTask(FitFocalPlaneTask):
    """Fit a `BlockedOversampledSpline`

    The `BlockedOversampledSpline` deals with variation over the focal plane by
    grouping fibers into discrete blocks.
    """
    ConfigClass = FitBlockedOversampledSplineConfig
    Function = BlockedOversampledSpline


class FitPolynomialPerFiberConfig(FitFocalPlaneConfig):
    """Configuration for fitting a `PolynomialPerFiber`

    The ``PolynomialPerFiber.fit`` method also needs ``minWavelength`` and
    ``maxWavelength`` input parameters, but those can be determined from the
    data.
    """
    order = Field(dtype=int, doc="Polynomial order")


class FitPolynomialPerFiberTask(FitFocalPlaneTask):
    """Fit a `PolynomialPerFiber`

    We fit a polynomial as a function of wavelength for each fiber individually.

    The ``PolynomialPerFiber.fit`` method also needs ``minWavelength`` and
    ``maxWavelength`` input parameters, but those can be determined from the
    data. This makes `FitPolynomialPerFiberTask`'s ``run`` method a little
    different from other flavors of `FitFocalPlaneTask`.
    """
    ConfigClass = FitPolynomialPerFiberConfig
    Function = PolynomialPerFiber


class FitFocalPlanePolynomialTask(FitFocalPlaneConfig):
    """Configuration for fitting a `FocalPlanPolynomial`"""
    order = Field(dtype=int, default=2, doc="Polynomial order")
    halfWidth = Field(dtype=float, default=250.0, doc="Half-width of the focal plane (mm)")


class FitFocalPlanePolynomialTask(FitFocalPlaneTask):
    """Fit a `FocalPlanPolynomial`
    The `FocalPlanPolynomial` is a polynomial function of position on the
    focal plane. It is not a function of wavelength.
    """
    ConfigClass = FitFocalPlanePolynomialTask
    Function = FocalPlanePolynomial


class FitConstantPerFiberConfig(FitFocalPlaneConfig):
    """Configuration for fitting a `ConstantPerFiber`"""
    pass


class FitConstantPerFiberTask(FitFocalPlaneTask):
    """Fit a `ConstantPerFiber`

    The `ConstantPerFiber` is a constant value for each fiber. It is not a
    function of wavelength or position.
    """
    ConfigClass = FitConstantPerFiberConfig
    Function = ConstantPerFiber


class FitFiberPolynomialsConfig(FitFocalPlaneConfig):
    """Configuration for fitting a `FiberPolynomials`

    The ``FiberPolynomials.fit`` method also needs ``minWavelength`` and
    ``maxWavelength`` input parameters, but those can be determined from the
    data.
    """
    order = Field(dtype=int, doc="Polynomial order")
    radius = Field(dtype=float, default=4.5, doc="Radius of the fiber patrol region (mm)")


class FitFiberPolynomialsTask(FitFocalPlaneTask):
    """Fit a `FiberPolynomials`

    We fit a polynomial as a function of wavelength for each fiber individually.
    """
    ConfigClass = FitFiberPolynomialsConfig
    Function = FiberPolynomials
