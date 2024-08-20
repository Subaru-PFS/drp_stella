from typing import Dict, Any
import numpy as np

from lsst.pex.config import Config, Field, ListField, RangeField
from lsst.pipe.base import Task

from pfs.datamodel import PfsConfig
from pfs.drp.stella.datamodel import PfsFiberArraySet
from .focalPlaneFunction import ConstantFocalPlaneFunction, OversampledSpline, BlockedOversampledSpline
from .focalPlaneFunction import PolynomialPerFiber

import lsstDebug

__all__ = ("FitFocalPlaneConfig", "FitFocalPlaneTask",
           "FitOversampledSplineConfig", "FitOversampledSplineTask",
           "FitBlockedOversampledSplineConfig", "FitBlockedOversampledSplineTask",
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
        numSamples = len(spectra)
        length = spectra.length
        fitParams = self.config.getFitParameters()
        fitParams.update(kwargs)

        wavelength = spectra.wavelength
        with np.errstate(invalid="ignore", divide="ignore"):
            values = spectra.flux/spectra.norm
            variance = (spectra.variance + self.config.sysErr*np.abs(spectra.flux))/spectra.norm**2
        mask = (spectra.mask & spectra.flags.get(*self.config.mask)) != 0

        # Robust fitting with rejection
        rejected = np.zeros((numSamples, length), dtype=bool)
        for ii in range(self.config.rejIterations):
            func = self.Function.fit(spectra, pfsConfig, self.config.mask, rejected=rejected, robust=True,
                                     **fitParams)
            funcEval = func(spectra.wavelength, pfsConfig)
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
        func = self.Function.fit(spectra, pfsConfig, self.config.mask, rejected=rejected, robust=False,
                                 **fitParams)
        funcEval = func(spectra.wavelength, pfsConfig)
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
    oversample = RangeField(dtype=float, default=1.1, min=1.0, doc="Oversampling factor")
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
    blockSize = Field(dtype=int, default=5,
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
