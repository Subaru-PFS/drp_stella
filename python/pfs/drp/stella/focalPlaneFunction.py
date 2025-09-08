from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import numpy as np
from pfs.datamodel import (
    PfsBlockedOversampledSpline,
    PfsConfig,
    PfsConstantFocalPlaneFunction,
    PfsFocalPlaneFunction,
    PfsOversampledSpline,
    PfsPolynomialPerFiber,
    PfsFluxCalib,
    PfsFocalPlanePolynomial,
    PfsConstantPerFiber,
    PfsFiberPolynomials,
)
from pfs.datamodel.utils import subclasses
from pfs.drp.stella.datamodel import PfsFiberArraySet
from pfs.drp.stella.interpolate import interpolateFlux, interpolateMask, interpolateVariance
from pfs.utils.fiberids import FiberIds
from scipy.interpolate import BSpline, InterpolatedUnivariateSpline, LSQUnivariateSpline, interp1d
from scipy.stats import binned_statistic

from .math import NormalizedPolynomial1D, NormalizedPolynomial2D, NormalizedPolynomialND
from .math import calculateMedian, solveLeastSquaresDesign
from .struct import Struct
from .utils.math import robustRms

from typing import Callable

if TYPE_CHECKING:
    import matplotlib


__all__ = (
    "FocalPlaneFunction",
    "ConstantFocalPlaneFunction",
    "OversampledSpline",
    "BlockedOversampledSpline",
    "PolynomialPerFiber",
    "FluxCalib",
    "FocalPlanePolynomial",
    "ConstantPerFiber",
    "FiberPolynomials",
)


class FocalPlaneFunction(ABC):
    """Spectral function on the focal plane

    A model of some spectrum as a function of position on the focal plane.

    This is an abstract base class. Subclasses need to override:
    * ``DamdClass`` class variable: subclass of
      `pfs.datamodel.PfsFocalPlaneFunction` that performs the I/O.
    * ``fitArrays`` class method: fit the function
    * ``evaluate`` method: evaluate the spectrum at a position.
    """

    DamdClass: Type[PfsFocalPlaneFunction]

    def __init__(self, *args, datamodel: Optional[PfsFocalPlaneFunction] = None, **kwargs):
        if datamodel is None:
            datamodel = self.DamdClass(*args, **kwargs)
        else:
            assert not args and not kwargs, "Provide only args,kwargs or datamodel; not both"
        super().__setattr__("_damdObj", datamodel)

    def __getattr__(self, name: str) -> Any:
        """Get attribute

        This forwards attribute lookups to the datamodel representation. The
        datamodel representation is holding the data that we care about, so this
        allows us to access it.
        """
        damdObj = self._damdObj
        if hasattr(damdObj, name):
            return getattr(damdObj, name)
        return super().__getattribute__(name)

    def asDatamodel(self) -> PfsFocalPlaneFunction:
        """Return the datamodel representation

        Returns
        -------
        datamodel : `pfs.datamodel.PfsFocalPlaneFunction`
            Datamodel representation of the function.
        """
        return self._damdObj

    @classmethod
    def fromDatamodel(cls, datamodel: PfsFocalPlaneFunction) -> "FocalPlaneFunction":
        """Construct from datamodel

        Parameters
        ----------
        datamodel : `pfs.datamodel.PfsFocalPlaneFunction`
            Datamodel representation of the function.

        Returns
        -------
        self : subclass of `FocalPlaneFunction`
            Constructed function.
        """
        subs = {ss.DamdClass: ss for ss in subclasses(cls)}
        return subs[type(datamodel)](datamodel=datamodel)

    @classmethod
    def fit(
        cls,
        spectra: PfsFiberArraySet,
        pfsConfig: PfsConfig,
        maskFlags: List[str],
        *,
        rejected: np.ndarray | None = None,
        robust: bool = False,
        **kwargs,
    ) -> "FocalPlaneFunction":
        """Fit a spectral function on the focal plane to spectra

        This method is intended as a convenience for users.

        Parameters
        ----------
        spectra : `PfsFiberArraySet`
            Spectra to fit. This should contain only the fibers to be fit.
        pfsConfig : `PfsConfig`
            Top-end configuration. This should contain only the fibers to be
            fit.
        maskFlags : iterable of `str`
            Mask flags to exclude from fit.
        rejected : `numpy.ndarray` of `bool`, shape ``(Nspectra, length)``
            Which pixels should be rejected?
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        **kwargs : `dict`
            Fitting parameters.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Fit function.
        """
        if len(spectra) != len(pfsConfig):
            raise RuntimeError(
                f"Length mismatch between spectra ({len(spectra)}) and pfsConfig ({len(pfsConfig)})"
            )
        if len(spectra) == 0:
            raise RuntimeError("No input spectra provided")
        if np.all(spectra.fiberId != pfsConfig.fiberId):
            raise RuntimeError("fiberId mismatch between spectra and pfsConfig")

        with np.errstate(invalid="ignore", divide="ignore"):
            values = spectra.flux / spectra.norm
            variance = spectra.variance / spectra.norm**2
        masks = spectra.mask & spectra.flags.get(*maskFlags) != 0
        masks |= ~np.isfinite(values) | ~np.isfinite(variance)
        if rejected is not None:
            masks |= rejected
        positions = pfsConfig.extractCenters(pfsConfig.fiberId)
        return cls.fitArrays(
            spectra.fiberId, spectra.wavelength, values, masks, variance, positions, robust=robust, **kwargs
        )

    @classmethod
    def fitMultiple(
        cls,
        spectraList: list[PfsFiberArraySet],
        pfsConfigList: list[PfsConfig],
        maskFlags: List[str],
        *,
        rejected: list[np.ndarray] | None = None,
        robust: bool = False,
        **kwargs,
    ) -> "FocalPlaneFunction":
        """Fit a spectral function on the focal plane to multiple spectra

        Parameters
        ----------
        spectraList : list of `PfsFiberArraySet`
            Spectra to fit. This should contain only the fibers to be fit.
        pfsConfigList : list of `PfsConfig`
            Top-end configurations. This should contain only the fibers to be
            fit.
        maskFlags : iterable of `str`
            Mask flags to exclude from fit.
        rejected : list of `np.ndarray` of `bool`, shape ``(Nspectra, length)``
            Pixels to reject from the fit.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        **kwargs : `dict`
            Fitting parameters.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Fit function.
        """
        fiberIdList = [spectra.fiberId for spectra in spectraList]
        wavelengthList = [spectra.wavelength for spectra in spectraList]
        with np.errstate(invalid="ignore", divide="ignore"):
            valuesList = [spectra.flux / spectra.norm for spectra in spectraList]
            varianceList = [spectra.variance / spectra.norm**2 for spectra in spectraList]
        maskList = [spectra.mask & spectra.flags.get(*maskFlags) != 0 for spectra in spectraList]
        if rejected is not None:
            maskList = [mm | rej for mm, rej in zip(maskList, rejected)]
        positionsList = [pfsConfig.extractCenters(pfsConfig.fiberId) for pfsConfig in pfsConfigList]
        return cls.fitMultipleArrays(
            fiberIdList,
            wavelengthList,
            valuesList,
            maskList,
            varianceList,
            positionsList,
            robust=robust,
            **kwargs,
        )

    @classmethod
    def fitMultipleArrays(
        cls,
        fiberIdList: List[np.ndarray],
        wavelengthList: List[np.ndarray],
        valuesList: List[np.ndarray],
        maskList: List[np.ndarray],
        varianceList: List[np.ndarray],
        positionsList: List[np.ndarray],
        *,
        robust: bool = False,
        **kwargs,
    ) -> "FocalPlaneFunction":
        """Fit a spectral function on the focal plane to multiple arrays

        This method is intended as a convenience for developers.

        Parameters
        ----------
        fiberIdList : list of `numpy.ndarray` of `float`, shape ``(N_i,)``
            Fiber identifier arrays.
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
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        **kwargs : `dict`
            Fitting parameters.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to input arrays.
        """
        numList = set([
            len(fiberIdList), len(wavelengthList), len(valuesList),
            len(maskList), len(varianceList), len(positionsList)
        ])
        if len(numList) != 1:
            raise RuntimeError("Length mismatch")
        num = numList.pop()
        if num == 0:
            raise RuntimeError("No input arrays provided")
        length = valuesList[0].shape[1]
        for ii in range(num):
            lengthList = set([
                fiberIdList[ii].shape[0],
                wavelengthList[ii].shape[0],
                valuesList[ii].shape[0],
                maskList[ii].shape[0],
                varianceList[ii].shape[0],
                positionsList[ii].shape[0],
            ])
            if len(lengthList) != 1:
                raise RuntimeError(f"Array {ii} length mismatch")

        numSamples = sum(len(values) for values in valuesList)

        fiberId = np.zeros(numSamples, dtype=int)
        wavelength = np.zeros((numSamples, length), dtype=float)
        values = np.zeros((numSamples, length), dtype=float)
        variance = np.zeros((numSamples, length), dtype=float)
        masks = np.zeros((numSamples, length), dtype=bool)
        positions = np.zeros((numSamples, 2), dtype=float)

        start = 0
        for ii in range(num):
            stop = start + len(valuesList[ii])
            select = slice(start, stop)
            fiberId[select] = fiberIdList[ii]
            wavelength[select, :] = wavelengthList[ii]
            values[select, :] = valuesList[ii]
            variance[select, :] = varianceList[ii]
            masks[select, :] = maskList[ii]
            positions[select, :] = positionsList[ii]
            start = stop

        return cls.fitArrays(fiberId, wavelength, values, masks, variance, positions, robust=robust, **kwargs)

    @classmethod
    @abstractmethod
    def fitArrays(
        cls,
        fiberId: np.ndarray,
        wavelengths: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        variances: np.ndarray,
        positions: np.ndarray,
        robust: bool = False,
        **kwargs,
    ) -> "FocalPlaneFunction":
        """Fit a spectral function on the focal plane to arrays

        This method is intended as a convenience for developers.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        **kwargs : `dict`
            Fitting parameters.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to input arrays.
        """
        raise NotImplementedError("Subclasses must override")

    def __call__(self, wavelengths: np.ndarray, pfsConfig: PfsConfig) -> Struct:
        """Evaluate the function for the provided fiberIds

        This method is intended as a convenience for users.

        For convenience, if the ``pfsConfig`` contains only a single fiber,
        we'll return a single array for each output.

        Parameters
        ----------
        wavelengths : `numpy.ndarray` of shape ``(M,)`` or ``(N, M)``
            Wavelength array.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration. This should contain only the fibers of
            interest.

        Returns
        -------
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Vector function evaluated at each position.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Indicates whether the value at each position is valid.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variances for each position.
        """
        isSingle = False  # Only a single fiber?
        if len(np.array(wavelengths).shape) == 1:
            isSingle = True
            wavelengths = np.array([wavelengths] * len(pfsConfig))
        positions = pfsConfig.extractCenters(pfsConfig.fiberId)
        result = self.evaluate(wavelengths, pfsConfig.fiberId, positions)

        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(result.values) | ~np.isfinite(result.variances) | (result.variances < 0)
        result.masks[bad] = True

        if isSingle:
            result.values = result.values[0]
            result.masks = result.masks[0]
            result.variances = result.variances[0]

        return result

    @abstractmethod
    def evaluate(self, wavelengths: np.ndarray, fiberIds: np.ndarray, positions: np.ndarray) -> Struct:
        """Evaluate the function at the provided positions

        This method is intended as a convenience for developers.

        This abstract method must be overridden by subclasses.

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
        raise NotImplementedError("Subclasses must override")

    @classmethod
    def readFits(cls, filename: str) -> "FocalPlaneFunction":
        """Read from FITS file

        Parameters
        ----------
        filename : `str`
            Filename to read.

        Returns
        -------
        self : subclass of `FocalPlaneFunction`
            Function read from FITS file.
        """
        subs = {ss.DamdClass: ss for ss in subclasses(cls)}
        if hasattr(cls, "DamdClass"):
            subs[cls.DamdClass] = cls
        func = PfsFocalPlaneFunction.readFits(filename)
        return subs[type(func)](datamodel=func)

    def writeFits(self, filename: str):
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        self._damdObj.writeFits(filename)


class ConstantFocalPlaneFunction(FocalPlaneFunction):
    """Constant function over the focal plane

    This implementation is something of a placeholder, as it simply returns a
    constant vector as a function of wavelength. No attention is paid to the
    position of the fibers on the focal plane.

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`, shape ``(N,)``
        Wavelengths for each value.
    value : `numpy.ndarray` of `float`, shape ``(N,)``
        Value at each wavelength.
    mask : `numpy.ndarray` of `bool`, shape ``(N,)``
        Indicate whether values should be ignored.
    variance : `numpy.ndarray` of `float`, shape ``(N,)``
        Variance in value at each wavelength.
    """

    DamdClass = PfsConstantFocalPlaneFunction

    @classmethod
    def fitArrays(
        cls,
        fiberId: np.ndarray,
        wavelengths: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        variances: np.ndarray,
        positions: np.ndarray,
        robust: bool = False,
        **kwargs,
    ) -> "ConstantFocalPlaneFunction":
        """Fit a spectral function on the focal plane to arrays

        This implementation is something of a placeholder, as no attention is
        paid to the position of the fibers on the focal plane. Essentially, this
        is a coadd.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelength : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array. The wavelength array for all the inputs must be
            identical.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        **kwargs : `dict`
            Fitting parameters.

        Returns
        -------
        fit : `FocalPlaneFunction`
            Function fit to input arrays.
        """
        if not np.all([np.array_equal(wavelengths[0], wl) for wl in wavelengths[1:]]):
            raise RuntimeError("Wavelength arrays not identical")

        if robust:
            # This is the equivalent of "np.median(array, axis=0)" with masks applied
            # (which numpy's MaskedArray can't handle).
            medianValues = np.array([calculateMedian(vv, mm) for vv, mm in zip(values.T, masks.T)])
            medianVariances = np.array([calculateMedian(vv, mm) for vv, mm in zip(variances.T, masks.T)])

            return ConstantFocalPlaneFunction(
                wavelengths[0], medianValues, np.logical_and.reduce(masks, axis=0), medianVariances
            )

        with np.errstate(invalid="ignore", divide="ignore"):
            weight = 1.0 / variances
            good = ~masks & (variances > 0)
            weight[~good] = 0.0

            sumWeights = np.sum(weight, axis=0)
            coaddValues = np.sum(np.where(good, values, 0.0) * weight, axis=0) / sumWeights
            coaddVariance = np.where(sumWeights > 0, 1.0 / sumWeights, np.inf)
            coaddMask = sumWeights <= 0
        return ConstantFocalPlaneFunction(wavelengths[0], coaddValues, coaddMask, coaddVariance)

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
        assert len(wavelengths) == len(positions)

        doResample = [
            wl.shape != self.wavelength.shape or not np.all(wl == self.wavelength) for wl in wavelengths
        ]

        values = [
            interpolateFlux(self.wavelength, self.value, wl) if resamp else self.value
            for wl, resamp in zip(wavelengths, doResample)
        ]
        masks = [
            interpolateMask(self.wavelength, self.mask, wl).astype(bool) if resamp else self.mask
            for wl, resamp in zip(wavelengths, doResample)
        ]
        variances = [
            interpolateVariance(self.wavelength, self.variance, wl) if resamp else self.variance
            for wl, resamp in zip(wavelengths, doResample)
        ]
        return Struct(values=np.array(values), masks=np.array(masks), variances=np.array(variances))


class OversampledSpline(FocalPlaneFunction):
    """An oversampled spline in the wavelength dimension, without regard to
    focal plane position

    Parameters
    ----------
    knots : `numpy.ndarray` of `float`
        Spline knots.
    coeffs : `numpy.ndarray` of `float`
        Spline coefficients.
    splineOrder : `int`
        Order of spline.
    wavelength : `numpy.ndarray` of `float`, shape ``(N,)``
        Wavelength array for variance estimation.
    variance : `numpy.ndarray` of `float`, shape ``(N,)``
        Variance array.
    defaultValue : `float`
        Default value.
    """

    DamdClass = PfsOversampledSpline

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._values = BSpline(self.knots, self.coeffs, self.splineOrder, extrapolate=False)
        self._variance = interp1d(
            self.wavelength,
            self.variance,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            copy=True,
            assume_sorted=True,
        )

    @classmethod
    def getKnots(cls, data: np.ndarray, numKnots: int, order: int = 3) -> np.ndarray:
        """Generate a suitable set of interior knots

        Choosing a suitable set of interior knots seems to be something of an
        art. It might be straightforward when we have multiple instances of
        regular sampling, but when rejection is introduced it becomes a "knotty
        problem". This function contains tricks we've found that get us past
        the ``fpchec`` function called by ``LSQUnivariateSpline`` when we
        operate on real data.

        Parameters
        ----------
        data : `numpy.ndarray` of `float`
            Ordered array of data ordinates.
        numKnots : `int`
            Number of initial knots. Knots will be removed as needed.
        order : `int`, optional
            Spline order.

        Returns
        -------
        knots : `numpy.ndarray` of `float`
            Ordered array of interior knots to use for ``LSQUnivariateSpline``.
        """
        first = data[0]
        start = data[1]
        stop = data[-2]
        last = data[-1]
        interior = np.linspace(start, stop, numKnots)

        # Remove knots with no points nearby
        # These can trigger failures of the Schoenberg-Whitney condition:
        # E.g., imagine regularly spaced knots and order=3, and there's data
        # up to knot 10, no data between knot 10 and knot 13, three data points
        # between knot 13 and knot 14, no data between knot 14 and knot 17, and
        # data after knot 17. Then the simple enforcement of Schoenberg-Whitney
        # below doesn't work because there's always 3 data points between knots
        # separated by 4 (e.g., between knot 10 and 14 there are 3 data points),
        # but scipy's ``fpchec`` says this violates Schoenberg-Whitney. So we
        # trim the knots with no empty points nearby, since they aren't really
        # contributing anything, and can only cause trouble.
        bins = np.zeros(interior.size + 1, dtype=float)
        bins[1:-1] = 0.5 * (interior[:-1] + interior[1:])
        bins[0] = first - np.finfo(float).eps
        bins[-1] = last + np.finfo(float).eps
        counts = np.histogram(data, bins)[0]
        select = counts > 0
        interior = interior[select]

        # Force Schoenberg-Whitney
        # This implementation courtesy of Mike Jarvis.
        # It's not an exact encoding of the Schoenberg-Whitney condition as used
        # in ``fpchec``, but when things are well-behaved this seems to satisfy
        # that condition.
        knots = np.concatenate(([first] * (order + 1), interior, [last] * (order + 1)))
        counts = np.histogram(data, knots)[0]
        cumsum = np.cumsum(counts)
        kCount = cumsum[order + 1 :] - cumsum[: -(order + 1)]
        select = kCount >= order
        return knots[order + 1 : -(order + 2)][select[: -(order + 1)]]

    @staticmethod
    def _fpchec(knots: np.ndarray, data: np.ndarray, order: int) -> bool:
        """Check that we can fit a least-squares spline

        This is a python implementation of the ``fpchec`` function from
        ``scipy/scipy/interpolate/fitpack/fpchec.f``. This function is called
        in the process of fitting a ``LSQUnivariateSpline``, and if the
        conditions are not met then fitting fails and the user is left to
        figure out on their own which condition failed. This implementation
        splits the different conditions out, allowing the user to identify
        which specific condition failed.

        This code is included for debugging, and is unused by default.
        Condition 5b (Schoenberg-Whitney) in particular is not coded efficiently
        for python.

        The conditions checked are below. Note that the conditions as enumerated
        here use Fortran indexing! ``t`` are the ``knots``, ``x`` are the
        ``data``, and ``k`` is the ``order``.

          1) k+1 <= n-k-1 <= m
          2) t(1) <= t(2) <= ... <= t(k+1)
             t(n-k) <= t(n-k+1) <= ... <= t(n)
          3) t(k+1) < t(k+2) < ... < t(n-k)
          4) t(k+1) <= x(i) <= t(n-k)
          5) the conditions specified by schoenberg and whitney must hold
             for at least one subset of data points, i.e. there must be a
             subset of data points y(j) such that
                 t(j) < y(j) < t(j+k+1), j=1,2,...,n-k-1

        Parameters
        ----------
        knots : `numpy.ndarray` of `float`
            Ordered array of interior knots.
        data : `numpy.ndarray` of `float`
            Ordered array of data ordinates.
        order : `int`
            Spline order.

        Returns
        -------
        fpchecResult : `bool`
            Result from running scipy's ``fpchec``, translated into a boolean
            indicating whether the inputs pass the gauntlet of conditions.
        """
        knots = np.concatenate(([data[0]] * (order + 1), knots, [data[-1]] * (order + 1)))

        numKnots = knots.size
        numData = data.size

        # Condition 1: there's enough data for the knots and spline order
        nk1 = numKnots - order - 1
        if order + 1 > nk1 or nk1 > numData:
            print("Condition 1 fails: k+1 <= n-k-1 <= m")
        # Condition 2: the exterior knots are increasing or constant
        lowKnots = knots[: order + 1]  # Low exterior knots
        highKnots = knots[-order - 1 :]  # High exterior knots
        if np.any(lowKnots[:-1] > lowKnots[1:]) or np.any(highKnots[:-1] > highKnots[1:]):
            print("Condition 2 fails: t(1) <= t(2) <= ... <= t(k+1); t(n-k) <= t(n-k+1) <= ... <= t(n)")
        # Condition 3: the interior knots are strictly increasing
        interior = knots[order:-order]  # Interior knots
        if np.any(interior[:-1] >= interior[1:]):
            print("Condition 3 fails: t(k+1) < t(k+2) < ... < t(n-k)")
        # Condition 4: all data bounded by interior knots
        if np.any((data < knots[order]) | (data > knots[-order - 1])):
            print("Condition 4 fails: t(k+1) <= x(i) <= t(n-k)")
        # Condition 5: Schoenberg-Whitney
        if data[0] >= knots[order + 1] or data[-1] <= knots[-order - 2]:
            print("Condition 5a fails: x(1) < t(k+2); x(m) > t(n-k-1)")
        if order >= 2:
            # Same implementation as in fpchec.f in scipy, minus the "goto"s.
            ii = 1
            ll = order + 2
            for jj in range(2, nk1):
                tj = knots[jj - 1]
                ll = ll + 1
                tl = knots[ll - 1]
                ii += 1
                while ii <= numData and data[ii - 1] <= tj:
                    ii += 1
                if ii > numData or data[ii - 1] >= tl:
                    print("Condition 5b fails: t(j) < x(j) < t(j+k+1), j=1,2,...,n-k-1")
                    break

        from scipy.interpolate.dfitpack import fpchec

        return fpchec(data, knots, order) == 0

    @classmethod
    def fitArrays(
        cls,
        fiberId: np.ndarray,
        wavelengths: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        variances: np.ndarray,
        positions: np.ndarray,
        robust: bool = False,
        oversample: float = 1.25,
        splineOrder: int = 3,
        defaultValue: float = 0.0,
        **kwargs,
    ) -> "OversampledSpline":
        """Fit a spectral function on the focal plane to arrays

        We currently throw everything into a single block, and don't respect
        the position on the focal plane.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        oversample : `float`
            Oversampling factor for spline.
        splineOrder : `int`
            Order of spline.
        defaultValue : `float`
            Default value for out-of-range data.

        Returns
        -------
        fit : `OversampledSpline`
            Function fit to input arrays.
        """
        if kwargs:
            raise RuntimeError(f"Unrecognised parameters: {kwargs}")
        use = ~masks
        if not np.any(use):
            raise RuntimeError("No unmasked data to fit")
        floatEpsilon = np.finfo(float).eps
        length = wavelengths.shape[1]
        wlUse = wavelengths[use]
        wlMin = wlUse.min() - floatEpsilon
        wlMax = wlUse.max() + floatEpsilon

        indices = np.argsort(wlUse)
        xx = wlUse[indices]
        yy = values[use][indices]
        var = variances[use][indices]

        dx = (wlMax - wlMin) / length / oversample
        knotMin = xx[1]
        knotMax = xx[-2]
        numKnots = int((knotMax - knotMin) / dx)
        knots = cls.getKnots(xx, numKnots, splineOrder)
        bins = np.concatenate(([wlMin], knots, [wlMax]))
        centers = 0.5 * (bins[:-1] + bins[1:])

        if robust:
            yMedian = binned_statistic(xx, yy, statistic="median", bins=bins)[0]
            xMedian = binned_statistic(xx, xx, statistic="median", bins=bins)[0]
            goodBins = np.isfinite(yMedian)
            spline = InterpolatedUnivariateSpline(
                xMedian[goodBins], yMedian[goodBins], k=splineOrder, bbox=(wlMin, wlMax)
            )
            knots = np.concatenate(
                ([wlMin] * (splineOrder - 1), centers[goodBins], [wlMax] * (splineOrder - 1))
            )
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                spline = LSQUnivariateSpline(xx, yy, t=knots, k=splineOrder, bbox=(wlMin, wlMax))
            knots = np.concatenate(([wlMin] * (splineOrder + 1), knots, [wlMax] * (splineOrder + 1)))

        coeffs = spline.get_coeffs()

        # Measure the noise in the fit
        residual = yy - spline(xx)
        variance = binned_statistic(xx, residual, statistic=robustRms if robust else "std", bins=bins)[0] ** 2
        # Remove noise originating from the input data
        variance -= binned_statistic(xx, var, statistic="mean", bins=bins)[0]
        variance = np.clip(variance, 0.0, None)

        return cls(knots, coeffs, splineOrder, centers, variance, defaultValue)

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
        values = self._values(wavelengths)
        variance = self._variance(wavelengths)
        masks = ~(np.isfinite(values) & np.isfinite(variance))
        outOfRange = (wavelengths < self.knots[0]) | (wavelengths > self.knots[-1])
        values[outOfRange] = self.defaultValue
        masks |= outOfRange
        return Struct(values=values, masks=masks, variances=variance)


class BlockedOversampledSpline(FocalPlaneFunction):
    """Oversampled splines defined in blocks of fiberId

    Parameters
    ----------
    splines : `dict` [`float`: `OversampledSpline`]
        Splines for each block index.
    """

    DamdClass = PfsBlockedOversampledSpline

    def __init__(self, *args, datamodel: Optional[PfsFocalPlaneFunction] = None, **kwargs):
        super().__init__(*args, datamodel=datamodel, **kwargs)
        self._splines = {ff: OversampledSpline(datamodel=func) for ff, func in self._damdObj.splines.items()}

    @classmethod
    def fitArrays(
        cls,
        fiberId: np.ndarray,
        wavelengths: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        variances: np.ndarray,
        positions: np.ndarray,
        robust: bool = False,
        blockSize: int = 20,
        **kwargs,
    ) -> FocalPlaneFunction:
        """Fit a spectral function on the focal plane to arrays

        We use fiberId as the sole variable that correlates with changes in the
        spectral function (no attention is paid to the actual position on the
        focal plane).

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        blockSize : `int`, optional
            Size of fiberId blocks.
        **kwargs
            Additional argument for `OversampledSpline`: ``oversample``,
            ``splineOrder`` and ``defaultValue``.

        Returns
        -------
        fit : `BlockedOversampledSpline`
            Function fit to input arrays.
        """
        numFibers = len(fiberId)
        numBlocks = int(np.ceil(numFibers / blockSize))
        blocks = (np.arange(numFibers, dtype=float) * numBlocks / numFibers).astype(int)

        splines: Dict[float, OversampledSpline] = {}
        for bb in range(numBlocks):
            select = blocks == bb
            ff = fiberId[select].mean()
            splines[ff] = OversampledSpline.fitArrays(
                fiberId[select],
                wavelengths[select],
                values[select],
                masks[select],
                variances[select],
                positions[select],
                robust=robust,
                **kwargs,
            )
        return cls(splines)

    def evaluate(self, wavelengths: np.ndarray, fiberIds: np.ndarray, positions: np.ndarray) -> Struct:
        """Evaluate the function at the provided positions

        We interpolate the solution using the two nearest splines.

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
        values = np.empty_like(wavelengths, dtype=float)
        variances = np.empty_like(wavelengths, dtype=float)
        masks = np.empty_like(wavelengths, dtype=bool)
        for ii, ff in enumerate(fiberIds):
            wavelengthArray = np.array([wavelengths[ii]])
            fiberIdArray = np.array([ff])
            positionsArray = np.array([positions[ii]])
            if len(self.fiberId) == 1:
                result = next(iter(self._splines.values())).evaluate(
                    wavelengthArray, fiberIdArray, positionsArray
                )
                thisValue = result.values[0]
                thisMask = result.masks[0]
                thisVariance = result.variances[0]
            else:
                index = np.searchsorted(self.fiberId, ff)
                iAbove = max(1, min(index, len(self.fiberId) - 1))
                iBelow = iAbove - 1
                weightBelow = (self.fiberId[iAbove] - ff) / (self.fiberId[iAbove] - self.fiberId[iBelow])
                weightAbove = 1 - weightBelow
                below = self._splines[self.fiberId[iBelow]].evaluate(
                    wavelengthArray, fiberIdArray, positionsArray
                )
                above = self._splines[self.fiberId[iAbove]].evaluate(
                    wavelengthArray, fiberIdArray, positionsArray
                )
                thisValue = below.values[0] * weightBelow + above.values[0] * weightAbove
                thisMask = below.masks[0] | above.masks[0]
                thisVariance = below.variances[0] * weightBelow**2 + above.variances[0] * weightAbove**2
            values[ii] = thisValue
            masks[ii] = thisMask
            variances[ii] = thisVariance
        bad = np.isnan(variances)
        variances[bad] = 0.0
        masks[bad] = True
        return Struct(values=values, masks=masks, variances=variances)


class PolynomialPerFiber(FocalPlaneFunction):
    """A polynomial in wavelength for each fiber independently.

    Parameters
    ----------
    coeffs : `dict` [`int`: `numpy.ndarray` of `float`]
        Polynomial coefficients, indexed by fiberId.
    rms : `dict` [`int`: `float`]
        RMS of residuals from fit, indexed by fiberId.
    minWavelength : `float`
        Minimum wavelength, for normalising the polynomial inputs.
    maxWavelength : `float`
        Maximum wavelength, for normalising the polynomial inputs.
    """

    DamdClass = PfsPolynomialPerFiber

    def __getitem__(self, fiberId: int) -> NormalizedPolynomial1D:
        """Return the polynomial for a particular fiber"""
        return NormalizedPolynomial1D(self.coeffs[fiberId], self.minWavelength, self.maxWavelength)

    @property
    def fiberId(self) -> np.ndarray:
        """Return the fiberIds"""
        return np.array(sorted(list(self.coeffs.keys())), dtype=int)

    @classmethod
    def fitArrays(
        cls,
        fiberId: np.ndarray,
        wavelengths: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        variances: np.ndarray,
        positions: np.ndarray,
        robust: bool = False,
        order: int = 3,
        minWavelength: float = np.nan,
        maxWavelength: float = np.nan,
        **kwargs,
    ) -> FocalPlaneFunction:
        """Fit a spectral function on the focal plane to arrays

        We fit a polynomial in wavelength to each fiber independently.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        order : `int`
            Order of polynomials.
        minWavelength : `float`
            Minimum wavelength, for normalising the polynomial inputs.
        maxWavelength : `float`
            Maximum wavelength, for normalising the polynomial inputs.

        Returns
        -------
        fit : `PolynomialPerFiber`
            Function fit to input arrays.
        """
        if kwargs:
            raise RuntimeError(f"Unrecognised parameters: {kwargs}")
        if not (np.isfinite(minWavelength) and np.isfinite(maxWavelength)):
            raise RuntimeError("Need to specify minWavelength and maxWavelength")
        errors = np.sqrt(variances)

        poly = NormalizedPolynomial1D(order, minWavelength, maxWavelength)
        numParams = poly.getNParameters()
        select = np.isfinite(values) & np.isfinite(variances) & ~masks
        coeffs = {}
        rms = {}
        for ii, ff in enumerate(fiberId):
            choose = select[ii]
            if choose.sum() < numParams:
                coeffs[ff] = np.full(numParams, np.nan, dtype=float)
                rms[ff] = np.nan
                continue
            design = poly.calculateDesignMatrix(wavelengths[ii][choose])
            coeffs[ff] = solveLeastSquaresDesign(design, values[ii][choose], errors[ii][choose])
            residuals = design @ coeffs[ff] - values[ii][choose]
            if robust:
                rms[ff] = robustRms(residuals)
            else:
                with np.errstate(divide="ignore"):
                    weights = 1.0 / errors[ii][choose] ** 2
                    rms[ff] = np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))

        return cls(coeffs, rms, minWavelength, maxWavelength)

    def evaluate(self, wavelengths: np.ndarray, fiberIds: np.ndarray, positions: np.ndarray) -> Struct:
        """Evaluate the function at the provided positions

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
        values = np.array([self[ff](wavelengths[ii]) for ii, ff in enumerate(fiberIds)])
        rms = np.array([np.full_like(wavelengths[ii], self.rms[ff]) for ii, ff in enumerate(fiberIds)])
        masks = ~np.isfinite(values) | ~np.isfinite(rms)
        return Struct(values=values, variances=rms**2, masks=masks)


class FluxCalib(FocalPlaneFunction):
    r"""Flux calibration vector such that pfsMerged divided by fluxCalib
    will be the calibrated spectra.

    This is the product of a ConstantFocalPlaneFunction ``h(\lambda)``
    multiplied by the exponential of a trivariate polynomial
    ``g(x, y, \lambda)``, where ``(x, y)`` is the fiber position.
    ``h(\lambda)`` represents the average shape of flux calibration vectors
    up to ``exp g(x, y, \lambda)``. ``exp g(x, y, \lambda)``, which is
    expected to be almost independent of ``\lambda``, represents the overall
    height of a flux calibration vector at ``(x, y)``. The height varies from
    fiber to fiber (or, according to ``(x, y)``) because of imperfect fiber
    positioning. ``g(x, y, \lambda)`` indeed depends slightly on ``\lambda``
    because seeing depends on wavelength.

    Parameters
    ----------
    polyParams : `numpy.ndarray` of `float`
        Parameters used by ``NormalizedPolynomialND`` in ``drp_stella``.
        These parameters define ``g(x, y, \lambda)``.
    polyMin : `numpy.ndarray` of `float`, shape ``(3,)``
        Vertex of the rectangular-parallelepipedal domain of the polynomial
        at which ``(x, y, \lambda)`` are minimal.
    polyMax : `numpy.ndarray` of `float`, shape ``(3,)``
        Vertex of the rectangular-parallelepipedal domain of the polynomial
        at which ``(x, y, \lambda)`` are maximal.
    constantFocalPlaneFunction : `PfsConstantFocalPlaneFunction`
        ``h(\lambda)`` as explaned above.
    polyNewNorm : `bool`, optional
        Whether the polynomial uses the new normalization scheme (default:
        ``True``).
    """

    DamdClass = PfsFluxCalib

    def __init__(self, *args, datamodel: Optional[PfsFluxCalib] = None, **kwargs):
        super().__init__(*args, datamodel=datamodel, **kwargs)
        self.constantFocalPlaneFunction = ConstantFocalPlaneFunction(
            datamodel=self._damdObj.constantFocalPlaneFunction,
        )
        self.poly = NormalizedPolynomialND(
            self._damdObj.polyParams,
            self._damdObj.polyMin,
            self._damdObj.polyMax,
            self._damdObj.polyNewNorm
        )

    @classmethod
    def fitArrays(
        cls,
        fiberId: np.ndarray,
        wavelengths: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        variances: np.ndarray,
        positions: np.ndarray,
        *,
        robust: bool,
        fitter: Callable,
        **kwargs,
    ) -> "FluxCalib":
        """Fit a spectral function on the focal plane to arrays

        We leave an actual fitting algorithm to``fitter`` argument
        because the algorithm we currently have requires many things that
        ``focalPlaneFunction.py`` should not know.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array. The wavelength array for all the inputs must be
            identical.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(N, 2)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        fitter : `Callable`
            A function that actually performs fitting.
            This function is called like
            ``fitter(fiberId, wavelengths, ..., positions, robust=robust, **kwargs)``
            and returns an instance of `FluxCalib`.
        **kwargs : `dict`
            Fitting parameters.

        Returns
        -------
        fit : `FluxCalib`
            Function fit to input arrays.
        """
        return fitter(fiberId, wavelengths, values, masks, variances, positions, robust=robust, **kwargs)

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
        assert len(wavelengths) == len(positions)
        nFibers, nSamples = wavelengths.shape

        x = np.empty(shape=(nFibers, nSamples, 3), dtype=float)
        x[:, :, :2] = positions.reshape(nFibers, 1, 2)
        x[:, :, 2] = wavelengths

        scales = np.exp(self.poly(x.reshape(-1, 3))).reshape(nFibers, nSamples)

        retvalue = self.constantFocalPlaneFunction.evaluate(wavelengths, fiberIds, positions)
        retvalue.values *= scales
        retvalue.variances *= np.square(scales)

        return retvalue


class FocalPlanePolynomial(FocalPlaneFunction):
    """A 2D polynomial on the focal plane

    There is no wavelength dependence.

    Parameters
    ----------
    datamodel : `PfsFocalPlanePolynomial`
        Datamodel representation. Either this may be specified, or the other
        parameters must be specified.
    coeffs : `numpy.ndarray` of `float`
        Coefficients of the polynomial.
    halfWidth : `float`
        Half-width of the focal plane, in mm.
    rms : `float`
        RMS of the fit.
    """

    DamdClass = PfsFocalPlanePolynomial

    def __init__(self, *args, datamodel: Optional[PfsFocalPlanePolynomial] = None, **kwargs):
        super().__init__(*args, datamodel=datamodel, **kwargs)
        from lsst.geom import Box2D, Point2D
        halfWidth = self.halfWidth
        box = Box2D(Point2D(-halfWidth, -halfWidth), Point2D(halfWidth, halfWidth))
        self.polynomial = NormalizedPolynomial2D(self.coeffs, box)

    @classmethod
    def fitArrays(
        cls,
        fiberId: np.ndarray,
        wavelengths: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        variances: np.ndarray,
        positions: np.ndarray,
        robust: bool = False,
        order: int = 2,
        halfWidth: float = 250.0,
        **kwargs,
    ) -> FocalPlaneFunction:
        """Fit a polynomial on the focal plane to arrays

        This is wavelength-independent, so if there are multiple wavelengths
        we reduce to a single wavelength by taking the median.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        order : `int`
            Order of polynomial.
        halfWidth : `float`
            Half-width of the focal plane, in mm.

        Returns
        -------
        fit : `FocalPlanePolynomial`
            Function fit to input arrays.
        """
        if kwargs:
            raise RuntimeError(f"Unrecognised parameters: {kwargs}")
        positions = positions.astype(np.float64)  # Because we're passing into pybind

        from lsst.geom import Box2D, Point2D

        box = Box2D(Point2D(-halfWidth, -halfWidth), Point2D(halfWidth, halfWidth))
        poly = NormalizedPolynomial2D(order, box)

        numFibers = wavelengths.shape[0]
        numPixels = wavelengths.shape[1]
        if numPixels > 1:
            bad = masks | ~np.isfinite(values) | ~np.isfinite(variances)
            values = np.ma.median(np.ma.masked_where(bad, values), axis=1).filled(np.nan)
            masks = ~np.isfinite(values)
            if not robust:
                errors = np.ma.median(np.ma.masked_where(bad, np.sqrt(variances)), axis=1).filled(np.nan)
        else:
            values = np.reshape(values, numFibers)
            masks = np.reshape(masks, numFibers)
            if not robust:
                errors = np.reshape(np.sqrt(variances), numFibers)

        if robust:
            errors = np.ones_like(values)

        good = ~masks
        design = poly.calculateDesignMatrix(positions[good, 0], positions[good, 1])
        coeffs = solveLeastSquaresDesign(design, values[good], errors[good])
        residuals = design @ coeffs - values[good]
        if robust:
            rms = robustRms(residuals)
        else:
            with np.errstate(divide="ignore"):
                weights = 1.0 / errors[good] ** 2
                rms = np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))

        return cls(coeffs=coeffs, halfWidth=halfWidth, rms=rms)

    def eval(self, positions: np.ndarray) -> Struct:
        """Evaluate the function at the provided positions

        This provides a single value per position.

        Parameters
        ----------
        positions : `numpy.ndarray` of shape ``(N, 2)``
            Focal-plane positions at which to evaluate.

        Returns
        -------
        values : `numpy.ndarray` of `float`
            Function evaluated at each position.
        masks : `numpy.ndarray` of `bool`
            Indicates whether the value at each position is valid.
        variances : `numpy.ndarray` of `float`
            Variance for each position.
        """
        positions = positions.astype(np.float64)  # Because we're passing into pybind
        values = self.polynomial(positions[:, 0], positions[:, 1])
        masks = np.isnan(values)
        variances = np.full_like(values, self.rms**2)
        return Struct(values=values, masks=masks, variances=variances)

    def evaluate(self, wavelengths: np.ndarray, fiberIds: np.ndarray, positions: np.ndarray) -> Struct:
        """Evaluate the function at the provided positions

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
        result = self.eval(positions)
        numPixels = wavelengths.shape[1]
        return Struct(
            values=np.tile(result.values, (numPixels, 1)).T,
            masks=np.tile(result.masks, (numPixels, 1)).T,
            variances=np.tile(result.variances, (numPixels, 1)).T,
        )

    def plot(
        self,
        wavelength: float,
        pfsConfig: PfsConfig,
        axes: "matplotlib.Axes | None" = None,
        vmin: float = 0.98,
        vmax: float = 1.02,
        cmap: "matplotlib.colors.Colormap | None" = None,
    ) -> "matplotlib.Axes":
        """Plot on the focal plane

        Parameters
        ----------
        wavelength : `float`
            Wavelength at which to plot.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        axes : `matplotlib.Axes`, optional
            Axes object to plot on. If not specified, a new figure is created.
        vmin, vmax : `float`, optional
            Minimum and maximum values for the color scale.
        cmap : `matplotlib.colors.Colormap`, optional
            Colormap to use. If not specified, a default colormap is used.

        Returns
        -------
        axes : `matplotlib.Axes`
            Axes object.
        """
        from matplotlib.colors import Normalize
        from pfs.datamodel import TargetType

        if axes is None:
            import matplotlib.pyplot as plt
            _, axes = plt.subplots()
        if cmap is None:
            import matplotlib.cm
            cmap = matplotlib.cm.coolwarm

        norm = Normalize(vmin=vmin, vmax=vmax)

        values = self.evaluate(wavelength, pfsConfig.fiberId, pfsConfig.pfiCenter).values

        xx = pfsConfig.pfiCenter[:, 0]
        yy = pfsConfig.pfiCenter[:, 1]
        good = np.isfinite(values)

        axes.scatter(xx[good], yy[good], marker="o", c=values[good], cmap=cmap, norm=norm, s=10)

        allRms = robustRms(values[good])
        axes.text(0.05, 0.05, f"RMS = {allRms:.3f}", transform=axes.transAxes, ha="left", va="bottom")

        select = pfsConfig.getSelection(targetType=TargetType.SKY)
        select &= good
        if np.any(select):
            axes.scatter(
                xx[select],
                yy[select],
                marker="o",
                edgecolors="grey",
                s=30,
                facecolors="none",
                ls="-",
                lw=0.5,
                label="Sky",
                alpha=0.4,
            )

            skyRms = robustRms(values[select])
            axes.text(
                0.95, 0.05, f"Sky RMS = {skyRms:.3f}", transform=axes.transAxes, ha="right", va="bottom"
            )

        axes.set_xlabel("X (mm)")
        axes.set_ylabel("Y (mm)")
        axes.legend()

        return axes


class ConstantPerFiber(FocalPlaneFunction):
    """A constant value for each fiber

    Parameters
    ----------
    fiberId : `np.ndarray` of `int`
        Fiber identifiers.
    values : `np.ndarray` of `float`
        Constant value for each fiber.
    rms : `np.ndarray` of `float`
        RMS of the fit for each fiber.
    """

    DamdClass = PfsConstantPerFiber

    def __len__(self) -> int:
        """Number of fibers with values"""
        return len(self.fiberId)

    @classmethod
    def concatenate(cls, *args: "ConstantPerFiber") -> "ConstantPerFiber":
        """Concatenate multiple ConstantPerFiber instances

        Parameters
        ----------
        *args : `ConstantPerFiber`
            Instances to concatenate.

        Returns
        -------
        concatenated : `ConstantPerFiber`
            Concatenated instance.
        """
        fiberId = np.concatenate([a.fiberId for a in args])
        if np.unique(fiberId).size != fiberId.size:
            raise ValueError("Cannot concatenate ConstantPerFiber with overlapping fiberId")
        value = np.concatenate([a.value for a in args])
        rms = np.concatenate([a.rms for a in args])
        indices = np.argsort(fiberId)
        return cls(fiberId=fiberId[indices], value=value[indices], rms=rms[indices])

    def eval(self, fiberIds: np.ndarray) -> Struct:
        """Evaluate the function

        A simpler version of `evaluate` that does not depend on wavelength or
        position.

        Parameters
        ----------
        fiberIds : `numpy.ndarray` of `int` of shape ``(N,)``
            Fiber identifiers.

        Returns
        -------
        values : `numpy.ndarray` of `float`, shape ``(N,)``
            Vector function evaluated at each position.
        masks : `numpy.ndarray` of `bool`, shape ``(N,)``
            Indicates whether the value at each position is valid.
        variances : `numpy.ndarray` of `float`, shape ``(N,)``
            Variances for each position.
        """
        values = np.full_like(fiberIds, np.nan, dtype=float)
        variances = np.full_like(fiberIds, np.nan, dtype=float)

        indices = np.searchsorted(self.fiberId, fiberIds)
        good = (indices >= 0) & (indices < len(self.fiberId)) & (self.fiberId[indices] == fiberIds)
        indices = indices[good]
        values[good] = self.value[indices]
        variances[good] = self.rms[indices]**2

        masks = ~np.isfinite(values) | ~np.isfinite(variances)
        shape = (len(fiberIds), 1)

        return Struct(
            values=values.reshape(shape),
            masks=masks.reshape(shape),
            variances=variances.reshape(shape),
        )

    def evaluate(self, wavelengths: np.ndarray, fiberIds: np.ndarray, positions: np.ndarray) -> Struct:
        """Evaluate the function at the provided positions

        Note that this returns a single value per fiber, not a value per
        wavelength. This is because the function is not wavelength-dependent.

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
        values : `numpy.ndarray` of `float`, shape ``(N,)``
            Vector function evaluated at each position.
        masks : `numpy.ndarray` of `bool`, shape ``(N,)``
            Indicates whether the value at each position is valid.
        variances : `numpy.ndarray` of `float`, shape ``(N,)``
            Variances for each position.
        """
        return self.eval(fiberIds)

    @classmethod
    def fitArrays(
        cls,
        fiberId: np.ndarray,
        wavelengths: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        variances: np.ndarray,
        positions: np.ndarray,
        robust: bool = False,
        **kwargs,
    ) -> FocalPlaneFunction:
        """Fit a constant per fiber to arrays

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.

        Returns
        -------
        fit : `ConstantPerFiber`
            Function fit to input arrays.
        """
        if kwargs:
            raise RuntimeError(f"Unrecognised parameters: {kwargs}")

        bad = masks | ~np.isfinite(values) | ~np.isfinite(variances)

        if robust:
            values = np.ma.median(np.ma.masked_where(bad, values), axis=1).filled(np.nan)
            rms = np.ma.median(np.ma.masked_where(bad, np.sqrt(variances)), axis=1).filled(np.nan)
        else:
            weights = 1.0/variances
            values = np.ma.average(np.ma.masked_where(bad, values), axis=1, weights=weights).filled(np.nan)
            rms = np.ma.sqrt(1.0/np.ma.sum(np.ma.masked_where(bad, weights), axis=1)).filled(np.nan)

        return cls(fiberId=fiberId, value=values, rms=rms)


class FiberPolynomials(FocalPlaneFunction):
    """A polynomial in position for each fiber independently.

    Parameters
    ----------
    fiberId : `np.ndaray`
        Fiber identifiers.
    coeffs : list of `numpy.ndarray` of `float`
        Polynomial coefficients for each fiber.
    xCenter, yCenter : `np.ndarray` of `float`
        Center of each fiber, in mm.
    radius : `float`
        Radius of the fiber patrol regions, in mm.
    rms : `np.ndarray` of `float`
        RMS of residuals from fit for each fiber.
    """

    DamdClass = PfsFiberPolynomials

    fiberId: np.ndarray
    coeffs: list[np.ndarray]
    xCenter: np.ndarray
    yCenter: np.ndarray
    radius: float
    rms: np.ndarray

    def __init__(self, *args, datamodel: Optional[PfsFiberPolynomials] = None, **kwargs):
        super().__init__(*args, datamodel=datamodel, **kwargs)

        from lsst.geom import Box2D, Point2D
        self.polynomials = {
            ff: NormalizedPolynomial2D(
                coeff.astype(np.float64),
                range=Box2D(
                    Point2D(xc - self.radius, yc - self.radius), Point2D(xc + self.radius, yc + self.radius)
                ),
            )
            for ff, coeff, xc, yc in zip(self.fiberId, self.coeffs, self.xCenter, self.yCenter)
        }
        self.variance = {ff: rms**2 for ff, rms in zip(self.fiberId, self.rms)}

    def evaluateSingle(
        self, fiberId: int, x: float | np.ndarray, y: float | np.ndarray
    ) -> float | np.ndarray:
        """Evaluate fiber throughput correction for a single fiber

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.
        x, y : `float` or array
            Position of the fiber on the focal plane, in mm.

        Returns
        -------
        correction : `float`
            Fiber throughput correction factor.
        """
        if np.isscalar(x) and np.isscalar(y):
            if fiberId not in self.polynomials:
                return np.nan
            return self.polynomials[fiberId](x, y)

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if fiberId not in self.polynomials:
            return np.full_like(x, np.nan, dtype=float)
        return self.polynomials[fiberId](x, y)

    def evaluate(self, wavelengths: np.ndarray, fiberIds: np.ndarray, positions: np.ndarray) -> Struct:
        """Evaluate the function at the provided positions

        Note that this returns a single value per fiber, not a value per
        wavelength. This is because the function is not wavelength-dependent.

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
        values : `numpy.ndarray` of `float`, shape ``(N,)``
            Vector function evaluated at each position.
        masks : `numpy.ndarray` of `bool`, shape ``(N,)``
            Indicates whether the value at each position is valid.
        variances : `numpy.ndarray` of `float`, shape ``(N,)``
            Variances for each position.
        """
        uniqueFiberIds = np.unique(fiberIds)
        values = np.full_like(fiberIds, np.nan, dtype=float)
        variances = np.full_like(fiberIds, np.nan, dtype=float)
        for ff in uniqueFiberIds:
            select = fiberIds == ff
            values[select] = self.evaluateSingle(ff, positions[select, 0], positions[select, 1])
            variances[select] = self.variance.get(ff, np.nan)
        masks = ~np.isfinite(values) | ~np.isfinite(variances)
        return Struct(
            values=values.reshape(wavelengths.shape),
            variances=variances.reshape(wavelengths.shape),
            masks=masks.reshape(wavelengths.shape),
        )

    @classmethod
    def fitArrays(
        cls,
        fiberId: np.ndarray,
        wavelengths: np.ndarray,
        values: np.ndarray,
        masks: np.ndarray,
        variances: np.ndarray,
        positions: np.ndarray,
        robust: bool = False,
        order: int = 2,
        fiberMap : FiberIds | None = None,
        radius: float = 4.5,
        **kwargs,
    ) -> FocalPlaneFunction:
        """Fit a constant per fiber to arrays

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
            Wavelength array.
        values : `numpy.ndarray` of `float`, shape ``(N, M)``
            Values to fit.
        masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
            Boolean array indicating values to ignore from the fit.
        variances : `numpy.ndarray` of `float`, shape ``(N, M)``
            Variance values to use in fit.
        positions : `numpy.ndarray` of `float`, shape ``(2, N)``
            Focal-plane positions of fibers.
        robust : `bool`
            Perform robust fit? A robust fit should provide an accurate answer
            in the presense of outliers, even if the answer is less precise
            than desired. A non-robust fit should provide the most precise
            answer while assuming there are no outliers.
        order : `int`
            Polynomial order.
        xCenter, yCenter : `numpy.ndarray` of `float`, shape ``(N,)``
            Center of each fiber, in mm. If not specified, these are taken
            from the mean of `positions`.

        Returns
        -------
        fit : `FiberPolynomials`
            Function fit to input arrays.
        """
        from lsst.geom import Box2D, Point2D

        if fiberMap is None:
            fiberMap = FiberIds()
        uniqueFiberId = np.unique(fiberId)
        indices = np.searchsorted(fiberMap.fiberId, uniqueFiberId)
        found = (indices >= 0) & (indices < len(fiberMap.fiberId))
        found &= (fiberMap.fiberId[indices] == uniqueFiberId)
        if not np.all(found):
            missing = uniqueFiberId[~found]
            raise RuntimeError(f"Some fiberId values not found in fiberMap: {missing}")

        length = values.shape[1]
        if length > 1:
            constants = ConstantPerFiber.fitArrays(
                fiberId,
                wavelengths,
                values,
                masks,
                variances,
                positions,
                robust=robust,
            )
            constValues = constants.values
            constMasks = constants.masks
            constErrors = np.sqrt(constants.variances)
        else:
            constValues = np.reshape(values, len(fiberId))
            constMasks = np.reshape(masks, len(fiberId))
            constErrors = np.sqrt(np.reshape(variances, len(fiberId)))

        num = len(uniqueFiberId)
        coeffs = []
        xCenter = np.full(num, np.nan, dtype=float)
        yCenter = np.full(num, np.nan, dtype=float)
        rms = np.full(num, np.nan, dtype=float)
        for ii, (ff, index) in enumerate(zip(uniqueFiberId, indices)):
            select = fiberId == ff
            xCenter[ii] = fiberMap.x[index]
            yCenter[ii] = fiberMap.y[index]

            xx = positions[select, 0]
            yy = positions[select, 1]
            zz = constValues[select]
            err = constErrors[select]
            mm = constMasks[select]
            good = ~mm & np.isfinite(zz) & np.isfinite(xx) & np.isfinite(yy) & np.isfinite(err) & (err > 0.0)
            box = Box2D(
                Point2D(xCenter[ii] - radius, yCenter[ii] - radius),
                Point2D(xCenter[ii] + radius, yCenter[ii] + radius),
            )
            poly = NormalizedPolynomial2D(order, box)
            if not np.any(good):
                coeffs.append(np.full(poly.getNParameters(), np.nan, dtype=float))
                rms[ii] = np.nan
                continue

            design = poly.calculateDesignMatrix(xx[good].astype(np.float64), yy[good].astype(np.float64))
            solution = solveLeastSquaresDesign(
                design, zz[good].astype(np.float64), err[good].astype(np.float64)
            )
            coeffs.append(solution)
            residuals = design @ solution - zz[good]
            if robust:
                rms[ii] = robustRms(residuals)
            else:
                rms[ii] = np.std(residuals)

        return cls(
            fiberId=uniqueFiberId,
            coeffs=coeffs,
            xCenter=xCenter,
            yCenter=yCenter,
            radius=radius,
            rms=rms,
        )
