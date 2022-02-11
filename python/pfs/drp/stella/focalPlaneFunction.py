from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np
from scipy.interpolate import LSQUnivariateSpline, InterpolatedUnivariateSpline, BSpline, interp1d
from scipy.stats import binned_statistic
import astropy.io.fits

from lsst.pipe.base import Struct

from pfs.datamodel import PfsConfig
from pfs.drp.stella.datamodel import PfsFiberArraySet
from pfs.drp.stella.interpolate import interpolateFlux, interpolateVariance, interpolateMask
from .math import NormalizedPolynomial1D, solveLeastSquaresDesign
from .utils.math import robustRms

__all__ = ("FocalPlaneFunction", "ConstantFocalPlaneFunction", "OversampledSpline",
           "BlockedOversampledSpline", "PolynomialPerFiber")


class FocalPlaneFunction(ABC):
    """Spectral function on the focal plane

    A model of some spectrum as a function of position on the focal plane.

    This is an abstract base class. Subclasses need to override:
    * ``fitArrays`` class method: fit the function
    * ``evaluate`` method: evaluate the spectrum at a position.
    * ``toFits`` method: write to a FITS file.
    * ``fromFits`` class method: construct from a FITS file.
    """
    @classmethod
    def fit(cls, spectra: PfsFiberArraySet, pfsConfig: PfsConfig, maskFlags: List[str],
            *, rejected=None, robust: bool = False, **kwargs) -> "FocalPlaneFunction":
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
                f"Length mismatch between spectra ({len(spectra)}) and pfsConfig ({len(pfsConfig)})")
        if len(spectra) == 0:
            raise RuntimeError("No input spectra provided")
        if np.all(spectra.fiberId != pfsConfig.fiberId):
            raise RuntimeError("fiberId mismatch between spectra and pfsConfig")

        with np.errstate(invalid="ignore", divide="ignore"):
            values = spectra.flux/spectra.norm
            variance = spectra.variance/spectra.norm**2
        masks = spectra.mask & spectra.flags.get(*maskFlags) != 0
        masks |= ~np.isfinite(values) | ~np.isfinite(variance)
        if rejected is not None:
            masks |= rejected
        positions = pfsConfig.extractCenters(pfsConfig.fiberId)
        return cls.fitArrays(spectra.fiberId, spectra.wavelength, values, masks, variance,
                             positions, robust=robust, **kwargs)

    @classmethod
    @abstractmethod
    def fitArrays(cls, fiberId, wavelengths, values, masks, variances, positions, robust: bool = False,
                  **kwargs) -> "FocalPlaneFunction":
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

    def __call__(self, wavelengths, pfsConfig: PfsConfig):
        """Evaluate the function for the provided fiberIds

        This method is intended as a convenience for users.

        For convenience, if the ``pfsConfig`` contains only a single fiber,
        we'll return a single array for each output.

        Parameters
        ----------
        wavelengths : iterable (length ``N``) of `numpy.ndarray` of shape ``(M,)``
            Wavelength arrays.
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
        if len(wavelengths.shape) == 1:
            isSingle = True
            wavelengths = np.array([wavelengths]*len(pfsConfig))
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
    def evaluate(self, wavelengths, fiberIds, positions):
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
        subclasses = {ss.__name__: ss for ss in cls.__subclasses__()}  # ignores sub-sub-classes
        with astropy.io.fits.open(filename) as fits:
            name = fits[0].header["pfs_focalPlaneFunction_class"]
            if name not in subclasses:
                raise RuntimeError(f"Unrecognised pfs_focalPlaneFunction_class value: {name}")
            return subclasses[name].fromFits(fits)

    def writeFits(self, filename: str) -> None:
        """Write to FITS file

        Parameters
        ----------
        filename : `str`
            Name of file to which to write.
        """
        header = astropy.io.fits.Header()
        header["HIERARCH pfs_focalPlaneFunction_class"] = type(self).__name__
        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(header=header)])
        self.toFits(fits)
        with open(filename, "wb") as fd:
            fits.writeto(fd)

    @classmethod
    @abstractmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "FocalPlaneFunction":
        """Construct from FITS file

        This abstract class method must be overridden by subclasses.

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        raise NotImplementedError("Subclasses must override")

    @abstractmethod
    def toFits(self, fits: astropy.io.fits.HDUList) -> None:
        """Write to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file to which to write.
        """
        raise NotImplementedError("Subclasses must override")


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
    def __init__(self, wavelength, value, mask, variance):
        self.wavelength = wavelength
        self.value = value
        self.mask = mask
        self.variance = variance

    @classmethod
    def fitArrays(cls, fiberId, wavelength, values, masks, variances, positions, robust: bool = False,
                  **kwargs) -> "ConstantFocalPlaneFunction":
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
        if not np.all(np.equal.reduce(wavelength, axis=0)):
            raise RuntimeError("Wavelength arrays not identical")

        if robust:
            return ConstantFocalPlaneFunction(wavelength[0], np.median(values, axis=0),
                                              np.logical_and.reduce(masks, axis=0),
                                              np.median(variances, axis=0))

        with np.errstate(invalid="ignore", divide="ignore"):
            weight = 1.0/variances
            good = ~masks & (variances > 0)
            weight[~good] = 0.0

            sumWeights = np.sum(weight, axis=0)
            coaddValues = np.sum(values*weight, axis=0)/sumWeights
            coaddVariance = np.where(sumWeights > 0, 1.0/sumWeights, np.inf)
            coaddMask = sumWeights <= 0
        return ConstantFocalPlaneFunction(wavelength[0], coaddValues, coaddMask, coaddVariance)

    def evaluate(self, wavelengths, fiberIds, positions) -> Struct:
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

        doResample = [wl.shape != self.wavelength.shape or not np.all(wl == self.wavelength) for
                      wl in wavelengths]

        values = [interpolateFlux(self.wavelength, self.value, wl, jacobian=False) if resamp else self.value
                  for wl, resamp in zip(wavelengths, doResample)]
        masks = [interpolateMask(self.wavelength, self.mask, wl).astype(bool) if resamp else self.mask for
                 wl, resamp in zip(wavelengths, doResample)]
        variances = [interpolateVariance(self.wavelength, self.variance, wl, jacobian=False) if
                     resamp else self.variance for wl, resamp in zip(wavelengths, doResample)]
        return Struct(values=np.array(values), masks=np.array(masks), variances=np.array(variances))

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "ConstantFocalPlaneFunction":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        wavelength = fits["WAVELENGTH"].data
        value = fits["VALUE"].data
        mask = fits["MASK"].data.astype(bool)
        variance = fits["VARIANCE"].data
        return cls(wavelength, value, mask, variance)

    def toFits(self, fits: astropy.io.fits.HDUList) -> None:
        """Write to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file to which to write.
        """
        fits.append(astropy.io.fits.ImageHDU(self.wavelength, name="WAVELENGTH"))
        fits.append(astropy.io.fits.ImageHDU(self.value, name="VALUE"))
        fits.append(astropy.io.fits.ImageHDU(self.mask.astype(np.uint8), name="MASK"))
        fits.append(astropy.io.fits.ImageHDU(self.variance, name="VARIANCE"))


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
    """
    def __init__(self, knots, coeffs, splineOrder: int, wavelength, variance, defaultValue: float):
        self.knots = knots
        self.coeffs = coeffs
        self.splineOrder = splineOrder
        self.wavelength = wavelength
        self.variance = variance
        self.defaultValue = defaultValue
        self._values = BSpline(knots, coeffs, splineOrder, extrapolate=False)
        self._variance = interp1d(wavelength, variance, kind="linear", bounds_error=False,
                                  fill_value=np.nan, copy=True, assume_sorted=True)

    @classmethod
    def getKnots(cls, data, numKnots, order=3):
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
        bins[1:-1] = 0.5*(interior[:-1] + interior[1:])
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
        knots = np.concatenate(([first]*(order + 1), interior, [last]*(order + 1)))
        counts = np.histogram(data, knots)[0]
        cumsum = np.cumsum(counts)
        kCount = cumsum[order + 1:] - cumsum[:-(order + 1)]
        select = kCount >= order
        return knots[order + 1:-(order + 2)][select[:-(order + 1)]]

    @staticmethod
    def _fpchec(knots, data, order):
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
        knots = np.concatenate(([data[0]]*(order + 1), knots, [data[-1]]*(order + 1)))

        numKnots = knots.size
        numData = data.size

        # Condition 1: there's enough data for the knots and spline order
        nk1 = numKnots - order - 1
        if order + 1 > nk1 or nk1 > numData:
            print("Condition 1 fails: k+1 <= n-k-1 <= m")
        # Condition 2: the exterior knots are increasing or constant
        lowKnots = knots[:order + 1]  # Low exterior knots
        highKnots = knots[-order - 1:]  # High exterior knots
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
    def fitArrays(cls, fiberId, wavelengths, values, masks, variances, positions, robust: bool = False,
                  oversample: float = 3.0, splineOrder: int = 3, defaultValue: float = 0.0
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
        use = ~masks
        floatEpsilon = np.finfo(float).eps
        length = wavelengths.shape[1]
        wlUse = wavelengths[use]
        wlMin = wlUse.min() - floatEpsilon
        wlMax = wlUse.max() + floatEpsilon

        indices = np.argsort(wlUse)
        xx = wlUse[indices]
        yy = values[use][indices]
        var = variances[use][indices]

        dx = (wlMax - wlMin)/length/oversample
        knotMin = xx[1]
        knotMax = xx[-2]
        numKnots = int((knotMax - knotMin)/dx)
        knots = cls.getKnots(xx, numKnots, splineOrder)
        bins = np.concatenate(([wlMin], knots, [wlMax]))
        centers = 0.5*(bins[:-1] + bins[1:])

        if robust:
            yMedian = binned_statistic(xx, yy, statistic="median", bins=bins)[0]
            goodBins = np.isfinite(yMedian)
            spline = InterpolatedUnivariateSpline(centers[goodBins], yMedian[goodBins], k=splineOrder,
                                                  bbox=(wlMin, wlMax))
            knots = np.concatenate(([wlMin]*(splineOrder - 1), centers[goodBins], [wlMax]*(splineOrder - 1)))
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                spline = LSQUnivariateSpline(xx, yy, w=1/var, t=knots, k=splineOrder, bbox=(wlMin, wlMax))
            knots = np.concatenate(([wlMin]*(splineOrder + 1), knots, [wlMax]*(splineOrder + 1)))

        coeffs = spline.get_coeffs()

        # Measure the noise in the fit
        residual = yy - spline(xx)
        variance = binned_statistic(xx, residual, statistic='std', bins=bins)[0]**2
        # Remove noise originating from the input data
        variance -= binned_statistic(xx, var, statistic='mean', bins=bins)[0]
        variance = np.clip(variance, 0.0, None)

        return cls(knots, coeffs, splineOrder, centers, variance, defaultValue)

    def evaluate(self, wavelengths, fiberIds, positions) -> Struct:
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

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "OversampledSpline":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        hdu = fits["SPLINE"]
        splineOrder = hdu.header["ORDER"]
        defaultValue = float(hdu.header["DEFAULT"])
        knots = hdu.data["knots"][0]
        coeffs = hdu.data["coeffs"][0]
        wavelength = hdu.data["wavelength"][0]
        variance = hdu.data["variance"][0]
        return cls(knots, coeffs, splineOrder, wavelength, variance, defaultValue)

    def toFits(self, fits: astropy.io.fits.HDUList) -> None:
        """Write to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file to which to write.
        """
        header = astropy.io.fits.Header()
        header["ORDER"] = self.splineOrder
        header["DEFAULT"] = self.defaultValue if np.isfinite(self.defaultValue) else str(self.defaultValue)
        numKnots = len(self.knots)
        numCoeffs = len(self.coeffs)
        numVar = len(self.variance)
        table = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column("knots", format=f"{numKnots}D", array=[self.knots]),
             astropy.io.fits.Column("coeffs", format=f"{numCoeffs}D", array=[self.coeffs]),
             astropy.io.fits.Column("wavelength", format=f"{numVar}D", array=[self.wavelength]),
             astropy.io.fits.Column("variance", format=f"{numVar}D", array=[self.variance]),
             ], header=header, name="SPLINE")
        fits.append(table)


class BlockedOversampledSpline(FocalPlaneFunction):
    """Oversampled splines defined in blocks of fiberId

    Parameters
    ----------
    splines : `dict` [`float`: `OversampledSpline`]
        Splines for each block index.
    """
    def __init__(self, splines: Dict[float, OversampledSpline]):
        self.splines = splines
        self.fiberId = np.sort(np.array(list(splines.keys())))

    @classmethod
    def fitArrays(cls, fiberId, wavelengths, values, masks, variances, positions, *, robust: bool = False,
                  blockSize: int = 5, **kwargs) -> "BlockedOversampledSpline":
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
        numBlocks = int(np.ceil(numFibers/blockSize))
        blocks = (np.arange(numFibers, dtype=float)*numBlocks/numFibers).astype(int)

        splines = {}
        for bb in range(numBlocks):
            select = blocks == bb
            ff = fiberId[select].mean()
            splines[ff] = OversampledSpline.fitArrays(fiberId[select], wavelengths[select], values[select],
                                                      masks[select], variances[select], positions[select],
                                                      robust=robust, **kwargs)
        return cls(splines)

    def evaluate(self, wavelengths, fiberIds, positions) -> Struct:
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
            if len(self.fiberId) == 1:
                result = next(iter(self.splines.values())).evaluate([wavelengths[ii]], [ff], [positions[ii]])
                thisValue = result.values[0]
                thisMask = result.masks[0]
                thisVariance = result.variances[0]
            else:
                index = np.searchsorted(self.fiberId, ff)
                iAbove = max(1, min(index, len(self.fiberId) - 1))
                iBelow = iAbove - 1
                weightBelow = (self.fiberId[iAbove] - ff)/(self.fiberId[iAbove] - self.fiberId[iBelow])
                weightAbove = 1 - weightBelow
                below = self.splines[self.fiberId[iBelow]].evaluate([wavelengths[ii]], [ff], [positions[ii]])
                above = self.splines[self.fiberId[iAbove]].evaluate([wavelengths[ii]], [ff], [positions[ii]])
                thisValue = below.values[0]*weightBelow + above.values[0]*weightAbove
                thisMask = below.masks[0] | above.masks[0]
                thisVariance = below.variances[0]*weightBelow + above.variances[0]*weightAbove
            values[ii] = thisValue
            masks[ii] = thisMask
            variances[ii] = thisVariance
        bad = np.isnan(variances)
        variances[bad] = 0.0
        masks[bad] = True
        return Struct(values=values, masks=masks, variances=variances)

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "BlockedOversampledSpline":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        hdu = fits["BLOCKSPLINE"]
        splines = {row["fiberId"]: OversampledSpline(row["knots"], row["coeffs"], row["splineOrder"],
                                                     row["wavelength"], row["variance"],
                                                     row["defaultValue"]) for row in hdu.data}
        return cls(splines)

    def toFits(self, fits: astropy.io.fits.HDUList) -> None:
        """Write to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file to which to write.
        """
        header = astropy.io.fits.Header()
        fiberId = list(self.splines.keys())
        splineOrder = [sp.splineOrder for sp in self.splines.values()]
        defaultValue = [sp.defaultValue for sp in self.splines.values()]
        knots = [sp.knots for sp in self.splines.values()]
        coeffs = [sp.coeffs for sp in self.splines.values()]
        wavelength = [sp.wavelength for sp in self.splines.values()]
        variance = [sp.variance for sp in self.splines.values()]
        table = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column("fiberId", format="D", array=fiberId),
             astropy.io.fits.Column("splineOrder", format="J", array=splineOrder),
             astropy.io.fits.Column("defaultValue", format="D", array=defaultValue),
             astropy.io.fits.Column("knots", format="PD()", array=knots),
             astropy.io.fits.Column("coeffs", format="PD()", array=coeffs),
             astropy.io.fits.Column("wavelength", format="PD()", array=wavelength),
             astropy.io.fits.Column("variance", format="PD()", array=variance),
             ], header=header, name="BLOCKSPLINE")
        fits.append(table)


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
    def __init__(self, coeffs: Dict[int, Any], rms: Dict[int, float],
                 minWavelength: float, maxWavelength: float):
        assert(set(coeffs.keys()) == set(rms.keys()))
        self._coeffs = coeffs
        self._rms = rms
        self.minWavelength = minWavelength
        self.maxWavelength = maxWavelength

    @property
    def fiberId(self):
        """Fiber identifiers with a corresponding polynomial"""
        return np.array(sorted(self._coeffs.keys()), dtype=int)

    def __getitem__(self, fiberId: int) -> NormalizedPolynomial1D:
        """Return the polynomial for a particular fiber"""
        return NormalizedPolynomial1D(self._coeffs[fiberId], self.minWavelength, self.maxWavelength)

    @classmethod
    def fitArrays(cls, fiberId, wavelengths, values, masks, variances, positions, *, robust: bool = False,
                  order: int = 3, minWavelength: float = np.nan, maxWavelength: float = np.nan
                  ) -> "PolynomialPerFiber":
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
            design = np.array([poly.getDFuncDParameters(wl) for wl in wavelengths[ii][choose]])
            coeffs[ff] = solveLeastSquaresDesign(design, values[ii][choose], errors[ii][choose])
            residuals = design @ coeffs[ff] - values[ii][choose]
            if robust:
                rms[ff] = robustRms(residuals)
            else:
                weights = 1.0/errors[ii][choose]**2
                rms[ff] = np.sqrt(np.sum(weights*residuals**2)/np.sum(weights))

        return cls(coeffs, rms, minWavelength, maxWavelength)

    def evaluate(self, wavelengths, fiberIds, positions) -> Struct:
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
        rms = np.array([np.full_like(wavelengths[ii], self._rms[ff]) for ii, ff in enumerate(fiberIds)])
        masks = ~np.isfinite(values) | ~np.isfinite(rms)
        return Struct(values=values, variances=rms**2, masks=masks)

    def toFits(self, fits: astropy.io.fits.HDUList) -> None:
        """Write to FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file to which to write.
        """
        fiberId = self.fiberId

        header = astropy.io.fits.Header()
        header["MIN_WL"] = self.minWavelength
        header["MAX_WL"] = self.maxWavelength
        coeffs = [self._coeffs[ff] for ff in fiberId]
        rms = [self._rms[ff] for ff in fiberId]

        fits.append(
            astropy.io.fits.BinTableHDU.from_columns(
                [astropy.io.fits.Column("fiberId", format="J", array=fiberId),
                 astropy.io.fits.Column("coeffs", format="PD()", array=coeffs),
                 astropy.io.fits.Column("rms", format="D", array=rms),
                 ], header=header, name="POLYPERFIBER")
        )

    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "BlockedOversampledSpline":
        """Construct from FITS file

        Parameters
        ----------
        fits : `astropy.io.fits.HDUList`
            FITS file from which to read.

        Returns
        -------
        self : cls
            Constructed focal plane function.
        """
        hdu = fits["POLYPERFIBER"]
        minWavelength = hdu.header["MIN_WL"]
        maxWavelength = hdu.header["MAX_WL"]
        fiberId = hdu.data["fiberId"]
        coeffs = dict(zip(fiberId, hdu.data["coeffs"]))
        rms = dict(zip(fiberId, hdu.data["rms"]))

        return cls(coeffs, rms, minWavelength, maxWavelength)
