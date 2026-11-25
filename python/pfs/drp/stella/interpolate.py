import numpy as np
import scipy.sparse
from scipy.interpolate import interp1d

from lsst.pipe.base import Struct

from .math import interpolate as interpolateImpl
from .math import interpolateFlux as interpolateFluxImpl
from .math import interpolateCovariance as interpolateCovarianceImpl
from .spline import SplineD

__all__ = [
    "COVAR_DIAGONALS",
    "calculateDispersion",
    "maskCovariance",
    "numCovarForSampling",
    "Resampler",
    "resampleWindow",
    "interpolate",
    "interpolateMaskedVariance",
    "interpolateFlux",
    "interpolateVariance",
    "interpolateMask",
]


COVAR_DIAGONALS = 3
"""Number of covariance diagonals in the datamodel's COVAR: the diagonal, and
offsets of 1 and 2 (a 5-banded matrix)"""


def calculateDispersion(wavelength):
    """Calculate dispersion (nm/pixel) as a function of wavelength

    This might be the inverse of what you consider "dispersion": this is
    wavelength increment per pixel, not pixels per unit wavelength.

    Parameters
    ----------
    wavelength : array_like of `float`
        Wavelength value for each pixel.

    Returns
    -------
    dispersion : `numpy.ndarray` of `float`
        Wavelength increment value for each pixel.
    """
    dispersion = np.empty_like(wavelength)
    dispersion[..., 0] = wavelength[..., 1] - wavelength[..., 0]
    dispersion[..., -1] = wavelength[..., -1] - wavelength[..., -2]
    dispersion[..., 1:-1] = 0.5*(wavelength[..., 2:] - wavelength[..., 0:-2])
    return np.abs(dispersion)


def interpolateIndices(fromWavelength: np.ndarray, toWavelength: np.ndarray) -> np.ndarray:
    """Return indices to interpolate from one wavelength grid to another

    Indices are floating-point values.

    Parameters
    ----------
    fromWavelength : array-like of `float`
        Source wavelength array.
    toWavelength : array-like of `float`
        Target wavelength array.

    Returns
    -------
    indices : `numpy.ndarray` of `float`
        Indices into the source wavelength array for each value in the target
        wavelength array.
    """
    indices = np.arange(fromWavelength.size, dtype=np.float64)
    spline = SplineD(fromWavelength.astype(np.float64), indices)
    good = (toWavelength >= fromWavelength[0]) & (toWavelength <= fromWavelength[-1])
    result = np.full_like(toWavelength, np.nan, dtype=np.float64)
    result[good] = spline(toWavelength[good].astype(np.float64))
    return result


def numCovarForSampling(
    fromWavelengths: list[np.ndarray],
    toWavelength: np.ndarray,
    order: int = 3,
    numInputDiagonals: int = 1,
) -> int:
    """Number of covariance diagonals needed to hold all of the covariance of
    a resampling, for several source samplings at once

    This is a conservative (cheap) form of `Resampler.numCovarNeeded`, from
    the largest source pixel and the smallest target pixel, for choosing one
    number for several spectra (e.g., all the arms and spectrographs of an
    exposure, whose combination needs a single shape).

    Parameters
    ----------
    fromWavelengths : iterable of `np.ndarray` of `float`
        Source wavelength arrays, each 1-D or 2-D (a row per fiber).
    toWavelength : `np.ndarray` of `float`
        Target wavelength array.
    order : `int`, optional
        Interpolation order (1 is linear; higher orders are Lanczos).
    numInputDiagonals : `int`, optional
        Number of non-zero diagonals of the source covariance (1 for
        uncorrelated source pixels).

    Returns
    -------
    numCovar : `int`
        Number of diagonals.
    """
    toWavelength = np.asarray(toWavelength, dtype=np.float64)
    if len(toWavelength) < 2:
        return 1
    halfSize = max(order, 1)
    reach = 2*halfSize + max(numInputDiagonals, 1) - 1
    toSpacing = np.diff(toWavelength)
    numCovar = 1
    for fromWavelength in fromWavelengths:
        for row in np.atleast_2d(np.asarray(fromWavelength, dtype=np.float64)):
            good = np.isfinite(row)
            if good.sum() < 2:
                continue
            row = row[good]
            lo, hi = min(row[0], row[-1]), max(row[0], row[-1])
            overlap = (toWavelength[:-1] >= lo) & (toWavelength[1:] <= hi)
            if not np.any(overlap):
                continue
            spacing = np.min(np.abs(toSpacing[overlap]))/np.max(np.abs(np.diff(row)))  # source pixels
            if spacing <= 0:
                continue
            numCovar = max(numCovar, int(np.ceil(reach/spacing - 1.0e-9)))
    return min(numCovar, len(toWavelength))


def maskCovariance(covariance: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Zero the elements of a banded covariance involving bad pixels

    This is the treatment of the variance of bad pixels before resampling
    (replaced by zero), extended to the off-diagonal elements.

    Parameters
    ----------
    covariance : `np.ndarray` of `float`, shape ``(numDiagonals, num)``
        Banded covariance: ``covariance[k][i]`` is the covariance of pixels
        ``i`` and ``i + k``.
    bad : `np.ndarray` of `bool`, shape ``(num,)``
        Bad pixels.

    Returns
    -------
    result : `np.ndarray` of `float`, shape ``(numDiagonals, num)``
        Covariance with the elements involving bad pixels (and non-finite
        elements) set to zero.
    """
    result = np.where(np.isfinite(covariance), covariance, 0.0).astype(covariance.dtype)
    num = result.shape[1]
    for kk in range(result.shape[0]):
        if kk >= num:
            break
        either = bad[:num - kk] | bad[kk:]
        result[kk, :num - kk][either] = 0.0
    return result


class Resampler:
    """Resample arrays from one wavelength sampling to another

    This is the resampling used for spectra (`interpolate`, `interpolateFlux`),
    with the source indices of the target wavelengths computed once. The same
    operation can then be applied to several arrays sampled like the spectrum:
    its flux, variance and sky, or a line-spread function kernel.

    Parameters
    ----------
    fromWavelength : `np.ndarray` of `float`
        Source wavelength array.
    toWavelength : `np.ndarray` of `float`
        Target wavelength array.
    order : `int`, optional
        The legacy specification of the kernel: 1 means linear interpolation;
        higher orders use a Lanczos kernel of the given order. Default 3, unless
        ``kernel`` is given (it's an error to give both).
    kernel : `str`, optional
        Interpolation kernel: ``"linear"`` (a triangle, half-width one source
        pixel), or ``"lanczos"`` (of order ``lanczosOrder``).
    lanczosOrder : `int`, optional
        Order (half-width, in source pixels) of the Lanczos kernel; at least 2.
    fill : `float`, optional
        Fill value, for target wavelengths outside the source range.
    """

    kernels = ("linear", "lanczos")
    """Supported interpolation kernels"""

    def __init__(
        self,
        fromWavelength: np.ndarray,
        toWavelength: np.ndarray,
        *,
        order: int | None = None,
        kernel: str | None = None,
        lanczosOrder: int = 3,
        fill: float = 0.0,
    ):
        self.fromWavelength = fromWavelength
        self.toWavelength = toWavelength
        self.order = self.kernelOrder(order, kernel, lanczosOrder)
        self.fill = fill
        self.indices = interpolateIndices(fromWavelength, toWavelength)

    @classmethod
    def kernelOrder(cls, order: int | None = None, kernel: str | None = None, lanczosOrder: int = 3) -> int:
        """The order of the C++ implementation for a kernel specification

        Parameters are as for the constructor.

        Returns
        -------
        order : `int`
            1 for linear interpolation; otherwise the order of the Lanczos
            kernel.
        """
        if kernel is None:
            order = 3 if order is None else order
            if order < 1:
                raise ValueError(f"Interpolation order ({order}) must be positive")
            return order
        if order is not None:
            raise ValueError("Specify the interpolation kernel or the order, not both")
        if kernel == "linear":
            return 1
        if kernel == "lanczos":
            if lanczosOrder < 2:
                raise ValueError(f"Lanczos order ({lanczosOrder}) must be at least 2 (1 is linear)")
            return lanczosOrder
        raise ValueError(f"Unrecognised interpolation kernel {kernel!r}: expected one of {cls.kernels}")

    @property
    def kernelName(self) -> str:
        """Name of the interpolation kernel: ``"linear"`` or ``"lanczos"``"""
        return "linear" if self.order <= 1 else "lanczos"

    def numCovarNeeded(self, numInputDiagonals: int = 1) -> int:
        """Number of covariance diagonals of the result needed to hold all of
        its covariance

        Two output pixels are correlated only if their kernels share source
        pixels (or source pixels correlated with each other), i.e., if they are
        less than ``2*halfSize + numInputDiagonals - 1`` source pixels apart.
        This depends on the kernel and the ratio of the output to the input
        pixel spacing (for non-uniform grids, the smallest output spacing).

        Parameters
        ----------
        numInputDiagonals : `int`, optional
            Number of non-zero diagonals of the source covariance (1 for
            uncorrelated source pixels).

        Returns
        -------
        numCovar : `int`
            Number of diagonals: the variance, and the covariances at offsets
            ``1..numCovar - 1``.
        """
        indices = np.asarray(self.indices, dtype=np.float64)
        indices = indices[np.isfinite(indices)]
        if len(indices) < 2:
            return 1
        spacing = np.min(np.abs(np.diff(indices)))
        if spacing <= 0:
            return len(indices)
        reach = 2*self.halfSize + max(numInputDiagonals, 1) - 1
        return int(min(len(indices), max(1, np.ceil(reach/spacing - 1.0e-9))))

    @property
    def halfSize(self) -> int:
        """Half-width of the interpolation kernel (source pixels)"""
        return max(self.order, 1)

    def __call__(
        self,
        flux: np.ndarray,
        mask: np.ndarray,
        variance: np.ndarray,
        *,
        minWeight: float = 0.1,
        badMask: int = 0,
        fillMask: int = 0,
        numCovar: int = 1,
        covariance: np.ndarray | None = None,
    ) -> Struct:
        """Resample a spectrum: flux, mask and covariance

        Parameters and results are as for `interpolate`, with the addition of
        ``covariance``.

        Parameters
        ----------
        covariance : `np.ndarray` of `float`, shape ``(numDiagonals, numIn)``, optional
            Banded covariance of the source pixels (``covariance[0]`` is the
            ``variance``). If it has non-zero off-diagonal elements (e.g., the
            spectrum has been resampled before), the covariance of the result
            is calculated from it (see `propagateCovariance`); otherwise the
            source pixels are taken to be uncorrelated.
        """
        if covariance is not None:
            covariance = np.atleast_2d(covariance)
            if covariance.shape[1] != len(variance):
                raise RuntimeError(
                    f"Shape mismatch between covariance {covariance.shape} and variance {variance.shape}"
                )
        correlated = covariance is not None and covariance.shape[0] > 1 and np.any(covariance[1:] != 0)
        resultMask = interpolateMask(self.fromWavelength, mask, self.toWavelength, fillMask)
        length = len(self.toWavelength)
        resultFlux = np.empty(length, dtype=flux.dtype)
        interpMask = np.empty(length, dtype=bool)
        resultCovariance = np.empty((numCovar, length), dtype=variance.dtype)
        interpolateImpl(
            resultFlux,
            interpMask,
            resultCovariance,
            flux,
            (mask & badMask) != 0,
            variance,
            self.indices,
            self.fill,
            self.order,
            minWeight,
        )
        resultMask[interpMask] |= fillMask
        if correlated:
            resultCovariance = self.propagateCovariance(
                covariance, mask, badMask, numCovar
            ).astype(resultCovariance.dtype)
        return Struct(
            wavelength=self.toWavelength,
            flux=resultFlux,
            mask=resultMask,
            covariance=resultCovariance,
            variance=resultCovariance[0],
        )

    def resampleVariance(
        self,
        mask: np.ndarray,
        variance: np.ndarray,
        *,
        minWeight: float = 0.1,
        badMask: int = 0,
        fillMask: int = 0,
        numCovar: int = 1,
    ) -> Struct:
        """Resample the variance and mask of a spectrum, without the flux

        This treats bad pixels exactly as ``__call__`` does. Parameters and
        results are as for `interpolateMaskedVariance`.
        """
        resultMask = interpolateMask(self.fromWavelength, mask, self.toWavelength, fillMask)
        length = len(self.toWavelength)
        interpMask = np.empty(length, dtype=bool)
        resultCovariance = np.empty((numCovar, length), dtype=variance.dtype)
        interpolateCovarianceImpl(
            resultCovariance,
            interpMask,
            variance,
            (mask & badMask) != 0,
            self.indices,
            self.fill,
            self.order,
            minWeight,
        )
        resultMask[interpMask] |= fillMask
        return Struct(
            wavelength=self.toWavelength,
            mask=resultMask,
            covariance=resultCovariance,
            variance=resultCovariance[0],
        )

    def kernel(self, x: np.ndarray) -> np.ndarray:
        """The interpolation kernel, as used by the C++ implementation

        ``order <= 1`` is a triangle (linear interpolation); higher orders are
        a Lanczos kernel of that order (``sinc(x) sinc(x/order)``).

        Parameters
        ----------
        x : `np.ndarray` of `float`
            Offsets from the target position (source pixels).

        Returns
        -------
        values : `np.ndarray` of `float`
            Kernel values.
        """
        if self.order <= 1:
            return np.clip(1.0 - np.abs(x), 0.0, None)
        return np.sinc(x)*np.sinc(x/self.order)

    def matrix(self, mask: np.ndarray | None = None, badMask: int = 0) -> scipy.sparse.csr_matrix:
        """The resampling as a linear operator

        The resampled values are ``matrix @ values``: each row is the
        normalised kernel of an output pixel over the source pixels, as used by
        ``__call__`` (source pixels with ``mask & badMask`` set are excluded and
        the kernel renormalised over the others). Rows of output pixels that
        can't be interpolated are zero.

        This is the operator through which other quantities are resampled
        consistently with the spectrum: the covariance (``K C K^T``; see
        `propagateCovariance`), or realisations of the noise (``K n``).

        Parameters
        ----------
        mask : `np.ndarray` of `int`, optional
            Source mask.
        badMask : `int`, optional
            Bitmask of source pixels to exclude.

        Returns
        -------
        matrix : `scipy.sparse.csr_matrix`, shape ``(numOut, numIn)``
            The resampling operator.
        """
        numIn = len(self.fromWavelength)
        numOut = len(self.toWavelength)
        targets = np.asarray(self.indices, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(targets) & (targets >= 0) & (targets <= numIn - 1)
        rows = np.flatnonzero(valid)
        target = targets[valid]
        # The C++ window: source pixels ceil(target) - halfSize (inclusive) to ceil(target) + halfSize
        halfSize = self.halfSize
        columns = np.ceil(target).astype(np.int64)[:, np.newaxis] + np.arange(-halfSize, halfSize)
        use = (columns >= 0) & (columns < numIn)
        if mask is not None and badMask:
            use &= (np.asarray(mask)[np.clip(columns, 0, numIn - 1)] & badMask) == 0
        values = np.where(use, self.kernel(columns - target[:, np.newaxis]), 0.0)
        sumValues = values.sum(axis=1)
        good = sumValues > 0
        values[good] /= sumValues[good, np.newaxis]
        values[~good] = 0.0
        use &= good[:, np.newaxis]
        rowIndex = np.broadcast_to(rows[:, np.newaxis], columns.shape)
        return scipy.sparse.csr_matrix(
            (values[use], (rowIndex[use], columns[use])), shape=(numOut, numIn)
        )

    def propagateCovariance(
        self,
        covariance: np.ndarray,
        mask: np.ndarray | None = None,
        badMask: int = 0,
        numCovar: int = 1,
    ) -> np.ndarray:
        """Propagate a banded covariance through the resampling: ``K C K^T``

        Unlike the C++ implementation (used by ``__call__`` for uncorrelated
        source pixels), this accounts for correlations between the source
        pixels, e.g. from an earlier resampling. It is exact, up to the number
        of diagonals retained.

        Parameters
        ----------
        covariance : `np.ndarray` of `float`, shape ``(numDiagonals, numIn)``
            Banded source covariance: ``covariance[k][i]`` is the covariance
            of source pixels ``i`` and ``i + k``. Values must be finite (bad
            pixels should be masked, or their values replaced, by the caller).
        mask : `np.ndarray` of `int`, optional
            Source mask.
        badMask : `int`, optional
            Bitmask of source pixels to exclude.
        numCovar : `int`, optional
            Number of diagonals of the result.

        Returns
        -------
        result : `np.ndarray` of `float`, shape ``(numCovar, numOut)``
            Banded covariance of the resampled values: ``result[k][i]`` is the
            covariance of output pixels ``i`` and ``i + k``. As for the C++
            implementation, output pixels that can't be interpolated, and
            offsets beyond the end of the array, have the ``fill`` value.
        """
        covariance = np.atleast_2d(np.asarray(covariance, dtype=np.float64))
        numIn = covariance.shape[1]
        diagonals = [covariance[0]]
        offsets = [0]
        for kk in range(1, covariance.shape[0]):
            if kk >= numIn:
                break
            diagonals += [covariance[kk, :numIn - kk]]*2
            offsets += [kk, -kk]
        cov = scipy.sparse.diags(diagonals, offsets, shape=(numIn, numIn), format="csr")
        operator = self.matrix(mask, badMask)
        product = (operator @ cov @ operator.T).tocsr()
        numOut = operator.shape[0]
        result = np.full((numCovar, numOut), self.fill, dtype=np.float64)
        valid = np.diff(operator.indptr) > 0
        for kk in range(numCovar):
            if kk >= numOut:
                break
            values = product.diagonal(kk)
            select = valid[:numOut - kk]
            result[kk, :numOut - kk][select] = values[select]
        return result

    def resampleFlux(self, values: np.ndarray) -> np.ndarray:
        """Resample a flux-like array, with no mask (as `interpolateFlux`)

        Parameters
        ----------
        values : `np.ndarray` of `float`
            Values at the source wavelengths.

        Returns
        -------
        result : `np.ndarray` of `float`
            Values at the target wavelengths.
        """
        return interpolateFluxImpl(values, self.indices, self.fill, self.order)

    def resampleWindow(self, values: np.ndarray, start: int) -> tuple[int, np.ndarray]:
        """Resample an array that is zero outside a short window

        This is used to resample a line-spread function kernel. It gives the
        same values as `resampleFlux` on the full array (with ``values`` at
        source pixels ``start:start + len(values)`` and zeros elsewhere), but
        only does the work near the window.

        Parameters
        ----------
        values : `np.ndarray` of `float`
            Values of the window.
        start : `int`
            Source pixel of ``values[0]``.

        Returns
        -------
        outStart : `int`
            Target pixel of ``result[0]``.
        result : `np.ndarray` of `float`
            Resampled values at target pixels ``outStart:outStart + len(result)``:
            every target pixel that the window can affect.
        """
        return resampleWindow(values, start, self.indices, len(self.fromWavelength), self.order, self.fill)


def resampleWindow(
    values: np.ndarray, start: int, indices: np.ndarray, numIn: int | None, order: int, fill: float = 0.0
) -> tuple[int, np.ndarray]:
    """Resample an array that is zero outside a short window

    This is the operation of `Resampler.resampleWindow`, for target pixels
    whose source indices are provided directly.

    Parameters
    ----------
    values : `np.ndarray` of `float`
        Values of the window.
    start : `int`
        Source pixel of ``values[0]``.
    indices : `np.ndarray` of `float`
        Source indices of a contiguous run of target pixels (NaN where the
        target is outside the source). Monotonic increasing.
    numIn : `int` or `None`
        Length of the source array (interpolation windows are clipped to it).
        If `None`, the source array is unbounded: ``start`` may be negative,
        and target pixels beyond the ends of the source are resampled as if
        the source continued (with the provided ``indices``).
    order : `int`
        Interpolation order (as for `Resampler`).
    fill : `float`, optional
        Fill value.

    Returns
    -------
    outStart : `int`
        Index into ``indices`` of the target pixel of ``result[0]``.
    result : `np.ndarray` of `float`
        Resampled values at the target pixels ``outStart:outStart + len(result)``
        of ``indices``: every one of them that the window can affect.
    """
    halfSize = max(order, 1)
    stop = start + len(values)
    # The target pixels that the window affects have source indices within halfSize of it. Their
    # interpolation windows reach at most 2*halfSize from it, so pad that far (plus one)
    pad = 2*halfSize + 1
    lo = start - pad if numIn is None else max(0, start - pad)
    hi = stop + pad if numIn is None else min(numIn, stop + pad)
    array = np.zeros(hi - lo, dtype=np.float64)
    array[start - lo:stop - lo] = values
    with np.errstate(invalid="ignore"):
        select = (indices >= start - halfSize) & (indices <= stop - 1 + halfSize)
    if not np.any(select):
        return 0, np.zeros(0, dtype=np.float64)
    outIndex = np.flatnonzero(select)
    outStart, outStop = outIndex[0], outIndex[-1] + 1
    return int(outStart), interpolateFluxImpl(array, indices[outStart:outStop] - lo, fill, order)


def interpolate(
    fromWavelength: np.ndarray,
    fromFlux: np.ndarray,
    fromMask: np.ndarray,
    fromVariance: np.ndarray,
    toWavelength: np.ndarray,
    *,
    fill: float = 0.0,
    order: int = 3,
    minWeight: float = 0.1,
    badMask: int = 0,
    fillMask: int = 0,
    numCovar: int = 1,
) -> Struct:
    """Interpolate a spectrum

    Parameters
    ----------
    fromWavelength : `np.ndarray` of `float`
        Source wavelength array.
    fromFlux : `np.ndarray` of `float`
        Source flux array.
    fromMask : `np.ndarray` of `integer`
        Source mask array.
    fromVariance : `np.ndarray` of `float`
        Source variance array.
    toWavelength : `np.ndarray` of `float`
        Target wavelength array.
    fill: `float`, optional
        Fill value.
    order : `int`, optional
        Interpolation order to use. Less than or equal to 1 means linear
        interpolation; higher orders use Lanczos interpolation of the given
        order.
    minWeight : `float`, optional
        The minimum sum of weights to accept; if the sum of weights is less than
        this value, the output will be masked.
    badMask : `int`, optional
        Value to use to identify bad pixels in the input mask. This is used only
        for Lanczos interpolation.
    fillMask : `int`, optional
        Value to use to mark pixels in the output mask that failed
        interpolation.
    numCovar : `int`, optional
        Number of covariance diagonals to interpolate. 1 means just the variance;
        2 means variance and first off-diagonal; etc.

    Returns
    -------
    spectrum : `Struct`
        Struct with fields:
        - wavelength: `numpy.ndarray` of `float`
            Target wavelength array.
        - flux: `numpy.ndarray` of `float`
            Target flux array.
        - mask: `numpy.ndarray` of `int`
            Target mask array.
        - variance: `numpy.ndarray` of `float`
            Target variance array.
        - covariance: `numpy.ndarray` of `float`, shape ``(numCovar, N)``
            Target covariance array.
    """
    resampler = Resampler(fromWavelength, toWavelength, order=order, fill=fill)
    return resampler(fromFlux, fromMask, fromVariance, minWeight=minWeight, badMask=badMask,
                     fillMask=fillMask, numCovar=numCovar)


def interpolateMaskedVariance(
    fromWavelength: np.ndarray,
    fromMask: np.ndarray,
    fromVariance: np.ndarray,
    toWavelength: np.ndarray,
    *,
    fill: float = 0.0,
    order: int = 3,
    minWeight: float = 0.1,
    badMask: int = 0,
    fillMask: int = 0,
    numCovar: int = 1,
) -> Struct:
    """Interpolate the variance and mask of a spectrum

    This is the same as ``interpolate``, including the treatment of bad
    pixels, but without the flux.

    Parameters
    ----------
    fromWavelength : `np.ndarray` of `float`
        Source wavelength array.
    fromMask : `np.ndarray` of `integer`
        Source mask array.
    fromVariance : `np.ndarray` of `float`
        Source variance array.
    toWavelength : `np.ndarray` of `float`
        Target wavelength array.
    fill: `float`, optional
        Fill value.
    order : `int`, optional
        Interpolation order to use. Less than or equal to 1 means linear
        interpolation; higher orders use Lanczos interpolation of the given
        order.
    minWeight : `float`, optional
        The minimum sum of weights to accept; if the sum of weights is less than
        this value, the output will be masked.
    badMask : `int`, optional
        Value to use to identify bad pixels in the input mask. This is used only
        for Lanczos interpolation.
    fillMask : `int`, optional
        Value to use to mark pixels in the output mask that failed
        interpolation.
    numCovar : `int`, optional
        Number of covariance diagonals to interpolate. 1 means just the variance;
        2 means variance and first off-diagonal; etc.

    Returns
    -------
    spectrum : `Struct`
        Struct with fields:
        - wavelength: `numpy.ndarray` of `float`
            Target wavelength array.
        - mask: `numpy.ndarray` of `int`
            Target mask array.
        - variance: `numpy.ndarray` of `float`
            Target variance array.
        - covariance: `numpy.ndarray` of `float`, shape ``(numCovar, N)``
            Target covariance array.
    """
    resampler = Resampler(fromWavelength, toWavelength, order=order, fill=fill)
    return resampler.resampleVariance(fromMask, fromVariance, minWeight=minWeight, badMask=badMask,
                                      fillMask=fillMask, numCovar=numCovar)


def interpolateFlux(
    fromWavelength: np.ndarray,
    fromFlux: np.ndarray,
    toWavelength: np.ndarray,
    *,
    fill: float = 0.0,
    order: int = 3,
) -> np.ndarray:
    """Interpolate a flux-like spectrum

    Basic linear interpolation, suitable for fluxes and flux-like (e.g., maybe
    variances) quantities.

    Parameters
    ----------
    fromWavelength : `np.ndarray` of `float`
        Source wavelength array.
    fromFlux : `np.ndarray` of `float`
        Source flux(-like) array.
    toWavelength : `np.ndarray` of `float`
        Target wavelength array.
    fill : `float`, optional
        Fill value.
    order : `int`, optional
        Interpolation order to use. Less than or equal to 1 means linear
        interpolation; higher orders use Lanczos interpolation of the given
        order.

    Returns
    -------
    toFlux : `numpy.ndarray` of `float`
        Target flux-(like) array.
    """
    return Resampler(fromWavelength, toWavelength, order=order, fill=fill).resampleFlux(fromFlux)


def interpolateCovariance(
    fromWavelength: np.ndarray,
    fromVariance: np.ndarray,
    toWavelength: np.ndarray,
    *,
    fill: float = 0.0,
    order: int = 3,
    numCovar: int = 1,
) -> np.ndarray:
    """Interpolate covariance of a spectrum

    Assumes that the input covariance is diagonal (i.e., just a variance array).

    Parameters
    ----------
    fromWavelength : array-like of `float`
        Source wavelength array.
    fromVariance : array-like of `float`
        Source variance array.
    toWavelength : array-like of `float`
        Target wavelength array.
    fill : `float`, optional
        Fill value.
    order : `int`, optional
        Interpolation order to use. Less than or equal to 1 means linear
        interpolation; higher orders use Lanczos interpolation of the given
        order.
    numCovar : `int`, optional
        Number of covariance diagonals to return. 1 means just the variance;
        2 means variance and first off-diagonal; etc.

    Returns
    -------
    toCovariance : `numpy.ndarray` of `float`
        Target covariance array, shape (N, numCovar) where N is the length of
        the target wavelength array.
    """
    indices = interpolateIndices(fromWavelength, toWavelength)
    return interpolateCovarianceImpl(fromVariance, indices, fill, order, numCovar)


def interpolateVariance(
    fromWavelength: np.ndarray,
    fromVariance: np.ndarray,
    toWavelength: np.ndarray,
    fill: float = 0.0,
    order: int = 3,
) -> np.ndarray:
    """Interpolate a variance-like spectrum

    Like ``interpolateFlux``, except we square all the coefficients that get
    applied.

    Parameters
    ----------
    fromWavelength : array-like of `float`
        Source wavelength array.
    fromVariance : array-like of `float`
        Source variance array.
    toWavelength : array-like of `float`
        Target wavelength array.
    fill : `float`, optional
        Fill value.
    order : `int`, optional
        Interpolation order to use. Less than or equal to 1 means linear
        interpolation; higher orders use Lanczos interpolation of the given
        order.

    Returns
    -------
    toVariance : `numpy.ndarray` of `float`
        Target variance array.
    """
    return interpolateCovariance(
        fromWavelength, fromVariance, toWavelength, fill=fill, order=order, numCovar=1
    )[0]


def interpolateMask(fromWavelength, fromMask, toWavelength, fill=0):
    """Interpolate a mask spectrum

    Linear interpolation for masks.

    Parameters
    ----------
    fromWavelength : array-like of `float`
        Source wavelength array.
    fromMask : array-like of `int`
        Source mask array.
    toWavelength : array-like of `float`
        Target wavelength array.
    fill : `float`, optional
        Fill value.

    Returns
    -------
    toMask : `numpy.ndarray` of `int`
        Target mask array.
    """
    length = len(fromWavelength)
    with np.errstate(invalid="ignore"):
        index = interp1d(fromWavelength, np.arange(length), kind="linear", bounds_error=False,
                         fill_value=-1, copy=True, assume_sorted=True)(toWavelength)
    intIndex = index.astype(int)
    result = np.empty(toWavelength.shape, dtype=fromMask.dtype)
    intIndex[(intIndex == index) & (index > 0)] -= 1  # Linear interpolation takes the index before
    select = (intIndex >= 0) & (intIndex < length - 1)
    result[select] = fromMask[intIndex[select]] | fromMask[intIndex[select] + 1]
    result[~select] = fill
    return result
