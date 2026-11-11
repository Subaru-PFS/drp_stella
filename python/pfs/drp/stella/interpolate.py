import numpy as np
from scipy.interpolate import interp1d

from lsst.pipe.base import Struct

from .math import interpolate as interpolateImpl
from .math import interpolateFlux as interpolateFluxImpl
from .math import interpolateCovariance as interpolateCovarianceImpl
from .spline import SplineD

__all__ = ["calculateDispersion", "Resampler", "resampleWindow", "interpolate", "interpolateFlux",
           "interpolateVariance", "interpolateMask"]


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


class Resampler:
    """Resample rows from one wavelength sampling to another

    This is the resampling used for spectra (`interpolate`, `interpolateFlux`),
    with the source indices of the target wavelengths computed once. The same
    operation can then be applied to several rows sampled like the spectrum:
    its flux, variance and sky, or a line-spread function kernel.

    Parameters
    ----------
    fromWavelength : `np.ndarray` of `float`
        Source wavelength array.
    toWavelength : `np.ndarray` of `float`
        Target wavelength array.
    order : `int`, optional
        Interpolation order to use. Less than or equal to 1 means linear
        interpolation; higher orders use Lanczos interpolation of the given
        order.
    fill : `float`, optional
        Fill value, for target wavelengths outside the source range.
    """

    def __init__(self, fromWavelength: np.ndarray, toWavelength: np.ndarray, *, order: int = 3,
                 fill: float = 0.0):
        self.fromWavelength = fromWavelength
        self.toWavelength = toWavelength
        self.order = order
        self.fill = fill
        self.indices = interpolateIndices(fromWavelength, toWavelength)

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
    ) -> Struct:
        """Resample a spectrum: flux, mask and covariance

        Parameters and results are as for `interpolate`.
        """
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
        return Struct(
            wavelength=self.toWavelength,
            flux=resultFlux,
            mask=resultMask,
            covariance=resultCovariance,
            variance=resultCovariance[0],
        )

    def resampleFlux(self, values: np.ndarray) -> np.ndarray:
        """Resample a flux-like row, with no mask (as `interpolateFlux`)

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
        """Resample a row that is zero except for a short window

        This gives the same values as `resampleFlux` on the full row (with
        ``values`` at source pixels ``start:start + len(values)`` and zeros
        elsewhere), but only does the work near the window. It is used to
        resample a line-spread function kernel.

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
    """Resample a row that is zero except for a short window

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
        Length of the source row (interpolation windows are clipped to it).
        If `None`, the source row is unbounded: ``start`` may be negative,
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
    row = np.zeros(hi - lo, dtype=np.float64)
    row[start - lo:stop - lo] = values
    with np.errstate(invalid="ignore"):
        select = (indices >= start - halfSize) & (indices <= stop - 1 + halfSize)
    if not np.any(select):
        return 0, np.zeros(0, dtype=np.float64)
    outIndex = np.flatnonzero(select)
    outStart, outStop = outIndex[0], outIndex[-1] + 1
    return int(outStart), interpolateFluxImpl(row, indices[outStart:outStop] - lo, fill, order)


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
