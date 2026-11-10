"""The line-spread function of a combination of resampled spectra

A merged (`MergeArmsTask`) or coadded (`CoaddSpectraTask`) spectrum is a
weighted sum of input spectra, each resampled to a common wavelength sampling.
For a line at output position ``c``, the combined spectrum's response at output
pixel ``j`` is::

    K(j; c) = sum_a W_a(j) [R_a k_a(t_a(c))](j)

where, for each input ``a``:

- ``t_a`` maps output pixels to (fractional) input pixels;
- ``k_a(t)`` is the input's LSF for a line at input pixel ``t``;
- ``R_a`` is the input's resampling operator (the same one applied to the flux);
- ``W_a`` are the normalised weights (``sum_a W_a = 1``).

The weights are *smooth*: each input's weights are a robust running mean (in
bins) of the weights that the combination uses for the flux. Per-pixel weights
would give bright lines a different LSF from faint ones. Masks are ignored
here: the LSF is that of a combination with no masked pixels.

Each resampled input kernel is normalised to unit area before it's weighted;
this is equivalent to assuming that the input's gain (e.g., the change of units
from counts per input pixel to flux per output pixel) is constant across the
kernel. Only its gradient across the few pixels of the kernel is neglected.
"""

from dataclasses import dataclass
import warnings
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from lsst.pex.config import Config, Field

from .interpolate import interpolateIndices, resampleWindow
from .lsf import FloatingPoint, Kernel1D, Lsf, NoLsfsError, convolveInBlocks

__all__ = ("IndexMap", "SmoothWeight", "CombinedLsfInput", "CombinedLsf", "CombinedLsfConfig",
           "makeCombinedLsf")


class IndexMap:
    """Map from output pixels to (fractional) input pixels

    This is an approximation to the source indices of a resampling
    (`pfs.drp.stella.interpolate.interpolateIndices`), over the output pixels
    that the input covers: the indices sampled at a set of output pixels and
    interpolated with a cubic spline. A global polynomial isn't adequate: the
    wavelength solutions have structure on scales of hundreds of pixels at the
    1e-3 pixel level, and sharper structure at the ends of the detector.

    Parameters
    ----------
    positions : array_like of `int`
        Output pixels at which the indices are sampled (increasing); the first
        and last define the range covered by the input.
    samples : array_like of `float`
        Source indices at the ``positions``.
    maxError : `float`
        Maximum error of the approximation (input pixels), for information.
    """

    def __init__(self, positions: NDArray[np.int32], samples: NDArray[np.float64], maxError: float):
        self.positions = np.asarray(positions, dtype=np.int32)
        self.samples = np.asarray(samples, dtype=np.float64)
        self.maxError = float(maxError)
        if len(self.positions) != len(self.samples) or len(self.positions) == 0:
            raise ValueError(f"Number of samples ({len(self.samples)}) doesn't match positions "
                             f"({len(self.positions)})")
        self.minIndex = int(self.positions[0])
        self.maxIndex = int(self.positions[-1])
        if len(self.positions) == 1:
            self._spline = None
        else:
            bc = "not-a-knot" if len(self.positions) > 3 else "natural"
            self._spline = CubicSpline(self.positions.astype(np.float64), self.samples, bc_type=bc)

    @classmethod
    def fromIndices(cls, indices: NDArray[np.float64], tolerance: float = 1.0e-4, step: int = 64
                    ) -> Optional["IndexMap"]:
        """Approximate resampling indices

        The indices are sampled every ``step`` output pixels (and at the last
        covered pixel). Then, while the error exceeds ``tolerance`` anywhere,
        the intervals between samples containing such pixels are bisected.

        Parameters
        ----------
        indices : `numpy.ndarray` of `float`
            Source index for each output pixel (NaN where not covered). The
            covered pixels must be contiguous.
        tolerance : `float`
            Maximum acceptable error (input pixels).
        step : `int`
            Initial sampling step (output pixels).

        Returns
        -------
        indexMap : `IndexMap` or `None`
            Approximation, or `None` if no output pixels are covered.
        """
        covered = np.flatnonzero(np.isfinite(indices))
        if covered.size == 0:
            return None
        minIndex, maxIndex = covered[0], covered[-1]
        pixels = np.arange(minIndex, maxIndex + 1, dtype=np.float64)
        values = indices[minIndex:maxIndex + 1]
        positions = np.arange(minIndex, maxIndex + 1, max(step, 1))
        if positions[-1] != maxIndex:
            positions = np.append(positions, maxIndex)
        while True:
            result = cls(positions, indices[positions], 0.0)
            error = np.abs(result(pixels) - values)
            result.maxError = float(np.max(error))
            if result.maxError < tolerance:
                return result
            # Bisect the intervals containing pixels that fail
            interval = np.searchsorted(positions, pixels[error >= tolerance].astype(int), side="right") - 1
            interval = np.unique(np.clip(interval, 0, len(positions) - 2))
            lower = positions[interval]
            upper = positions[interval + 1]
            split = upper - lower > 1
            if not np.any(split):
                return result
            positions = np.union1d(positions, (lower[split] + upper[split])//2)

    @classmethod
    def fromWavelength(cls, inWavelength: NDArray[np.float64], outWavelength: NDArray[np.float64],
                       tolerance: float = 1.0e-4, step: int = 64) -> Optional["IndexMap"]:
        """Approximate the resampling indices between wavelength samplings

        Parameters
        ----------
        inWavelength : `numpy.ndarray` of `float`
            Wavelength of each input pixel.
        outWavelength : `numpy.ndarray` of `float`
            Wavelength of each output pixel.
        tolerance : `float`
            Maximum acceptable error (input pixels).
        step : `int`
            Initial sampling step (output pixels).

        Returns
        -------
        indexMap : `IndexMap` or `None`
            Approximation, or `None` if no output pixels are covered.
        """
        return cls.fromIndices(interpolateIndices(inWavelength, outWavelength), tolerance, step)

    def __call__(self, pixels: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
        """Evaluate: return input pixel for output pixel(s); NaN outside the range"""
        if np.isscalar(pixels):
            if not self.contains(pixels):
                return np.nan
            return float(self._spline(pixels)) if self._spline is not None else float(self.samples[0])
        pixels = np.asarray(pixels, dtype=np.float64)
        if self._spline is None:
            result = np.full(pixels.shape, self.samples[0])
        else:
            result = self._spline(pixels)
        result[(pixels < self.minIndex) | (pixels > self.maxIndex)] = np.nan
        return result

    def derivative(self, pixel: float) -> float:
        """Return the derivative (input pixels per output pixel)"""
        if self._spline is None:
            return 1.0
        return float(self._spline(pixel, 1))

    def contains(self, pixel: float) -> bool:
        """Is the output pixel covered by the input?"""
        return self.minIndex <= pixel <= self.maxIndex

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.positions, self.samples, self.maxError)


class SmoothWeight:
    """Smooth weight of an input as a function of output pixel

    The weight is linearly interpolated between the centers of bins, and zero
    outside the range covered by the input.

    Parameters
    ----------
    minIndex, maxIndex : `int`
        Range of output pixels (inclusive) covered by the input.
    binSize : `int`
        Size of bins (output pixels); the bins start at ``minIndex``, and the
        last may be partial.
    values : array_like of `float`
        Weight for each bin.
    """

    def __init__(self, minIndex: int, maxIndex: int, binSize: int, values: NDArray[np.float32]):
        self.minIndex = int(minIndex)
        self.maxIndex = int(maxIndex)
        self.binSize = int(binSize)
        self.values = np.asarray(values, dtype=np.float32)
        starts = np.arange(len(self.values))*self.binSize + self.minIndex
        stops = np.minimum(starts + self.binSize - 1, self.maxIndex)
        self.centers = 0.5*(starts + stops)

    @classmethod
    def fit(
        cls,
        weight: NDArray[np.float64],
        select: NDArray[np.bool_],
        minIndex: int,
        maxIndex: int,
        binSize: int = 50,
        rejNSigma: float = 3.0,
        rejIter: int = 2,
    ) -> "SmoothWeight":
        """Fit to the per-pixel weights of the combination

        In each bin, the weights of the selected pixels are clipped about the
        median, with a robust standard deviation (1.4826*MAD), and averaged.
        Clipping removes, e.g., bright sky and object lines, whose photon noise
        lowers the weight. Bins with no selected pixels are interpolated from
        their neighbours.

        Parameters
        ----------
        weight : `numpy.ndarray` of `float`
            Weight for each output pixel.
        select : `numpy.ndarray` of `bool`
            Output pixels where the weight was used.
        minIndex, maxIndex : `int`
            Range of output pixels (inclusive) covered by the input.
        binSize : `int`
            Size of bins (output pixels).
        rejNSigma : `float`
            Rejection threshold (robust standard deviations).
        rejIter : `int`
            Number of rejection iterations.

        Returns
        -------
        self : `SmoothWeight`
            Smooth weight.
        """
        num = maxIndex - minIndex + 1
        numBins = (num + binSize - 1)//binSize
        values = np.full(numBins*binSize, np.nan, dtype=np.float64)
        use = select[minIndex:maxIndex + 1] & np.isfinite(weight[minIndex:maxIndex + 1])
        values[:num][use] = weight[minIndex:maxIndex + 1][use]
        values = values.reshape(numBins, binSize)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN bins
            for _ in range(rejIter):
                median = np.nanmedian(values, axis=1)
                sigma = 1.4826*np.nanmedian(np.abs(values - median[:, np.newaxis]), axis=1)
                reject = np.abs(values - median[:, np.newaxis]) > rejNSigma*sigma[:, np.newaxis]
                reject &= (sigma > 0)[:, np.newaxis]
                values[reject] = np.nan
            mean = np.nanmean(values, axis=1)
        good = np.isfinite(mean)
        self = cls(minIndex, maxIndex, binSize, np.zeros(numBins, dtype=np.float32))
        if np.any(good):
            self.values[:] = np.interp(self.centers, self.centers[good], mean[good])
        return self

    @classmethod
    def constant(cls, minIndex: int, maxIndex: int, value: float = 1.0) -> "SmoothWeight":
        """Construct a constant weight over a range"""
        return cls(minIndex, maxIndex, maxIndex - minIndex + 1, [value])

    def __call__(self, pixels: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the weight at output pixels"""
        pixels = np.asarray(pixels, dtype=np.float64)
        result = np.interp(pixels, self.centers, self.values.astype(np.float64))
        result[(pixels < self.minIndex) | (pixels > self.maxIndex)] = 0.0
        return result

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.minIndex, self.maxIndex, self.binSize, self.values)


@dataclass
class CombinedLsfInput:
    """An input to a `CombinedLsf`

    Parameters
    ----------
    lsf : `pfs.drp.stella.Lsf`
        Line-spread function of the input, on its own pixels.
    indexMap : `IndexMap`
        Map from output pixels to input pixels.
    weight : `SmoothWeight`
        Weight of the input as a function of output pixel.
    order : `int`
        Order of the resampling (as for `pfs.drp.stella.interpolate.Resampler`).
    """
    lsf: Lsf
    indexMap: IndexMap
    weight: SmoothWeight
    order: int

    def computeWindow(self, center: float) -> tuple[int, NDArray[np.float64]] | None:
        """Return the input's LSF for a line at an output position, resampled

        Parameters
        ----------
        center : `float`
            Output pixel of the line.

        Returns
        -------
        start : `int`
            Output pixel of ``values[0]``.
        values : `numpy.ndarray` of `float`
            Resampled LSF, normalised to unit sum.

        Or `None` if the result is empty.
        """
        inCenter = self.indexMap(center)
        inStart, inValues = self.lsf.computeWindow(inCenter)
        inStop = inStart + len(inValues)
        # Output pixels whose interpolation windows can reach the input window; pad generously
        halfSize = max(self.order, 1)
        scale = 1.0/max(abs(self.indexMap.derivative(center)), 1.0e-3)  # output pixels per input pixel
        lo = int(np.floor(center + (inStart - halfSize - 1 - inCenter)*scale)) - 2
        hi = int(np.ceil(center + (inStop + halfSize - inCenter)*scale)) + 3
        lo = max(lo, self.indexMap.minIndex)
        hi = min(hi, self.indexMap.maxIndex + 1)
        if hi <= lo:
            return None
        indices = self.indexMap(np.arange(lo, hi, dtype=np.float64))
        offset, values = resampleWindow(inValues, inStart, indices, self.lsf.length, self.order)
        norm = np.sum(values)
        if values.size == 0 or not np.isfinite(norm) or norm <= 0:
            return None
        return lo + offset, values/norm


class CombinedLsf(Lsf):
    """Line-spread function of a weighted combination of resampled spectra

    See the module documentation for the model.

    Parameters
    ----------
    length : `int`
        Length of the combined spectrum.
    inputs : iterable of `CombinedLsfInput`
        Inputs to the combination.
    """

    def __init__(self, length: int, inputs: Sequence[CombinedLsfInput]):
        super().__init__(length)
        self.inputs = list(inputs)

    def _getInputs(self, center: float) -> list[CombinedLsfInput]:
        """Return the inputs that cover an output position"""
        return [inp for inp in self.inputs if inp.indexMap.contains(center)]

    def computeWindow(self, center: float) -> tuple[int, NDArray[np.float64]]:
        """Return the LSF for a line at a (sub-pixel) position

        Parameters
        ----------
        center : `float`, pixels
            Position of the line.

        Returns
        -------
        start : `int`
            Index of ``values[0]`` in the full array.
        values : `numpy.ndarray` of `float`
            LSF, normalised to unit sum.

        Raises
        ------
        NoLsfsError
            If there are no inputs covering ``center``.
        """
        inputs = self._getInputs(center)
        windows = [inp.computeWindow(center) for inp in inputs]
        used = [(inp, ww) for inp, ww in zip(inputs, windows) if ww is not None]
        if not used:
            raise NoLsfsError(f"No LSFs at {center}")
        start = min(ww[0] for _, ww in used)
        stop = max(ww[0] + len(ww[1]) for _, ww in used)
        pixels = np.arange(start, stop, dtype=np.float64)
        numerator = np.zeros(stop - start, dtype=np.float64)
        denominator = np.zeros(stop - start, dtype=np.float64)
        unweighted = np.zeros(stop - start, dtype=np.float64)
        for inp, (ss, values) in used:
            weight = inp.weight(pixels)
            denominator += weight
            select = slice(ss - start, ss - start + len(values))
            numerator[select] += weight[select]*values
            unweighted[select] += values
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(denominator > 0, numerator/denominator, 0.0)
        norm = np.sum(result)
        if not np.isfinite(norm) or norm <= 0:
            # No weight anywhere (e.g., all inputs fully masked): use equal weights
            result = unweighted
            norm = np.sum(result)
        return start, result/norm

    def computeArray(self, center: float) -> NDArray[np.float64]:
        """Return an array with the LSF inserted at the nominated position

        Parameters
        ----------
        center : `float`, pixels
            Position at which to insert the LSF.

        Returns
        -------
        array : `numpy.ndarray` of `float`, shape ``(length,)``
            Array containing the LSF inserted.
        """
        start, values = self.computeWindow(center)
        array = np.zeros(self.length, dtype=np.float64)
        stop = min(start + len(values), self.length)
        array[start:stop] = values[:stop - start]
        return array

    def computeKernel(self, center: float) -> Kernel1D:
        """Return a kernel for the nominated position

        The kernel is that for a line centered on the pixel nearest to
        ``center`` (as for `ExtractionLsf`); use ``computeArray`` or
        ``computeWindow`` for sub-pixel positions.

        Parameters
        ----------
        center : `float`, pixels
            Position at which to evaluate the LSF.

        Returns
        -------
        kernel : `Kernel1D`
            Kernel with a model of the LSF at the nominated position.

        Raises
        ------
        NoLsfsError
            If there are no inputs covering ``center``.
        """
        pixel = int(np.floor(center + 0.5))
        start, values = self.computeWindow(pixel)
        if not start <= pixel < start + len(values):
            raise NoLsfsError(f"LSF at {pixel} does not cover its center")
        return Kernel1D(values, pixel - start, normalize=False)  # Already normalised

    def convolve(self, array: NDArray[FloatingPoint], sampling: int = 100) -> NDArray[FloatingPoint]:
        """Convolve an array by the LSF

        Parameters
        ----------
        array : `numpy.ndarray`, floating-point, shape ``(M,)``
            Array to convolve.
        sampling : `int`
            Sampling length for convolution, pixels.

        Returns
        -------
        convolved : `numpy.ndarray`, floating-point, shape ``(M,)``
            Convolved array.
        """
        return convolveInBlocks(self, array, sampling)

    def warp(self, inWavelength: NDArray[FloatingPoint], outWavelength: NDArray[FloatingPoint]
             ) -> "CombinedLsf":
        """Warp the LSF to a new wavelength frame

        The result is this LSF, linearly interpolated to the new frame: a
        `CombinedLsf` with a single input. This is correct both for a coarser
        frame (it's a further resampling) and for a finer one (e.g., to convolve
        a model spectrum before resampling it to this frame).

        Parameters
        ----------
        inWavelength : `numpy.ndarray` of `float`
            Wavelength array in the same frame as for ``lsf``.
        outWavelength : `numpy.ndarray` of `float`
            Wavelength array in the target frame.

        Returns
        -------
        warpedLsf : `CombinedLsf`
            LSF in the warped frame.
        """
        if len(inWavelength) != self.length:
            raise RuntimeError(f"Length mismatch between LSF ({self.length}) and wavelength "
                               f"({len(inWavelength)})")
        indexMap = IndexMap.fromWavelength(np.asarray(inWavelength), np.asarray(outWavelength))
        if indexMap is None:
            return CombinedLsf(len(outWavelength), [])
        weight = SmoothWeight.constant(indexMap.minIndex, indexMap.maxIndex)
        return CombinedLsf(len(outWavelength), [CombinedLsfInput(self, indexMap, weight, 1)])

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.length, self.inputs)


class CombinedLsfConfig(Config):
    """Configuration for constructing a `CombinedLsf`"""
    weightBinSize = Field(
        dtype=int,
        default=50,
        doc="Size of bins (output pixels) for the smooth weight of each input: the flux weights are averaged "
            "(with clipping) in bins, and linearly interpolated between them",
    )
    weightRejNSigma = Field(
        dtype=float,
        default=3.0,
        doc="Threshold (robust standard deviations, 1.4826*MAD) for clipping weights when averaging, "
            "e.g. at bright lines whose own photon noise lowers the weight",
    )
    weightRejIter = Field(dtype=int, default=2, doc="Number of clipping iterations when averaging weights")
    indexTolerance = Field(
        dtype=float,
        default=1.0e-4,
        doc="Maximum error (input pixels) of the approximation to the resampling of each input (the "
            "source indices, sampled and interpolated with a cubic spline)",
    )
    indexStep = Field(
        dtype=int,
        default=64,
        doc="Initial sampling step (output pixels) of the resampling indices; intervals are bisected where "
            "the error exceeds indexTolerance",
    )


def makeCombinedLsf(
    lsfList: Sequence[Lsf | None],
    inWavelengths: Sequence[NDArray[np.float64]],
    outWavelength: NDArray[np.float64],
    weights: Sequence[NDArray[np.float64]],
    selects: Sequence[NDArray[np.bool_]],
    order: int,
    config: CombinedLsfConfig,
) -> CombinedLsf:
    """Construct the LSF of a combination of resampled spectra

    Parameters
    ----------
    lsfList : iterable of `pfs.drp.stella.Lsf` or `None`
        LSF of each input, on its own pixels; `None` for none (the input is
        then left out).
    inWavelengths : iterable of `numpy.ndarray` of `float`
        Wavelength of each pixel, for each input.
    outWavelength : `numpy.ndarray` of `float`
        Wavelength of each pixel of the combination.
    weights : iterable of `numpy.ndarray` of `float`
        Weight of each output pixel (after resampling), for each input, as
        used for the flux.
    selects : iterable of `numpy.ndarray` of `bool`
        Output pixels where the weights were used, for each input.
    order : `int`
        Order of the resampling (as for `pfs.drp.stella.interpolate.Resampler`).
    config : `CombinedLsfConfig`
        Configuration.

    Returns
    -------
    lsf : `CombinedLsf`
        Line-spread function of the combination.
    """
    inputs = []
    for lsf, inWavelength, weight, select in zip(lsfList, inWavelengths, weights, selects):
        if lsf is None:
            continue
        indexMap = IndexMap.fromWavelength(inWavelength, outWavelength, config.indexTolerance,
                                           config.indexStep)
        if indexMap is None:
            continue
        smooth = SmoothWeight.fit(weight, select, indexMap.minIndex, indexMap.maxIndex,
                                  config.weightBinSize, config.weightRejNSigma, config.weightRejIter)
        inputs.append(CombinedLsfInput(lsf, indexMap, smooth, order))
    return CombinedLsf(len(outWavelength), inputs)
