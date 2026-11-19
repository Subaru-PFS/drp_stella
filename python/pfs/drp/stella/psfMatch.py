from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.fft
import scipy.ndimage

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ip.diffim import diffimLib
from lsst.ip.diffim.makeKernelBasisList import makeKernelBasisList
from lsst.ip.diffim.psfMatch import PsfMatchConfigDF
from lsst.ip.diffim.psfMatch import PsfMatchTask as BasePsfMatchTask
from lsst.utils.timer import timeMethod
from pfs.datamodel import FiberStatus, PfsConfig, TargetType

from .DetectorMap import DetectorMap
from .referenceLine import ReferenceLineSet, ReferenceLineStatus
from .repair import maskLines
from .selectFibers import SelectFibersTask
from .warp import warpImage

if TYPE_CHECKING:
    import matplotlib

__all__ = (
    "PsfMatchConfig",
    "PsfMatchResult",
    "PsfMatchTask",
    "peakSignalToNoise",
    "subtractRunningMedian",
    "measureOffset",
    "shiftMaskedImage",
    "offsetKernel",
    "warpMaskedImage",
    "calculateWarpShifts",
    "plotSpatialKernel",
    "calculateFirstMoments",
    "plotFirstMoments",
    "plotPsfMatchResult",
)


# Variance assigned to pixels excluded from the kernel fit, giving them
# effectively zero weight. This is finite (rather than ``inf``) so that
# convolving the variance doesn't produce ``inf*0 = NaN``.
EXCLUDED_VARIANCE = 1.0e30


def peakSignalToNoise(maskedImage: afwImage.MaskedImage, ignore: Optional[np.ndarray] = None) -> float:
    """Return the peak (per-pixel) signal-to-noise ratio of a masked image

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        Image for which to calculate the peak signal-to-noise.
    ignore : `numpy.ndarray` of `bool`, optional
        Pixels to ignore (same shape as ``maskedImage``).

    Returns
    -------
    peakSnr : `float`
        Maximum value of ``image/sqrt(variance)`` over all finite,
        positive-variance pixels (other than those in ``ignore``).
        ``-inf`` if there are no such pixels.
    """
    image = maskedImage.image.array
    variance = maskedImage.variance.array
    good = np.isfinite(image) & np.isfinite(variance) & (variance > 0)
    if ignore is not None:
        good &= ~ignore
    if not good.any():
        return -np.inf
    return float(np.max(image[good] / np.sqrt(variance[good])))


def subtractRunningMedian(maskedImage: afwImage.MaskedImage, numRows: int) -> afwImage.MaskedImage:
    """Return a copy of an image with a running median along columns removed

    This removes the continuum of near-vertical spectral traces, leaving the
    lines. Non-finite pixels are treated as zero when calculating the median,
    and are left unchanged.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        Image from which to remove the running median; not modified.
    numRows : `int`
        Number of rows over which to calculate the running median.

    Returns
    -------
    filtered : `lsst.afw.image.MaskedImage`
        Copy of ``maskedImage`` with the running median subtracted.
    """
    filtered = maskedImage.clone()
    array = filtered.image.array
    finite = np.isfinite(array)
    median = scipy.ndimage.median_filter(np.where(finite, array, 0.0), size=(numRows, 1), mode="nearest")
    array[finite] -= median[finite]
    return filtered


def measureOffset(
    source: afwImage.MaskedImage,
    target: afwImage.MaskedImage,
    maxOffsetX: int,
    maxOffsetY: int,
    ignoreMask: int = 0,
    weights: Optional[Sequence[np.ndarray]] = None,
    minAliasSeparation: int = 3,
) -> pipeBase.Struct:
    """Measure the integer offset between two images by cross-correlation

    The cross-correlation ``sum_x weight(x) source(x) target(x + offset)`` is
    calculated with FFTs for each of the ``weights``, and normalized by its
    maximum within ``maxOffsetX`` and ``maxOffsetY``. The offset with the
    largest minimum (over the ``weights``) correlation is returned, so that
    ``target(x + offset)`` best matches ``source(x)``.

    For fiber spectrograph images, the comb of fibers makes the correlation
    nearly periodic in x (with the fiber pitch). The ``weights`` can be used
    to emphasize the breaks in the comb, which break the aliasing (see
    `PsfMatchTask._makeOffsetWeights`). Because a fiber with a break on one
    side suppresses only the alias in one direction, the fibers with breaks
    on each side are best given separate weights: the true offset correlates
    well with both, but each alias correlates poorly with one of them.

    Parameters
    ----------
    source, target : `lsst.afw.image.MaskedImage`
        Images to correlate; must have the same dimensions. These should
        have any continuum removed (e.g., with `subtractRunningMedian`), so
        that the correlation is determined by the lines.
    maxOffsetX, maxOffsetY : `int`
        Maximum offset (pixels) to consider in each dimension.
    ignoreMask : `int`, optional
        Mask bits for pixels to ignore (set to zero) in the correlation.
    weights : sequence of `numpy.ndarray` of `float`, optional
        Weights for each pixel of ``source``; each produces a separate
        correlation. If not provided, uniform weights are used.
    minAliasSeparation : `int`, optional
        Minimum separation in x (pixels) from the peak for the
        ``aliasRatio`` diagnostic.

    Returns
    -------
    dx, dy : `int`
        Offset (pixels) of ``target`` relative to ``source``.
    peak : `float`
        Correlation coefficient at the peak (with the first of the
        ``weights``).
    aliasRatio : `float`
        Ratio of the largest (combined) correlation at offsets at least
        ``minAliasSeparation`` from the peak in x (at any y offset) to the
        peak; values approaching unity indicate that the x offset is poorly
        determined. NaN if there are no such offsets.
    """
    if source.getDimensions() != target.getDimensions():
        raise ValueError(
            f"Dimensions differ: source {source.getDimensions()} vs target {target.getDimensions()}"
        )

    def prepare(image: afwImage.MaskedImage) -> np.ndarray:
        array = image.image.array.astype(np.float32)
        good = np.isfinite(array) & ((image.mask.array & ignoreMask) == 0)
        if not np.any(good):
            raise RuntimeError("No good pixels for measuring the offset")
        result = np.zeros_like(array)
        result[good] = array[good] - np.median(array[good])
        return result

    sourceArray = prepare(source)
    targetArray = prepare(target)
    shape = sourceArray.shape
    targetTransform = scipy.fft.rfft2(targetArray)
    targetNorm = np.sqrt(np.sum(targetArray.astype(float) ** 2))
    # Restrict to the allowed offsets; negative offsets wrap around
    yOffsets = np.arange(-maxOffsetY, maxOffsetY + 1)
    xOffsets = np.arange(-maxOffsetX, maxOffsetX + 1)
    yIndices = yOffsets % shape[0]
    xIndices = xOffsets % shape[1]

    if weights is None:
        weights = [None]
    combined = None
    coefficients = []
    for weight in weights:
        weighted = sourceArray if weight is None else sourceArray * weight.astype(np.float32)
        correlation = scipy.fft.irfft2(np.conj(scipy.fft.rfft2(weighted)) * targetTransform, s=shape)
        window = correlation[np.ix_(yIndices, xIndices)]
        windowMax = window.max()
        if windowMax <= 0:
            raise RuntimeError("No positive correlation for measuring the offset")
        norm = np.sqrt(np.sum(weighted.astype(float) ** 2)) * targetNorm
        coefficients.append(window / norm if norm > 0 else np.full_like(window, np.nan))
        normalized = window / windowMax
        combined = normalized if combined is None else np.minimum(combined, normalized)

    yIndex, xIndex = np.unravel_index(np.argmax(combined), combined.shape)
    best = combined[yIndex, xIndex]
    alias = np.abs(xOffsets - xOffsets[xIndex]) >= minAliasSeparation
    aliasRatio = float(combined[:, alias].max() / best) if np.any(alias) else np.nan
    return pipeBase.Struct(
        dx=int(xOffsets[xIndex]),
        dy=int(yOffsets[yIndex]),
        peak=float(coefficients[0][yIndex, xIndex]),
        aliasRatio=aliasRatio,
    )


def shiftMaskedImage(
    maskedImage: afwImage.MaskedImage, dx: int, dy: int
) -> Tuple[afwImage.MaskedImage, np.ndarray]:
    """Shift an image by an integer offset, keeping the same bounding box

    The result is ``shifted(x, y) = maskedImage(x + dx, y + dy)``. Pixels
    shifted in from outside the image have zero image, variance of
    ``EXCLUDED_VARIANCE`` and the ``NO_DATA`` mask plane set.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        Image to shift; not modified.
    dx, dy : `int`
        Offset (pixels).

    Returns
    -------
    shifted : `lsst.afw.image.MaskedImage`
        Shifted image.
    missing : `numpy.ndarray` of `bool`
        Pixels shifted in from outside the image.
    """
    height, width = maskedImage.image.array.shape
    shifted = maskedImage.clone()
    missing = np.ones((height, width), dtype=bool)
    targetY = slice(max(0, -dy), min(height, height - dy))
    targetX = slice(max(0, -dx), min(width, width - dx))
    sourceY = slice(max(0, dy), min(height, height + dy))
    sourceX = slice(max(0, dx), min(width, width + dx))
    missing[targetY, targetX] = False
    for target, source in (
        (shifted.image.array, maskedImage.image.array),
        (shifted.mask.array, maskedImage.mask.array),
        (shifted.variance.array, maskedImage.variance.array),
    ):
        target[targetY, targetX] = source[sourceY, sourceX]
    shifted.image.array[missing] = 0.0
    shifted.variance.array[missing] = EXCLUDED_VARIANCE
    shifted.mask.array[missing] = shifted.mask.getPlaneBitMask("NO_DATA")
    return shifted, missing


def warpMaskedImage(
    maskedImage: afwImage.MaskedImage,
    fromDetectorMap: DetectorMap,
    toDetectorMap: DetectorMap,
    warpingKernel: str = "lanczos3",
) -> Tuple[afwImage.MaskedImage, np.ndarray]:
    """Warp an image from the frame of one detectorMap to that of another

    Each point (fiberId, wavelength) is moved from its position in
    ``fromDetectorMap`` to its position in ``toDetectorMap``, using
    `pfs.drp.stella.warp.warpImage`. Pixels without data (mapped from outside
    the input) have zero image, variance of ``EXCLUDED_VARIANCE`` and the
    ``NO_DATA`` mask plane set.

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        Image to warp, in the frame of ``fromDetectorMap``; not modified.
    fromDetectorMap : `DetectorMap`
        Mapping of fiberId,wavelength to x,y for ``maskedImage``.
    toDetectorMap : `DetectorMap`
        Mapping of fiberId,wavelength to x,y for the output frame.
    warpingKernel : `str`, optional
        Name of the warping kernel (e.g., ``lanczos3``).

    Returns
    -------
    warped : `lsst.afw.image.MaskedImage`
        Warped image, with the bounding box of ``toDetectorMap``.
    missing : `numpy.ndarray` of `bool`
        Pixels without data.
    """
    warped = warpImage(maskedImage, fromDetectorMap, toDetectorMap, warpingKernel)
    missing = ~np.isfinite(warped.image.array) | ~np.isfinite(warped.variance.array)
    warped.image.array[missing] = 0.0
    warped.variance.array[missing] = EXCLUDED_VARIANCE
    warped.mask.array[missing] = warped.mask.getPlaneBitMask("NO_DATA")
    return warped, missing


def calculateWarpShifts(
    fromDetectorMap: DetectorMap,
    toDetectorMap: DetectorMap,
    fiberId: np.ndarray,
    row: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the shift of the warp between two detectorMaps

    For each point (``fiberId``, ``row``) on a trace in ``fromDetectorMap``,
    we calculate the shift of the point with the same fiberId and wavelength
    in ``toDetectorMap``.

    Parameters
    ----------
    fromDetectorMap, toDetectorMap : `DetectorMap`
        Mappings of fiberId,wavelength to x,y.
    fiberId : `numpy.ndarray` of `int`
        Fiber identifiers.
    row : `numpy.ndarray` of `float`
        Rows on the trace in ``fromDetectorMap``.

    Returns
    -------
    dx, dy : `numpy.ndarray` of `float`
        Shift (``toDetectorMap`` minus ``fromDetectorMap``).
    """
    fiberId = np.asarray(fiberId, dtype=np.int32)
    row = np.asarray(row, dtype=float)
    wavelength = fromDetectorMap.findWavelength(fiberId, row).astype(float)
    xFrom = fromDetectorMap.getXCenter(fiberId, row)
    toPoints = toDetectorMap.findPoint(fiberId, wavelength)
    return toPoints[:, 0] - xFrom, toPoints[:, 1] - row


def offsetKernel(
    kernel: afwMath.LinearCombinationKernel, dx: int, dy: int
) -> afwMath.LinearCombinationKernel:
    """Fold an integer offset into a spatially varying kernel

    Each basis kernel is embedded in a larger kernel (enlarged by twice the
    offset in each dimension, so it remains centered) at a position that
    moves the kernel's effect by the offset. If ``kernel`` matches ``source``
    to ``shifted``, where ``shifted(x) = target(x + offset)`` (see
    `shiftMaskedImage`), then the result matches ``source`` to ``target``,
    and its first moment differs from that of ``kernel`` by ``-offset``
    (because `lsst.afw.math.convolve` is a correlation). The spatial model is
    unchanged, so it is evaluated at the position in ``target`` rather than
    in ``shifted``; this is negligible for a model that varies slowly
    compared to the offset.

    Parameters
    ----------
    kernel : `lsst.afw.math.LinearCombinationKernel`
        Spatially varying kernel, with odd dimensions.
    dx, dy : `int`
        Offset (pixels).

    Returns
    -------
    offsetKernel : `lsst.afw.math.LinearCombinationKernel`
        Kernel including the offset.
    """
    if dx == 0 and dy == 0:
        return kernel
    width, height = kernel.getDimensions()
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError(f"Kernel dimensions must be odd; got {width}x{height}")
    newWidth = width + 2 * abs(dx)
    newHeight = height + 2 * abs(dy)
    # Kernel pixel k (relative to the center) moves to k - offset
    xStart = abs(dx) - dx
    yStart = abs(dy) - dy
    image = afwImage.ImageD(kernel.getDimensions())
    basisList = []
    for basis in kernel.getKernelList():
        basis.computeImage(image, False)
        newImage = afwImage.ImageD(newWidth, newHeight)
        newImage.array[yStart : yStart + height, xStart : xStart + width] = image.array
        basisList.append(afwMath.FixedKernel(newImage))
    newKernel = afwMath.LinearCombinationKernel(basisList, kernel.getSpatialFunctionList())
    newKernel.setSpatialParameters(kernel.getSpatialParameters())
    return newKernel


class PsfMatchResult(pipeBase.Struct):
    """Results of `PsfMatchTask.run`

    This is a `lsst.pipe.base.Struct` that supports pickling. The
    ``spatialSolution`` and ``kernelCellSet`` elements are internals of the
    fit that cannot be persisted, so they are set to `None` when unpickled.
    """

    _transient = ("spatialSolution", "kernelCellSet")

    def __reduce__(self):
        # Struct is a SimpleNamespace, whose __reduce__ ignores __getstate__
        state = self.__dict__.copy()
        for name in self._transient:
            if name in state:
                state[name] = None
        return (type(self), (), state)


class PsfMatchConfig(pexConfig.Config):
    """Configuration for PsfMatchTask"""

    kernel = pexConfig.ConfigChoiceField(
        doc="Kernel type",
        typemap=dict(DF=PsfMatchConfigDF),
        default="DF",
    )

    @property
    def dfKernel(self) -> PsfMatchConfigDF:
        """Return the kernel configuration if ``kernel.name == "DF"``, else raise"""
        if self.kernel.name != "DF":
            raise RuntimeError(f"kernel.name is {self.kernel.name}, not DF")
        return self.kernel["DF"]

    xStampSize = pexConfig.Field(
        dtype=int,
        default=64,
        check=lambda xx: xx > 0,
        doc="Width of each tiling stamp (pixels)",
    )
    yStampSize = pexConfig.Field(
        dtype=int,
        default=64,
        check=lambda yy: yy > 0,
        doc="Height of each tiling stamp (pixels)",
    )
    minPeakSignalToNoise = pexConfig.Field(
        dtype=float,
        default=5.0,
        check=lambda snr: snr >= 0,
        doc="Minimum peak signal-to-noise for a stamp to be used",
    )
    maxStamps = pexConfig.Field(
        dtype=int,
        optional=True,
        default=None,
        check=lambda num: num is None or num > 0,
        doc="Maximum number of stamps to use (None: use all that pass selection)",
    )
    seed = pexConfig.Field(
        dtype=int,
        default=0,
        doc="Fallback seed for random stamp subsampling, used only if the source exposure "
        "has no usable visitInfo.id (see PsfMatchTask.run's seed parameter)",
    )
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        default=["SAT", "NO_DATA", "EDGE"],
        doc="Mask planes that disqualify a stamp",
    )
    excludeMaskPlanes = pexConfig.ListField(
        dtype=str,
        default=["BAD", "CR", "INTRP"],
        doc="Mask planes for pixels to exclude from the kernel fit (grown by half the kernel size to "
        "account for the convolution), without disqualifying the stamp",
    )
    doContinuumFilter = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Fit the kernel to images from which a running median (of continuumFilterRows rows) has been "
        "subtracted in each column, so that the kernel is determined by the lines rather than the "
        "continuum (which may vary differently, e.g., for sky)? The kernel is applied to the original "
        "image",
    )
    continuumFilterRows = pexConfig.Field(
        dtype=int,
        default=41,
        check=lambda rows: rows > 0,
        doc="Number of rows for the running median used when doContinuumFilter",
    )
    lineMaskStatus = pexConfig.ListField(
        dtype=str,
        default=["BAD"],
        doc="ReferenceLineStatus flags for lines to exclude from the kernel fit (only when a line list "
        "is provided); a line is excluded if any of these flags is set",
    )
    lineMaskRadius = pexConfig.Field(
        dtype=int,
        default=3,
        check=lambda radius: radius >= 0,
        doc="Radius (pixels) around excluded lines to exclude from the kernel fit; this is grown by "
        "half the kernel size to account for the convolution",
    )
    maskFibers = pexConfig.ConfigurableField(
        target=SelectFibersTask,
        doc="Select fibers to mask (only when a pfsConfig is provided): all pixels (lines and continuum) "
        "on these fibers are excluded from the kernel fit",
    )
    fiberMaskRadius = pexConfig.Field(
        dtype=int,
        default=5,
        check=lambda radius: radius >= 0,
        doc="Half-width (pixels, in the spatial dimension) around the trace of excluded fibers to exclude "
        "from the kernel fit; this is grown by half the kernel size to account for the convolution",
    )
    doLineScale = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Before fitting the kernel (only when a line list is provided), scale each line in the source "
        "image by its flux ratio (target/source, relative to the median over all lines), so that the kernel "
        "sum isn't affected by changes in the relative strengths of lines between exposures (e.g., for "
        "sky)? The kernel is applied to the original image. This is intended for use with "
        "doContinuumFilter, as any continuum around the lines is also scaled",
    )
    lineRatioHalfHeight = pexConfig.Field(
        dtype=int,
        default=6,
        check=lambda size: size >= 0,
        doc="Half-height (pixels, in the dispersion dimension) of the box around each line on each fiber "
        "over which to sum the flux when measuring line flux ratios; this should be large enough that the "
        "sum is insensitive to the PSF",
    )
    lineRatioHalfWidth = pexConfig.Field(
        dtype=int,
        default=3,
        check=lambda size: size >= 0,
        doc="Half-width (pixels, in the spatial dimension) of the box around each line on each fiber over "
        "which to sum the flux when measuring line flux ratios",
    )
    lineRatioMinSignalToNoise = pexConfig.Field(
        dtype=float,
        default=20.0,
        check=lambda snr: snr >= 0,
        doc="Minimum signal-to-noise ratio of a line's summed flux (in both source and target) for its flux "
        "ratio to be used",
    )
    lineRatioRejection = pexConfig.Field(
        dtype=float,
        default=5.0,
        check=lambda rej: rej > 0,
        doc="Rejection threshold (in robust standard deviations from the median) for line flux ratios",
    )
    lineScaleHalfHeight = pexConfig.Field(
        dtype=int,
        default=8,
        check=lambda size: size >= 0,
        doc="Half-height (pixels, in the dispersion dimension) of the region around each line that is scaled "
        "by its flux ratio; pixels farther from any line with a good flux ratio are not scaled",
    )
    warpingKernel = pexConfig.Field(
        dtype=str,
        default="lanczos3",
        doc="Warping kernel for warping the target onto the frame of the source (when detectorMaps for both "
        "are provided to run)",
    )
    doOffset = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Measure the integer offset between source and target by cross-correlation, and shift the "
        "target by it before fitting the kernel? The offset is folded back into the output kernel, so "
        "kernel.active.kernelSize need only span the residual shift (and the PSF change)",
    )
    maxOffsetX = pexConfig.Field(
        dtype=int,
        default=4,
        check=lambda offset: offset >= 0,
        doc="Maximum integer offset (pixels) in x (the spatial dimension) to consider when doOffset. The "
        "comb of fibers makes the correlation nearly periodic in x, so this should be small compared to "
        "the fiber pitch, as a backstop to offsetEdgeFraction",
    )
    maxOffsetY = pexConfig.Field(
        dtype=int,
        default=15,
        check=lambda offset: offset >= 0,
        doc="Maximum integer offset (pixels) in y (the dispersion dimension) to consider when doOffset",
    )
    offsetEdgeFraction = pexConfig.Field(
        dtype=float,
        default=0.5,
        check=lambda fraction: 0 <= fraction < 1,
        doc="Fraction of the total weight in the offset cross-correlation to give to pixels on 'edge' "
        "fibers (those next to a dark or missing fiber, or at the end of the slit), which break the "
        "aliasing from the comb of fibers; 0 means no up-weighting. Requires a detectorMap; a pfsConfig "
        "identifies fibers without flux",
    )
    offsetDarkFiberStatus = pexConfig.ListField(
        dtype=str,
        default=["BROKENFIBER", "BLOCKED", "BLACKSPOT", "UNILLUMINATED"],
        doc="FiberStatus values for fibers without flux, which break the comb of fibers (for identifying "
        "edge fibers when measuring the offset)",
    )
    offsetDarkTargetType = pexConfig.ListField(
        dtype=str,
        default=["ENGINEERING", "BLACKSPOT"],
        doc="TargetType values for fibers without flux, which break the comb of fibers (for identifying "
        "edge fibers when measuring the offset)",
    )
    lineMaskPlane = pexConfig.Field(
        dtype=str,
        default="PSFMATCH_EXCLUDED",
        doc="Mask plane for pixels excluded from the kernel fit; must be in kernel.active.badMaskPlanes "
        "and not in badMaskPlanes",
    )

    def setDefaults(self):
        super().setDefaults()
        self.dfKernel.kernelSize = 9
        self.dfKernel.scaleByFwhm = False
        self.dfKernel.constantVarianceWeighting = False
        self.dfKernel.useRegularization = True
        # The fibers form a regular comb in x, which leaves the kernel's x structure degenerate: kernel
        # weight can move by one fiber pitch without changing the model. Only breaks in the comb
        # (missing or dark fibers, and the ends of the slit) constrain that, but most stamps contain no
        # break. With PCA, each stamp's kernel is solved independently, so the stamps without a break
        # contribute arbitrary values along the degenerate direction to the PCA basis, which biases the
        # kernel's x first moment and its x shape. Without PCA, the stamps' normal equations are
        # combined directly in the spatial fit, so the stamps with breaks constrain the degenerate
        # direction everywhere, while all stamps constrain everything else.
        self.dfKernel.usePcaForSpatialKernel = False
        self.dfKernel.numPrincipalComponents = 10
        self.dfKernel.spatialKernelOrder = 1
        self.dfKernel.spatialBgOrder = 3
        self.dfKernel.fitForBackground = True
        self.dfKernel.sizeCellX = self.xStampSize
        self.dfKernel.sizeCellY = self.yStampSize
        self.dfKernel.badMaskPlanes = list(self.dfKernel.badMaskPlanes) + [self.lineMaskPlane]
        self.maskFibers.targetType = ["FLUXSTD"]
        self.maskFibers.fiberStatus = []  # Regardless of status

    def validate(self):
        super().validate()
        if self.lineMaskPlane not in self.kernel.active.badMaskPlanes:
            raise pexConfig.FieldValidationError(
                PsfMatchConfig.lineMaskPlane,
                self,
                f"lineMaskPlane ({self.lineMaskPlane}) must be in kernel.active.badMaskPlanes",
            )
        if self.lineMaskPlane in self.badMaskPlanes:
            raise pexConfig.FieldValidationError(
                PsfMatchConfig.lineMaskPlane,
                self,
                f"lineMaskPlane ({self.lineMaskPlane}) must not be in badMaskPlanes",
            )
        if self.lineMaskPlane in self.excludeMaskPlanes:
            raise pexConfig.FieldValidationError(
                PsfMatchConfig.lineMaskPlane,
                self,
                f"lineMaskPlane ({self.lineMaskPlane}) must not be in excludeMaskPlanes",
            )
        both = set(self.excludeMaskPlanes) & set(self.badMaskPlanes)
        if both:
            raise pexConfig.FieldValidationError(
                PsfMatchConfig.excludeMaskPlanes,
                self,
                f"Mask planes {sorted(both)} are in both excludeMaskPlanes and badMaskPlanes",
            )
        for name in self.lineMaskStatus:
            if name not in ReferenceLineStatus.__members__:
                raise pexConfig.FieldValidationError(
                    PsfMatchConfig.lineMaskStatus, self, f"Unrecognized ReferenceLineStatus: {name}"
                )


class PsfMatchTask(BasePsfMatchTask):
    """Match the PSF of one fiber-spectrograph exposure to another

    This differs from the standard `lsst.ip.diffim` PSF-matching tasks
    (which are designed for direct imaging) in how candidate stamps are
    selected: instead of running a star finder to locate a modest number of
    isolated point sources, we don't have (or want) star-like candidates, so
    we tile the entire overlap of the two exposures with stamps, discard
    stamps with low peak signal-to-noise, and (optionally) randomly discard
    a subset of the remaining stamps to bound the number used (and hence the
    runtime) while keeping the retained stamps spread across the image.

    We use the delta-function kernel basis with regularization
    (`lsst.ip.diffim.psfMatch.PsfMatchConfigDF`), and fit the spatial model
    without PCA (see the comment in `PsfMatchConfig.setDefaults`); the
    actual kernel solve is inherited unchanged from the base class's
    ``_solve`` method.

    If detectorMaps for both ``source`` and ``target`` are provided to
    ``run``, the ``target`` is warped onto the frame of the ``source`` (each
    point fiberId,wavelength is moved from its position in the ``target`` to
    its position in the ``source``) before matching, so the kernel need only
    represent the PSF change (and any error in the detectorMaps). The kernel
    then doesn't include the shift of the warp: the caller must add it (see
    `calculateWarpShifts`). Resampling with a Lanczos kernel barely changes
    the PSF of a well-sampled image (the second moment changes by less than
    0.1% for a Gaussian with sigma >= 1 pixel), so the kernel remains
    appropriate for the unwarped images.

    Otherwise, the two exposures are assumed to share a pixel grid, and any
    relative shift between them (e.g., from differing trace or wavelength
    solutions) is absorbed by the fitted kernel itself, which may therefore
    end up off-center. With ``config.doOffset`` (the default), the integer part of
    a large shift is measured first by cross-correlating the images, the
    ``target`` is shifted by it (so the frame of ``source``, in which the
    ``detectorMap`` applies, is retained), and the offset is folded back
    into the fitted kernel (which is enlarged accordingly). The comb of
    fibers makes the cross-correlation nearly periodic in x, so the x offset
    is limited to ``config.maxOffsetX`` (small compared to the fiber pitch),
    and, if a ``detectorMap`` is provided, the fibers next to breaks in the
    comb (which distinguish the true offset from its aliases) are up-weighted
    (``config.offsetEdgeFraction``). Either way,
    ``kernel.active.kernelSize`` must be large enough to span the remaining
    shift (which may vary over the image) plus the PSF change.

    If a line list (and detectorMap) is provided to ``run``, the pixels
    around lines with any of the ``config.lineMaskStatus`` flags set are
    excluded from the kernel fit. This is useful for sky exposures, where
    the strengths of lines may vary. The exclusion is achieved by setting a
    very large variance (giving zero weight in the fit) and the
    ``config.lineMaskPlane`` mask plane (excluding them from candidate
    assessment) on copies of the stamps; the inputs are not modified.
    Similarly, if a ``pfsConfig`` is provided to ``run`` for either (or
    both) of ``source`` and ``target``, the pixels on fibers selected by
    the ``maskFibers`` subtask in either are excluded, since the spectra of
    bright objects can differ between exposures. By default, only
    ``FLUXSTD`` fibers are masked: many science targets are faint and so
    essentially sky, and fibers that are not where they were intended
    (e.g., ``NOTCONVERGED``) still see sky. Pixels with any of the
    ``config.excludeMaskPlanes`` mask planes set (e.g., cosmic rays) are
    excluded in the same way, rather than disqualifying the entire stamp
    (which is what happens for ``config.badMaskPlanes``).

    The continuum and the lines may vary differently between exposures
    (e.g., for sky), in which case no kernel can match both, and the fitted
    background can't follow the structure of the continuum between and on
    the fibers. With ``config.doContinuumFilter`` (the default), the kernel
    is fit to images from which the continuum has been removed with a
    running median along each column, so that the kernel is determined by
    the lines. The kernel is then applied to the original ``source``, so the
    continuum difference remains in the matched exposure, to be dealt with
    separately.

    Similarly, the relative strengths of the lines may vary between
    exposures (e.g., for sky, where different bands vary differently). If a
    line list is provided and ``config.doLineScale`` is set (the default),
    the flux ratio (target/source) of each line that isn't excluded is
    measured by summing over all fibers, and the lines in (a copy of) the
    ``source`` used for the fit are scaled by their ratio relative to the
    median. The kernel then accounts for the median ratio and the PSF
    change, and its sum isn't driven by the line-to-line variations, which
    remain in the matched exposure.

    Notes
    -----
    For additional debug logging from the inherited kernel solver, set:

    .. code-block:: py

        import lsst.utils.logging as logUtils
        logUtils.trace_set_at("lsst.ip.diffim", 4)
    """

    ConfigClass = PsfMatchConfig
    _DefaultName = "psfMatch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kConfig = self.config.kernel.active
        self.makeSubtask("maskFibers")
        # The kernel solver's candidate statistics require that all of its badMaskPlanes exist.
        afwImage.Mask.addMaskPlane(self.config.lineMaskPlane)

    @timeMethod
    def run(
        self,
        source: afwImage.Exposure,
        target: afwImage.Exposure,
        *,
        seed: Optional[int] = None,
        refLines: Optional[ReferenceLineSet] = None,
        detectorMap: Optional[DetectorMap] = None,
        sourcePfsConfig: Optional[PfsConfig] = None,
        targetPfsConfig: Optional[PfsConfig] = None,
        sourceDetectorMap: Optional[DetectorMap] = None,
        targetDetectorMap: Optional[DetectorMap] = None,
    ) -> PsfMatchResult:
        """Match the PSF of ``source`` to ``target``

        If ``sourceDetectorMap`` and ``targetDetectorMap`` are provided, the
        ``target`` is warped onto the frame of the ``source`` before matching;
        otherwise, ``source`` and ``target`` must already share a pixel grid.

        Parameters
        ----------
        source : `lsst.afw.image.Exposure`
            Exposure to be convolved to match ``target``.
        target : `lsst.afw.image.Exposure`
            Exposure to which ``source`` is matched.
        seed : `int`, optional
            Seed for the random subsampling of stamps down to
            ``config.maxStamps``. If not provided, we use ``source``'s
            ``visitInfo.id`` if available, else ``config.seed``.
        refLines : `ReferenceLineSet`, optional
            Line list. If provided, lines with any of the
            ``config.lineMaskStatus`` flags set are excluded from the
            kernel fit, and (if ``config.doLineScale``) the other lines are
            scaled by their flux ratios before fitting. Requires
            ``detectorMap``.
        detectorMap : `DetectorMap`, optional
            Mapping of fiberId,wavelength to x,y, used to locate
            ``refLines`` and fiber traces (which are assumed to be at the
            same position in both ``source`` and ``target``).
        sourcePfsConfig, targetPfsConfig : `pfs.datamodel.PfsConfig`, optional
            Top-end configurations for ``source`` and ``target``. If
            either is provided, pixels on fibers selected by the
            ``maskFibers`` subtask in any of those provided are excluded from
            the kernel fit. Requires ``detectorMap``.
        sourceDetectorMap, targetDetectorMap : `DetectorMap`, optional
            Mappings of fiberId,wavelength to x,y for ``source`` and
            ``target`` (e.g., fit to each exposure). If provided (both are
            required), the ``target`` is warped onto the frame of the
            ``source`` before matching, so the kernel doesn't include the
            shift of the warp (see `calculateWarpShifts`).

        Returns
        -------
        result : `PsfMatchResult`
            Results of the PSF matching, which supports pickling (but see
            below regarding ``spatialSolution`` and ``kernelCellSet``),
            with the following elements:

        matchedExposure : `lsst.afw.image.Exposure`
            ``source``, convolved by the psf-matching kernel with the
            fitted background added, so as to resemble ``target`` (or, if
            warped, ``warpedTarget``).
        psfMatchingKernel : `lsst.afw.math.LinearCombinationKernel`
            Spatially varying kernel that matches ``source`` to ``target``,
            including the ``offset``.
        backgroundModel : `lsst.afw.math.Function2D`
            Spatially varying background difference between ``source`` (as
            matched) and ``target``.
        spatialSolution : `lsst.ip.diffim.SpatialKernelSolution`
            Kernel solution produced by the fit. Not pickled: `None` when
            unpickled.
        kernelCellSet : `lsst.afw.math.SpatialCellSet`
            Cell set of candidates used in the fit. Not pickled: `None`
            when unpickled.
        basisList : `list` of `lsst.afw.math.Kernel`
            Kernel basis functions used in the fit.
        metadata : `lsst.daf.base.PropertySet`
            Metadata accumulated while fitting.
        selection : `lsst.pipe.base.Struct`
            Stamp-selection diagnostics: ``numTiles``, ``numRejectedMask``,
            ``numRejectedSnr``, ``numCandidates``, ``numUsed``,
            ``numLinesExcluded`` (`None` unless ``refLines`` was
            provided), ``numFibersExcluded`` (`None` unless a pfsConfig
            was provided) and ``excludedFraction`` (fraction of pixels
            excluded; `None` if neither applies).
        lineRatios : `lsst.pipe.base.Struct` or `None`
            Line flux ratios measured for ``config.doLineScale`` (see
            ``_measureLineRatios``), or `None` if no line list was provided
            or ``config.doLineScale`` is not set.
        offset : `tuple` of `int`
            Integer offset (x, y) of ``target`` relative to ``source`` that
            was removed before fitting the kernel, and folded back into
            ``psfMatchingKernel``; ``(0, 0)`` unless ``config.doOffset``.
        warpedTarget : `lsst.afw.image.MaskedImage` or `None`
            ``target`` warped onto the frame of the ``source``, or `None` if
            not warped.
        """

        self.log.info(
            "Building kernel candidates from %s source and %s target exposures",
            source.getDimensions(),
            target.getDimensions(),
        )
        if seed is None:
            visitInfo = source.getInfo().getVisitInfo()
            if visitInfo is not None:
                seed = visitInfo.id
                self.log.debug("Using seed=%d from source visitInfo.id", seed)
            else:
                seed = self.config.seed
                self.log.debug("Using fallback seed=%d from config.seed", seed)

        bbox = source.getBBox()
        bbox.clip(target.getBBox())

        sourceFit = source.maskedImage
        targetMaskedImage = target.maskedImage
        warpedTarget = None
        warpMissing = None
        if sourceDetectorMap is not None or targetDetectorMap is not None:
            if sourceDetectorMap is None or targetDetectorMap is None:
                raise ValueError("Both sourceDetectorMap and targetDetectorMap are required for warping")
            self.log.info("Warping target onto the frame of the source")
            warpedTarget, warpMissing = warpMaskedImage(
                targetMaskedImage, targetDetectorMap, sourceDetectorMap, self.config.warpingKernel
            )
            if warpedTarget.getBBox() != targetMaskedImage.getBBox():
                raise RuntimeError(
                    f"Warped target bbox {warpedTarget.getBBox()} doesn't match target bbox "
                    f"{targetMaskedImage.getBBox()}"
                )
            targetMaskedImage = warpedTarget
        targetFit = targetMaskedImage
        if self.config.doContinuumFilter:
            self.log.info(
                "Fitting kernel to images with %d-row running median subtracted",
                self.config.continuumFilterRows,
            )
            sourceFit = subtractRunningMedian(sourceFit, self.config.continuumFilterRows)
            targetFit = subtractRunningMedian(targetFit, self.config.continuumFilterRows)

        # Exclusions of lines and fibers are in the frame of the source (where the detectorMap applies)
        exclude = None
        numLinesExcluded = None
        numFibersExcluded = None
        excludedFraction = None
        if refLines is not None:
            if detectorMap is None:
                raise ValueError("detectorMap is required when refLines is provided")
            lineExclusion = self._makeLineExclusion(bbox, refLines, detectorMap)
            exclude = lineExclusion.exclude
            numLinesExcluded = lineExclusion.numLines
        pfsConfigList = [
            pfsConfig for pfsConfig in (sourcePfsConfig, targetPfsConfig) if pfsConfig is not None
        ]
        if pfsConfigList:
            if detectorMap is None:
                raise ValueError("detectorMap is required when a pfsConfig is provided")
            fiberExclusion = self._makeFiberExclusion(bbox, detectorMap, pfsConfigList)
            exclude = fiberExclusion.exclude if exclude is None else exclude | fiberExclusion.exclude
            numFibersExcluded = fiberExclusion.numFibers
        if warpMissing is not None:
            exclude = warpMissing if exclude is None else exclude | warpMissing

        # Shift the target onto the source, so the kernel need only span the residual shift. Everything
        # that uses the detectorMap (lines, fibers) is then in the frame of the source.
        offset = (0, 0)
        if self.config.doOffset:
            offset = self._measureOffset(bbox, sourceFit, targetFit, detectorMap, pfsConfigList, exclude)
            dx, dy = offset
            if dx != 0 or dy != 0:
                targetMaskedImage, missing = shiftMaskedImage(targetMaskedImage, dx, dy)
                if self.config.doContinuumFilter:
                    targetFit = shiftMaskedImage(targetFit, dx, dy)[0]
                else:
                    targetFit = targetMaskedImage
                exclude = missing if exclude is None else exclude | missing

        maskExclusion = self._makeMaskExclusion(bbox, source.maskedImage, targetMaskedImage)
        if maskExclusion is not None:
            exclude = maskExclusion if exclude is None else exclude | maskExclusion
        if exclude is not None:
            excludedFraction = float(exclude.mean())
            self.log.info("Excluding %.1f%% of pixels from kernel fit", 100 * excludedFraction)

        lineRatios = None
        if refLines is not None and self.config.doLineScale:
            goodLines = refLines[~self._selectExcludedLines(refLines)]
            lineRatios = self._measureLineRatios(bbox, sourceFit, targetFit, goodLines, detectorMap, exclude)
            if np.any(lineRatios.good):
                sourceFit = self._scaleLines(bbox, sourceFit, lineRatios, detectorMap)

        buildResult = self._buildCellSet(sourceFit, targetFit, seed=seed, exclude=exclude)
        self.log.info(
            "Built %d kernel candidates from %d tiles (%d rejected: mask, %d rejected: S/N)",
            buildResult.numUsed,
            buildResult.numTiles,
            buildResult.numRejectedMask,
            buildResult.numRejectedSnr,
        )

        basisList = makeKernelBasisList(self.kConfig)
        spatialSolution, spatialKernel, spatialBackground = self._solve(buildResult.kernelCellSet, basisList)
        self.log.info("Solved for spatial PSF-matching kernel using %d candidates", buildResult.numUsed)
        # The background model is in the frame of the source, but it's smooth, so the offset doesn't matter.
        spatialKernel = offsetKernel(spatialKernel, *offset)

        convolutionControl = afwMath.ConvolutionControl()
        convolutionControl.setDoNormalize(False)
        matchedMaskedImage = afwImage.MaskedImageF(source.getBBox())
        afwMath.convolve(matchedMaskedImage, source.maskedImage, spatialKernel, convolutionControl)
        matchedMaskedImage += spatialBackground

        matchedExposure = afwImage.ExposureF(matchedMaskedImage, source.getWcs())
        matchedExposure.info.id = source.info.id
        matchedExposure.setFilter(source.getFilter())
        matchedExposure.getInfo().setVisitInfo(source.getInfo().getVisitInfo())
        if source.getDetector() is not None:
            matchedExposure.setDetector(source.getDetector())
        if target.getPhotoCalib() is not None:
            matchedExposure.setPhotoCalib(target.getPhotoCalib())
        if target.hasPsf():
            matchedExposure.setPsf(target.getPsf())

        self.log.info("PSF matching complete")
        return PsfMatchResult(
            matchedExposure=matchedExposure,
            psfMatchingKernel=spatialKernel,
            backgroundModel=spatialBackground,
            spatialSolution=spatialSolution,
            kernelCellSet=buildResult.kernelCellSet,
            basisList=basisList,
            metadata=self.metadata,
            selection=pipeBase.Struct(
                numTiles=buildResult.numTiles,
                numRejectedMask=buildResult.numRejectedMask,
                numRejectedSnr=buildResult.numRejectedSnr,
                numCandidates=buildResult.numCandidates,
                numUsed=buildResult.numUsed,
                numLinesExcluded=numLinesExcluded,
                numFibersExcluded=numFibersExcluded,
                excludedFraction=excludedFraction,
            ),
            lineRatios=lineRatios,
            offset=offset,
            warpedTarget=warpedTarget,
        )

    def _measureOffset(
        self,
        bbox: geom.Box2I,
        sourceFit: afwImage.MaskedImage,
        targetFit: afwImage.MaskedImage,
        detectorMap: Optional[DetectorMap],
        pfsConfigList: Iterable[PfsConfig],
        exclude: Optional[np.ndarray],
    ) -> Tuple[int, int]:
        """Measure the integer offset of the target relative to the source

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the region of interest.
        sourceFit, targetFit : `lsst.afw.image.MaskedImage`
            Images to correlate (with any continuum removed).
        detectorMap : `DetectorMap`, optional
            Mapping of fiberId,wavelength to x,y for the source, used to
            up-weight edge fibers (see ``_makeOffsetWeights``).
        pfsConfigList : iterable of `pfs.datamodel.PfsConfig`
            Top-end configurations, identifying fibers without flux.
        exclude : `numpy.ndarray` of `bool`, optional
            Pixels of the source to exclude from the correlation.

        Returns
        -------
        dx, dy : `int`
            Offset (pixels) of the target relative to the source.
        """
        ignoreMask = sourceFit.mask.getPlaneBitMask(
            list(self.config.badMaskPlanes) + list(self.config.excludeMaskPlanes)
        )
        width, height = bbox.getDimensions()
        weights = [np.ones((height, width), dtype=np.float32)]
        if detectorMap is not None and self.config.offsetEdgeFraction > 0:
            weights = self._makeOffsetWeights(bbox, detectorMap, pfsConfigList)
        if exclude is not None:
            for weight in weights:
                weight[exclude] = 0.0
        result = measureOffset(
            sourceFit, targetFit, self.config.maxOffsetX, self.config.maxOffsetY, ignoreMask, weights
        )
        self.log.info(
            "Measured offset of target relative to source: dx=%d dy=%d (peak %.3f, x alias ratio %.3f)",
            result.dx,
            result.dy,
            result.peak,
            result.aliasRatio,
        )
        if abs(result.dx) == self.config.maxOffsetX or abs(result.dy) == self.config.maxOffsetY:
            self.log.warning(
                "Offset (%d, %d) is at the limit of maxOffsetX=%d, maxOffsetY=%d",
                result.dx,
                result.dy,
                self.config.maxOffsetX,
                self.config.maxOffsetY,
            )
        if np.isfinite(result.aliasRatio) and result.aliasRatio > 0.9:
            self.log.warning("Offset in x is poorly determined: alias ratio %.3f", result.aliasRatio)
        self.metadata["offsetX"] = result.dx
        self.metadata["offsetY"] = result.dy
        self.metadata["offsetAliasRatio"] = result.aliasRatio
        return result.dx, result.dy

    def _makeOffsetWeights(
        self, bbox: geom.Box2I, detectorMap: DetectorMap, pfsConfigList: Iterable[PfsConfig] = ()
    ) -> List[np.ndarray]:
        """Make weights for the offset cross-correlation

        The fibers form a regular comb in x, so a shift of one fiber pitch
        matches each fiber with its neighbour, and the cross-correlation is
        nearly periodic in x. Only breaks in the comb distinguish the true
        offset from the aliases: a fiber next to a break lands on nothing
        when shifted by a pitch towards the break. We make two sets of
        weights, one for fibers with a break on the left and one for those
        with a break on the right, each up-weighting the pixels on those
        "edge" fibers so that they carry ``config.offsetEdgeFraction`` of the
        total weight on fibers. Each suppresses the aliases in one direction
        (see `measureOffset`).

        A break is any of:

        - a fiber without flux: one with a ``fiberStatus`` in
          ``config.offsetDarkFiberStatus`` or a ``targetType`` in
          ``config.offsetDarkTargetType`` in any of the ``pfsConfigList``,
          or (if any pfsConfigs are provided) not in all of them;
        - a jump in ``fiberId`` between neighbouring fibers (fibers missing
          from the ``detectorMap``; this includes the chip gap);
        - the end of the slit, or the edge of the image.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the region of interest.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y for the source.
        pfsConfigList : iterable of `pfs.datamodel.PfsConfig`, optional
            Top-end configurations, identifying fibers without flux.

        Returns
        -------
        weights : `list` of `numpy.ndarray` of `float`
            Weights for each pixel, with shape matching ``bbox``: for fibers
            with a break on the left, and on the right (or a single set of
            uniform weights, if there are no edge fibers).
        """
        width, height = bbox.getDimensions()
        uniform = [np.ones((height, width), dtype=np.float32)]
        fiberId = detectorMap.fiberId
        if len(fiberId) < 3:
            return uniform
        rows = np.arange(height)
        middle = np.full(len(fiberId), bbox.getMinY() + 0.5 * (height - 1), dtype=float)
        xMiddle = detectorMap.getXCenter(fiberId, middle)
        onImage = np.isfinite(xMiddle) & (xMiddle >= bbox.getMinX()) & (xMiddle <= bbox.getMaxX())
        order = np.argsort(np.where(onImage, xMiddle, np.inf))
        fiberId = fiberId[order]
        xMiddle = xMiddle[order]
        onImage = onImage[order]
        pitch = np.median(np.abs(np.diff(xMiddle[onImage])))
        radius = max(int(np.floor(0.5 * pitch)), 0)

        lit = onImage.copy()
        darkStatus = [FiberStatus[name] for name in self.config.offsetDarkFiberStatus]
        darkType = [TargetType[name] for name in self.config.offsetDarkTargetType]
        for pfsConfig in pfsConfigList:
            sorter = np.argsort(pfsConfig.fiberId)
            position = np.searchsorted(pfsConfig.fiberId, fiberId, sorter=sorter)
            position = np.clip(position, 0, len(sorter) - 1)
            index = sorter[position]
            present = pfsConfig.fiberId[index] == fiberId
            dark = ~present
            dark[present] = np.isin(pfsConfig.fiberStatus[index[present]], darkStatus) | np.isin(
                pfsConfig.targetType[index[present]], darkType
            )
            lit &= ~dark

        # Breaks between neighbouring fibers
        isGap = ~onImage[:-1] | ~onImage[1:] | (np.abs(np.diff(fiberId)) != 1)
        leftBreak = np.concatenate(([True], ~lit[:-1] | isGap))
        rightBreak = np.concatenate((~lit[1:] | isGap, [True]))

        # Pixel coordinates of each trace
        traces = []
        for ff in fiberId:
            xCenter = detectorMap.getXCenter(ff, (rows + bbox.getMinY()).astype(float))
            good = np.isfinite(xCenter)
            columns = np.floor(xCenter[good] + 0.5).astype(int) - bbox.getMinX()
            traceRows = rows[good]
            inside = (columns >= 0) & (columns < width)
            traces.append((traceRows[inside], columns[inside]))

        leftEdge = lit & leftBreak
        rightEdge = lit & rightBreak

        def traceMask(select: np.ndarray) -> np.ndarray:
            mask = np.zeros((height, width), dtype=bool)
            offsets = np.arange(-radius, radius + 1)
            for ii in np.flatnonzero(select):
                traceRows, columns = traces[ii]
                cols = columns[:, np.newaxis] + offsets[np.newaxis, :]
                rr = np.broadcast_to(traceRows[:, np.newaxis], cols.shape)
                inside = (cols >= 0) & (cols < width)
                mask[rr[inside], cols[inside]] = True
            return mask

        litMask = traceMask(lit)
        self.log.info(
            "Offset weights: %d/%d fibers dark; %d with a break on the left, %d on the right "
            "(pitch %.2f pixels)",
            (~lit).sum(),
            len(fiberId),
            leftEdge.sum(),
            rightEdge.sum(),
            pitch,
        )
        fraction = self.config.offsetEdgeFraction
        weights = []
        for edge in (leftEdge, rightEdge):
            edgeMask = traceMask(edge)
            numEdge = edgeMask.sum()
            numOther = (litMask & ~edgeMask).sum()
            if numEdge == 0 or numOther == 0:
                continue
            weight = np.ones((height, width), dtype=np.float32)
            weight[edgeMask] = max(fraction / (1 - fraction) * numOther / numEdge, 1.0)
            weights.append(weight)
        return weights if weights else uniform

    def _makeMaskExclusion(
        self, bbox: geom.Box2I, source: afwImage.MaskedImage, target: afwImage.MaskedImage
    ) -> Optional[np.ndarray]:
        """Determine which masked pixels to exclude from the kernel fit

        Pixels with any of the ``config.excludeMaskPlanes`` mask planes set
        in either ``source`` or ``target`` are excluded. The excluded region
        is grown by half the kernel size, so that excluded pixels in the
        ``source`` don't contribute (through the convolution) to pixels that
        are used in the fit.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the region of interest.
        source, target : `lsst.afw.image.MaskedImage`
            Images whose masks are to be checked.

        Returns
        -------
        exclude : `numpy.ndarray` of `bool`, or `None`
            Pixels to exclude, with shape matching ``bbox``; or `None` if
            there are no such pixels.
        """
        planes = [plane for plane in self.config.excludeMaskPlanes if plane in source.mask.getMaskPlaneDict()]
        if not planes:
            return None
        bitmask = source.mask.getPlaneBitMask(planes)
        flagged = ((source.mask[bbox].array | target.mask[bbox].array) & bitmask) != 0
        if not flagged.any():
            return None
        radius = self.kConfig.kernelSize // 2
        exclude = scipy.ndimage.binary_dilation(flagged, structure=np.ones((2 * radius + 1, 2 * radius + 1)))
        self.log.info(
            "Excluding %.1f%% of pixels from kernel fit due to mask planes %s", 100 * exclude.mean(), planes
        )
        return exclude

    def _selectExcludedLines(self, refLines: ReferenceLineSet) -> np.ndarray:
        """Select lines to exclude from the kernel fit

        Parameters
        ----------
        refLines : `ReferenceLineSet`
            Line list.

        Returns
        -------
        select : `numpy.ndarray` of `bool`
            Whether each line has any of the ``config.lineMaskStatus``
            flags set.
        """
        flags = 0
        for name in self.config.lineMaskStatus:
            flags |= ReferenceLineStatus[name]
        return (refLines.status & flags) != 0

    def _makeLineExclusion(
        self, bbox: geom.Box2I, refLines: ReferenceLineSet, detectorMap: DetectorMap
    ) -> pipeBase.Struct:
        """Determine which pixels to exclude from the kernel fit

        Pixels within ``config.lineMaskRadius`` of lines with any of the
        ``config.lineMaskStatus`` flags set are excluded. The radius is
        grown by half the kernel size, so that excluded pixels in the
        ``source`` don't contribute (through the convolution) to pixels
        that are used in the fit.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the region of interest.
        refLines : `ReferenceLineSet`
            Line list.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.

        Returns
        -------
        exclude : `numpy.ndarray` of `bool`
            Pixels to exclude, with shape matching ``bbox``.
        numLines : `int`
            Number of lines excluded.
        fraction : `float`
            Fraction of pixels excluded.
        """
        select = self._selectExcludedLines(refLines)
        numLines = int(select.sum())

        mask = afwImage.Mask(bbox)
        radius = self.config.lineMaskRadius + self.kConfig.kernelSize // 2
        maskLines(mask, detectorMap, refLines[select], radius, maskPlane=self.config.lineMaskPlane)
        exclude = (mask.array & mask.getPlaneBitMask(self.config.lineMaskPlane)) != 0
        fraction = float(exclude.mean())
        self.log.info(
            "Excluding %d/%d lines (%s) from kernel fit: %.1f%% of pixels",
            numLines,
            len(refLines),
            "|".join(self.config.lineMaskStatus),
            100 * fraction,
        )
        return pipeBase.Struct(exclude=exclude, numLines=numLines, fraction=fraction)

    def _makeFiberExclusion(
        self, bbox: geom.Box2I, detectorMap: DetectorMap, pfsConfigList: Iterable[PfsConfig]
    ) -> pipeBase.Struct:
        """Determine which pixels on masked fibers to exclude from the fit

        All pixels (lines and continuum) within ``config.fiberMaskRadius``
        (in the spatial dimension) of the trace of each fiber in the
        ``detectorMap`` that is selected by the ``maskFibers`` subtask in
        any of the ``pfsConfigList`` are excluded. The radius is grown by
        half the kernel size, so that excluded pixels in the ``source``
        don't contribute (through the convolution) to pixels that are used
        in the fit.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the region of interest.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        pfsConfigList : iterable of `pfs.datamodel.PfsConfig`
            Top-end configurations.

        Returns
        -------
        exclude : `numpy.ndarray` of `bool`
            Pixels to exclude, with shape matching ``bbox``.
        numFibers : `int`
            Number of fibers excluded.
        fraction : `float`
            Fraction of pixels excluded.
        """
        mask = np.zeros(len(detectorMap), dtype=bool)
        for pfsConfig in pfsConfigList:
            mask |= np.isin(detectorMap.fiberId, self.maskFibers.run(pfsConfig).fiberId)
        excludeFiberId = detectorMap.fiberId[mask]
        numFibers = len(excludeFiberId)

        width, height = bbox.getDimensions()
        exclude = np.zeros((height, width), dtype=bool)
        radius = self.config.fiberMaskRadius + self.kConfig.kernelSize // 2
        rows = np.arange(height)
        offsets = np.arange(-radius, radius + 1)
        for fiberId in excludeFiberId:
            xCenter = detectorMap.getXCenter(fiberId, (rows + bbox.getMinY()).astype(float))
            good = np.isfinite(xCenter)
            columns = np.floor(xCenter[good] + 0.5).astype(int) - bbox.getMinX()
            columns = columns[:, np.newaxis] + offsets[np.newaxis, :]
            fiberRows = np.broadcast_to(rows[good][:, np.newaxis], columns.shape)
            inside = (columns >= 0) & (columns < width)
            exclude[fiberRows[inside], columns[inside]] = True

        fraction = float(exclude.mean())
        self.log.info(
            "Excluding %d/%d masked fibers from kernel fit: %.1f%% of pixels",
            numFibers,
            len(detectorMap),
            100 * fraction,
        )
        return pipeBase.Struct(exclude=exclude, numFibers=numFibers, fraction=fraction)

    def _measureLineRatios(
        self,
        bbox: geom.Box2I,
        source: afwImage.MaskedImage,
        target: afwImage.MaskedImage,
        refLines: ReferenceLineSet,
        detectorMap: DetectorMap,
        exclude: Optional[np.ndarray] = None,
    ) -> pipeBase.Struct:
        """Measure the flux ratio (target/source) of each line

        The flux of each line is summed over a box (with half-dimensions
        ``config.lineRatioHalfWidth`` and ``config.lineRatioHalfHeight``)
        around the line on each fiber in the ``detectorMap``, and over all
        fibers, ignoring pixels that are excluded, non-finite, or have any
        of the ``config.badMaskPlanes`` mask planes set. Boxes that extend
        beyond ``bbox`` are not used. Summing over all fibers assumes that
        the flux ratio of each line is common to all fibers (as for sky).

        Ratios are rejected if either sum has a signal-to-noise ratio below
        ``config.lineRatioMinSignalToNoise``, or if the ratio is more than
        ``config.lineRatioRejection`` robust standard deviations from the
        median.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the region of interest.
        source, target : `lsst.afw.image.MaskedImage`
            Images in which to measure the line fluxes.
        refLines : `ReferenceLineSet`
            Lines to measure.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        exclude : `numpy.ndarray` of `bool`, optional
            Pixels to ignore, with shape matching ``bbox``.

        Returns
        -------
        wavelength : `numpy.ndarray` of `float`
            Wavelength of each line (nm).
        ratio : `numpy.ndarray` of `float`
            Flux ratio (target/source) of each line.
        signalToNoise : `numpy.ndarray` of `float`
            Signal-to-noise ratio of each line's summed flux (the lesser of
            ``source`` and ``target``).
        good : `numpy.ndarray` of `bool`
            Whether each line's ratio is good.
        median : `float`
            Median ratio of the good lines (NaN if there are none).
        relative : `numpy.ndarray` of `float`
            Ratio relative to ``median`` for the good lines; NaN for others.
        """
        width, height = bbox.getDimensions()
        badBitMask = source.mask.getPlaneBitMask(self.config.badMaskPlanes)
        arrays = (source.image[bbox].array, target.image[bbox].array)
        bad = ((source.mask[bbox].array | target.mask[bbox].array) & badBitMask) != 0
        bad |= ~np.isfinite(arrays[0]) | ~np.isfinite(arrays[1])
        if exclude is not None:
            bad |= exclude
        arrays += (source.variance[bbox].array, target.variance[bbox].array)
        arrays = [np.where(bad, 0.0, array) for array in arrays]

        halfWidth = self.config.lineRatioHalfWidth
        halfHeight = self.config.lineRatioHalfHeight
        rowOffsets = np.arange(-halfHeight, halfHeight + 1)
        colOffsets = np.arange(-halfWidth, halfWidth + 1)
        wavelength = refLines.wavelength
        sums = np.zeros((len(arrays), len(refLines)))
        for fiberId in detectorMap.fiberId:
            points = detectorMap.findPoint(fiberId, wavelength)
            with np.errstate(invalid="ignore"):
                xCenter = np.floor(points[:, 0] + 0.5) - bbox.getMinX()
                yCenter = np.floor(points[:, 1] + 0.5) - bbox.getMinY()
                use = (
                    np.isfinite(xCenter)
                    & np.isfinite(yCenter)
                    & (xCenter >= halfWidth)
                    & (xCenter < width - halfWidth)
                    & (yCenter >= halfHeight)
                    & (yCenter < height - halfHeight)
                )
            if not np.any(use):
                continue
            rows = yCenter[use].astype(int)[:, np.newaxis, np.newaxis] + rowOffsets[np.newaxis, :, np.newaxis]
            cols = xCenter[use].astype(int)[:, np.newaxis, np.newaxis] + colOffsets[np.newaxis, np.newaxis, :]
            for ii, array in enumerate(arrays):
                sums[ii, use] += array[rows, cols].sum(axis=(1, 2))

        sourceFlux, targetFlux, sourceVariance, targetVariance = sums
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = targetFlux / sourceFlux
            signalToNoise = np.minimum(
                sourceFlux / np.sqrt(sourceVariance), targetFlux / np.sqrt(targetVariance)
            )
            good = np.isfinite(ratio) & (signalToNoise >= self.config.lineRatioMinSignalToNoise)
        median = np.nan
        if np.any(good):
            median = np.median(ratio[good])
            lq, uq = np.percentile(ratio[good], [25.0, 75.0])
            sigma = 0.741 * (uq - lq)
            if sigma > 0:
                good &= np.abs(ratio - median) <= self.config.lineRatioRejection * sigma
            median = np.median(ratio[good])
        relative = np.where(good, ratio / median, np.nan)

        if np.any(good):
            self.log.info(
                "Measured flux ratios for %d/%d lines: median %.3f, relative range %.3f..%.3f",
                good.sum(),
                len(refLines),
                median,
                np.min(relative[good]),
                np.max(relative[good]),
            )
        else:
            self.log.warning("No good line flux ratios from %d lines; not scaling lines", len(refLines))
        return pipeBase.Struct(
            wavelength=wavelength,
            ratio=ratio,
            signalToNoise=signalToNoise,
            good=good,
            median=median,
            relative=relative,
        )

    def _scaleLines(
        self,
        bbox: geom.Box2I,
        maskedImage: afwImage.MaskedImage,
        lineRatios: pipeBase.Struct,
        detectorMap: DetectorMap,
    ) -> afwImage.MaskedImage:
        """Return a copy of an image with the lines scaled by their ratios

        Each pixel within ``bbox`` is associated with the nearest fiber
        (in the spatial dimension) and, on that fiber, the nearest line
        with a good flux ratio (in the dispersion dimension). If that line
        is within ``config.lineScaleHalfHeight`` rows, the pixel (and its
        variance) is scaled by the line's relative flux ratio; otherwise,
        the pixel is not modified.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the region of interest.
        maskedImage : `lsst.afw.image.MaskedImage`
            Image to scale; not modified.
        lineRatios : `lsst.pipe.base.Struct`
            Line flux ratios, from ``_measureLineRatios``.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.

        Returns
        -------
        scaled : `lsst.afw.image.MaskedImage`
            Copy of ``maskedImage`` with the lines scaled.
        """
        width, height = bbox.getDimensions()
        rows = np.arange(height, dtype=float) + bbox.getMinY()
        wavelength = lineRatios.wavelength[lineRatios.good]
        relative = lineRatios.relative[lineRatios.good]

        # Scale as a function of row for each fiber
        numFibers = len(detectorMap)
        xCenter = np.full((numFibers, height), np.nan)
        fiberScale = np.ones((numFibers, height))
        for ii, fiberId in enumerate(detectorMap.fiberId):
            xCenter[ii] = detectorMap.getXCenter(fiberId, rows)
            lineRows = detectorMap.findPoint(fiberId, wavelength)[:, 1]
            select = np.isfinite(lineRows)
            if not np.any(select):
                continue
            distance = np.abs(rows[np.newaxis, :] - lineRows[select][:, np.newaxis])
            nearest = np.argmin(distance, axis=0)
            nearestDistance = distance[nearest, np.arange(height)]
            fiberScale[ii] = np.where(
                nearestDistance <= self.config.lineScaleHalfHeight, relative[select][nearest], 1.0
            )

        # Assign each pixel to the nearest fiber
        scale = np.ones((height, width))
        columns = np.arange(width, dtype=float) + bbox.getMinX()
        for yy in range(height):
            finite = np.isfinite(xCenter[:, yy])
            if not np.any(finite):
                continue
            order = np.argsort(xCenter[finite, yy])
            xFiber = xCenter[finite, yy][order]
            midpoints = 0.5 * (xFiber[:-1] + xFiber[1:])
            scale[yy] = fiberScale[finite, yy][order][np.searchsorted(midpoints, columns)]

        scaled = maskedImage.clone()
        scaled.image[bbox].array *= scale
        scaled.variance[bbox].array *= scale**2
        self.log.info(
            "Scaled %d lines by their relative flux ratios before fitting: %.1f%% of pixels modified",
            len(wavelength),
            100 * np.mean(scale != 1.0),
        )
        return scaled

    def _buildCellSet(
        self,
        source: afwImage.MaskedImage,
        target: afwImage.MaskedImage,
        seed: int = 0,
        exclude: Optional[np.ndarray] = None,
    ) -> pipeBase.Struct:
        """Build a SpatialCellSet of tiled kernel candidates

        The overlap of ``source`` and ``target`` is tiled with
        non-overlapping stamps (``config.xStampSize`` x
        ``config.yStampSize``). Stamps with any bad mask plane set, or with
        peak signal-to-noise below ``config.minPeakSignalToNoise``, are
        discarded. If more stamps survive than ``config.maxStamps``, a
        random subset is kept (using ``seed``), so that the retained
        stamps remain spread across the image rather than being biased
        toward the very brightest ones.

        Parameters
        ----------
        source : `lsst.afw.image.MaskedImage`
            Image to be convolved to match ``target``.
        target : `lsst.afw.image.MaskedImage`
            Image to which ``source`` is matched.
        seed : `int`
            Seed for random subsampling of stamps.
        exclude : `numpy.ndarray` of `bool`, optional
            Pixels to exclude from the kernel fit, over the overlap of
            ``source`` and ``target``. If provided, the stamps are copies
            in which the excluded pixels have the ``config.lineMaskPlane``
            mask plane set and a very large variance, and these pixels
            are ignored when measuring the peak signal-to-noise.

        Returns
        -------
        kernelCellSet : `lsst.afw.math.SpatialCellSet`
            Cell set of candidates to be used by ``self._solve``.
        numTiles : `int`
            Total number of tiles generated.
        numRejectedMask : `int`
            Number of tiles rejected due to a bad mask plane.
        numRejectedSnr : `int`
            Number of tiles rejected due to low peak signal-to-noise.
        numCandidates : `int`
            Number of tiles surviving quality selection, before any random
            subsampling.
        numUsed : `int`
            Number of tiles actually used, after random subsampling.
        """
        bbox = source.getBBox()
        bbox.clip(target.getBBox())
        xStampSize, yStampSize = self.config.xStampSize, self.config.yStampSize
        width, height = bbox.getDimensions()
        numCellsX = width // xStampSize
        numCellsY = height // yStampSize
        if numCellsX < 1 or numCellsY < 1:
            raise RuntimeError(
                f"Overlap region {bbox} is too small for stamp size ({xStampSize}, {yStampSize})"
            )
        marginX = width - numCellsX * xStampSize
        marginY = height - numCellsY * yStampSize
        x0 = bbox.getMinX() + marginX // 2
        y0 = bbox.getMinY() + marginY // 2
        if exclude is not None and exclude.shape != (height, width):
            raise ValueError(f"Shape of exclude ({exclude.shape}) doesn't match overlap region {bbox}")
        self.log.info(
            "Tiling %s overlap region into %d x %d = %d candidate stamps of %d x %d pixels",
            bbox,
            numCellsX,
            numCellsY,
            numCellsX * numCellsY,
            xStampSize,
            yStampSize,
        )

        badBitMask = source.mask.getPlaneBitMask(self.config.badMaskPlanes)
        excludeBitMask = source.mask.getPlaneBitMask(self.config.lineMaskPlane)
        candidates = []
        numRejectedMask = 0
        numRejectedSnr = 0
        for row in range(numCellsY):
            for col in range(numCellsX):
                tileBBox = geom.Box2I(
                    geom.Point2I(x0 + col * xStampSize, y0 + row * yStampSize),
                    geom.Extent2I(xStampSize, yStampSize),
                )
                sourceStamp = afwImage.MaskedImageF(source, tileBBox)
                targetStamp = afwImage.MaskedImageF(target, tileBBox)
                if (
                    np.bitwise_and(sourceStamp.mask.array, badBitMask).any()
                    or np.bitwise_and(targetStamp.mask.array, badBitMask).any()
                ):
                    numRejectedMask += 1
                    self.log.debug("Rejecting stamp at %s: bad mask plane set", tileBBox)
                    continue
                stampExclude = None
                if exclude is not None:
                    xStart = tileBBox.getMinX() - bbox.getMinX()
                    yStart = tileBBox.getMinY() - bbox.getMinY()
                    stampExclude = exclude[yStart:yStart + yStampSize, xStart:xStart + xStampSize]
                    if stampExclude.any():
                        # Copy, so we don't modify the inputs
                        sourceStamp = afwImage.MaskedImageF(sourceStamp, deep=True)
                        targetStamp = afwImage.MaskedImageF(targetStamp, deep=True)
                        for stamp in (sourceStamp, targetStamp):
                            stamp.mask.array[stampExclude] |= excludeBitMask
                            stamp.variance.array[stampExclude] = EXCLUDED_VARIANCE
                snr = min(
                    peakSignalToNoise(sourceStamp, stampExclude),
                    peakSignalToNoise(targetStamp, stampExclude),
                )
                if snr < self.config.minPeakSignalToNoise:
                    numRejectedSnr += 1
                    self.log.debug(
                        "Rejecting stamp at %s: peak S/N %.2f < %.2f",
                        tileBBox,
                        snr,
                        self.config.minPeakSignalToNoise,
                    )
                    continue
                center = geom.Box2D(tileBBox).getCenter()
                candidates.append((center.getX(), center.getY(), sourceStamp, targetStamp, snr))

        numCandidates = len(candidates)
        if self.config.maxStamps is not None and numCandidates > self.config.maxStamps:
            rng = np.random.RandomState(seed)
            keep = np.sort(rng.choice(numCandidates, size=self.config.maxStamps, replace=False))
            candidates = [candidates[index] for index in keep]
            self.log.info(
                "Randomly subsampled %d candidate stamps down to maxStamps=%d (seed=%d)",
                numCandidates,
                self.config.maxStamps,
                seed,
            )

        kernelCellSet = afwMath.SpatialCellSet(
            geom.Box2I(bbox), self.kConfig.sizeCellX, self.kConfig.sizeCellY
        )
        ps = pexConfig.makePropertySet(self.kConfig)
        for posX, posY, sourceStamp, targetStamp, snr in candidates:
            candidate = diffimLib.makeKernelCandidate(posX, posY, sourceStamp, targetStamp, ps)
            self.log.debug("Candidate %d at (%.1f, %.1f): peak S/N=%.2f", candidate.getId(), posX, posY, snr)
            kernelCellSet.insertCandidate(candidate)

        return pipeBase.Struct(
            kernelCellSet=kernelCellSet,
            numTiles=numCellsX * numCellsY,
            numRejectedMask=numRejectedMask,
            numRejectedSnr=numRejectedSnr,
            numCandidates=numCandidates,
            numUsed=len(candidates),
        )


def plotSpatialKernel(
    kernel: afwMath.Kernel,
    bbox: geom.Box2I,
    numRows: int = 3,
    numCols: int = 3,
    *,
    doNormalize: bool = True,
    margin: float = 0.1,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentile: float = 100.0,
    symmetric: bool = False,
    cmap: str = "viridis",
    colorbar: Optional[str] = "shared",
    markCenter: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    fig: Optional["matplotlib.figure.Figure"] = None,
    axes: Optional[np.ndarray] = None,
) -> Tuple["matplotlib.figure.Figure", np.ndarray]:
    """Plot a spatially varying kernel over a grid of image positions

    Useful for visually inspecting a `PsfMatchTask.run` result's
    ``psfMatchingKernel`` in a notebook: evaluates the kernel on a
    ``numRows`` x ``numCols`` grid of positions spanning ``bbox`` and
    plots each resulting kernel image in its own subplot.

    Parameters
    ----------
    kernel : `lsst.afw.math.Kernel`
        Spatially varying kernel to plot (e.g., ``psfMatchingKernel``
        from `PsfMatchTask.run`).
    bbox : `lsst.geom.Box2I`
        Bounding box of the region the kernel was fit over (e.g., the
        ``kernelCellSet``'s bounding box, or the source/target
        exposure's bounding box); used to lay out the grid of
        positions at which to evaluate ``kernel``.
    numRows : `int`
        Number of rows in the grid of positions (and of subplots).
    numCols : `int`
        Number of columns in the grid of positions (and of subplots).
    doNormalize : `bool`
        Normalize each kernel image to unit sum before plotting? The
        default of `True` makes the kernel *shape* easy to compare
        across positions; set `False` to instead see the fitted flux
        ratio between the two exposures, as `PsfMatchTask.run` uses
        when convolving. Either way, the range (over the grid of
        positions) of the pre-normalization kernel sum is reported in
        the figure title.
    margin : `float`
        Fraction of ``bbox``'s width/height by which to inset the
        grid from its edges, avoiding positions where the spatial
        polynomial is typically least well constrained.
    vmin, vmax : `float`, optional
        Colormap stretch limits. Either left as `None` (the default)
        is set automatically from ``percentile`` (and ``symmetric``)
        over all the plotted kernel images.
    percentile : `float`
        Percentile used to set ``vmin``/``vmax`` automatically, when
        not given explicitly: with ``symmetric=True``, the percentile
        of ``abs(value)``; otherwise the ``percentile`` and
        ``100 - percentile`` points of the raw values. Defaults to
        ``100.0``, i.e. a plain min/max stretch.
    symmetric : `bool`
        Force the automatic stretch to be symmetric about zero?
        Defaults to `False`: kernel pixel values are not typically
        distributed symmetrically about zero (unlike, e.g., a
        source-minus-template difference image), so a diverging
        stretch centered on zero is usually not a good default. Set
        `True` (and pass a diverging ``cmap`` such as ``"RdBu_r"``) if
        you specifically want that.
    cmap : `str` or `matplotlib.colors.Colormap`
        Colormap to use.
    colorbar : `str`, optional
        One of ``"shared"`` (a single colorbar for the whole figure),
        ``"each"`` (one colorbar per subplot), or `None` (no
        colorbar).
    markCenter : `bool`
        Draw crosshairs at the geometric center of each kernel image?
        This makes it easy to see how far the fitted kernel has moved
        off-center, which is expected: see `PsfMatchTask`'s docstring.
    figsize : `tuple` of `float`, optional
        Figure size, passed to ``matplotlib.pyplot.subplots``.
    fig : `matplotlib.figure.Figure`, optional
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`, optional
        Existing figure and grid of axes to plot into (e.g., reusing
        the result of a previous call), instead of creating a new
        one. If either is provided, both must be, and their shape
        must match ``(numRows, numCols)``.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot.
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`
        Grid of axes, of shape ``(numRows, numCols)``.
    """
    import matplotlib.pyplot as plt

    marginX = margin * bbox.getWidth()
    marginY = margin * bbox.getHeight()
    xPositions = np.linspace(bbox.getMinX() + marginX, bbox.getMaxX() - marginX, numCols)
    # Top row corresponds to the highest y, to match the image's on-sky orientation.
    yPositions = np.linspace(bbox.getMaxY() - marginY, bbox.getMinY() + marginY, numRows)

    images = []
    sums = []
    for yPos in yPositions:
        imageRow = []
        sumRow = []
        for xPos in xPositions:
            image = afwImage.ImageD(kernel.getDimensions())
            # Always compute the pre-normalization sum (computeImage returns 1.0
            # instead, once normalized, so normalization -- if wanted -- is applied
            # separately below) so the figure title reports it regardless of doNormalize.
            kernelSum = kernel.computeImage(image, False, xPos, yPos)
            array = image.array
            if doNormalize:
                if kernelSum == 0:
                    raise ValueError(f"Cannot normalize kernel at ({xPos}, {yPos}): kernel sum is 0")
                array = array / kernelSum
            imageRow.append(array.copy())
            sumRow.append(kernelSum)
        images.append(imageRow)
        sums.append(sumRow)

    if vmin is None or vmax is None:
        allValues = np.concatenate([array.ravel() for imageRow in images for array in imageRow])
        if symmetric:
            limit = np.percentile(np.abs(allValues), percentile)
            autoVmin, autoVmax = -limit, limit
        else:
            autoVmin = np.percentile(allValues, 100 - percentile)
            autoVmax = np.percentile(allValues, percentile)
        vmin = autoVmin if vmin is None else vmin
        vmax = autoVmax if vmax is None else vmax

    if fig is None or axes is None:
        fig, axes = plt.subplots(numRows, numCols, figsize=figsize, sharex=True, sharey=True, squeeze=False)

    mappable = None
    for row in range(numRows):
        for col in range(numCols):
            axis = axes[row, col]
            mappable = axis.imshow(images[row][col], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(f"({xPositions[col]:.0f}, {yPositions[row]:.0f})", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
            if markCenter:
                centerX = (kernel.getWidth() - 1) / 2.0
                centerY = (kernel.getHeight() - 1) / 2.0
                axis.axhline(centerY, color="k", ls=":", lw=0.5)
                axis.axvline(centerX, color="k", ls=":", lw=0.5)

    if colorbar == "shared":
        fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.8)
    elif colorbar == "each":
        for axis in axes.ravel():
            fig.colorbar(axis.images[0], ax=axis)
    elif colorbar is not None:
        raise ValueError(f"Unrecognized colorbar option: {colorbar!r}")

    minSum = min(min(sumRow) for sumRow in sums)
    maxSum = max(max(sumRow) for sumRow in sums)
    fig.suptitle(f"PSF-matching kernel (pre-normalization sum: {minSum:.3f}-{maxSum:.3f})")

    return fig, axes


def calculateFirstMoments(
    kernel: afwMath.Kernel,
    x: np.ndarray,
    y: np.ndarray,
) -> pipeBase.Struct:
    """Calculate the zeroth and first moments of a spatially varying kernel

    The kernel is realized at each of the given ``(xx, yy)`` positions,
    and its zeroth moment (the pre-normalization kernel sum, i.e. the
    fitted flux ratio) and first moments in x and y (the flux-weighted
    centroid of the realized kernel image, measured relative to its
    geometric center) are calculated. The first moments are useful for
    diagnosing how far a PSF-matching kernel has shifted off-center, as
    a function of position in the image (see `PsfMatchTask`'s
    docstring).

    Parameters
    ----------
    kernel : `lsst.afw.math.Kernel`
        Spatially varying kernel to realize (e.g., ``psfMatchingKernel``
        from `PsfMatchTask.run`).
    x, y : `numpy.ndarray`
        Arrays (of the same shape) of x,y positions at which to realize
        ``kernel``.

    Returns
    -------
    kernelSum : `numpy.ndarray`
        Zeroth moment (pre-normalization sum of the kernel image) at
        each position. Same shape as ``x`` input.
    x : `numpy.ndarray`
        First moment in x: the flux-weighted centroid of the kernel
        image at each position, in pixels relative to the kernel's
        geometric center. Same shape as ``x`` input.
    y : `numpy.ndarray`
        First moment in y, as ``x`` but in the y direction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape; got {x.shape} and {y.shape}")

    width, height = kernel.getDimensions()
    centerX = (width - 1) / 2.0
    centerY = (height - 1) / 2.0
    pixelX, pixelY = np.meshgrid(np.arange(width) - centerX, np.arange(height) - centerY)

    zeroth = np.zeros(x.shape, dtype=float)
    xMoment = np.zeros(x.shape, dtype=float)
    yMoment = np.zeros(x.shape, dtype=float)

    image = afwImage.ImageD(kernel.getDimensions())
    for index in np.ndindex(x.shape):
        kernelSum = kernel.computeImage(image, False, float(x[index]), float(y[index]))
        if kernelSum == 0:
            raise ValueError(f"Cannot calculate moments at ({x[index]}, {y[index]}): kernel sum is 0")
        array = image.array
        zeroth[index] = kernelSum
        xMoment[index] = np.sum(array * pixelX) / kernelSum
        yMoment[index] = np.sum(array * pixelY) / kernelSum

    return pipeBase.Struct(kernelSum=zeroth, x=xMoment, y=yMoment)


def plotFirstMoments(
    kernel: afwMath.Kernel,
    bbox: geom.Box2I,
    numX: int = 9,
    numY: int = 9,
    *,
    margin: float = 0.1,
    cmap: str = "viridis",
    figsize: Optional[Tuple[float, float]] = (12, 4),
    fig: Optional["matplotlib.figure.Figure"] = None,
    axes: Optional[np.ndarray] = None,
) -> Tuple["matplotlib.figure.Figure", np.ndarray]:
    """Plot the spatial variation of a kernel's zeroth and first moments

    Realizes the kernel on a ``numY`` x ``numX`` grid of positions
    spanning ``bbox``, calculates the zeroth and first moments at each
    position (see `calculateFirstMoments`), and plots their spatial
    variation as three `matplotlib.pyplot.imshow` panels. This is a more
    compact alternative to `plotSpatialKernel` for visualizing how far a
    PSF-matching kernel has shifted off-center, and by how much its flux
    varies, as a function of position in the image.

    Parameters
    ----------
    kernel : `lsst.afw.math.Kernel`
        Spatially varying kernel to plot (e.g., ``psfMatchingKernel``
        from `PsfMatchTask.run`).
    bbox : `lsst.geom.Box2I`
        Bounding box of the region the kernel was fit over; used to lay
        out the grid of positions at which to realize ``kernel``.
    numX : `int`
        Number of realizations in x (and hence columns of the grid).
    numY : `int`
        Number of realizations in y (and hence rows of the grid).
    margin : `float`
        Fraction of ``bbox``'s width/height by which to inset the grid
        from its edges, avoiding positions where the spatial polynomial
        is typically least well constrained.
    cmap : `str` or `matplotlib.colors.Colormap`
        Colormap to use for each panel.
    figsize : `tuple` of `float`, optional
        Figure size, passed to ``matplotlib.pyplot.subplots``.
    fig : `matplotlib.figure.Figure`, optional
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`, optional
        Existing figure and row of 3 axes to plot into, instead of
        creating new ones. If either is provided, both must be.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot.
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`
        Row of 3 axes, for the zeroth moment, first moment in x and
        first moment in y, respectively.
    """
    import matplotlib.pyplot as plt

    marginX = margin * bbox.getWidth()
    marginY = margin * bbox.getHeight()
    xPositions = np.linspace(bbox.getMinX() + marginX, bbox.getMaxX() - marginX, numX)
    yPositions = np.linspace(bbox.getMinY() + marginY, bbox.getMaxY() - marginY, numY)
    xx, yy = np.meshgrid(xPositions, yPositions)

    moments = calculateFirstMoments(kernel, xx, yy)

    if (fig is None) != (axes is None):
        raise ValueError("fig and axes must be provided together")
    if fig is None:
        fig, axes = plt.subplots(
            1, 3, figsize=figsize, sharex=True, sharey=True, squeeze=False, layout="constrained"
        )
        axes = axes[0]
    elif axes.shape != (3,):
        raise ValueError(f"axes must have shape (3,); got {axes.shape}")

    extent = (bbox.getMinX(), bbox.getMaxX(), bbox.getMinY(), bbox.getMaxY())
    panels = (
        ("zeroth moment", moments.kernelSum),
        ("first moment (x)", moments.x),
        ("first moment (y)", moments.y),
    )
    for axis, (title, data) in zip(axes, panels):
        mappable = axis.imshow(data, origin="lower", cmap=cmap, extent=extent, aspect="auto")
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("x position (pixels)")
        axis.set_ylabel("y position (pixels)")
        fig.colorbar(mappable, ax=axis, shrink=0.8, label=title)

    fig.suptitle("PSF-matching kernel moments")

    return fig, axes


def plotPsfMatchResult(
    source: afwImage.Exposure,
    target: afwImage.Exposure,
    result: pipeBase.Struct,
    *,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    stretchAlgorithm: str = "zscale",
    percentile: float = 99.5,
    symmetric: bool = False,
    zscaleSamples: int = 1000,
    zscaleContrast: float = 0.25,
    diffVmin: Optional[float] = None,
    diffVmax: Optional[float] = None,
    diffPercentile: float = 99.5,
    diffSymmetric: bool = True,
    cmap: str = "viridis",
    diffCmap: str = "RdBu_r",
    titles: Tuple[str, str, str, str] = ("source", "target", "convolved", "target - convolved"),
    figsize: Optional[Tuple[float, float]] = None,
    fig: Optional["matplotlib.figure.Figure"] = None,
    axes: Optional[np.ndarray] = None,
) -> Tuple["matplotlib.figure.Figure", np.ndarray]:
    """Plot a `PsfMatchTask.run` result

    Static 2x2 view of ``source``, ``target``, ``result.matchedExposure``
    and their difference, sharing one colormap stretch between
    ``source``/``target``/``convolved`` and a separate stretch for the
    difference. Pan/zoom is linked across all four panels (via
    ``sharex``/``sharey``); no mouse/key event handling is attached, so
    the matplotlib toolbar's own pan/zoom works normally.

    For an interactive version of this plot (marking, click-drag stretch
    adjustment), use
    `pfs.drp.stella.utils.interactiveDisplay.PsfMatchDiagnostic` directly.

    Parameters
    ----------
    source : `lsst.afw.image.Exposure`
        Exposure passed to `PsfMatchTask.run` as ``source``.
    target : `lsst.afw.image.Exposure`
        Exposure passed to `PsfMatchTask.run` as ``target``.
    result : `lsst.pipe.base.Struct`
        Result of `PsfMatchTask.run`.
    vmin, vmax : `float`, optional
        Stretch limits for ``source``/``target``/``convolved``. Either
        left as `None` (the default) is set automatically according to
        ``stretchAlgorithm``.
    stretchAlgorithm : `str`
        Algorithm used to compute ``vmin``/``vmax`` automatically, if
        either is `None`: ``"zscale"`` (the default; the classic ds9/IRAF
        algorithm, computed per-image and then combined by taking the
        widest limits of the three) or ``"percentile"`` (see
        ``percentile``, ``symmetric``).
    percentile : `float`
        Percentile used to set ``vmin``/``vmax`` automatically, if
        ``stretchAlgorithm`` is ``"percentile"``.
    symmetric : `bool`
        If ``stretchAlgorithm`` is ``"percentile"``, force the automatic
        ``source``/``target``/``convolved`` stretch to be symmetric about
        zero?
    zscaleSamples : `int`
        Number of pixels to sample, if ``stretchAlgorithm`` is
        ``"zscale"``.
    zscaleContrast : `float`
        Contrast parameter, if ``stretchAlgorithm`` is ``"zscale"``.
    diffVmin, diffVmax : `float`, optional
        Stretch limits for the difference image; as ``vmin``/``vmax``
        but for the difference.
    diffPercentile : `float`
        As ``percentile``, but for the difference image.
    diffSymmetric : `bool`
        As ``symmetric``, but for the difference image. Defaults to
        `True`, since a residual is naturally zero-centered.
    cmap, diffCmap : `str` or `matplotlib.colors.Colormap`
        Colormaps for the shared and difference stretch groups.
    titles : `tuple` of `str`
        Panel titles, in the order (source, target, convolved,
        difference).
    figsize : `tuple` of `float`, optional
        Figure size, passed to ``matplotlib.pyplot.subplots``.
    fig : `matplotlib.figure.Figure`, optional
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`, optional
        Existing figure and 2x2 grid of axes to plot into, instead of
        creating new ones. If either is provided, both must be.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot.
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`
        Grid of axes, of shape ``(2, 2)``.
    """
    import matplotlib.pyplot as plt
    import lsst.afw.display.rgb as afwRgb
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    convolved = result.matchedExposure
    sourceArr = source.image.array
    targetArr = target.image.array
    convolvedArr = convolved.image.array
    if sourceArr.shape != targetArr.shape or sourceArr.shape != convolvedArr.shape:
        raise ValueError(
            f"source ({sourceArr.shape}), target ({targetArr.shape}) and convolved "
            f"({convolvedArr.shape}) must have the same shape"
        )
    diffArr = targetArr - convolvedArr

    bbox = source.getBBox()
    height, width = sourceArr.shape
    extent = (bbox.getMinX(), bbox.getMinX() + width, bbox.getMinY(), bbox.getMinY() + height)

    if vmin is None or vmax is None:
        if stretchAlgorithm == "zscale":
            z1s, z2s = zip(
                *(
                    afwRgb.getZScale(
                        afwImage.ImageF(np.ascontiguousarray(arr, dtype=np.float32)),
                        zscaleSamples,
                        zscaleContrast,
                    )
                    for arr in (sourceArr, targetArr, convolvedArr)
                )
            )
            autoVmin, autoVmax = min(z1s), max(z2s)
        elif stretchAlgorithm == "percentile":
            allValues = np.concatenate([sourceArr.ravel(), targetArr.ravel(), convolvedArr.ravel()])
            finite = allValues[np.isfinite(allValues)]
            if symmetric:
                limit = np.percentile(np.abs(finite), percentile)
                autoVmin, autoVmax = -limit, limit
            else:
                autoVmin = np.percentile(finite, 100 - percentile)
                autoVmax = np.percentile(finite, percentile)
        else:
            raise ValueError(
                f"Unrecognized stretchAlgorithm: {stretchAlgorithm!r}; expected 'zscale' or 'percentile'"
            )
        vmin = autoVmin if vmin is None else vmin
        vmax = autoVmax if vmax is None else vmax

    if diffVmin is None or diffVmax is None:
        diffFinite = diffArr[np.isfinite(diffArr)]
        if diffSymmetric:
            diffLimit = np.percentile(np.abs(diffFinite), diffPercentile)
            autoDiffVmin, autoDiffVmax = -diffLimit, diffLimit
        else:
            autoDiffVmin = np.percentile(diffFinite, 100 - diffPercentile)
            autoDiffVmax = np.percentile(diffFinite, diffPercentile)
        diffVmin = autoDiffVmin if diffVmin is None else diffVmin
        diffVmax = autoDiffVmax if diffVmax is None else diffVmax

    if (fig is None) != (axes is None):
        raise ValueError("fig and axes must be provided together")
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    elif axes.shape != (2, 2):
        raise ValueError(f"axes must have shape (2, 2); got {axes.shape}")

    axSource, axTarget = axes[0, 0], axes[0, 1]
    axConvolved, axDiff = axes[1, 0], axes[1, 1]

    sharedNorm = Normalize(vmin=vmin, vmax=vmax)
    diffNorm = Normalize(vmin=diffVmin, vmax=diffVmax)

    imSource = axSource.imshow(sourceArr, origin="lower", cmap=cmap, norm=sharedNorm, extent=extent)
    axTarget.imshow(targetArr, origin="lower", cmap=cmap, norm=sharedNorm, extent=extent)
    axConvolved.imshow(convolvedArr, origin="lower", cmap=cmap, norm=sharedNorm, extent=extent)
    imDiff = axDiff.imshow(diffArr, origin="lower", cmap=diffCmap, norm=diffNorm, extent=extent)

    for axis, title in zip((axSource, axTarget, axConvolved, axDiff), titles):
        axis.set_title(title)

    # Attach each colorbar directly beside one specific Axes (via a divider), rather than
    # letting fig.colorbar(..., ax=[...]) steal room from the whole figure for a list of
    # Axes: the "shared" group spans both columns (it includes axTarget, top-right), so a
    # colorbar auto-placed to the right of that whole group lands on top of axDiff
    # (bottom-right), which is not part of the group and so never gets shrunk to make room
    # for it.
    sharedCax = make_axes_locatable(axTarget).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(imSource, cax=sharedCax)
    diffCax = make_axes_locatable(axDiff).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(imDiff, cax=diffCax)

    return fig, axes
