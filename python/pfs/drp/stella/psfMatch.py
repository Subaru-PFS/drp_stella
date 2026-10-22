from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

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

from .DetectorMap import DetectorMap
from .referenceLine import ReferenceLineSet, ReferenceLineStatus
from .repair import maskLines

if TYPE_CHECKING:
    import matplotlib

__all__ = (
    "PsfMatchConfig",
    "PsfMatchResult",
    "PsfMatchTask",
    "peakSignalToNoise",
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
        default=["BAD", "SAT", "CR", "INTRP", "NO_DATA", "EDGE"],
        doc="Mask planes that disqualify a stamp",
    )
    lineMaskStatus = pexConfig.ListField(
        dtype=str,
        default=["BAD"],
        doc="ReferenceLineStatus flags for lines to exclude from the kernel fit (only when a line list "
        "is provided); a line is excluded if any of these flags is set",
    )
    lineMaskRadius = pexConfig.Field(
        dtype=int,
        default=5,
        check=lambda radius: radius >= 0,
        doc="Radius (pixels) around excluded lines to exclude from the kernel fit; this is grown by "
        "half the kernel size to account for the convolution",
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
        self.dfKernel.usePcaForSpatialKernel = True
        self.dfKernel.numPrincipalComponents = 10
        self.dfKernel.spatialKernelOrder = 3
        self.dfKernel.spatialBgOrder = 3
        self.dfKernel.fitForBackground = True
        self.dfKernel.sizeCellX = self.xStampSize
        self.dfKernel.sizeCellY = self.yStampSize
        self.dfKernel.badMaskPlanes = list(self.dfKernel.badMaskPlanes) + [self.lineMaskPlane]

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

    We use the delta-function kernel basis with regularization and PCA
    spatial fitting (`lsst.ip.diffim.psfMatch.PsfMatchConfigDF`); the actual
    kernel solve is inherited unchanged from the base class's ``_solve``
    method.

    We do not warp ``source`` onto ``target`` before matching: the two
    exposures are assumed to already share a pixel grid, and any relative
    shift between them (e.g., from differing trace or wavelength solutions)
    is expected to be absorbed by the fitted kernel itself, which may
    therefore end up off-center. For this to work, ``kernel.active.kernelSize``
    must be set large enough to span the largest expected relative shift
    between ``source`` and ``target``.

    If a line list (and detectorMap) is provided to ``run``, the pixels
    around lines with any of the ``config.lineMaskStatus`` flags set are
    excluded from the kernel fit. This is useful for sky exposures, where
    the strengths of lines may vary. The exclusion is achieved by setting a
    very large variance (giving zero weight in the fit) and the
    ``config.lineMaskPlane`` mask plane (excluding them from candidate
    assessment) on copies of the stamps; the inputs are not modified.

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
    ) -> PsfMatchResult:
        """Match the PSF of ``source`` to ``target``

        ``source`` and ``target`` must already share a pixel grid: this
        Task does not warp either exposure.

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
            kernel fit. Requires ``detectorMap``.
        detectorMap : `DetectorMap`, optional
            Mapping of fiberId,wavelength to x,y, used to locate
            ``refLines`` (which are assumed to be at the same position in
            both ``source`` and ``target``).

        Returns
        -------
        result : `PsfMatchResult`
            Results of the PSF matching, which supports pickling (but see
            below regarding ``spatialSolution`` and ``kernelCellSet``),
            with the following elements:

        matchedExposure : `lsst.afw.image.Exposure`
            ``source``, convolved by the psf-matching kernel with the
            fitted background added, so as to resemble ``target``.
        psfMatchingKernel : `lsst.afw.math.LinearCombinationKernel`
            Spatially varying kernel that matches ``source`` to ``target``.
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
            ``numRejectedSnr``, ``numCandidates``, ``numUsed``, and (if
            ``refLines`` was provided; otherwise `None`)
            ``numLinesExcluded`` and ``excludedFraction`` (fraction of
            pixels excluded).
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

        exclusion = None
        if refLines is not None:
            if detectorMap is None:
                raise ValueError("detectorMap is required when refLines is provided")
            bbox = source.getBBox()
            bbox.clip(target.getBBox())
            exclusion = self._makeLineExclusion(bbox, refLines, detectorMap)

        buildResult = self._buildCellSet(
            source.maskedImage,
            target.maskedImage,
            seed=seed,
            exclude=exclusion.exclude if exclusion is not None else None,
        )
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
                numLinesExcluded=exclusion.numLines if exclusion is not None else None,
                excludedFraction=exclusion.fraction if exclusion is not None else None,
            ),
        )

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
        flags = 0
        for name in self.config.lineMaskStatus:
            flags |= ReferenceLineStatus[name]
        select = (refLines.status & flags) != 0
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
