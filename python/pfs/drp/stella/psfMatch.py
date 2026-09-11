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

if TYPE_CHECKING:
    import matplotlib

__all__ = ("PsfMatchConfig", "PsfMatchTask", "peakSignalToNoise", "plotSpatialKernel")


def peakSignalToNoise(maskedImage: afwImage.MaskedImage) -> float:
    """Return the peak (per-pixel) signal-to-noise ratio of a masked image

    Parameters
    ----------
    maskedImage : `lsst.afw.image.MaskedImage`
        Image for which to calculate the peak signal-to-noise.

    Returns
    -------
    peakSnr : `float`
        Maximum value of ``image/sqrt(variance)`` over all finite,
        positive-variance pixels. ``-inf`` if there are no such pixels.
    """
    image = maskedImage.image.array
    variance = maskedImage.variance.array
    good = np.isfinite(image) & np.isfinite(variance) & (variance > 0)
    if not good.any():
        return -np.inf
    return float(np.max(image[good] / np.sqrt(variance[good])))


class PsfMatchConfig(pexConfig.Config):
    """Configuration for PsfMatchTask"""

    kernel = pexConfig.ConfigChoiceField(
        doc="Kernel type",
        typemap=dict(DF=PsfMatchConfigDF),
        default="DF",
    )
    xStampSize = pexConfig.Field(
        dtype=int,
        default=128,
        check=lambda xx: xx > 0,
        doc="Width of each tiling stamp (pixels)",
    )
    yStampSize = pexConfig.Field(
        dtype=int,
        default=128,
        check=lambda yy: yy > 0,
        doc="Height of each tiling stamp (pixels)",
    )
    minPeakSignalToNoise = pexConfig.Field(
        dtype=float,
        default=25.0,
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

    @timeMethod
    def run(
        self,
        source: afwImage.Exposure,
        target: afwImage.Exposure,
        *,
        seed: Optional[int] = None,
    ) -> pipeBase.Struct:
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

        Returns
        -------
        matchedExposure : `lsst.afw.image.Exposure`
            ``source``, convolved by the psf-matching kernel with the
            fitted background added, so as to resemble ``target``.
        psfMatchingKernel : `lsst.afw.math.LinearCombinationKernel`
            Spatially varying kernel that matches ``source`` to ``target``.
        backgroundModel : `lsst.afw.math.Function2D`
            Spatially varying background difference between ``source`` (as
            matched) and ``target``.
        spatialSolution : `lsst.ip.diffim.SpatialKernelSolution`
            Kernel solution produced by the fit.
        kernelCellSet : `lsst.afw.math.SpatialCellSet`
            Cell set of candidates used in the fit.
        basisList : `list` of `lsst.afw.math.Kernel`
            Kernel basis functions used in the fit.
        metadata : `lsst.daf.base.PropertySet`
            Metadata accumulated while fitting.
        selection : `lsst.pipe.base.Struct`
            Stamp-selection diagnostics: ``numTiles``, ``numRejectedMask``,
            ``numRejectedSnr``, ``numCandidates``, ``numUsed``.
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

        buildResult = self._buildCellSet(source.maskedImage, target.maskedImage, seed=seed)
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
        return pipeBase.Struct(
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
            ),
        )

    def _buildCellSet(
        self,
        source: afwImage.MaskedImage,
        target: afwImage.MaskedImage,
        seed: int = 0,
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
                snr = min(peakSignalToNoise(sourceStamp), peakSignalToNoise(targetStamp))
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
    percentile: float = 99.5,
    symmetric: bool = True,
    cmap: str = "RdBu_r",
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
        ``100 - percentile`` points of the raw values.
    symmetric : `bool`
        Force the automatic stretch to be symmetric about zero? This
        is usually what you want for a diverging ``cmap``, since a
        delta-function-basis kernel typically has a positive core
        and negative side lobes.
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
            kernelSum = kernel.computeImage(image, doNormalize, xPos, yPos)
            imageRow.append(image.array.copy())
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
