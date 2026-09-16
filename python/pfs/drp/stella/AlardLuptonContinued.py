from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

import lsst.afw.image as afwImage
from lsst.afw.image import MaskedImage
from lsst.utils import continueClass

from .AlardLupton import AlardLuptonResult

if TYPE_CHECKING:
    import matplotlib

__all__ = ["AlardLuptonResult"]


@continueClass  # noqa: F811 redefinition
class AlardLuptonResult:  # noqa: F811 (redefinition)
    def plotSpatialKernel(
        self,
        *,
        doNormalize: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        percentile: float = 100.0,
        symmetric: bool = False,
        cmap: str = "viridis",
        colorbar: Optional[str] = "shared",
        markCenter: bool = True,
        annotate: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        fig: Optional["matplotlib.figure.Figure"] = None,
        axes: Optional[np.ndarray] = None,
    ) -> Tuple["matplotlib.figure.Figure", np.ndarray]:
        """Plot the fitted kernel in each region

        Unlike `pfs.drp.stella.psfMatch.plotSpatialKernel` (which evaluates
        a single spatially-continuous kernel at a chosen grid of
        positions), the kernel here is already discrete: a separate kernel
        was fit independently in each of ``numRegionsX`` x ``numRegionsY``
        regions (see `fitAlardLuptonKernel`), so we plot exactly those, one
        subplot per region, laid out to match each region's position in
        the image.

        Parameters
        ----------
        doNormalize : `bool`
            Normalize each kernel image to unit sum before plotting? As
            with `pfs.drp.stella.psfMatch.plotSpatialKernel`, this makes
            the kernel *shape* easy to compare across regions; set
            `False` to instead see the fitted flux ratio between the two
            images. A region whose fit failed is always shown
            un-normalized (as an all-zero kernel), since its sum cannot
            be normalized.
        vmin, vmax : `float`, optional
            Colormap stretch limits. Either left as `None` (the default)
            is set automatically from ``percentile`` (and ``symmetric``)
            over all the plotted (successful) kernel images.
        percentile : `float`
            Percentile used to set ``vmin``/``vmax`` automatically, when
            not given explicitly: with ``symmetric=True``, the
            percentile of ``abs(value)``; otherwise the ``percentile``
            and ``100 - percentile`` points of the raw values. Defaults
            to ``100.0``, i.e. a plain min/max stretch.
        symmetric : `bool`
            Force the automatic stretch to be symmetric about zero?
        cmap : `str` or `matplotlib.colors.Colormap`
            Colormap to use.
        colorbar : `str`, optional
            One of ``"shared"`` (a single colorbar for the whole
            figure), ``"each"`` (one colorbar per subplot), or `None`
            (no colorbar).
        markCenter : `bool`
            Draw crosshairs at the geometric center of each kernel image?
        annotate : `bool`
            Label each subplot with its region's bounding box and, if
            the fit succeeded, its reduced chi^2 and number of rejected
            pixels (or ``"FAILED"`` if it did not), to help judge the
            quality of the fit in each region.
        figsize : `tuple` of `float`, optional
            Figure size, passed to ``matplotlib.pyplot.subplots``.
        fig : `matplotlib.figure.Figure`, optional
        axes : `numpy.ndarray` of `matplotlib.axes.Axes`, optional
            Existing figure and grid of axes to plot into, instead of
            creating new ones. If either is provided, both must be, and
            their shape must match ``(numRegionsY, numRegionsX)``.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot.
        axes : `numpy.ndarray` of `matplotlib.axes.Axes`
            Grid of axes, of shape ``(numRegionsY, numRegionsX)``.
        """
        import matplotlib.pyplot as plt

        numCols = self.numRegionsX
        numRows = self.numRegionsY
        if len(self.solutions) != numRows * numCols:
            raise ValueError(
                f"Number of solutions ({len(self.solutions)}) does not match "
                f"numRegionsX * numRegionsY ({numCols} * {numRows})"
            )

        # self.solutions is in row-major (y, x) order with index 0 at the
        # smallest y; flip vertically so the highest y ends up on top,
        # matching the image's on-sky (origin="lower") orientation, as in
        # pfs.drp.stella.psfMatch.plotSpatialKernel.
        grid = [[None] * numCols for _ in range(numRows)]
        for index, solution in enumerate(self.solutions):
            regionY, regionX = divmod(index, numCols)
            row = numRows - 1 - regionY
            array = np.asarray(solution.kernel, dtype=float)
            kernelSum = solution.getKernelSum()
            if doNormalize and solution.success:
                if kernelSum == 0:
                    raise ValueError(f"Cannot normalize kernel for region {solution.bbox}: kernel sum is 0")
                array = array / kernelSum
            grid[row][regionX] = (array, kernelSum, solution)

        if vmin is None or vmax is None:
            allValues = np.concatenate(
                [grid[row][col][0].ravel() for row in range(numRows) for col in range(numCols)]
            )
            finite = allValues[np.isfinite(allValues)]
            if symmetric:
                limit = np.percentile(np.abs(finite), percentile)
                autoVmin, autoVmax = -limit, limit
            else:
                autoVmin = np.percentile(finite, 100 - percentile)
                autoVmax = np.percentile(finite, percentile)
            vmin = autoVmin if vmin is None else vmin
            vmax = autoVmax if vmax is None else vmax

        if (fig is None) != (axes is None):
            raise ValueError("fig and axes must be provided together")
        if fig is None:
            fig, axes = plt.subplots(
                numRows, numCols, figsize=figsize, sharex=True, sharey=True, squeeze=False
            )
        elif axes.shape != (numRows, numCols):
            raise ValueError(f"axes must have shape ({numRows}, {numCols}); got {axes.shape}")

        mappable = None
        for row in range(numRows):
            for col in range(numCols):
                array, kernelSum, solution = grid[row][col]
                axis = axes[row, col]
                mappable = axis.imshow(array, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_xticks([])
                axis.set_yticks([])
                if markCenter:
                    center = solution.kernelHalfWidth
                    axis.axhline(center, color="k", ls=":", lw=0.5)
                    axis.axvline(center, color="k", ls=":", lw=0.5)
                if annotate:
                    bbox = solution.bbox
                    title = f"({bbox.getMinX()}:{bbox.getMaxX() + 1}, {bbox.getMinY()}:{bbox.getMaxY() + 1})"
                    if solution.success:
                        title += f"\n$\\chi^2$/dof={solution.getReducedChi2():.2f} rej={solution.numRejected}"
                    else:
                        title += "\nFAILED"
                    axis.set_title(title, fontsize=7)

        if colorbar == "shared":
            fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.8)
        elif colorbar == "each":
            for axis in axes.ravel():
                fig.colorbar(axis.images[0], ax=axis)
        elif colorbar is not None:
            raise ValueError(f"Unrecognized colorbar option: {colorbar!r}")

        sums = [
            grid[row][col][1] for row in range(numRows) for col in range(numCols) if grid[row][col][2].success
        ]
        if sums:
            fig.suptitle(
                f"Alard-Lupton kernel, {numCols}x{numRows} regions "
                f"(pre-normalization sum range: {min(sums):.3f}-{max(sums):.3f})"
            )
        else:
            fig.suptitle(f"Alard-Lupton kernel, {numCols}x{numRows} regions (all regions failed)")

        return fig, axes

    def plotPsfMatchResult(
        self,
        source: MaskedImage,
        target: MaskedImage,
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
        showRegions: bool = True,
        showRejected: bool = True,
        showPartial: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        fig: Optional["matplotlib.figure.Figure"] = None,
        axes: Optional[np.ndarray] = None,
    ) -> Tuple["matplotlib.figure.Figure", np.ndarray]:
        """Plot a `fitAlardLuptonKernel` result

        Static 2x2 view of ``source``, ``target``, ``self.convolved``
        (the matched source), and ``self.difference``, sharing one
        colormap stretch between ``source``/``target``/``convolved`` and
        a separate stretch for the difference. Pan/zoom is linked across
        all four panels (via ``sharex``/``sharey``).

        This mirrors `pfs.drp.stella.psfMatch.plotPsfMatchResult`, but
        for a `fitAlardLuptonKernel` result: since the kernel here is fit
        independently in a grid of regions rather than as a single
        spatially-varying kernel, this also (optionally) overlays the
        region boundaries, so that any per-region discontinuity in the
        difference is easy to spot, and marks pixels rejected during the
        fit.

        Parameters
        ----------
        source : `lsst.afw.image.MaskedImage`
            Image passed to `fitAlardLuptonKernel` as ``source``.
        target : `lsst.afw.image.MaskedImage`
            Image passed to `fitAlardLuptonKernel` as ``target``.
        vmin, vmax : `float`, optional
            Stretch limits for ``source``/``target``/``convolved``.
            Either left as `None` (the default) is set automatically
            according to ``stretchAlgorithm``, using only ``source``
            and ``target`` (``self.convolved`` has gaps -- ``NaN`` --
            where `fitAlardLuptonKernel` could not compute a model, e.g.
            near the image edge or in a failed region).
        stretchAlgorithm : `str`
            Algorithm used to compute ``vmin``/``vmax`` automatically,
            if either is `None`: ``"zscale"`` (the default; the classic
            ds9/IRAF algorithm, computed per-image and then combined by
            taking the widest limits of the two) or ``"percentile"``
            (see ``percentile``, ``symmetric``).
        percentile : `float`
            Percentile used to set ``vmin``/``vmax`` automatically, if
            ``stretchAlgorithm`` is ``"percentile"``.
        symmetric : `bool`
            If ``stretchAlgorithm`` is ``"percentile"``, force the
            automatic ``source``/``target``/``convolved`` stretch to be
            symmetric about zero?
        zscaleSamples : `int`
            Number of pixels to sample, if ``stretchAlgorithm`` is
            ``"zscale"``.
        zscaleContrast : `float`
            Contrast parameter, if ``stretchAlgorithm`` is ``"zscale"``.
        diffVmin, diffVmax : `float`, optional
            Stretch limits for the difference image; as
            ``vmin``/``vmax`` but for the difference.
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
        showRegions : `bool`
            Draw the boundaries of the ``numRegionsX`` x ``numRegionsY``
            regions the kernel was fit in?
        showRejected : `bool`
            Mark, with a red hatched overlay on the difference panel,
            pixels rejected during the fit (the ``DIFFIM_REJECTED`` mask
            plane)? The overlay has no fill, so the underlying
            difference values remain visible through the hatching.
        showPartial : `bool`
            Mark, with a blue hatched overlay on the difference panel,
            pixels computed from only part of the kernel footprint
            because some source pixels within it were unusable (the
            ``DIFFIM_PARTIAL`` mask plane)? As with ``showRejected``,
            the overlay has no fill, so the underlying difference
            values remain visible through the hatching.
        figsize : `tuple` of `float`, optional
            Figure size, passed to ``matplotlib.pyplot.subplots``.
        fig : `matplotlib.figure.Figure`, optional
        axes : `numpy.ndarray` of `matplotlib.axes.Axes`, optional
            Existing figure and 2x2 grid of axes to plot into, instead
            of creating new ones. If either is provided, both must be.

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

        sourceArr = source.image.array
        targetArr = target.image.array
        diffArr = np.array(self.difference.image.array, dtype=float)  # copy: we set NO_DATA to NaN below
        convolvedArr = np.array(self.convolved.image.array, dtype=float)  # copy: ditto
        if sourceArr.shape != targetArr.shape or sourceArr.shape != diffArr.shape:
            raise ValueError(
                f"source ({sourceArr.shape}), target ({targetArr.shape}) and this result's "
                f"difference ({diffArr.shape}) must have the same shape"
            )
        if self.numRegionsX * self.numRegionsY != len(self.solutions):
            raise ValueError(
                f"Number of solutions ({len(self.solutions)}) does not match "
                f"numRegionsX * numRegionsY ({self.numRegionsX} * {self.numRegionsY})"
            )

        noDataBitMask = 1 << self.difference.mask.getMaskPlane("NO_DATA")
        noData = (self.difference.mask.array & noDataBitMask) != 0
        diffArr[noData] = np.nan
        convolvedArr[noData] = np.nan

        bbox = self.difference.getBBox()
        height, width = diffArr.shape
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
                        for arr in (sourceArr, targetArr)
                    )
                )
                autoVmin, autoVmax = min(z1s), max(z2s)
            elif stretchAlgorithm == "percentile":
                allValues = np.concatenate([sourceArr.ravel(), targetArr.ravel()])
                finite = allValues[np.isfinite(allValues)]
                if symmetric:
                    limit = np.percentile(np.abs(finite), percentile)
                    autoVmin, autoVmax = -limit, limit
                else:
                    autoVmin = np.percentile(finite, 100 - percentile)
                    autoVmax = np.percentile(finite, percentile)
            else:
                raise ValueError(
                    f"Unrecognized stretchAlgorithm: {stretchAlgorithm!r}; "
                    "expected 'zscale' or 'percentile'"
                )
            vmin = autoVmin if vmin is None else vmin
            vmax = autoVmax if vmax is None else vmax

        if diffVmin is None or diffVmax is None:
            diffFinite = diffArr[np.isfinite(diffArr)]
            if diffFinite.size == 0:
                autoDiffVmin, autoDiffVmax = -1.0, 1.0
            elif diffSymmetric:
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

        def hatchMaskPlane(maskPlane: str, color: str, hatch: str) -> None:
            """Hatch (rather than fill) the pixels flagged with maskPlane

            Hatching leaves the underlying difference values visible;
            filling would hide exactly the values we want to see.
            """
            bitMask = 1 << self.difference.mask.getMaskPlane(maskPlane)
            flagged = (self.difference.mask.array & bitMask) != 0
            if not flagged.any():
                return
            rowIndices, colIndices = np.mgrid[0:height, 0:width]
            xCenters = bbox.getMinX() + colIndices + 0.5
            yCenters = bbox.getMinY() + rowIndices + 0.5
            with plt.rc_context({"hatch.color": color}):
                axDiff.contourf(
                    xCenters,
                    yCenters,
                    flagged.astype(float),
                    levels=[0.5, 1.5],
                    colors="none",
                    hatches=[hatch],
                )

        if showRejected:
            hatchMaskPlane("DIFFIM_REJECTED", "red", "////")

        if showPartial:
            hatchMaskPlane("DIFFIM_PARTIAL", "blue", "....")

        if showRegions and (self.numRegionsX > 1 or self.numRegionsY > 1):
            numRegionsX, numRegionsY = self.numRegionsX, self.numRegionsY
            xBoundaries = [self.solutions[index].bbox.getMinX() for index in range(numRegionsX)]
            xBoundaries.append(self.solutions[numRegionsX - 1].bbox.getMaxX() + 1)
            yBoundaries = [self.solutions[index * numRegionsX].bbox.getMinY() for index in range(numRegionsY)]
            yBoundaries.append(self.solutions[(numRegionsY - 1) * numRegionsX].bbox.getMaxY() + 1)
            for axis in (axSource, axTarget, axConvolved, axDiff):
                for xx in xBoundaries[1:-1]:
                    axis.axvline(xx, color="grey", ls="--", lw=0.5)
                for yy in yBoundaries[1:-1]:
                    axis.axhline(yy, color="grey", ls="--", lw=0.5)

        for axis, title in zip((axSource, axTarget, axConvolved, axDiff), titles):
            axis.set_title(title)

        # See pfs.drp.stella.psfMatch.plotPsfMatchResult for why each colorbar is attached
        # via a divider on one specific Axes, rather than fig.colorbar(..., ax=[...]).
        sharedCax = make_axes_locatable(axTarget).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(imSource, cax=sharedCax)
        diffCax = make_axes_locatable(axDiff).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(imDiff, cax=diffCax)

        numRegions = len(self.solutions)
        numSuccess = sum(solution.success for solution in self.solutions)
        totalConsidered = self.numFit + self.numRejected
        fig.suptitle(
            f"{numSuccess}/{numRegions} region(s) succeeded; "
            f"{self.numRejected}/{totalConsidered} pixels rejected; "
            f"chi^2={self.chi2:.1f}"
        )

        return fig, axes
