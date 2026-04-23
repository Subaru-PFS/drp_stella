import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from scipy.spatial import Delaunay, Voronoi

from pfs.datamodel.pfsConfig import FiberStatus, TargetType


__all__ = ["plot2dSpectrumSlice", "plotPFISkyBlocking", "plotScatterModel", "plotIllumModel"]


def plot2dSpectrumSlice(exposure, pfsConfig, detectorMap, title="", r0=1980, r1=2020,
                        nrows=3, ncols=2, overlap=50):
    """Plot a slice through all the columns of a 2-D spectrum in
    nrows x ncols panels

    exposure : `lsst.afw.image.Image` 2-D spectrum
    pfsConfig: FfsConfig for exposure
    detectorMap: DetectorMap for exposure
    r0, r1: `int`  Use mean of flux in rows r0..r1
    nrows: `int` Number of subplots per row (passed to plt.subplots)
    ncols: `int` Number of subplots per column  (passed to plt.subplots)
    overlap: `int` how much to overlap the pixel range in the panels
    """
    fig, axs = plt.subplots(nrows, ncols, sharey=True)
    axs = axs.flatten()
    n = len(axs)   # number of panels
    xlen = exposure.getWidth()//n + 1

    for i in range(n):
        plt.sca(axs[i])
        axs[i].label_outer()

        plt.plot(np.median(exposure.image.array[r0:r1 + 1, :], axis=0))

        for fid in detectorMap.fiberId:
            if fid not in pfsConfig.fiberId:
                continue

            xc = detectorMap.getXCenter(fid)[(r0 + r1)//2]
            ind = pfsConfig.selectFiber(fid)[0]
            color = 'red' if pfsConfig.targetType[ind] == TargetType.SUNSS_DIFFUSE else 'green'
            ls = ':' if pfsConfig.fiberStatus[ind] == FiberStatus.BROKENFIBER else '-'

            plt.axvline(xc, ls=ls, color=color, alpha=0.25, zorder=-1)

        x01 = np.array([i*xlen - overlap, (i + 1)*xlen + overlap])
        if i == 0:
            x01 += overlap
        elif i == n - 1:
            x01 -= overlap
        plt.xlim(*x01)

        plt.axhline(0, ls='-', color='black', alpha=0.5)

    ax = fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel('Column', labelpad=10)  # Use argument `labelpad` to move label downwards.
    ax.set_ylabel('Mean flux', labelpad=20)
    plt.suptitle(title + f"  rows {r0}..{r1}", y=1.0)

    plt.tight_layout()

    return fig


def plotPFISkyBlocking(pfsConfig, nBlocks, ax=None, seed=42):
    """Visualise how sky fibers on the PFI are partitioned into spatial blocks.

    Clusters sky fibers into ``nBlocks`` groups by (x, y) position using
    k-means, then draws:

    - all fibers (grey) for context
    - sky fibers coloured by block assignment
    - Voronoi cells of the block centroids (the actual spatial partition)
    - Delaunay triangulation of the centroids (the interpolation structure)
    - per-block fiber count (SNR proxy: noise ∝ 1/√n)

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig`
        Fiber configuration for the exposure.
    nBlocks : `int`
        Number of spatial blocks (= number of Delaunay nodes).
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw into; created if not provided.
    seed : `int`, optional
        Random seed for k-means initialisation reproducibility.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
    centroids : `np.ndarray`, shape (nBlocks, 2)
        Block centroid positions [mm].
    labels : `np.ndarray`, shape (nSkyFibers,)
        Block index for each selected sky fiber.
    """
    # --- select sky fibers with known positions ---
    isSky = pfsConfig.getSelection(targetType=TargetType.SKY)
    hasPos = ~np.isnan(pfsConfig.pfiCenter).any(axis=1)
    select = isSky & hasPos
    if np.sum(select) < nBlocks:
        raise ValueError(
            f"Only {np.sum(select)} sky fibers with positions; cannot form {nBlocks} blocks."
        )

    xyAll = pfsConfig.pfiCenter                 # (nFibers, 2), mm
    xySky = pfsConfig.pfiCenter[select]         # (nSky, 2)

    # --- k-means clustering ---
    centroids, labels = kmeans2(xySky.astype(np.float64), nBlocks, minit='points', iter=50, rng=seed)

    # --- Delaunay triangulation of centroids ---
    tri = Delaunay(centroids)

    # --- Voronoi diagram of centroids (finite regions only) ---
    vor = Voronoi(centroids)

    # --- plot ---
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # all fibers as grey context
    ax.scatter(xyAll[:, 0], xyAll[:, 1], c='lightgrey', s=4, zorder=1, label='all fibers')

    # sky fibers coloured by block
    cmap = plt.cm.get_cmap('tab20', nBlocks)
    sc = ax.scatter(xySky[:, 0], xySky[:, 1], c=labels, cmap=cmap,
                    vmin=-0.5, vmax=nBlocks - 0.5, s=18, zorder=3, label='sky fibers')

    # Voronoi edges (finite segments only)
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            p0, p1 = vor.vertices[simplex]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'k-', lw=0.5, alpha=0.4, zorder=2)

    # Delaunay triangulation
    ax.triplot(centroids[:, 0], centroids[:, 1], tri.simplices,
               'k-', lw=1.0, alpha=0.7, zorder=4)

    # centroids with per-block fiber count
    counts = np.bincount(labels, minlength=nBlocks)
    for i, (cx, cy) in enumerate(centroids):
        ax.scatter(cx, cy, c='black', s=60, marker='+', zorder=5)
        ax.text(cx, cy, f' {counts[i]}', fontsize=6, va='center', zorder=6,
                color='black')

    margin = 20  # mm
    ax.set_xlim(xySky[:, 0].min() - margin, xySky[:, 0].max() + margin)
    ax.set_ylim(xySky[:, 1].min() - margin, xySky[:, 1].max() + margin)
    ax.set_xlabel('PFI x [mm]')
    ax.set_ylabel('PFI y [mm]')
    ax.set_aspect('equal')
    ax.set_title(
        f'{nBlocks} blocks — {np.sum(select)} sky fibers — '
        f'median {int(np.median(counts))} fibers/block '
        f'(SNR gain ×{np.sqrt(np.median(counts)):.1f})'
    )
    plt.colorbar(sc, ax=ax, label='block index')

    return ax, centroids, labels


SP_COLORS = {1: "C0", 2: "C1", 3: "C2", 4: "C3"}


def plotScatterModel(dfs2, mergedSpec, waveBins, ylim=(0.93, 1.22)):
    """Per-spectrograph scatter model vs fiberId, one panel per wavelength bin.

    Parameters
    ----------
    dfs2      : DataFrame output of ``fitModels``. Should carry ``visit`` and
                ``quartzVisit`` columns (populated by ``evaluate``) so the
                suptitle can reference them.
    mergedSpec: merged spectra object with .spectrograph attribute
    waveBins  : sequence of (wmin, wmax) tuples. Bin centres are derived as
                ``int(round((wmin + wmax) / 2))`` to match the ``wavelength``
                column produced by ``evaluate``.
    ylim      : y-axis limits

    Returns
    -------
    fig : Figure
    """
    visit, quartzVisit = _extractVisitIds(dfs2)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(22, 20), sharey=True)
    title = "Scatter model: twilight/quartz vs fiberId"
    if visit is not None or quartzVisit is not None:
        title += f"  (twilight={visit}, quartz={quartzVisit})"
    fig.suptitle(title, fontsize=18)

    for ax, (wmin, wmax) in zip(axs.flat, waveBins):
        wv = int(round((wmin + wmax) / 2))
        dfi = dfs2[dfs2.wavelength == wv]

        for spectrograph in [1, 2, 3, 4]:
            specMask = mergedSpec.spectrograph == spectrograph
            perSpec = dfi[specMask]
            color = SP_COLORS[spectrograph]

            ax.plot(perSpec.fiberId, perSpec.illumCorr, ".", color=color,
                    alpha=0.3, ms=2)
            ax.plot(perSpec.fiberId, perSpec.scatModel, "-", color=color,
                    lw=1.5, label=f"sp{spectrograph}")

        ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
        ax.set_title(f"{wv} nm  [{wmin:g}–{wmax:g}]", fontsize=14)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.3)

    for ax in axs[-1]:
        ax.set_xlabel("fiberId", fontsize=13)
    for ax in axs[:, 0]:
        ax.set_ylabel("twilight / quartz (norm.)", fontsize=13)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles[:4], labels[:4], loc="lower center", ncol=4,
               fontsize=12, frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return fig


def plotIllumModel(dfs2, x, y, waveBins, vmin=-5, vmax=5):
    """Residual 2D illumination on the PFI focal plane, one panel per wavelength bin.

    Parameters
    ----------
    dfs2      : DataFrame output of ``fitModels``. Should carry ``visit`` and
                ``quartzVisit`` columns (populated by ``evaluate``) for title
                annotation.
    x, y      : PFI coordinates (mm), same ordering as dfs2 fibers per wavelength
    waveBins  : sequence of (wmin, wmax) tuples. Bin centres are derived as
                ``int(round((wmin + wmax) / 2))`` to match the ``wavelength``
                column produced by ``evaluate``.
    vmin, vmax: colour scale limits in percent

    Returns
    -------
    fig : Figure
    """
    visit, quartzVisit = _extractVisitIds(dfs2)

    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(24, 22))
    title = "Residual illumination after scatter correction [% from unity]"
    if visit is not None or quartzVisit is not None:
        title += f"  (twilight={visit}, quartz={quartzVisit})"
    fig.suptitle(title, fontsize=18)

    for ax, (wmin, wmax) in zip(axs.flat, waveBins):
        wv = int(round((wmin + wmax) / 2))
        dfi = dfs2[dfs2.wavelength == wv]
        normalized = 100 * (dfi.scatCorr.to_numpy() - 1)

        sc = ax.scatter(x, y, c=normalized, cmap="bwr", vmin=vmin, vmax=vmax,
                        s=6, rasterized=True)
        cb = fig.colorbar(sc, ax=ax, label="%", shrink=0.8)
        cb.ax.tick_params(labelsize=12)
        cb.set_label("%", fontsize=13)
        ax.set_title(f"{wv} nm  [{wmin:g}–{wmax:g}]", fontsize=14)
        ax.tick_params(labelsize=11)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    for ax in axs[-1]:
        ax.set_xlabel("PFI x [mm]", fontsize=13)
    for ax in axs[:, 0]:
        ax.set_ylabel("PFI y [mm]", fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def _extractVisitIds(dfs2):
    """Return (visit, quartzVisit) from a dfs2 DataFrame, or (None, None)."""
    def _unique(col):
        if col not in dfs2.columns:
            return None
        vals = dfs2[col].dropna().unique()
        if len(vals) != 1:
            return None
        try:
            v = int(vals[0])
        except (TypeError, ValueError):
            return None
        return v if v >= 0 else None
    return _unique("visit"), _unique("quartzVisit")
