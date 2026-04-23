"""Tune ``ScatteredLightModel`` parameters (``frac1``/``frac2`` and optionally
``powerLaw*``/``soften*``) against a set of post-ISR frames.

Single module, three entry points:

* :class:`TuneScatteredLightTask` — a Gen3 ``PipelineTask`` with dimensions
  ``(instrument, arm, spectrograph)``; one quantum per camera so ``pipetask
  run --processes 8`` fans out to all visible cameras in parallel.
* :func:`tuneCamerasInParallel` — notebook convenience that runs per-camera
  tuning across a ``multiprocessing.Pool``; each worker builds its own butler
  so nothing large crosses the queue.
* :func:`tuneScatteredLight` — pure-Python function taking pre-loaded
  ``postISRCCDs``, ``detectorMap``, ``fiberProfiles``; used by the Task and
  directly from notebooks.

:func:`diagnoseScatteredLight` produces a 4-panel comparison plot for a
(preferably held-out) set of frames.

Approach
--------
Pixels outside ±``halfIllum`` of every illuminated fiber centre should come
back to ~0 after a correct scatter correction. Bin those pixels by distance
to the nearest fiber centre; compute mean residual scatter/illum per bin
averaged over bright rows and frames; minimise the RMS over bins.

The FFT inversion ``clean = IFFT[FFT(img) / (1 + FFT(K))]`` equals the
production forward-subtract to first order in the (few-percent) scatter
fraction, so the tuned parameters transfer to the pipeline as-is.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
from typing import List, Mapping, Optional, Sequence

import numpy as np

from lsst.pex.config import Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base import QuantumContext, Struct
from lsst.pipe.base.connections import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
)
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from ..LayeredDetectorMapContinued import LayeredDetectorMap
from ..scatteredLight import ScatteredLightConfig

__all__ = (
    "TuneScatteredLightConnections",
    "TuneScatteredLightConfig",
    "TuneScatteredLightTask",
    "tuneCamerasInParallel",
    "tuneScatteredLight",
    "diagnoseScatteredLight",
)

_log = logging.getLogger(__name__)

DEFAULT_BIN_EDGES = (6, 10, 15, 25, 40, 80)
_KERNEL_PARAMS = ("frac1", "frac2", "powerLaw1", "powerLaw2", "soften1", "soften2")


# ── PipelineTask ────────────────────────────────────────────────────────────


class TuneScatteredLightConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "arm", "spectrograph"),
):
    """Connections for ``TuneScatteredLightTask``."""

    postISRCCD = InputConnection(
        name="postISRCCD",
        doc="Post-ISR exposures (pre-scatter-correction) used for tuning.",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    detectorMap = PrerequisiteConnection(
        name="detectorMap_calib",
        doc="Detector map (used for fiber x-centers).",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Fiber profiles (used to identify illuminated fibers).",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    tuneResult = OutputConnection(
        name="scatteredLightTune",
        doc="Tuned scattered-light parameters and grid diagnostics.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "arm", "spectrograph"),
    )


class TuneScatteredLightConfig(
    PipelineTaskConfig, pipelineConnections=TuneScatteredLightConnections,
):
    """Configuration for ``TuneScatteredLightTask``.

    Leave any ``*Grid`` field empty (default) to keep that parameter at the
    ``ScatteredLightConfig`` value for this camera; set a non-empty list to
    scan it.
    """

    frac1Grid = ListField(
        dtype=float,
        default=list(np.arange(0.010, 0.101, 0.005).round(4)),
        doc="Grid of frac1 values to scan (empty = keep fixed at config).",
    )
    frac2Grid = ListField(
        dtype=float,
        default=list(np.arange(0.005, 0.101, 0.005).round(4)),
        doc="Grid of frac2 values to scan (empty = keep fixed at config).",
    )
    powerLaw1Grid = ListField(
        dtype=float, default=[], optional=True,
        doc="Grid of powerLaw1 values (empty = keep fixed at config).",
    )
    powerLaw2Grid = ListField(
        dtype=float, default=[], optional=True,
        doc="Grid of powerLaw2 values (empty = keep fixed at config).",
    )
    soften1Grid = ListField(
        dtype=float, default=[], optional=True,
        doc="Grid of soften1 values (empty = keep fixed at config).",
    )
    soften2Grid = ListField(
        dtype=float, default=[], optional=True,
        doc="Grid of soften2 values (empty = keep fixed at config).",
    )
    halfIllum = Field(
        dtype=int, default=5,
        doc="Half-width of the illuminated-pixel mask around each fiber (columns).",
    )
    brightPercentile = Field(
        dtype=float, default=60.0,
        doc="Per-frame: rows whose mean on-fiber flux exceeds this percentile "
            "of positive per-row values are used in the metric.",
    )
    binEdges = ListField(
        dtype=int,
        default=list(DEFAULT_BIN_EDGES),
        doc="Distance bin edges (px from nearest illuminated fiber).",
    )


class TuneScatteredLightTask(PipelineTask):
    """Tune ``ScatteredLightModel`` ``frac1``/``frac2`` (and optionally
    kernel-shape parameters) against a set of post-ISR frames for one camera.

    Quantum dimensions are ``(instrument, arm, spectrograph)``. One quantum
    per camera, so the pipeline framework parallelises across cameras itself.
    """

    ConfigClass = TuneScatteredLightConfig
    _DefaultName = "tuneScatteredLight"

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        arm = outputRefs.tuneResult.dataId["arm"]
        spectrograph = outputRefs.tuneResult.dataId["spectrograph"]
        camera = f"{arm}{spectrograph}"

        postISRCCDs = [butler.get(ref) for ref in inputRefs.postISRCCD]
        detectorMap = butler.get(inputRefs.detectorMap)
        fiberProfiles = butler.get(inputRefs.fiberProfiles)
        visits = [int(ref.dataId["visit"]) for ref in inputRefs.postISRCCD]

        result = self.run(
            postISRCCDs=postISRCCDs,
            detectorMap=detectorMap,
            fiberProfiles=fiberProfiles,
            camera=camera,
            visits=visits,
        )
        butler.put(result.tuneResult, outputRefs.tuneResult)

    def run(
        self,
        postISRCCDs,
        detectorMap,
        fiberProfiles,
        camera: str,
        visits: Optional[List[int]] = None,
    ) -> Struct:
        """Tune scattered-light parameters for one camera.

        Parameters
        ----------
        postISRCCDs : sequence of `lsst.afw.image.Exposure`
        detectorMap : `pfs.drp.stella.DetectorMap`
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
        camera : `str`
            e.g. ``"b1"``. Used to look up current defaults from
            ``ScatteredLightConfig`` and for logging.
        visits : list[int], optional
            Visits the frames correspond to. Recorded in the output for
            provenance. If not given, frames are labelled by index.

        Returns
        -------
        struct : `Struct`
            ``tuneResult`` : plain-Python dict (butler ``StructuredDataDict``)
            with keys ``best``, ``best_rms``, ``best_bins``, ``grids``,
            ``rms_grid``, ``bin_grid``, ``binLabels``, ``raw_bins``, ``meta``.
            Numpy arrays are converted to lists so the dict is JSON-safe.
        """
        tuneGrids = {}
        for paramName, field in (
            ("frac1", self.config.frac1Grid),
            ("frac2", self.config.frac2Grid),
            ("powerLaw1", self.config.powerLaw1Grid),
            ("powerLaw2", self.config.powerLaw2Grid),
            ("soften1", self.config.soften1Grid),
            ("soften2", self.config.soften2Grid),
        ):
            vals = list(field) if field else []
            if vals:
                tuneGrids[paramName] = np.asarray(vals, dtype=np.float64)

        if not tuneGrids:
            raise ValueError(
                "At least one of frac1Grid/frac2Grid/... must be non-empty",
            )

        self.log.info(
            "[%s] tuning %s on %d frames",
            camera,
            ",".join(f"{k}({len(v)})" for k, v in tuneGrids.items()),
            len(postISRCCDs),
        )

        result = tuneScatteredLight(
            postISRCCDs=postISRCCDs,
            detectorMap=detectorMap,
            fiberProfiles=fiberProfiles,
            camera=camera,
            scatConfig=ScatteredLightConfig(),
            tuneGrids=tuneGrids,
            halfIllum=self.config.halfIllum,
            brightPercentile=self.config.brightPercentile,
            binEdges=tuple(self.config.binEdges),
        )

        best = result["best"]
        self.log.info(
            "[%s] best frac1=%.4f frac2=%.4f  RMS=%.5f",
            camera, best["frac1"], best["frac2"], result["best_rms"],
        )

        payload = _toJsonSafe(result)
        payload["visits"] = list(visits) if visits else list(range(len(postISRCCDs)))
        payload["camera"] = camera

        return Struct(tuneResult=payload)


# ── parallel per-camera helper (notebook use) ───────────────────────────────


def _tuneOneCameraWorker(args):
    """Worker for parallel per-camera tuning (importable at module level so it
    survives both ``fork`` and ``spawn``).

    Each worker builds its own butler and loads data itself — nothing large
    is pickled across the pool queue, avoiding ``VisitInfo`` pickling issues.
    """
    from lsst.daf.butler import Butler

    (
        camera, useVisits, calibVisit, repo, collections, configKwargs,
    ) = args

    butler = Butler(repo, collections=collections)
    dataId = dict(arm=camera[0], spectrograph=int(camera[1:]))

    detMap = butler.get("detectorMap_calib", dataId, visit=calibVisit)
    profiles = butler.get("fiberProfiles", dataId, visit=calibVisit)

    postISRs = []
    for v in useVisits:
        did = dict(dataId)
        did["visit"] = int(v)
        postISR = butler.get("postISRCCD", did)
        calexp = butler.get("calexp", did)
        postISR.mask.array[:] = calexp.mask.array[:]
        postISRs.append(postISR)

    config = TuneScatteredLightConfig()
    for key, val in (configKwargs or {}).items():
        setattr(config, key, val)
    task = TuneScatteredLightTask(config=config)
    struct = task.run(postISRs, detMap, profiles, camera=camera, visits=list(useVisits))
    return camera, struct.tuneResult


def tuneCamerasInParallel(
    repo: str,
    collections,
    cameras,
    useVisits,
    calibVisit: int,
    maxWorkers: int = 8,
    configKwargs: Optional[dict] = None,
    context: str = "spawn",
) -> dict:
    """Run ``TuneScatteredLightTask`` over multiple cameras in parallel processes.

    Each worker creates its own butler from ``(repo, collections)``, so no
    Exposure / DetectorMap objects cross the pool queue (avoids LSST C++
    pickling pitfalls, e.g. ``VisitInfo``).

    Parameters
    ----------
    repo : `str`
        Gen3 butler repo path (or URI).
    collections : sequence of `str`
        Input collections to search for postISRCCD + calibrations.
    cameras : sequence of `str`
        e.g. ``['b1','r1','b2','r2','b3','r3','b4','r4']``.
    useVisits : sequence of `int`
        Visits whose postISRCCD frames drive the tuning. Same set is used
        for every camera.
    calibVisit : `int`
        Visit used to resolve the detectorMap and fiberProfiles calibrations.
    maxWorkers : `int`
        Process pool size.
    configKwargs : dict[str, ...], optional
        Overrides applied to ``TuneScatteredLightConfig`` inside each worker
        (e.g., ``{"frac1Grid": [...], "brightPercentile": 70}``).
    context : `str`
        ``"spawn"`` (default, safest) or ``"fork"``.

    Returns
    -------
    results : dict[str, dict]
        Maps camera name to the ``tuneResult`` dict returned by the task.
    """
    jobs = [
        (c, list(useVisits), int(calibVisit), repo, list(collections), configKwargs)
        for c in cameras
    ]
    results: dict = {}
    with mp.get_context(context).Pool(maxWorkers) as pool:
        for camera, payload in pool.imap_unordered(_tuneOneCameraWorker, jobs):
            results[camera] = payload
            best = payload.get("best", {})
            print(
                f"[{camera}] frac1={best.get('frac1', float('nan')):.4f} "
                f"frac2={best.get('frac2', float('nan')):.4f}  "
                f"RMS={payload.get('best_rms', float('nan')):.5f}"
            )
    return {c: results.get(c) for c in cameras}


# ── core fitting function (used by Task and directly from notebooks) ────────


def tuneScatteredLight(
    postISRCCDs,
    detectorMap,
    fiberProfiles,
    camera: Optional[str] = None,
    scatConfig: Optional[ScatteredLightConfig] = None,
    tuneGrids: Optional[Mapping[str, np.ndarray]] = None,
    halfIllum: int = 5,
    brightPercentile: float = 60.0,
    binEdges: Sequence[int] = DEFAULT_BIN_EDGES,
) -> dict:
    """Optimize scattered-light kernel parameters against a set of frames.

    Parameters
    ----------
    postISRCCDs : sequence of `lsst.afw.image.Exposure`
        Post-ISR frames (pre-scatter-subtraction).
    detectorMap : `pfs.drp.stella.DetectorMap`
    fiberProfiles : `pfs.drp.stella.FiberProfileSet`
        Used to identify illuminated fibers via ``.fiberId``.
    camera : `str`, optional
        e.g. ``"b1"``. Used to look up current defaults from ``scatConfig``.
    scatConfig : `ScatteredLightConfig`, optional
        Source of any parameter not being tuned (kernel shape, halfSize).
        Defaults to a fresh ``ScatteredLightConfig()``.
    tuneGrids : dict[str, ndarray], optional
        Maps parameter name to 1-D grid of candidate values. Valid keys:
        ``frac1, frac2, powerLaw1, powerLaw2, soften1, soften2``.
        Default: tune ``frac1`` (0.010..0.100) and ``frac2`` (0.005..0.100).
    halfIllum : `int`
    brightPercentile : `float`
    binEdges : sequence of `int`

    Returns
    -------
    result : `dict`
        ``best``       : dict of best parameter values (all 6 kernel params)
        ``best_rms``   : float, RMS of bin residuals at best point
        ``best_bins``  : ndarray, bin residuals at best point
        ``grids``      : dict[str, ndarray] of the tune grids
        ``rms_grid``   : n-D array of RMS values over the tune grid
        ``bin_grid``   : (n+1)-D array, last axis = distance bins
        ``binLabels``  : list[str]
        ``raw_bins``   : ndarray, bin residuals with no correction
        ``meta``       : dict with shape info, halfSize, camera, etc.
    """
    scatConfig = scatConfig or ScatteredLightConfig()
    if tuneGrids is None:
        tuneGrids = dict(
            frac1=np.arange(0.010, 0.101, 0.005),
            frac2=np.arange(0.005, 0.101, 0.005),
        )
    _validateTuneGrids(tuneGrids)

    prep = _prepare(postISRCCDs, detectorMap, fiberProfiles,
                    halfIllum=halfIllum, brightPercentile=brightPercentile,
                    binEdges=binEdges)

    fixed = {p: float(scatConfig.getValue(p, camera or "")) for p in _KERNEL_PARAMS}
    halfSize = int(scatConfig.halfSize)

    names = list(tuneGrids.keys())
    grids = {n: np.asarray(tuneGrids[n], dtype=np.float64) for n in names}
    shape = tuple(len(grids[n]) for n in names)

    k1_cache = _kernelCache("k1", names, grids, fixed, halfSize, prep)
    k2_cache = _kernelCache("k2", names, grids, fixed, halfSize, prep)

    _log.info("Tune grid: %s -> %d points", dict(zip(names, shape)), int(np.prod(shape)))

    rms_grid = np.full(shape, np.nan)
    bin_grid = np.full(shape + (len(prep["binLabels"]),), np.nan)

    for flatIdx in np.ndindex(*shape):
        params = dict(fixed)
        for n, i in zip(names, flatIdx):
            params[n] = float(grids[n][i])
        K1_hat = k1_cache[(params["powerLaw1"], params["soften1"])]
        K2_hat = k2_cache[(params["powerLaw2"], params["soften2"])]
        bins = _meanBinResiduals(prep, K1_hat, K2_hat,
                                 params["frac1"], params["frac2"])
        rms_grid[flatIdx] = float(np.sqrt(np.nanmean(bins ** 2)))
        bin_grid[flatIdx] = bins

    imin = np.unravel_index(np.nanargmin(rms_grid), rms_grid.shape)
    best = dict(fixed)
    for n, i in zip(names, imin):
        best[n] = float(grids[n][i])

    raw_bins = _meanBinResiduals(prep, None, None, 0.0, 0.0)

    _log.info(
        "Best: %s  RMS=%.5f  bins=%s",
        {n: f"{best[n]:.4f}" for n in names},
        float(rms_grid[imin]),
        ", ".join(f"{lbl}={v:+.4f}"
                  for lbl, v in zip(prep["binLabels"], bin_grid[imin])),
    )

    return dict(
        best=best,
        best_rms=float(rms_grid[imin]),
        best_bins=bin_grid[imin],
        grids=grids,
        rms_grid=rms_grid,
        bin_grid=bin_grid,
        binLabels=prep["binLabels"],
        raw_bins=raw_bins,
        meta=dict(
            camera=camera,
            halfSize=halfSize,
            halfIllum=halfIllum,
            brightPercentile=brightPercentile,
            binEdges=list(binEdges),
            nFrames=len(postISRCCDs),
            H=prep["H"], W=prep["W"], xGap=prep["xGap"],
            nIllumFibers=prep["nIllumFibers"],
        ),
    )


def diagnoseScatteredLight(
    postISRCCDs,
    detectorMap,
    fiberProfiles,
    params: Mapping[str, float],
    currentParams: Optional[Mapping[str, float]] = None,
    camera: Optional[str] = None,
    scatConfig: Optional[ScatteredLightConfig] = None,
    halfIllum: int = 5,
    brightPercentile: float = 60.0,
    binEdges: Sequence[int] = DEFAULT_BIN_EDGES,
    plotPath: Optional[str] = None,
    show: bool = False,
):
    """Generate diagnostic plots on a (preferably held-out) set of frames.

    Compares three configurations:
      - raw (no scatter subtraction)
      - current (from ``currentParams``, or ``scatConfig.getValue(..., camera)``)
      - proposed (``params``)

    Returns
    -------
    summary : `dict`
    fig : `matplotlib.figure.Figure`
    """
    import matplotlib.pyplot as plt

    scatConfig = scatConfig or ScatteredLightConfig()

    def fill(p):
        full = {n: float(scatConfig.getValue(n, camera or "")) for n in _KERNEL_PARAMS}
        for k, v in (p or {}).items():
            if k not in _KERNEL_PARAMS:
                raise ValueError(f"unknown kernel parameter: {k!r}")
            full[k] = float(v)
        return full

    proposed = fill(params)
    current = fill(currentParams or {})
    halfSize = int(scatConfig.halfSize)

    prep = _prepare(postISRCCDs, detectorMap, fiberProfiles,
                    halfIllum=halfIllum, brightPercentile=brightPercentile,
                    binEdges=binEdges)

    def k_hats(p):
        return (
            _unitKernelFft(p["powerLaw1"], p["soften1"], halfSize,
                           (prep["H"], prep["wp"])),
            _unitKernelFft(p["powerLaw2"], p["soften2"], halfSize,
                           (prep["H"], prep["wp"])),
        )

    K1_cur, K2_cur = k_hats(current)
    K1_prop, K2_prop = k_hats(proposed)

    configs = [
        ("raw", 0.0, 0.0, None, None, "k"),
        ("current", current["frac1"], current["frac2"], K1_cur, K2_cur, "#d62728"),
        ("proposed", proposed["frac1"], proposed["frac2"], K1_prop, K2_prop, "#2ca02c"),
    ]

    per_visit = {name: [] for name, *_ in configs}
    per_row = {name: [] for name, *_ in configs}
    for frame in prep["frames"]:
        for name, f1, f2, K1, K2, _ in configs:
            clean = _deconvolveOne(frame, K1, K2, f1, f2, prep)
            per_visit[name].append(_binRatios(clean, frame, prep["illum_mask"]))
            per_row[name].append(_perRowRatio(clean, frame, prep["illum_mask"]))

    mean_bins = {name: np.nanmean(per_visit[name], axis=0) for name, *_ in configs}
    rms = {name: float(np.sqrt(np.nanmean(mean_bins[name] ** 2)))
           for name, *_ in configs}

    _log.info("Diagnostic summary (N=%d frames):", len(postISRCCDs))
    for name, *_ in configs:
        _log.info(
            "  %-8s RMS=%.5f  bins=%s", name, rms[name],
            ", ".join(f"{lbl}={v:+.4f}"
                      for lbl, v in zip(prep["binLabels"], mean_bins[name])),
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    bin_centers = 0.5 * (np.array(binEdges[:-1]) + np.array(binEdges[1:]))

    ax = axes[0, 0]
    for name, f1, f2, _, _, color in configs:
        ax.plot(bin_centers, mean_bins[name], "-o", color=color, lw=1.3,
                label=f"{name} ({f1:.3f},{f2:.3f}) RMS={rms[name]:.4f}")
    ax.axhline(0, color="k", lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("distance from fiber (px)")
    ax.set_ylabel("scatter / illum")
    ax.set_title("Per-bin residual (mean over frames)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    x = np.arange(len(prep["binLabels"]))
    width = 0.28
    for k, (name, f1, f2, *_, color) in enumerate(configs):
        ax.bar(x + (k - 1) * width, mean_bins[name], width,
               color=color, alpha=0.9, label=name)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(prep["binLabels"], fontsize=8)
    ax.set_xlabel("distance bin")
    ax.set_ylabel("scatter / illum")
    ax.set_title("Per-bin residual (bar view)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 0]
    H = prep["H"]
    for name, f1, f2, _, _, color in configs:
        stacked = np.nanmean(per_row[name], axis=0)
        ax.plot(np.arange(H), stacked, color=color, lw=0.5, alpha=0.8, label=name)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("row")
    ax.set_ylabel("scatter / illum (mean over frames)")
    ax.set_title("Per-row residual (mean over frames)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    nv = len(postISRCCDs)
    xs = np.arange(nv)
    width = 0.28
    rms_per_visit = {
        name: [float(np.sqrt(np.nanmean(per_visit[name][i] ** 2))) for i in range(nv)]
        for name, *_ in configs
    }
    for k, (name, f1, f2, *_, color) in enumerate(configs):
        ax.bar(xs + (k - 1) * width, rms_per_visit[name], width,
               color=color, alpha=0.9, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"#{i}" for i in range(nv)], fontsize=8)
    ax.set_xlabel("frame index")
    ax.set_ylabel("per-frame RMS")
    ax.set_title("RMS per frame")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    title = f"Scattered-light diagnostic ({camera or 'unknown camera'}, N={nv})"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if plotPath is not None:
        fig.savefig(plotPath, dpi=120)
        _log.info("Saved %s", plotPath)
    if not show:
        plt.close(fig)

    return dict(
        mean_bins=mean_bins, rms=rms,
        rms_per_visit=rms_per_visit,
        per_visit_bins=per_visit,
        binLabels=prep["binLabels"],
    ), fig


# ── internals ───────────────────────────────────────────────────────────────


def _validateTuneGrids(tuneGrids):
    if not tuneGrids:
        raise ValueError("tuneGrids must have at least one parameter")
    for n in tuneGrids:
        if n not in _KERNEL_PARAMS:
            raise ValueError(
                f"unknown parameter {n!r}; must be one of {_KERNEL_PARAMS}"
            )


def _prepare(postISRCCDs, detectorMap, fiberProfiles,
             halfIllum, brightPercentile, binEdges):
    """Build shared mask + distance map and per-frame FFT/masks/bright rows."""
    bbox = detectorMap.getBBox()
    H = bbox.getHeight()
    W = bbox.getWidth()
    if isinstance(detectorMap, LayeredDetectorMap):
        xGap = -int(detectorMap.rightCcd.getTranslation().getX() + 0.5)
    else:
        xGap = 0
    ccdSplit = W // 2
    wp = W + xGap

    illumIds = {int(fid) for fid in fiberProfiles.fiberId}
    illum_centers = []
    illum_mask = np.zeros((H, W), dtype=bool)
    rows = np.arange(H, dtype=np.int64)
    for fid in detectorMap.fiberId:
        if int(fid) not in illumIds:
            continue
        xc = detectorMap.getXCenter(int(fid))
        illum_centers.append(xc.astype(np.float32))
        ix = np.round(xc).astype(np.int64)
        for dr in range(-halfIllum, halfIllum + 1):
            illum_mask[rows, np.clip(ix + dr, 0, W - 1)] = True
    if not illum_centers:
        raise RuntimeError("No illuminated fibers in profiles.fiberId")

    illum_arr = np.asarray(illum_centers, dtype=np.float32)
    col_arr = np.arange(W, dtype=np.float32)
    dist_map = np.full((H, W), np.inf, dtype=np.float32)
    for r in range(H):
        sc = np.sort(illum_arr[:, r])
        ins = np.clip(np.searchsorted(sc, col_arr), 1, len(sc) - 1)
        dist_map[r] = np.minimum.reduce([
            np.abs(col_arr - sc[ins - 1]),
            np.abs(col_arr - sc[ins]),
            np.abs(col_arr - sc[0]),
            np.abs(col_arr - sc[-1]),
        ])

    binLabels = [f"{binEdges[i]}-{binEdges[i+1]}px"
                 for i in range(len(binEdges) - 1)]

    frames = []
    for idx, exp in enumerate(postISRCCDs):
        img = exp.image.array.astype(np.float64)
        bad = exp.mask.array.astype(np.uint32) != 0
        scatter_mask = (~illum_mask) & (~bad)
        bin_masks = [
            scatter_mask
            & (dist_map >= binEdges[i]) & (dist_map < binEdges[i + 1])
            for i in range(len(binEdges) - 1)
        ]
        illum_per_row = np.array([
            img[r, illum_mask[r]].mean() if illum_mask[r].any() else 0.0
            for r in range(H)
        ])
        positive = illum_per_row[illum_per_row > 0]
        thr = np.percentile(positive, brightPercentile) if positive.size else 0.0
        bright_rows = np.where(illum_per_row > thr)[0]

        imgp = np.zeros((H, wp))
        imgp[:, :ccdSplit] = img[:, :ccdSplit]
        imgp[:, ccdSplit + xGap:] = img[:, ccdSplit:]
        img_fft = np.fft.rfft2(imgp)

        _log.info(
            "  frame %d: shape=%dx%d  median=%.1f  bright rows=%d (thr=%.1f)",
            idx, H, W, float(np.median(img)), len(bright_rows), thr,
        )
        frames.append(dict(
            img=img, scatter_mask=scatter_mask, bin_masks=bin_masks,
            illum_per_row=illum_per_row, bright_rows=bright_rows,
            img_fft=img_fft,
        ))

    _log.info(
        "Prepared %d frames, %d illuminated fibers, xGap=%d, illum mask %.1f%%",
        len(frames), len(illum_centers), xGap, illum_mask.mean() * 100,
    )

    return dict(
        H=H, W=W, wp=wp, ccdSplit=ccdSplit, xGap=xGap,
        illum_mask=illum_mask, dist_map=dist_map,
        binEdges=list(binEdges), binLabels=binLabels,
        frames=frames, nIllumFibers=len(illum_centers),
    )


def _kernelCache(which, tunedNames, grids, fixed, halfSize, prep):
    """Precompute unit FFTs for all (PL, soften) combos needed by the grid."""
    if which == "k1":
        plName, softName = "powerLaw1", "soften1"
    else:
        plName, softName = "powerLaw2", "soften2"
    pls = grids[plName].tolist() if plName in tunedNames else [fixed[plName]]
    softs = grids[softName].tolist() if softName in tunedNames else [fixed[softName]]
    cache = {}
    for pl, s in itertools.product(pls, softs):
        cache[(float(pl), float(s))] = _unitKernelFft(
            pl, s, halfSize, (prep["H"], prep["wp"]),
        )
    return cache


def _unitKernelFft(powerLaw, soften, halfSize, shape):
    """FFT of unit-normalized power-law kernel, built in padded layout."""
    h, w = shape
    iy = np.arange(h, dtype=np.float32)
    ix = np.arange(w, dtype=np.float32)
    cy = np.where(iy <= h // 2, iy, iy - h)
    cx = np.where(ix <= w // 2, ix, ix - w)
    dy, dx = np.meshgrid(cy, cx, indexing="ij")
    in_range = (np.abs(dy) <= halfSize) & (np.abs(dx) <= halfSize)
    rr2 = np.maximum(dx**2 + soften**2 + dy**2, 1.0)
    k = np.where(in_range, rr2.astype(np.float64) ** (-powerLaw / 2), 0.0)
    del dy, dx, rr2, in_range
    k /= k.sum()
    return np.fft.rfft2(k)


def _deconvolveOne(frame, K1_hat, K2_hat, frac1, frac2, prep):
    """Deconvolve one frame. If frac1=frac2=0, returns the raw image unchanged."""
    if frac1 == 0.0 and frac2 == 0.0:
        return frame["img"]
    Khat = frac1 * K1_hat + frac2 * K2_hat
    dp = np.fft.irfft2(frame["img_fft"] / (1.0 + Khat), s=(prep["H"], prep["wp"]))
    H, W, ccdSplit, xGap = prep["H"], prep["W"], prep["ccdSplit"], prep["xGap"]
    res = np.empty((H, W))
    res[:, :ccdSplit] = dp[:, :ccdSplit]
    res[:, ccdSplit:] = dp[:, ccdSplit + xGap:]
    return res


def _binRatios(img, frame, illum_mask):
    br = frame["bright_rows"]
    il = np.array([
        img[r, illum_mask[r]].mean() if illum_mask[r].any() else np.nan
        for r in br
    ])
    out = np.zeros(len(frame["bin_masks"]))
    for i, m in enumerate(frame["bin_masks"]):
        sc = np.array([
            img[r, m[r]].mean() if m[r].any() else np.nan for r in br
        ])
        out[i] = np.nanmean(sc / il)
    return out


def _perRowRatio(img, frame, illum_mask):
    H = img.shape[0]
    sc = np.array([
        img[r, frame["scatter_mask"][r]].mean()
        if frame["scatter_mask"][r].any() else np.nan
        for r in range(H)
    ])
    il = np.array([
        img[r, illum_mask[r]].mean() if illum_mask[r].any() else np.nan
        for r in range(H)
    ])
    return sc / il


def _meanBinResiduals(prep, K1_hat, K2_hat, frac1, frac2):
    """Mean per-bin residual averaged over frames."""
    per_visit = []
    for frame in prep["frames"]:
        clean = _deconvolveOne(frame, K1_hat, K2_hat, frac1, frac2, prep)
        per_visit.append(_binRatios(clean, frame, prep["illum_mask"]))
    return np.nanmean(per_visit, axis=0)


def _toJsonSafe(obj):
    """Recursively convert numpy arrays / scalars to plain Python."""
    if isinstance(obj, dict):
        return {k: _toJsonSafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_toJsonSafe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
