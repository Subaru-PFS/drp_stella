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
    useGrossBg = Field(
        dtype=bool, default=True,
        doc="Subtract a per-row linear background (anchored at the CCD edges "
            "beyond the slit boundary) before kernel optimisation, so the "
            "kernel is tuned only against fiber-correlated scatter.",
    )
    grossBgEdgeWidth = Field(
        dtype=int, default=20,
        doc="Width (in cols) of the outer-edge anchor bands used for the "
            "gross bg fit. Only the leftmost N cols of CCD1 and the "
            "rightmost N cols of CCD2 are used (the inner-edges next to "
            "the CCD gap are too close to fibers and would absorb halo).",
    )
    useFineBg = Field(
        dtype=bool, default=False,
        doc="Inside the kernel-evaluation cost, also apply the iterative "
            "fine 2-D polynomial bg refinement (in physical pixel coords) "
            "after FFT-IF. ~3× more expensive per grid point but tunes the "
            "kernel against the FULL production pipeline residual.",
    )
    fineBgIter = Field(
        dtype=int, default=1,
        doc="Number of fine-bg refinement iterations per FFT-IF call.",
    )
    fineBgDegCol = Field(
        dtype=int, default=2,
        doc="Degree of the fine bg polynomial in the (physical) column.",
    )
    fineBgDegRow = Field(
        dtype=int, default=3,
        doc="Degree of the fine bg polynomial in the row direction.",
    )
    useIterative = Field(
        dtype=bool, default=False,
        doc="Use the production-style iterative pipeline as the cost: in "
            "physical-pixel space, alternate between a gross deg=2 bg fit "
            "(through 3 anchors: ccd1-outer, middle, ccd2-outer) and the "
            "FFT inverse filter for `nIter` iterations. Cost is then the "
            "residual at 8 specific scatter regions (4 per CCD).",
    )
    nIter = Field(
        dtype=int, default=3,
        doc="Number of iterations of (grossBg, FFT-IF) when useIterative.",
    )
    outerRegionWeight = Field(
        dtype=float, default=5.0,
        doc="Weight multiplier on the outermost cost regions (CCD1 outer-left "
            "and CCD2 outer-right) in the weighted-RMS aggregation. Higher → "
            "the optimiser is more strongly anchored to bg ≈ 0 at the truly "
            "outer edges, which breaks the bg-vs-kernel-tail degeneracy "
            "(an additive bg shift biases the outer regions; the kernel tail "
            "biases them less). Set to 1.0 to disable.",
    )
    radialPenaltyWeight = Field(
        dtype=float, default=1.0,
        doc="Weight on the radial-shape penalty in the cost: "
            "|<bins[near]> − <bins[far]>| where near and far are regions with "
            "`mean_dist < radialNearThreshold` and `> radialFarThreshold` "
            "respectively. Penalises kernels whose radial shape mismatches "
            "the residual radial profile (complementary to the weighted RMS, "
            "which only sees overall amplitude). Set to 0.0 to disable.",
    )
    radialNearThreshold = Field(
        dtype=float, default=30.0,
        doc="Cost regions with mean distance-to-nearest-fiber below this "
            "value (px, in physical coords) are 'near' for the radial-shape "
            "penalty.",
    )
    radialFarThreshold = Field(
        dtype=float, default=60.0,
        doc="Cost regions with mean distance-to-nearest-fiber above this "
            "value (px, in physical coords) are 'far' for the radial-shape "
            "penalty.",
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
            useGrossBg=self.config.useGrossBg,
            grossBgEdgeWidth=self.config.grossBgEdgeWidth,
            useFineBg=self.config.useFineBg,
            fineBgIter=self.config.fineBgIter,
            fineBgDegCol=self.config.fineBgDegCol,
            fineBgDegRow=self.config.fineBgDegRow,
            useIterative=self.config.useIterative,
            nIter=self.config.nIter,
            outerRegionWeight=self.config.outerRegionWeight,
            radialPenaltyWeight=self.config.radialPenaltyWeight,
            radialNearThreshold=self.config.radialNearThreshold,
            radialFarThreshold=self.config.radialFarThreshold,
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
    useGrossBg: bool = True,
    grossBgEdgeWidth: int = 20,
    useFineBg: bool = False,
    fineBgIter: int = 1,
    fineBgDegCol: int = 2,
    fineBgDegRow: int = 3,
    useIterative: bool = False,
    nIter: int = 3,
    outerRegionWeight: float = 5.0,
    radialPenaltyWeight: float = 1.0,
    radialNearThreshold: float = 30.0,
    radialFarThreshold: float = 60.0,
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
                    binEdges=binEdges, useGrossBg=useGrossBg,
                    grossBgEdgeWidth=grossBgEdgeWidth,
                    outerRegionWeight=outerRegionWeight)
    fineBgKwargs = dict(
        useFineBg=useFineBg, fineBgIter=fineBgIter,
        fineBgDegCol=fineBgDegCol, fineBgDegRow=fineBgDegRow,
    )

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

    if useIterative:
        # Each grid point evaluated as: nIter-iteration physical-pixel
        # pipeline, cost = weighted RMS over the 8 cost regions plus a
        # radial-shape penalty (see `_aggregateCost`).
        n_regions = len(prep["cost_regions"])
        bin_grid = np.full(shape + (n_regions,), np.nan)
        cost_kw = dict(
            weights=prep["cost_region_weights"],
            dists=prep["cost_region_dists"],
            radial_weight=radialPenaltyWeight,
            near_thresh=radialNearThreshold,
            far_thresh=radialFarThreshold,
        )
        for flatIdx in np.ndindex(*shape):
            params = dict(fixed)
            for n, i in zip(names, flatIdx):
                params[n] = float(grids[n][i])
            K1_hat = k1_cache[(params["powerLaw1"], params["soften1"])]
            K2_hat = k2_cache[(params["powerLaw2"], params["soften2"])]
            bins = _meanRegionResiduals(prep, K1_hat, K2_hat,
                                         params["frac1"], params["frac2"],
                                         n_iter=nIter)
            rms_grid[flatIdx] = _aggregateCost(bins, **cost_kw)
            bin_grid[flatIdx] = bins
        raw_bins = _meanRegionResiduals(prep, None, None, 0.0, 0.0,
                                         n_iter=nIter)
    else:
        for flatIdx in np.ndindex(*shape):
            params = dict(fixed)
            for n, i in zip(names, flatIdx):
                params[n] = float(grids[n][i])
            K1_hat = k1_cache[(params["powerLaw1"], params["soften1"])]
            K2_hat = k2_cache[(params["powerLaw2"], params["soften2"])]
            bins = _meanBinResiduals(prep, K1_hat, K2_hat,
                                     params["frac1"], params["frac2"],
                                     **fineBgKwargs)
            rms_grid[flatIdx] = float(np.sqrt(np.nanmean(bins ** 2)))
            bin_grid[flatIdx] = bins
        raw_bins = _meanBinResiduals(prep, None, None, 0.0, 0.0)

    imin = np.unravel_index(np.nanargmin(rms_grid), rms_grid.shape)
    best = dict(fixed)
    for n, i in zip(names, imin):
        best[n] = float(grids[n][i])

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
    useGrossBg: bool = True,
    grossBgEdgeWidth: int = 20,
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
                    binEdges=binEdges, useGrossBg=useGrossBg,
                    grossBgEdgeWidth=grossBgEdgeWidth)

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


def _imgToPhysical(img, bad, ccd_split, x_gap):
    """Insert the physical CCD gap (xGap zero rows in the cross-disp dir)
    so cols become physical coords.  Returns (imgp, badp) of shape (H, W+xGap).
    Gap region: image=0, bad=True (excluded from fits)."""
    H, W = img.shape
    WP = W + x_gap
    imgp = np.zeros((H, WP), dtype=img.dtype)
    badp = np.ones((H, WP), dtype=bool)
    imgp[:, :ccd_split] = img[:, :ccd_split]
    badp[:, :ccd_split] = bad[:, :ccd_split]
    imgp[:, ccd_split + x_gap:] = img[:, ccd_split:]
    badp[:, ccd_split + x_gap:] = bad[:, ccd_split:]
    return imgp, badp


def _imgFromPhysical(imgp, ccd_split, x_gap):
    """Inverse of `_imgToPhysical`: drop the gap region, returning array coords."""
    H, WP = imgp.shape
    W = WP - x_gap
    out = np.empty((H, W), dtype=imgp.dtype)
    out[:, :ccd_split] = imgp[:, :ccd_split]
    out[:, ccd_split:] = imgp[:, ccd_split + x_gap:]
    return out


def _fillGapLinear(imgp, badp, ccd_split, x_gap, ramp_width=8):
    """Linearly-interpolate imgp across the physical CCD gap, per row.

    Why: the FFT-based inverse filter sees a sharp ~xGap-px zero step at the
    gap boundary. The step is broadband in Fourier space, and the kernel ends
    up "explaining" it — biasing the inner-edge cost regions. A linear ramp
    between the boundary medians removes that step at zero cost, since the
    gap region is dropped on output anyway (`_imgFromPhysical`).

    Per-row median over the ``ramp_width`` boundary columns on each side.
    Bad pixels are excluded; rows with no good boundary pixels keep 0.
    Does not modify ``badp`` — the gap stays flagged bad for fits.
    """
    if x_gap <= 0:
        return imgp
    gap_lo, gap_hi = ccd_split, ccd_split + x_gap
    left_lo = max(0, ccd_split - ramp_width)
    right_hi = min(imgp.shape[1], gap_hi + ramp_width)
    left_block = imgp[:, left_lo:ccd_split].astype(np.float64, copy=True)
    left_block[badp[:, left_lo:ccd_split]] = np.nan
    right_block = imgp[:, gap_hi:right_hi].astype(np.float64, copy=True)
    right_block[badp[:, gap_hi:right_hi]] = np.nan
    with np.errstate(invalid='ignore'):
        v_l = np.nanmedian(left_block, axis=1)
        v_r = np.nanmedian(right_block, axis=1)
    v_l = np.where(np.isfinite(v_l), v_l, 0.0)
    v_r = np.where(np.isfinite(v_r), v_r, 0.0)
    t = (np.arange(1, x_gap + 1, dtype=np.float64)) / (x_gap + 1.0)
    ramp = v_l[:, None] * (1.0 - t)[None, :] + v_r[:, None] * t[None, :]
    out = imgp.copy()
    out[:, gap_lo:gap_hi] = ramp
    return out


def _wienerInverse(img_fft, Khat, shape, eps=1e-3):
    """Wiener-regularised inverse of the (δ + K) blur.

    Standard inverse `irfft2(F / (1 + K̂))` blows up where ``1 + K̂`` is small
    (kernel hard-edge ringing in Fourier space). Wiener replaces the division
    by a regularised pseudo-inverse:

        clean = irfft2( F · (1 + K̂)* / (|1 + K̂|² + eps) )

    For a real-symmetric kernel ``K̂`` is real and ``conj = identity``, but we
    use ``np.conj`` defensively so the formula is correct if the layout is
    ever changed to a non-symmetric kernel.
    """
    denom_hat = 1.0 + Khat
    return np.fft.irfft2(
        img_fft * np.conj(denom_hat) / (np.abs(denom_hat) ** 2 + eps),
        s=shape,
    )


def _grossBgDeg2_3anchorsPhysical(imgp, badp, anchors_phys):
    """Per-row deg=2 polynomial through 3 (col_phys, median) anchors.

    Vectorised across rows. Rows where any anchor has < 3 good pixels are
    filled by linear interpolation along the row axis from neighbouring good
    rows (the previous behaviour of leaving them at zero produced a step into
    the FFT).
    """
    if len(anchors_phys) != 3:
        raise ValueError("Expected exactly 3 anchors")
    H, WP = imgp.shape
    cols_arr = np.arange(WP, dtype=np.float64)
    x_centers = np.array([0.5 * (lo + hi - 1) for lo, hi in anchors_phys],
                          dtype=np.float64)

    # Per-row median and good-pixel count at each anchor (vectorised).
    medians = np.full((H, 3), np.nan)
    counts = np.zeros((H, 3), dtype=np.int32)
    for k, (c_lo, c_hi) in enumerate(anchors_phys):
        sub = imgp[:, c_lo:c_hi].astype(np.float64, copy=True)
        sub[badp[:, c_lo:c_hi]] = np.nan
        with np.errstate(invalid='ignore'):
            medians[:, k] = np.nanmedian(sub, axis=1)
        counts[:, k] = (~badp[:, c_lo:c_hi]).sum(axis=1)

    valid = (counts >= 3).all(axis=1) & np.isfinite(medians).all(axis=1)
    if not valid.any():
        _log.warning("gross-bg deg2: no rows have 3 good pixels in all 3 "
                     "anchors; returning zero bg")
        return np.zeros_like(imgp)

    # Solve V·c = m for c on every valid row at once.  V is the same 3×3
    # Vandermonde for all rows so we factor once.
    V = np.vstack([np.ones(3), x_centers, x_centers ** 2]).T  # (3, 3)
    coefs = np.linalg.solve(V, medians[valid].T)              # (3, n_valid)

    pow_cols = np.vstack([np.ones(WP), cols_arr, cols_arr ** 2])  # (3, WP)
    bg_valid = coefs.T @ pow_cols                                 # (n_valid, WP)

    bg = np.zeros((H, WP), dtype=np.float64)
    bg[valid] = bg_valid

    # Fill invalid rows by linear interpolation along the row axis (per col).
    # Vectorised in 1 D (np.interp loops in C).
    if not valid.all():
        n_bad = int((~valid).sum())
        _log.debug("gross-bg deg2: filling %d/%d invalid rows by row interp",
                   n_bad, H)
        rows_all = np.arange(H, dtype=np.float64)
        valid_rows = rows_all[valid]
        invalid_rows = rows_all[~valid]
        for c in range(WP):
            bg[~valid, c] = np.interp(invalid_rows, valid_rows, bg[valid, c])

    return bg


def _grossBgPerRow(img, bad, left_cols, right_cols):
    """Per-row linear background from medians at the two far-edge anchors.

    Returns the per-pixel bg image (zero on rows where either anchor has too
    few good pixels)."""
    H, W = img.shape
    bg = np.zeros_like(img)
    cols_arr = np.arange(W, dtype=np.float64)
    l_lo, l_hi = left_cols
    r_lo, r_hi = right_cols
    if l_hi <= l_lo or r_hi <= r_lo:
        return bg
    x_l = 0.5 * (l_lo + l_hi - 1)
    x_r = 0.5 * (r_lo + r_hi - 1)
    for r in range(H):
        bad_row = bad[r]
        sel_l = ~bad_row[l_lo:l_hi]
        sel_r = ~bad_row[r_lo:r_hi]
        if sel_l.sum() < 3 or sel_r.sum() < 3:
            continue
        v_l = np.median(img[r, l_lo:l_hi][sel_l])
        v_r = np.median(img[r, r_lo:r_hi][sel_r])
        if not (np.isfinite(v_l) and np.isfinite(v_r)):
            continue
        slope = (v_r - v_l) / (x_r - x_l)
        bg[r] = v_l + slope * (cols_arr - x_l)
    return bg


def _prepare(postISRCCDs, detectorMap, fiberProfiles,
             halfIllum, brightPercentile, binEdges,
             useGrossBg=False, grossBgEdgeWidth=20,
             outerRegionWeight=5.0):
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
    # Distance map in PHYSICAL coords: a pixel at array col `ccdSplit+1` is
    # ~xGap further from a CCD1 fiber than the array distance suggests, so
    # array-space distances would underestimate the true sky-distance at the
    # inner edges and bias the inner bin assignments.
    col_arr = np.arange(W, dtype=np.float32)
    col_phys_arr = np.where(col_arr < ccdSplit, col_arr,
                             col_arr + xGap).astype(np.float32)
    illum_phys = np.where(illum_arr < ccdSplit, illum_arr,
                           illum_arr + xGap).astype(np.float32)
    dist_map = np.full((H, W), np.inf, dtype=np.float32)
    for r in range(H):
        sc = np.sort(illum_phys[:, r])
        ins = np.clip(np.searchsorted(sc, col_phys_arr), 1, len(sc) - 1)
        dist_map[r] = np.minimum.reduce([
            np.abs(col_phys_arr - sc[ins - 1]),
            np.abs(col_phys_arr - sc[ins]),
            np.abs(col_phys_arr - sc[0]),
            np.abs(col_phys_arr - sc[-1]),
        ])

    binLabels = [f"{binEdges[i]}-{binEdges[i+1]}px"
                 for i in range(len(binEdges) - 1)]

    # Far-edge bg anchors: outermost N cols on each side, but capped so we
    # never reach into the fiber halo zone (would absorb scatter into bg).
    # We use ONLY ccd1-left and ccd2-right (the truly outer edges); the
    # inner-edges next to the CCD gap are too close to fibers (~16 px on
    # ccd1-right, ~42 px on ccd2-left) and would absorb halo signal.
    xc_min = int(np.floor(illum_arr.min()))
    xc_max = int(np.ceil(illum_arr.max()))
    left_safe_max = max(0, xc_min - halfIllum - 1)
    right_safe_min = min(W, xc_max + halfIllum + 2)
    left_anchor = (0, min(grossBgEdgeWidth, left_safe_max))
    right_anchor = (max(W - grossBgEdgeWidth, right_safe_min), W)
    if useGrossBg:
        _log.info(
            "Gross bg anchors (outer %d cols): left [%d:%d], right [%d:%d]",
            grossBgEdgeWidth,
            left_anchor[0], left_anchor[1], right_anchor[0], right_anchor[1],
        )

    # Fine bg anchors (3 anchors used by _fineBg2D in physical pixel coords):
    # ccd1-left, ccd2-left at the gap, ccd2-right. ccd1-right is excluded
    # (only ~16 px from rightmost ccd1 fiber → inside halo).
    fine_anchors = [
        left_anchor,                                    # ccd1-left
        (ccdSplit, ccdSplit + grossBgEdgeWidth),        # ccd2-left
        right_anchor,                                   # ccd2-right
    ]

    # Gross-bg anchors for the iterative pipeline, in PHYSICAL coords (3 pts).
    # Middle anchor: pick whichever of CCD1-right or CCD2-left is FURTHER
    # from a fiber so we don't bias the kernel by absorbing halo into bg.
    ccd1_right_arr = ccdSplit - 1                            # last array col of CCD1
    ccd2_left_arr  = ccdSplit                                # first array col of CCD2
    fiber_mean_col = illum_arr.mean(axis=1)
    right_c1_arr = float(np.max(fiber_mean_col[fiber_mean_col < ccdSplit])
                          if (fiber_mean_col < ccdSplit).any() else 0)
    left_c2_arr  = float(np.min(fiber_mean_col[fiber_mean_col >= ccdSplit])
                          if (fiber_mean_col >= ccdSplit).any() else W)
    dist_ccd1_inner = ccd1_right_arr - right_c1_arr
    dist_ccd2_inner = left_c2_arr - ccd2_left_arr
    if dist_ccd2_inner > dist_ccd1_inner:
        mid_anchor_phys = (ccdSplit + xGap, ccdSplit + xGap + grossBgEdgeWidth)
        mid_label = f"CCD2-left (d={dist_ccd2_inner:.1f} > {dist_ccd1_inner:.1f})"
    else:
        mid_anchor_phys = (ccdSplit - grossBgEdgeWidth, ccdSplit)
        mid_label = f"CCD1-right (d={dist_ccd1_inner:.1f} >= {dist_ccd2_inner:.1f})"
    # Outer anchors in physical coords
    left_anchor_phys = left_anchor                       # ccd1-left (cols < ccdSplit, no shift)
    right_anchor_phys = (right_anchor[0] + xGap, right_anchor[1] + xGap)
    gross_bg_anchors_phys = [left_anchor_phys, mid_anchor_phys, right_anchor_phys]
    _log.info("Iterative gross-bg 3 anchors (phys): "
              "[%d:%d] [%d:%d] [%d:%d]  middle=%s",
              left_anchor_phys[0], left_anchor_phys[1],
              mid_anchor_phys[0], mid_anchor_phys[1],
              right_anchor_phys[0], right_anchor_phys[1], mid_label)

    # 8 cost regions for the iterative pipeline (4 per CCD), in array coords:
    # CCD1: outer-left, 2 wide-fiber-gap interiors, inner-right
    # CCD2: inner-left, 2 wide-fiber-gap interiors, outer-right
    #
    # Detect "wide" fiber gaps in COLUMN space (sorted by xCenter) rather than
    # by fiberId — adjacent fiberIds aren't necessarily adjacent in column
    # space (engineering fibers, broken fibers), and a 4-fiberId gap can be
    # tiny in cols. We require ≥ 30 px of clear sky (after halfIllum margin
    # on each side) for a region to qualify.
    xc_arr = np.array(sorted(
        float(np.median(detectorMap.getXCenter(int(fid))))
        for fid in detectorMap.fiberId if int(fid) in illumIds
    ))
    minGapPx = 2 * halfIllum + 8       # need ≥ 8 px clear sky after margins
    interior_ccd1, interior_ccd2 = [], []
    for k in np.where(np.diff(xc_arr) >= minGapPx)[0]:
        x_lo = xc_arr[k] + halfIllum
        x_hi = xc_arr[k + 1] - halfIllum
        if x_hi <= x_lo:
            continue
        if x_hi < ccdSplit:
            interior_ccd1.append((int(x_lo), int(x_hi)))
        elif x_lo >= ccdSplit:
            interior_ccd2.append((int(x_lo), int(x_hi)))
        # else: spans the gap, skip
    cost_regions = (
        [('ccd1 outer-left', left_anchor)]
        + [(f'ccd1 inner @{(b[0]+b[1])//2}', b) for b in interior_ccd1]
        + [('ccd1 inner-right', (max(0, int(right_c1_arr) + halfIllum + 1), ccdSplit))]
        + [('ccd2 inner-left', (ccdSplit, max(ccdSplit, int(left_c2_arr) - halfIllum - 1)))]
        + [(f'ccd2 inner @{(b[0]+b[1])//2}', b) for b in interior_ccd2]
        + [('ccd2 outer-right', right_anchor)]
    )

    # Per-region metadata for the cost aggregator:
    #   • mean_dist: median distance-to-nearest-fiber (in physical px) over
    #     the region's scatter pixels — used to bin regions as "near" vs
    #     "far" for the radial-shape penalty.
    #   • weight: multiplier on that region in the weighted-RMS aggregation;
    #     outermost regions are anchored at `outerRegionWeight` (default 5×)
    #     to break the bg-vs-kernel-tail degeneracy (an additive bg shift
    #     biases outer regions; the kernel tail mostly biases inner ones).
    cost_region_dists = np.zeros(len(cost_regions), dtype=np.float64)
    cost_region_weights = np.ones(len(cost_regions), dtype=np.float64)
    for k, (name, (c_lo, c_hi)) in enumerate(cost_regions):
        if c_hi <= c_lo:
            cost_region_dists[k] = np.nan
            continue
        sub_dist = dist_map[:, c_lo:c_hi]
        sub_scatter = (~illum_mask[:, c_lo:c_hi]) & np.isfinite(sub_dist)
        if sub_scatter.any():
            cost_region_dists[k] = float(np.median(sub_dist[sub_scatter]))
        else:
            cost_region_dists[k] = float(np.median(sub_dist))
        if name.endswith('outer-left') or name.endswith('outer-right'):
            cost_region_weights[k] = float(outerRegionWeight)
    _log.info("Cost regions (%d):", len(cost_regions))
    for (name, (c_lo, c_hi)), d, w in zip(
        cost_regions, cost_region_dists, cost_region_weights
    ):
        _log.info("    %-22s [%4d:%4d]  mean_dist=%5.1f px  weight=%.1f",
                  name, c_lo, c_hi, d, w)

    frames = []
    for idx, exp in enumerate(postISRCCDs):
        img = exp.image.array.astype(np.float64)
        bad = exp.mask.array.astype(np.uint32) != 0

        if useGrossBg:
            bg_gross = _grossBgPerRow(img, bad, left_anchor, right_anchor)
            img = img - bg_gross

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
        # Linear-fill the gap so the FFT-IF input has no broadband step.
        # Without this, the kernel inversion absorbs the gap discontinuity
        # and biases the inner-edge residuals (cf. iterative branch).
        badp_frame = np.ones((H, wp), dtype=bool)
        badp_frame[:, :ccdSplit] = bad[:, :ccdSplit]
        badp_frame[:, ccdSplit + xGap:] = bad[:, ccdSplit:]
        imgp = _fillGapLinear(imgp, badp_frame, ccdSplit, xGap)
        img_fft = np.fft.rfft2(imgp)

        _log.info(
            "  frame %d: shape=%dx%d  median=%.1f  bright rows=%d (thr=%.1f)"
            "%s",
            idx, H, W, float(np.median(img)), len(bright_rows), thr,
            "  (gross-bg subtracted)" if useGrossBg else "",
        )
        frames.append(dict(
            img=img, bad=bad, scatter_mask=scatter_mask, bin_masks=bin_masks,
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
        fine_anchors=fine_anchors,
        gross_bg_anchors_phys=gross_bg_anchors_phys,
        cost_regions=cost_regions,
        cost_region_dists=cost_region_dists,
        cost_region_weights=cost_region_weights,
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


def _deconvolveOne(frame, K1_hat, K2_hat, frac1, frac2, prep,
                    useFineBg=False, fineBgIter=1,
                    fineBgDegCol=2, fineBgDegRow=3):
    """Deconvolve one frame. If frac1=frac2=0, returns the raw image unchanged.

    With ``useFineBg=True`` the cleaned image is further refined by an
    iterative 2-D Chebyshev polynomial bg fit at the three trusted anchor
    regions (ccd1-left, ccd2-left, ccd2-right) in physical column coords,
    matching the production pipeline.
    """
    H, W, ccdSplit, xGap = prep["H"], prep["W"], prep["ccdSplit"], prep["xGap"]
    if frac1 == 0.0 and frac2 == 0.0:
        cleaned = frame["img"]
    else:
        Khat = frac1 * K1_hat + frac2 * K2_hat
        dp = _wienerInverse(frame["img_fft"], Khat, (H, prep["wp"]))
        cleaned = np.empty((H, W))
        cleaned[:, :ccdSplit] = dp[:, :ccdSplit]
        cleaned[:, ccdSplit:] = dp[:, ccdSplit + xGap:]

    if useFineBg and "fine_anchors" in prep and "bad" in frame:
        for _ in range(max(1, fineBgIter)):
            fine_bg = _fineBg2D(
                cleaned, frame["bad"], xGap, ccdSplit, prep["fine_anchors"],
                deg_col=fineBgDegCol, deg_row=fineBgDegRow,
            )
            cleaned = cleaned - fine_bg
    return cleaned


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


def _meanBinResiduals(prep, K1_hat, K2_hat, frac1, frac2,
                      useFineBg=False, fineBgIter=1,
                      fineBgDegCol=2, fineBgDegRow=3):
    """Mean per-bin residual averaged over frames."""
    per_visit = []
    for frame in prep["frames"]:
        clean = _deconvolveOne(
            frame, K1_hat, K2_hat, frac1, frac2, prep,
            useFineBg=useFineBg, fineBgIter=fineBgIter,
            fineBgDegCol=fineBgDegCol, fineBgDegRow=fineBgDegRow,
        )
        per_visit.append(_binRatios(clean, frame, prep["illum_mask"]))
    return np.nanmean(per_visit, axis=0)


def _iterativeCleanOne(frame, K1_hat, K2_hat, frac1, frac2, prep, n_iter=3,
                        wiener_eps=1e-3):
    """Iterative production-style pipeline in physical-pixel space.

    Pseudocode the user specified:
        fiberScatter = 0
        for _ in range(n_iter):
            bck = grossBg(postISR - fiberScatter)
            fiberScatter = (postISR - bck) - FFT_IF(postISR - bck)
        clean = postISR - bck - fiberScatter

    Both grossBg and FFT_IF operate on physical-pixel images (so the
    deg=2 polynomial through 3 anchors is naturally smooth across the
    physical CCD gap).  The final cleaned image is returned in array
    coords (gap region dropped).
    """
    H, ccdSplit, xGap, wp = (prep["H"], prep["ccdSplit"],
                              prep["xGap"], prep["wp"])
    img = frame["img"]              # array coords
    bad = frame["bad"]              # array coords
    imgp, badp = _imgToPhysical(img, bad, ccdSplit, xGap)
    anchors_phys = prep["gross_bg_anchors_phys"]
    gap_lo, gap_hi = ccdSplit, ccdSplit + xGap   # the 75-px gap region

    def _zeroGap(arr):
        """Force `arr` to 0 in the physical gap region (in-place safe)."""
        arr[:, gap_lo:gap_hi] = 0.0
        return arr

    if frac1 == 0.0 and frac2 == 0.0:
        bg_phys = _grossBgDeg2_3anchorsPhysical(imgp, badp, anchors_phys)
        _zeroGap(bg_phys)
        clean_phys = imgp - bg_phys
        return _imgFromPhysical(clean_phys, ccdSplit, xGap)

    Khat = frac1 * K1_hat + frac2 * K2_hat
    fiberScatter_phys = np.zeros_like(imgp)
    bg_phys = np.zeros_like(imgp)
    for _ in range(max(1, n_iter)):
        bg_phys = _grossBgDeg2_3anchorsPhysical(
            imgp - fiberScatter_phys, badp, anchors_phys
        )
        # bg_phys is naturally smooth across the gap (deg-2 polynomial in
        # physical coords). Compute denom = imgp - bg_phys, then linear-fill
        # the gap so the FFT-IF input has no broadband step. The gap fill is
        # only used by the FFT — bg_phys and fiberScatter_phys are zeroed in
        # the gap on output, and the gap is dropped by `_imgFromPhysical`.
        denom = imgp - bg_phys
        denom_for_fft = _fillGapLinear(denom, badp, ccdSplit, xGap)
        cleaned = _wienerInverse(np.fft.rfft2(denom_for_fft), Khat, (H, wp),
                                  eps=wiener_eps)
        fiberScatter_phys = denom_for_fft - cleaned
        _zeroGap(fiberScatter_phys)                 # don't leak deconvolution into gap

    _zeroGap(bg_phys)                               # gap stays 0 on output
    clean_phys = imgp - bg_phys - fiberScatter_phys
    return _imgFromPhysical(clean_phys, ccdSplit, xGap)


def _regionResiduals(clean, regions, scatter_mask, sigma=3.5):
    """Sigma-clipped median absolute residual at each named region.

    The cost metric is the *unsigned* residual size — `nanmedian(|v|)` after
    a MAD sigma-clip. Mean was wrong: signed cancellation lets a poor kernel
    score zero by oversubtracting some regions and undersubtracting others.
    Median(|v|) penalises any non-zero residual regardless of sign, and
    sigma-clipping guards against cosmic-ray contamination on arc visits.
    """
    out = np.full(len(regions), np.nan)
    for k, (_, (c_lo, c_hi)) in enumerate(regions):
        if c_hi <= c_lo:
            continue
        sub_clean = clean[:, c_lo:c_hi]
        sub_mask = scatter_mask[:, c_lo:c_hi]
        v = sub_clean[sub_mask]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        if mad > 0.0:
            keep = np.abs(v - med) < sigma * 1.4826 * mad
            v = v[keep]
        if v.size:
            out[k] = float(np.median(np.abs(v)))
    return out


def _meanRegionResiduals(prep, K1_hat, K2_hat, frac1, frac2, n_iter=3):
    """Mean per-region residual (averaged over frames) for the iterative
    pipeline.  Used as the kernel-tuning cost when ``useIterative=True``.
    """
    per_visit = []
    regions = prep["cost_regions"]
    for frame in prep["frames"]:
        clean = _iterativeCleanOne(frame, K1_hat, K2_hat, frac1, frac2,
                                    prep, n_iter=n_iter)
        per_visit.append(_regionResiduals(clean, regions, frame["scatter_mask"]))
    return np.nanmean(per_visit, axis=0)


def _aggregateCost(bins, weights, dists,
                    radial_weight=1.0, near_thresh=30.0, far_thresh=60.0):
    """Combine per-region residual magnitudes into a scalar cost.

    Two complementary terms:

    1. **Weighted RMS** of per-region magnitudes — penalises any non-zero
       residual amplitude. Outermost regions are anchored at higher weight
       (set in ``_prepare``) so the optimiser cannot trade a constant bg
       offset for a more extended kernel tail without paying a price at
       the truly outer (fiber-free) edges.

    2. **Radial-shape penalty** — ``|<bins[near]> − <bins[far]>|`` where
       ``near = mean_dist < near_thresh`` and ``far = mean_dist > far_thresh``.
       Penalises kernels whose radial shape mismatches the residual radial
       profile. Complementary to the RMS: a kernel whose total amplitude is
       right but whose tail exponent is wrong will have near-residual ≠
       far-residual, so the penalty fires even though the unsigned-RMS may
       look fine. This term is what breaks the bg-vs-extended-tail
       degeneracy diagnosed in the physics review.

    Set ``radial_weight=0`` to disable the radial term and recover plain
    weighted RMS.
    """
    bins = np.asarray(bins, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    d = np.asarray(dists, dtype=np.float64)
    finite = np.isfinite(bins) & np.isfinite(w) & (w > 0)
    if not finite.any():
        return float('nan')
    bf, wf, df = bins[finite], w[finite], d[finite]
    rms = float(np.sqrt(np.sum(wf * bf ** 2) / np.sum(wf)))
    near = bf[df < near_thresh]
    far = bf[df > far_thresh]
    if near.size and far.size and radial_weight > 0:
        shape_penalty = float(abs(np.nanmean(near) - np.nanmean(far)))
    else:
        shape_penalty = 0.0
    return rms + radial_weight * shape_penalty


def _fineBg2D(residual, bad, x_gap, ccd_split, anchors_array,
              deg_col=2, deg_row=3):
    """2-D Chebyshev polynomial fit of residual at the given anchor regions,
    with column coordinate in *physical* pixels (so the fit spans both CCDs
    smoothly across the physical gap).

    Parameters
    ----------
    residual : ndarray (H, W)
        Image to be fit (typically postISR − cleaned, or post-FFT-IF residual).
    bad : ndarray (H, W) bool
        Bad-pixel mask (True = exclude from fit).
    x_gap : int
        Physical gap (pixels) between the two CCD halves.
    ccd_split : int
        Array column at which CCD2 begins (= W//2 typically).
    anchors_array : list of (col_lo, col_hi)
        Array-coordinate column ranges to use as fit anchors.
    deg_col, deg_row : int
        Polynomial degrees in physical column / row direction.

    Returns
    -------
    bg : ndarray (H, W)
        Polynomial bg evaluated at every pixel of the (H, W) array grid.
    """
    H, W = residual.shape
    cols_arr = np.arange(W, dtype=np.float64)
    cols_phys = np.where(cols_arr < ccd_split, cols_arr, cols_arr + x_gap)
    WP_minus1 = float(W + x_gap - 1)
    H_minus1 = float(H - 1)

    # Build (x_phys, y, value) data points from anchor regions
    xs, ys, zs = [], [], []
    for col_lo, col_hi in anchors_array:
        col_lo, col_hi = int(col_lo), int(col_hi)
        if col_hi <= col_lo:
            continue
        sub_phys = cols_phys[col_lo:col_hi]   # (W_anchor,)
        sub_img = residual[:, col_lo:col_hi]  # (H, W_anchor)
        sub_bad = bad[:, col_lo:col_hi]
        # rows × cols meshgrid
        rs, cs = np.indices(sub_img.shape)
        finite = np.isfinite(sub_img) & ~sub_bad
        rs = rs[finite]; cs = cs[finite]; vals = sub_img[finite]
        xs.append(sub_phys[cs])
        ys.append(rs.astype(np.float64))
        zs.append(vals)
    if not xs:
        return np.zeros_like(residual)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)

    # Normalise to [-1, 1] for Chebyshev stability
    x_norm = 2 * xs / WP_minus1 - 1
    y_norm = 2 * ys / H_minus1 - 1

    Tx = np.polynomial.chebyshev.chebvander(x_norm, deg_col)
    Ty = np.polynomial.chebyshev.chebvander(y_norm, deg_row)
    M = (Tx[:, :, None] * Ty[:, None, :]).reshape(len(xs), -1)
    coef, *_ = np.linalg.lstsq(M, zs, rcond=None)
    coef_2d = coef.reshape(deg_col + 1, deg_row + 1)

    # Evaluate everywhere
    XN = 2 * cols_phys / WP_minus1 - 1
    YN = 2 * np.arange(H, dtype=np.float64) / H_minus1 - 1
    Tx_full = np.polynomial.chebyshev.chebvander(XN, deg_col)   # (W, deg_col+1)
    Ty_full = np.polynomial.chebyshev.chebvander(YN, deg_row)   # (H, deg_row+1)
    bg = np.einsum('ij,ci,rj->rc', coef_2d, Tx_full, Ty_full)
    return bg


def applyScatteredLightCorrection(
    img, bad, detectorMap, fiberProfiles, kernelParams,
    halfIllum=11, grossBgEdgeWidth=20,
    fineBgDegCol=2, fineBgDegRow=3,
    nFineIter=1, useIterative=True, nIter=3, wiener_eps=1e-3,
):
    """Apply the full scattered-light correction pipeline to one frame.

    Two pipelines are available, selected by ``useIterative``:

    * ``useIterative=True`` (default): the production-style **iterative**
      pipeline — physical-pixel space, alternate (deg-2 gross bg through 3
      anchors) and (Wiener FFT inverse filter, gap linearly interpolated on
      input) for ``nIter`` iterations. This matches the cost function used by
      ``tuneScatteredLight``, so kernel parameters tuned by that task are
      applied via the *same* code path.

    * ``useIterative=False``: legacy non-iterative pipeline for diagnostic
      comparison — per-row linear gross bg + Wiener FFT-IF + iterative fine
      2-D polynomial bg refinement at three anchors.

    Parameters
    ----------
    img : ndarray (H, W)
        postISRCCD image.
    bad : ndarray (H, W) bool
        Bad-pixel mask.
    detectorMap, fiberProfiles : pfs DRP objects
        Used for fiber positions / illum mask geometry.
    kernelParams : dict
        Keys ``frac1, frac2, powerLaw1, powerLaw2, soften1, soften2`` and
        ``halfSize``. (Typically the output of ``tuneScatteredLight``.)
    halfIllum : int
        Mask radius (cols) around each illuminated fiber.
    grossBgEdgeWidth : int
        Width of the outer-edge anchor bands for the gross bg sub.
    fineBgDegCol, fineBgDegRow : int
        Polynomial degrees for the fine 2-D bg fit (legacy branch only).
    nFineIter : int
        Fine-bg refinement iterations after the FFT-IF (legacy branch only).
    useIterative : bool
        Use the iterative production pipeline (default).
    nIter : int
        Number of iterations of (gross-bg, FFT-IF) when ``useIterative``.

    Returns
    -------
    cleaned : ndarray (H, W)
        Final scatter+bg-corrected image.
    diagnostics : dict
        Pipeline-dependent intermediate maps for inspection.
    """
    bbox = detectorMap.getBBox()
    H = bbox.getHeight()
    W = bbox.getWidth()
    if isinstance(detectorMap, LayeredDetectorMap):
        xGap = -int(round(detectorMap.rightCcd.getTranslation().getX()))
        xGap = -xGap if xGap < 0 else xGap   # always non-negative
    else:
        xGap = 0
    ccdSplit = W // 2
    wp = W + xGap

    # Illum mask + xc range (for anchor capping)
    illum_centers = []
    illumIds = {int(fid) for fid in fiberProfiles.fiberId}
    for fid in detectorMap.fiberId:
        if int(fid) in illumIds:
            illum_centers.append(detectorMap.getXCenter(int(fid)).astype(np.float32))
    if not illum_centers:
        raise RuntimeError("No illuminated fibers in profiles.fiberId")
    illum_arr = np.asarray(illum_centers, dtype=np.float32)
    xc_min = int(np.floor(illum_arr.min()))
    xc_max = int(np.ceil(illum_arr.max()))

    # Outer-edge bg anchors
    left_safe_max = max(0, xc_min - halfIllum - 1)
    right_safe_min = min(W, xc_max + halfIllum + 2)
    left_anchor_gross = (0, min(grossBgEdgeWidth, left_safe_max))
    right_anchor_gross = (max(W - grossBgEdgeWidth, right_safe_min), W)

    K1_hat = _unitKernelFft(
        kernelParams["powerLaw1"], kernelParams["soften1"],
        int(kernelParams.get("halfSize", 4096)), (H, wp),
    )
    K2_hat = _unitKernelFft(
        kernelParams["powerLaw2"], kernelParams["soften2"],
        int(kernelParams.get("halfSize", 4096)), (H, wp),
    )

    if useIterative:
        # Same anchor-selection logic as `_prepare`: pick the inner anchor
        # (CCD1-right or CCD2-left) that is FURTHER from a fiber.
        fiber_mean_col = illum_arr.mean(axis=1)
        right_c1_arr = float(np.max(fiber_mean_col[fiber_mean_col < ccdSplit])
                              if (fiber_mean_col < ccdSplit).any() else 0)
        left_c2_arr = float(np.min(fiber_mean_col[fiber_mean_col >= ccdSplit])
                              if (fiber_mean_col >= ccdSplit).any() else W)
        if (left_c2_arr - ccdSplit) > (ccdSplit - 1 - right_c1_arr):
            mid_anchor_phys = (ccdSplit + xGap, ccdSplit + xGap + grossBgEdgeWidth)
        else:
            mid_anchor_phys = (ccdSplit - grossBgEdgeWidth, ccdSplit)
        left_anchor_phys = left_anchor_gross
        right_anchor_phys = (right_anchor_gross[0] + xGap,
                              right_anchor_gross[1] + xGap)
        anchors_phys = [left_anchor_phys, mid_anchor_phys, right_anchor_phys]

        prep_lite = dict(H=H, ccdSplit=ccdSplit, xGap=xGap, wp=wp)
        frame = dict(img=img.astype(np.float64, copy=False), bad=bad)
        # Reuse `_iterativeCleanOne` with a minimal prep dict.
        prep_lite["gross_bg_anchors_phys"] = anchors_phys
        cleaned = _iterativeCleanOne(
            frame, K1_hat, K2_hat,
            float(kernelParams["frac1"]), float(kernelParams["frac2"]),
            prep_lite, n_iter=nIter, wiener_eps=wiener_eps,
        )
        return cleaned, dict(
            anchors_phys=anchors_phys,
            xGap=xGap, ccdSplit=ccdSplit, nIter=nIter,
        )

    # Legacy non-iterative path (preserved for diagnostic comparison).
    fine_anchors = [
        left_anchor_gross,
        (ccdSplit, ccdSplit + grossBgEdgeWidth),
        right_anchor_gross,
    ]
    gross_bg = _grossBgPerRow(img, bad, left_anchor_gross, right_anchor_gross)
    img_grossbg = img - gross_bg

    Khat = (kernelParams["frac1"] * K1_hat
            + kernelParams["frac2"] * K2_hat)
    imgp = np.zeros((H, wp))
    imgp[:, :ccdSplit] = img_grossbg[:, :ccdSplit]
    imgp[:, ccdSplit + xGap:] = img_grossbg[:, ccdSplit:]
    badp = np.ones((H, wp), dtype=bool)
    badp[:, :ccdSplit] = bad[:, :ccdSplit]
    badp[:, ccdSplit + xGap:] = bad[:, ccdSplit:]
    imgp = _fillGapLinear(imgp, badp, ccdSplit, xGap)
    cleaned_padded = _wienerInverse(np.fft.rfft2(imgp), Khat, (H, wp),
                                     eps=wiener_eps)
    cleaned = np.empty_like(img)
    cleaned[:, :ccdSplit] = cleaned_padded[:, :ccdSplit]
    cleaned[:, ccdSplit:] = cleaned_padded[:, ccdSplit + xGap:]
    cleaned_kernel_only = cleaned.copy()

    fine_bg = np.zeros_like(img)
    for _ in range(max(1, nFineIter)):
        fine_bg = _fineBg2D(
            cleaned, bad, xGap, ccdSplit, fine_anchors,
            deg_col=fineBgDegCol, deg_row=fineBgDegRow,
        )
        cleaned = cleaned - fine_bg

    return cleaned, dict(
        gross_bg=gross_bg,
        fine_bg=fine_bg,
        cleaned_kernel_only=cleaned_kernel_only,
    )


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
