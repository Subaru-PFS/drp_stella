"""
Characterise the PFS flat-field screen response using twilight / quartz ratios.

The twilight/quartz ratio per fiber is decomposed into two multiplicative components:

    measured ≈ scatModel(fiberId) × illumModel(x, y)

- ``scatModel`` : per-spectrograph, per-CCD-half polynomial — captures scattered
  light from the non-uniform quartz beam picked up by the spectral extraction.
- ``illumModel`` : degree-2 focal-plane polynomial — captures the large-scale
  spatial mismatch between screen and sky illumination.

The two are fitted iteratively; the outlier mask from the 2-D illumination fit is
fed back into the 1-D scatter fit to exclude star-contaminated fibers.

Parallelism follows the pattern established in ``dotRoach.py``:
- Camera extraction uses ``multiprocessing.Process`` (fork), so non-picklable
  LSST objects are passed by memory copy; results are exchanged via FITS files.
- Wavelength-bin fitting uses ``joblib.Parallel`` (all inputs are numpy/pandas).
"""

import multiprocessing
import os
import tempfile

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from joblib import Parallel, delayed

from pfs.datamodel import Identity, PfsArm

__all__ = [
    "buildRatioImage",
    "extractArm",
    "resampleFibers",
    "resampleArm",
    "evaluate",
    "evaluateRaw",
    "fitModels",
    "extractPfsArmsParallel",
    "extractPfsArmsNormParallel",
]

nRows, nCols = 4176, 4096


# ── ratio image ───────────────────────────────────────────────────────────────

def _buildRatioFromImages(postISR1, calexp1, postISR2, calexp2):
    """Build twilight/quartz ratio from pre-loaded images.

    Safe to call inside a forked subprocess — no butler access.

    Parameters
    ----------
    postISR1, calexp1 : twilight postISR and calexp
    postISR2, calexp2 : quartz postISR and calexp

    Returns
    -------
    ratio : MaskedImage clone of calexp1 with ratio image/variance/mask
    """
    postISR1.mask.array[:] = calexp1.mask.array[:]
    postISR2.mask.array[:] = calexp2.mask.array[:]

    img1, img2 = postISR1.image.array, postISR2.image.array
    var1, var2 = postISR1.variance.array, postISR2.variance.array

    with np.errstate(divide="ignore", invalid="ignore"):
        r = img1 / img2
        v = (var1 + r**2 * var2) / img2**2

    ratio = calexp1.clone()
    ratio.mask.array[:] = calexp1.mask.array | calexp2.mask.array
    bad = ~np.isfinite(r) | (img2 <= 0)
    r[bad] = np.nan
    v[bad] = np.nan
    ratio.mask.array[bad] |= ratio.mask.getPlaneBitMask("NO_DATA")
    ratio.image.array[:] = r
    ratio.variance.array[:] = v
    return ratio


def buildRatioImage(butler, dataId, quartzVisit):
    """Load images from the butler and return the twilight/quartz ratio.

    Must be called in the main process (butler access).

    Parameters
    ----------
    butler : Butler
    dataId : dict  e.g. ``dict(spectrograph=1, arm='b', visit=selVisit)``
    quartzVisit : int

    Returns
    -------
    ratio : MaskedImage
    """
    postISR1 = butler.get("postISRCCD", dataId)
    calexp1 = butler.get("calexp", dataId)
    postISR2 = butler.get("postISRCCD", dataId, visit=quartzVisit)
    calexp2 = butler.get("calexp", dataId, visit=quartzVisit)
    return _buildRatioFromImages(postISR1, calexp1, postISR2, calexp2)


# ── spectrum resampling ───────────────────────────────────────────────────────

def _interpolateDensity(fl_nm, w, m, wgrid):
    """Interpolate per-fiber flux density onto a common wavelength grid.

    Parameters
    ----------
    fl_nm : ndarray (nFibers, nWave)  flux density (counts/nm)
    w     : ndarray (nFibers, nWave)  per-fiber wavelength arrays (nm)
    m     : bool ndarray (nFibers, nWave)  True = masked (bad)
    wgrid : ndarray (nWaveOut,)

    Returns
    -------
    out : ndarray (nFibers, nWaveOut), NaN where insufficient good pixels
    """
    out = np.full((len(fl_nm), len(wgrid)), np.nan)
    for i, (density, wave, mask) in enumerate(zip(fl_nm, w, m)):
        good = ~mask
        if good.sum() < 2:
            continue
        out[i] = np.interp(wgrid, wave[good], density[good],
                           left=np.nan, right=np.nan)
    return out


def resampleFibers(fl, w, m, wgrid):
    """Resample per-fiber spectra onto a common wavelength grid (flux-conserving).

    Converts raw pixel flux to flux density (counts/nm) before interpolation
    so that fibers with different dispersions are comparable.

    Parameters
    ----------
    fl : ndarray (nFibers, nWave)  flux per native pixel (counts)
    w  : ndarray (nFibers, nWave)  per-fiber wavelength arrays (nm)
    m  : bool ndarray (nFibers, nWave)  True = masked (bad)
    wgrid : ndarray (nWaveOut,)

    Returns
    -------
    out : ndarray (nFibers, nWaveOut)  flux density (counts/nm),
          NaN where insufficient good pixels
    """
    fl_nm = fl / np.gradient(w, axis=1)
    return _interpolateDensity(fl_nm, w, m, wgrid)


def extractArm(pfsArms, arm, selfibs):
    """Concatenate pfsArm data across spectrographs and filter to selected fibers.

    The flux is the dimensionless twilight/quartz ratio (e-/px ÷ e-/px);
    no dispersion conversion is applied since it cancels in the ratio.

    Parameters
    ----------
    pfsArms : dict mapping camera name → pfsArm.
    arm     : 'b', 'r', or 'br'.  When 'br', blue and red are concatenated
              along the wavelength axis for each spectrograph.
    selfibs : iterable of fiberIds to keep (e.g. ``mergedSpec.fiberId``).

    Returns
    -------
    fl       : (n_fib, n_wave)  dimensionless twilight/quartz ratio
    w        : (n_fib, n_wave)  wavelength (nm)
    m        : (n_fib, n_wave)  bad-pixel mask (True = bad)
    fiberIds : (n_fib,)
    """
    arms = list(arm)  # 'br' → ['b', 'r'], 'b' → ['b']
    fl, w, m, fibs = [], [], [], []
    for sp in (1, 2, 3, 4):
        pas = [pfsArms[f"{a}{sp}"] for a in arms]
        if len(pas) > 1:
            for pa0, pa1 in zip(pas, pas[1:]):
                np.testing.assert_array_equal(pa0.fiberId, pa1.fiberId)
        fl.append(np.concatenate([pa.flux       for pa in pas], axis=1))
        w.append( np.concatenate([pa.wavelength for pa in pas], axis=1))
        m.append( np.concatenate([pa.mask != 0  for pa in pas], axis=1))
        fibs.append(pas[0].fiberId)

    fl   = np.concatenate(fl,   axis=0)
    w    = np.concatenate(w,    axis=0)
    m    = np.concatenate(m,    axis=0)
    fibs = np.concatenate(fibs, axis=0)

    maskFib = np.isin(fibs, selfibs)
    return fl[maskFib], w[maskFib], m[maskFib], fibs[maskFib]


def resampleArm(fl_nm, w, m, doNormalize=False):
    """Resample per-fiber flux density onto a common wavelength grid.

    The grid is built by integrating the per-fiber-averaged dispersion forward
    from the common minimum wavelength, which is more accurate than taking
    the per-pixel median wavelength directly.

    Parameters
    ----------
    fl_nm       : (n_fib, n_wave)  flux density (counts/nm); see extractArm.
    w           : (n_fib, n_wave)  per-fiber wavelength arrays (nm).
    m           : (n_fib, n_wave)  bad-pixel mask (True = bad).
    doNormalize : bool, default False.  If True, divide the resampled flux by
                  the sigma-clipped per-wavelength mean across fibers.

    Returns
    -------
    flResampled : (n_fib, n_wave_out)  flux density on common grid (counts/nm)
    wgrid       : (n_wave_out,)        common wavelength grid (nm)
    """
    wmin = np.nanmax([np.nanmin(wi) for wi in w])
    wmax = np.nanmin([np.nanmax(wi) for wi in w])

    disp  = np.nanmean(np.diff(w, axis=1), axis=0)
    wgrid = np.concatenate(([wmin], wmin + np.cumsum(disp)))
    wgrid = wgrid[(wgrid >= wmin) & (wgrid <= wmax)]

    flResampled = _interpolateDensity(fl_nm, w, m, wgrid)

    if doNormalize:
        med = sigma_clip(flResampled, sigma=3, axis=0).mean(axis=0)
        med = np.where(np.isfinite(med) & (med != 0), med, 1.0)
        flResampled = flResampled / med

    return flResampled, wgrid


# ── wavelength-bin statistics ─────────────────────────────────────────────────

def rebuildWaveBins(waveBins, span):
    """Build new (wmin, wmax) bins keeping the centres of ``waveBins`` but
    applying a common ``span`` (full width, nm).

    Parameters
    ----------
    waveBins : list of (wmin, wmax) tuples
        Original bins; their centres are preserved.
    span : float
        New full width (nm) for every bin.

    Returns
    -------
    list of (wmin, wmax) tuples, same length as ``waveBins``.
    """
    half = span / 2.0
    return [((wmin + wmax) / 2.0 - half, (wmin + wmax) / 2.0 + half)
            for wmin, wmax in waveBins]


def evaluate(wavegrid, array, fiberId, waveBins, sigma=3,
             visit=None, quartzVisit=None):
    """Sigma-clipped per-fiber statistics in each wavelength bin.

    Parameters
    ----------
    wavegrid : ndarray (nWave,)
    array    : ndarray (nFibers, nWave)
    fiberId  : ndarray (nFibers,)
    waveBins : list of (wmin, wmax) tuples
    sigma    : float  clipping threshold
    visit    : int, optional
        Twilight visit used for this analysis; stored on every row.
    quartzVisit : int, optional
        Quartz visit used as the normalisation; stored on every row.

    Returns
    -------
    pandas.DataFrame  columns: meanVals, stdVals, medianVals, fiberId,
                               wavelength, wavelength_m, visit, quartzVisit
    """
    dfs = []
    for wmin, wmax in waveBins:
        sel = (wavegrid >= wmin) & (wavegrid <= wmax)
        clipped = sigma_clip(array[:, sel], axis=1, sigma=sigma)
        df = pd.DataFrame({
            "meanVals": np.ma.filled(clipped.mean(axis=1), np.nan),
            "stdVals": np.ma.filled(clipped.std(axis=1), np.nan),
            "medianVals": np.ma.filled(np.ma.median(array[:, sel], axis=1), np.nan),
            "fiberId": fiberId,
        })
        df["wavelength"] = int(round((wmin + wmax) / 2))
        df["wavelength_m"] = wavegrid[sel].mean()
        df["visit"] = -1 if visit is None else int(visit)
        df["quartzVisit"] = -1 if quartzVisit is None else int(quartzVisit)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def evaluateRaw(w, fl_nm, fiberId, waveBins, sigma=3, visit=None, quartzVisit=None):
    """Per-fiber wavelength-bin statistics directly from native pixel data.

    Identical output format to ``evaluate()``, but works on per-fiber native
    wavelength grids without prior resampling to a common grid.  Use alongside
    ``evaluate()`` to compare resampled vs. native-pixel binning.

    Parameters
    ----------
    w        : (nFibers, nWave)  per-fiber wavelength arrays (nm); see extractArm.
    fl_nm    : (nFibers, nWave)  flux density (counts/nm); see extractArm.
    fiberId  : (nFibers,)
    waveBins : list of (wmin, wmax) tuples
    sigma    : float, clipping threshold
    visit, quartzVisit : int, optional

    Returns
    -------
    pandas.DataFrame  same columns as evaluate()
    """
    dfs = []
    for wmin, wmax in waveBins:
        rows = []
        for i, fib in enumerate(fiberId):
            sel = (w[i] >= wmin) & (w[i] <= wmax) & ~np.isnan(fl_nm[i])
            vals = fl_nm[i, sel]
            if len(vals) < 2:
                rows.append({"meanVals": np.nan, "stdVals": np.nan,
                             "medianVals": np.nan, "fiberId": fib})
                continue
            clipped = sigma_clip(vals, sigma=sigma)
            rows.append({
                "meanVals":   float(np.ma.filled(clipped.mean(), np.nan)),
                "stdVals":    float(np.ma.filled(clipped.std(),  np.nan)),
                "medianVals": float(np.ma.filled(np.ma.median(vals), np.nan)),
                "fiberId":    fib,
            })
        df = pd.DataFrame(rows)
        df["wavelength"]   = int(round((wmin + wmax) / 2))
        df["wavelength_m"] = (wmin + wmax) / 2.0
        df["visit"]        = -1 if visit is None else int(visit)
        df["quartzVisit"]  = -1 if quartzVisit is None else int(quartzVisit)
        dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


# ── model fitting ─────────────────────────────────────────────────────────────

def _fitScatterModel(illumCorr, fiberId, specMask, mask2d, degree, sigma):
    """Fit a single polynomial across one spectrograph.

    Fibers flagged in ``mask2d`` (stars, bad fibers from the 2-D fit) are
    excluded from the polynomial fit but still evaluated.
    """
    from alefur.math import recursivePolyFit

    fids = fiberId[specMask]
    use = ~mask2d[specMask]
    if use.sum() < degree + 1:
        return np.ones(specMask.sum())
    coeffs, _ = recursivePolyFit(fids[use], illumCorr[specMask][use], degree, sigma=sigma)
    return np.polyval(coeffs, fids)


def _fitOneWavelength(wv, dfs_wv, x, y, fiberId, spectrograph_arr,
                      illumModel_degree, scatModel_degree, sigma, nIter):
    """Fit scatter + illumination models for one wavelength bin (joblib target)."""
    from alefur.math import recursivePolyFit2d, evalPoly2d

    dfi = dfs_wv.copy()
    illumModel = np.ones(len(x))
    mask2d = np.zeros(len(x), dtype=bool)

    for _ in range(nIter):
        illumCorr = dfi.meanVals.to_numpy() / illumModel

        for spectrograph in [1, 2, 3, 4]:
            specMask = spectrograph_arr == spectrograph
            scat = _fitScatterModel(illumCorr, fiberId, specMask,
                                    mask2d, scatModel_degree, sigma)
            dfi.loc[dfi.index[specMask], "scatModel"] = scat

        scatCorr = dfi.meanVals.to_numpy() / dfi["scatModel"].to_numpy()
        coeffs2d, mask2d = recursivePolyFit2d(x, y, scatCorr, illumModel_degree, sigma=sigma)
        mask2d = ~mask2d   # recursivePolyFit2d returns good-pixel mask
        illumModel = evalPoly2d(coeffs2d, x, y, illumModel_degree)

    illumCorr = dfi.meanVals.to_numpy() / illumModel
    return wv, dfi, illumModel, scatCorr, illumCorr


def fitModels(dfs, mergedSpec,
              illumModel_degree=2, scatModel_degree=7, sigma=3, nIter=10, n_jobs=-1):
    """Iteratively decompose twilight/quartz into scatter and illumination models.

    Parameters
    ----------
    dfs              : DataFrame from ``evaluate``
    mergedSpec       : merged spectra object with .x .y .fiberId .spectrograph
    illumModel_degree: degree of 2-D focal-plane polynomial
    scatModel_degree : degree of per-spectrograph 1-D polynomial
    sigma            : sigma-clipping threshold
    nIter            : number of alternating iterations
    n_jobs           : joblib parallelism (-1 = all cores)

    Returns
    -------
    dfs2 : DataFrame with added columns scatModel, illumModel, scatCorr, illumCorr
    """
    x = mergedSpec.x
    y = mergedSpec.y
    fiberId = mergedSpec.fiberId
    spectrograph_arr = mergedSpec.spectrograph

    dfs2 = dfs.copy()
    for col in ("scatModel", "illumModel", "scatCorr", "illumCorr"):
        dfs2[col] = 1.0

    wvs = np.sort(dfs2.wavelength.unique())

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fitOneWavelength)(
            wv, dfs2[dfs2.wavelength == wv].copy(),
            x, y, fiberId, spectrograph_arr,
            illumModel_degree, scatModel_degree, sigma, nIter,
        )
        for wv in wvs
    )

    for wv, dfi, illumModel, scatCorr, illumCorr in results:
        dfs2.loc[dfi.index, "scatModel"] = dfi["scatModel"].to_numpy()
        dfs2.loc[dfi.index, "illumModel"] = illumModel
        dfs2.loc[dfi.index, "scatCorr"] = scatCorr
        dfs2.loc[dfi.index, "illumCorr"] = illumCorr

    return dfs2


# ── parallel camera extraction (dotRoach pattern) ────────────────────────────

_CAMERAS = [(sp, arm) for sp in [1, 2, 3, 4] for arm in "br"]


def _extractOnePfsArm(extractSpectra, image, calexp, fiberTrace, detectorMap,
                      dataId, useCalexp):
    """Extract one PfsArm from a postISR (or calexp) image.

    The ``calexp`` mask is copied onto the source image so the mask planes
    are consistent regardless of which dataset we extract from.
    """
    src = calexp if useCalexp else image
    src.mask.array[:] = calexp.mask.array[:]
    return extractSpectra.run(
        src.maskedImage, fiberTrace, detectorMap
    ).spectra.toPfsArm(Identity(**dataId))


def _extractCameraSimpleWorker(camKey, spectrograph, arm, postISR, calexp,
                               fiberTrace, detectorMap, selVisit, useCalexp,
                               tmpDir):
    """Extract spectra from one visit, write PfsArm to disk.

    Forked subprocess — LSST objects come in via memory copy, result goes
    out via FITS to avoid pickling.
    """
    from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask

    extractSpectra = ExtractSpectraTask(config=ExtractSpectraTask.ConfigClass())
    dataId = dict(spectrograph=spectrograph, arm=arm, visit=selVisit)

    pfsArm = _extractOnePfsArm(extractSpectra, postISR, calexp,
                               fiberTrace, detectorMap, dataId, useCalexp)
    pfsArm.writeFits(os.path.join(tmpDir, f"pfsArm_{camKey}.fits"))


def _extractCameraNormWorker(camKey, spectrograph, arm,
                             postISR_tw, calexp_tw, postISR_q, calexp_q,
                             fiberTrace, detectorMap, selVisit, useCalexp,
                             kernel_tw, kernel_q,
                             tmpDir):
    """Extract twilight and quartz separately, divide spectra, write PfsArm.

    Extracting separately and dividing cancels the fiberProfile normalisation,
    unlike extracting from a pre-divided ratio image.

    If ``kernel_tw`` / ``kernel_q`` are provided (FiberKernel objects from
    ``fitFiberKernel``), the per-fiber traces are convolved with the matching
    kernel before extraction. The forward model becomes ``image = K_X·(P·f)``,
    so extracting with ``K_X·P`` recovers the per-fiber flux for that SED
    instead of one biased by the SED-mismatched profile.
    """
    from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask

    extractSpectra = ExtractSpectraTask(config=ExtractSpectraTask.ConfigClass())
    dataId = dict(spectrograph=spectrograph, arm=arm, visit=selVisit)

    bbox = (calexp_tw if useCalexp else postISR_tw).getBBox()
    traces_tw = kernel_tw.convolve(fiberTrace, bbox) if kernel_tw else fiberTrace
    traces_q = kernel_q.convolve(fiberTrace, bbox) if kernel_q else fiberTrace

    pfsArm_tw = _extractOnePfsArm(extractSpectra, postISR_tw, calexp_tw,
                                  traces_tw, detectorMap, dataId, useCalexp)
    pfsArm_q = _extractOnePfsArm(extractSpectra, postISR_q, calexp_q,
                                 traces_q, detectorMap, dataId, useCalexp)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_flux = pfsArm_tw.flux / pfsArm_q.flux

    bad = ~np.isfinite(ratio_flux) | (pfsArm_q.flux <= 0)
    ratio_flux[bad] = np.nan
    pfsArm_tw.flux = ratio_flux
    pfsArm_tw.mask = pfsArm_tw.mask | pfsArm_q.mask
    pfsArm_tw.mask[bad] |= 1

    pfsArm_tw.writeFits(os.path.join(tmpDir, f"pfsArm_{camKey}.fits"))


def _runCamerasParallel(target, perCameraArgs):
    """Fork one worker per camera, collect resulting PfsArm files.

    Parameters
    ----------
    target : callable  worker entry point (must accept ``tmpDir`` last)
    perCameraArgs : dict  camKey -> tuple of args (without ``tmpDir``)

    Returns
    -------
    pfsArms : dict  camKey -> PfsArm
    """
    with tempfile.TemporaryDirectory() as tmpDir:
        jobs = []
        for camKey, args in perCameraArgs.items():
            p = multiprocessing.Process(target=target, args=(*args, tmpDir))
            jobs.append(p)
            p.start()
        for p in jobs:
            p.join()
        return {
            camKey: PfsArm.readFits(os.path.join(tmpDir, f"pfsArm_{camKey}.fits"))
            for camKey in perCameraArgs
        }


def extractPfsArmsParallel(butler, selVisit, fiberTraces, detectorMaps,
                           useCalexp=False):
    """Extract per-camera PfsArm spectra for one visit, in parallel.

    Butler reads are performed sequentially in the main process; per-camera
    extraction is forked (LSST objects passed by memory, results via FITS).

    Parameters
    ----------
    butler        : Butler
    selVisit      : int
    fiberTraces   : dict  camKey -> FiberTraceSet
    detectorMaps  : dict  camKey -> DetectorMap
    useCalexp     : bool
        If True, extract from ``calexp`` (scatter correction applied).
        If False (default), extract from ``postISRCCD`` (no scatter correction).

    Returns
    -------
    pfsArms : dict  camKey -> PfsArm
    """
    perCameraArgs = {}
    for spectrograph, arm in _CAMERAS:
        camKey = f"{arm}{spectrograph}"
        dataId = dict(spectrograph=spectrograph, arm=arm, visit=selVisit)
        postISR = butler.get("postISRCCD", dataId)
        calexp = butler.get("calexp", dataId)
        perCameraArgs[camKey] = (
            camKey, spectrograph, arm, postISR, calexp,
            fiberTraces[camKey], detectorMaps[camKey], selVisit, useCalexp,
        )
    return _runCamerasParallel(_extractCameraSimpleWorker, perCameraArgs)


def extractPfsArmsNormParallel(butler, selVisit, quartzVisit, fiberTraces,
                               detectorMaps, useCalexp=False, kernels=None):
    """Extract twilight/quartz ratio spectra for all cameras in parallel.

    Same parallel pattern as ``extractPfsArmsParallel`` but extracts ``selVisit``
    and ``quartzVisit`` separately per camera and divides the spectra.

    Parameters
    ----------
    butler        : Butler
    selVisit      : int  twilight visit
    quartzVisit   : int  quartz visit (denominator)
    fiberTraces   : dict  camKey -> FiberTraceSet
    detectorMaps  : dict  camKey -> DetectorMap
    useCalexp     : bool  see ``extractPfsArmsParallel``
    kernels       : dict, optional
        ``camKey -> {'kernel1': K_twilight, 'kernel2': K_quartz}`` as
        produced by ``fitFiberKernel``. When provided, the per-fiber traces
        are convolved with the SED-matched kernel before each extraction so
        the deblending solve sees the right forward model. Per-camera
        missing kernels are silently skipped (falls back to un-convolved
        traces).

    Returns
    -------
    pfsArms : dict  camKey -> PfsArm  (flux = twilight / quartz)
    """
    perCameraArgs = {}
    for spectrograph, arm in _CAMERAS:
        camKey = f"{arm}{spectrograph}"
        dataId = dict(spectrograph=spectrograph, arm=arm, visit=selVisit)
        per = (kernels or {}).get(camKey, {})
        perCameraArgs[camKey] = (
            camKey, spectrograph, arm,
            butler.get("postISRCCD", dataId),
            butler.get("calexp", dataId),
            butler.get("postISRCCD", dataId, visit=quartzVisit),
            butler.get("calexp", dataId, visit=quartzVisit),
            fiberTraces[camKey], detectorMaps[camKey], selVisit, useCalexp,
            per.get('kernel1'), per.get('kernel2'),
        )
    return _runCamerasParallel(_extractCameraNormWorker, perCameraArgs)
