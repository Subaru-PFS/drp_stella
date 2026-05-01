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
    "resampleFibers",
    "evaluate",
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

def resampleFibers(fl, w, m, wgrid):
    """Resample per-fiber spectra onto a common wavelength grid (flux-conserving).

    Native pixel flux is ``f(λ) × Δλ_per_native_pixel``; converting to flux
    density (``flux / Δλ``) before interpolation makes per-pixel values
    comparable across fibers, even when their dispersions differ.

    Parameters
    ----------
    fl : ndarray (nFibers, nWave)  flux per native pixel (counts)
    w  : ndarray (nFibers, nWave)  per-fiber wavelength arrays (nm)
    m  : bool ndarray (nFibers, nWave)  True = masked (bad)
    wgrid : ndarray (nWaveOut,)

    Returns
    -------
    out : ndarray (nFibers, nWaveOut)  flux density (counts / nm),
          NaN where insufficient good pixels
    """
    out = np.full((len(fl), len(wgrid)), np.nan)
    for i, (flux, wave, mask) in enumerate(zip(fl, w, m)):
        good = ~mask
        if good.sum() < 2:
            continue
        density = flux / np.gradient(wave)
        out[i] = np.interp(wgrid, wave[good], density[good],
                           left=np.nan, right=np.nan)
    return out


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
                             tmpDir):
    """Extract twilight and quartz separately, divide spectra, write PfsArm.

    Extracting separately and dividing cancels the fiberProfile normalisation,
    unlike extracting from a pre-divided ratio image.
    """
    from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask

    extractSpectra = ExtractSpectraTask(config=ExtractSpectraTask.ConfigClass())
    dataId = dict(spectrograph=spectrograph, arm=arm, visit=selVisit)

    pfsArm_tw = _extractOnePfsArm(extractSpectra, postISR_tw, calexp_tw,
                                  fiberTrace, detectorMap, dataId, useCalexp)
    pfsArm_q = _extractOnePfsArm(extractSpectra, postISR_q, calexp_q,
                                 fiberTrace, detectorMap, dataId, useCalexp)

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
                               detectorMaps, useCalexp=False):
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

    Returns
    -------
    pfsArms : dict  camKey -> PfsArm  (flux = twilight / quartz)
    """
    perCameraArgs = {}
    for spectrograph, arm in _CAMERAS:
        camKey = f"{arm}{spectrograph}"
        dataId = dict(spectrograph=spectrograph, arm=arm, visit=selVisit)
        perCameraArgs[camKey] = (
            camKey, spectrograph, arm,
            butler.get("postISRCCD", dataId),
            butler.get("calexp", dataId),
            butler.get("postISRCCD", dataId, visit=quartzVisit),
            butler.get("calexp", dataId, visit=quartzVisit),
            fiberTraces[camKey], detectorMaps[camKey], selVisit, useCalexp,
        )
    return _runCamerasParallel(_extractCameraNormWorker, perCameraArgs)
