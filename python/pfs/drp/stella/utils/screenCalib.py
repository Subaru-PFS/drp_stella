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
from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask

__all__ = [
    "buildRatioImage",
    "resampleFibers",
    "evaluate",
    "fitModels",
    "extractPfsArmsParallel",
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
    calexp1  = butler.get("calexp",     dataId)
    postISR2 = butler.get("postISRCCD", dataId, visit=quartzVisit)
    calexp2  = butler.get("calexp",     dataId, visit=quartzVisit)
    return _buildRatioFromImages(postISR1, calexp1, postISR2, calexp2)


# ── spectrum resampling ───────────────────────────────────────────────────────

def resampleFibers(fl, w, m, wgrid):
    """Resample per-fiber spectra onto a common wavelength grid.

    Parameters
    ----------
    fl : ndarray (nFibers, nWave)
    w  : ndarray (nFibers, nWave)  per-fiber wavelength arrays
    m  : bool ndarray (nFibers, nWave)  True = masked (bad)
    wgrid : ndarray (nWaveOut,)

    Returns
    -------
    out : ndarray (nFibers, nWaveOut)  NaN where insufficient good pixels
    """
    out = np.full((len(fl), len(wgrid)), np.nan)
    for i, (flux, wave, mask) in enumerate(zip(fl, w, m)):
        good = ~mask
        if good.sum() < 2:
            continue
        out[i] = np.interp(wgrid, wave[good], flux[good], left=np.nan, right=np.nan)
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
        sel     = (wavegrid >= wmin) & (wavegrid <= wmax)
        clipped = sigma_clip(array[:, sel], axis=1, sigma=sigma)
        df = pd.DataFrame({
            "meanVals":   np.ma.filled(clipped.mean(axis=1),                np.nan),
            "stdVals":    np.ma.filled(clipped.std(axis=1),                 np.nan),
            "medianVals": np.ma.filled(np.ma.median(array[:, sel], axis=1), np.nan),
            "fiberId":    fiberId,
        })
        df["wavelength"]   = int(round((wmin + wmax) / 2))
        df["wavelength_m"] = wavegrid[sel].mean()
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
    use  = ~mask2d[specMask]
    if use.sum() < degree + 1:
        return np.ones(specMask.sum())
    coeffs, _ = recursivePolyFit(fids[use], illumCorr[specMask][use], degree, sigma=sigma)
    return np.polyval(coeffs, fids)


def _fitOneWavelength(wv, dfs_wv, x, y, fiberId, spectrograph_arr,
                      illumModel_degree, scatModel_degree, sigma, nIter):
    """Fit scatter + illumination models for one wavelength bin (joblib target)."""
    from alefur.math import recursivePolyFit2d, evalPoly2d

    dfi        = dfs_wv.copy()
    illumModel = np.ones(len(x))
    mask2d     = np.zeros(len(x), dtype=bool)

    for _ in range(nIter):
        illumCorr = dfi.meanVals.to_numpy() / illumModel

        for spectrograph in [1, 2, 3, 4]:
            specMask = spectrograph_arr == spectrograph
            scat     = _fitScatterModel(illumCorr, fiberId, specMask,
                                        mask2d, scatModel_degree, sigma)
            dfi.loc[dfi.index[specMask], "scatModel"] = scat

        scatCorr         = dfi.meanVals.to_numpy() / dfi["scatModel"].to_numpy()
        coeffs2d, mask2d = recursivePolyFit2d(x, y, scatCorr, illumModel_degree, sigma=sigma)
        mask2d           = ~mask2d   # recursivePolyFit2d returns good-pixel mask
        illumModel       = evalPoly2d(coeffs2d, x, y, illumModel_degree)

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
    x                = mergedSpec.x
    y                = mergedSpec.y
    fiberId          = mergedSpec.fiberId
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
        dfs2.loc[dfi.index, "scatModel"]  = dfi["scatModel"].to_numpy()
        dfs2.loc[dfi.index, "illumModel"] = illumModel
        dfs2.loc[dfi.index, "scatCorr"]   = scatCorr
        dfs2.loc[dfi.index, "illumCorr"]  = illumCorr

    return dfs2


# ── parallel camera extraction (dotRoach pattern) ────────────────────────────

def _extractCameraWorker(camKey, spectrograph, arm, postISR_tw, calexp_tw,
                          postISR_q, calexp_q, fiberTrace, detectorMap,
                          selVisit, tmpDir):
    """Extract twilight and quartz separately then ratio the spectra, write pfsArm to disk.

    Extracting separately and dividing cancels the fiberProfile normalisation,
    unlike extracting from a pre-divided ratio image.
    Runs in a forked subprocess — receives LSST objects via fork memory copy,
    writes result as FITS to avoid pickling.
    """
    from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask

    extractSpectra = ExtractSpectraTask(config=ExtractSpectraTask.ConfigClass())
    dataId = dict(spectrograph=spectrograph, arm=arm, visit=selVisit)

    postISR_tw.mask.array[:] = calexp_tw.mask.array[:]
    postISR_q.mask.array[:]  = calexp_q.mask.array[:]

    pfsArm_tw = extractSpectra.run(
        postISR_tw.maskedImage, fiberTrace, detectorMap
    ).spectra.toPfsArm(Identity(**dataId))
    pfsArm_q = extractSpectra.run(
        postISR_q.maskedImage, fiberTrace, detectorMap
    ).spectra.toPfsArm(Identity(**dataId))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_flux = pfsArm_tw.flux / pfsArm_q.flux

    bad = ~np.isfinite(ratio_flux) | (pfsArm_q.flux <= 0)
    ratio_flux[bad] = np.nan
    pfsArm_tw.flux = ratio_flux
    pfsArm_tw.mask = pfsArm_tw.mask | pfsArm_q.mask
    pfsArm_tw.mask[bad] |= 1

    pfsArm_tw.writeFits(os.path.join(tmpDir, f"pfsArm_{camKey}.fits"))


def extractPfsArmsParallel(butler, selVisit, quartzVisit, fiberTraces, detectorMaps):
    """Extract twilight/quartz ratio spectra for all cameras in parallel.

    Butler reads are performed sequentially in the main process; per-camera
    computation is forked (LSST objects passed by memory, results via FITS).

    Parameters
    ----------
    butler        : Butler
    selVisit      : int  twilight visit
    quartzVisit   : int  quartz visit
    fiberTraces   : dict  camKey -> FiberTraceSet
    detectorMaps  : dict  camKey -> DetectorMap

    Returns
    -------
    pfsArms : dict  camKey -> PfsArm
    """
    cameras = [(sp, arm) for sp in [1, 2, 3, 4] for arm in "br"]

    # Load all images in main process before forking
    rawImages = {}
    for spectrograph, arm in cameras:
        camKey = f"{arm}{spectrograph}"
        dataId = dict(spectrograph=spectrograph, arm=arm, visit=selVisit)
        rawImages[camKey] = (
            butler.get("postISRCCD", dataId),
            butler.get("calexp",     dataId),
            butler.get("postISRCCD", dataId, visit=quartzVisit),
            butler.get("calexp",     dataId, visit=quartzVisit),
        )

    with tempfile.TemporaryDirectory() as tmpDir:
        jobs = []
        for spectrograph, arm in cameras:
            camKey                          = f"{arm}{spectrograph}"
            postISR_tw, calexp_tw, postISR_q, calexp_q = rawImages[camKey]
            p = multiprocessing.Process(
                target=_extractCameraWorker,
                args=(camKey, spectrograph, arm,
                      postISR_tw, calexp_tw, postISR_q, calexp_q,
                      fiberTraces[camKey], detectorMaps[camKey],
                      selVisit, tmpDir),
            )
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

        pfsArms = {
            f"{arm}{sp}": PfsArm.readFits(os.path.join(tmpDir, f"pfsArm_{arm}{sp}.fits"))
            for sp, arm in cameras
        }

    return pfsArms
