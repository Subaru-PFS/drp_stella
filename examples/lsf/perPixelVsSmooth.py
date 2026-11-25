"""Claude: PIPE2D-1993: how far is the smooth-weight CombinedLsf from the per-pixel weights' LSF?

For lines every few output pixels, the per-pixel kernel is sum_a w_a(j) r_a(j) / sum_a w_a(j), with w_a the
weights mergeArms uses for the flux (zero where not selected) and r_a the input's resampled LSF; it is
compared with CombinedLsf (smooth weights). Reports centroid shifts and fractional sigma differences, overall
and in the overlaps (PIPE2D-2006 removes the difference).

Usage: python perPixelVsSmooth.py [visit] [spectrograph] [numFibers]
"""
import os
import sys
import numpy as np
from lsst.daf.butler import Butler
from pfs.drp.stella.mergeArms import MergeArmsConfig, MergeArmsTask

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

visit = int(sys.argv[1]) if len(sys.argv) > 1 else 122771
spectrograph = int(sys.argv[2]) if len(sys.argv) > 2 else 1
numFibers = int(sys.argv[3]) if len(sys.argv) > 3 else 20
butler = Butler(REPO, collections="u/kiyoyabe/weekly_hilo/w.2026.04")
dataId = dict(visit=visit, spectrograph=spectrograph, instrument="PFS")
arms = [aa for aa in "brmn"
        if butler.registry.queryDatasets("pfsArm", arm=aa, **dataId).any(execute=True, exact=False)]
spectra = [butler.get("pfsArm", arm=aa, **dataId) for aa in arms]
lsfs = [butler.get("pfsArm.lsf", arm=aa, **dataId).toLegacy() for aa in arms]
task = MergeArmsTask(config=MergeArmsConfig())
wavelength = task.wavelength.run("m" in arms)
task.normalizeSpectra(spectra, wavelength)
merged = task.mergeSpectra(spectra, wavelength)
weights = merged.weights
lsfDict = task.mergeLsfs(lsfs, spectra, wavelength, weights)


def moments(start, values):
    pixels = np.arange(start, start + len(values))
    centroid = np.sum(pixels*values)/np.sum(values)
    return centroid, np.sqrt(np.sum((pixels - centroid)**2*values)/np.sum(values))


rows = []
step = len(spectra[0]) // numFibers
for ii in range(0, len(spectra[0]), max(step, 1)):
    ff = spectra[0].fiberId[ii]
    lsf = lsfDict[ff]
    perPixelWeight = [np.where(sel[ii], ww[ii], 0.0) for ww, sel in zip(weights.weights, weights.selects)]
    for center in np.arange(50, len(wavelength) - 50, 7.3):
        inputs = [(aa, inp) for aa, inp in enumerate(lsf.inputs) if inp.indexMap.contains(center)]
        if len(inputs) < 2:
            continue  # A single input: identical by construction
        try:
            start, smooth = lsf.computeWindow(center)
        except Exception:
            continue
        numerator = np.zeros(len(smooth))
        denominator = np.zeros(len(smooth))
        for aa, inp in inputs:
            window = inp.computeWindow(center)
            if window is None:
                continue
            ss, values = window
            ww = perPixelWeight[aa][start:start + len(smooth)]
            denominator += ww
            numerator[ss - start:ss - start + len(values)] += ww[ss - start:ss - start + len(values)]*values
        with np.errstate(invalid="ignore", divide="ignore"):
            perPixel = np.where(denominator > 0, numerator/denominator, 0.0)
        if np.sum(perPixel) <= 0:
            continue
        c1, s1 = moments(start, smooth)
        c2, s2 = moments(start, perPixel/np.sum(perPixel))
        rows.append((ff, wavelength[int(center)], c2 - c1, s2/s1 - 1))
rows = np.array(rows)
print(f"visit {visit} SM{spectrograph} arms {arms}: {len(rows)} line positions in overlaps, "
      f"{len(np.unique(rows[:, 0]))} fibers")
for lo, hi in ((0, 2000), (600, 680), (930, 980)):
    sel = (rows[:, 1] >= lo) & (rows[:, 1] < hi)
    if not np.any(sel):
        continue
    dc, ds = rows[sel, 2], rows[sel, 3]
    mad = lambda xx: 1.4826*np.median(np.abs(xx - np.median(xx)))  # noqa: E731
    print(f"  {lo:4.0f}-{hi:4.0f} nm: N={sel.sum():5d}  dcentroid median {np.median(dc):+.4f}"
          f" robust-rms {mad(dc):.4f} max|.| {np.abs(dc).max():.4f} px;"
          f"  dsigma/sigma median {np.median(ds):+.4f} robust-rms {mad(ds):.4f}"
          f" max|.| {np.abs(ds).max():.4f}")
