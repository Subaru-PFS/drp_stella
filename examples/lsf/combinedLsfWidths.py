"""Claude: PIPE2D-1993: compare the width of the old (CoaddLsf, warped GaussianLsf) and new (CombinedLsf) LSFs

Usage: python combinedLsfWidths.py [oldTag [newTag [outDir]]]
For pfsMergedLsf (visits 122771, 122783) and pfsCoaddLsf (all targets), plots the median (and 16-84% range)
over fibres/targets of the kernel standard deviation in nm, old and new, and their ratio.
"""
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lsst.daf.butler import Butler

matplotlib.use("Agg")

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

oldTag = sys.argv[1] if len(sys.argv) > 1 else "sortA"
newTag = sys.argv[2] if len(sys.argv) > 2 else "idx64"
outDir = sys.argv[3] if len(sys.argv) > 3 else "."
numObjects = 150  # Number of fibres/targets sampled per dataset
step = 2.0  # Wavelength step for evaluating the width (nm)
rng = np.random.default_rng(12345)


def stdev(lsf, center):
    """Kernel standard deviation (output pixels), or NaN if unavailable"""
    try:
        return lsf.computeKernel(center).computeStdev()
    except Exception:
        return np.nan


def widths(lsfDict, wavelengthDict, keys, grid):
    """Widths (nm) on the wavelength grid for the nominated keys: array (len(keys), len(grid))"""
    result = np.full((len(keys), len(grid)), np.nan)
    for ii, kk in enumerate(keys):
        wl = np.asarray(wavelengthDict[kk], float)
        pixels = np.interp(grid, wl, np.arange(len(wl)), left=np.nan, right=np.nan)
        dispersion = np.gradient(wl)
        for jj, cc in enumerate(pixels):
            if np.isfinite(cc):
                result[ii, jj] = stdev(lsfDict[kk], cc)*dispersion[int(round(cc))]
    return result


def overlaps(lsf, wavelength):
    """Wavelength ranges covered by more than one input of a CombinedLsf"""
    ranges = [(wavelength[inp.indexMap.minIndex], wavelength[inp.indexMap.maxIndex]) for inp in lsf.inputs]
    ranges.sort()
    clusters = [list(ranges[0])]  # Inputs from the same arm (several visits) start within a few nm
    for aa, bb in ranges[1:]:
        if aa - clusters[-1][0] < 50:
            clusters[-1][1] = max(clusters[-1][1], bb)
        else:
            clusters.append([aa, bb])
    ranges = clusters
    return [(ranges[ii + 1][0], ranges[ii][1]) for ii in range(len(ranges) - 1)
            if ranges[ii + 1][0] < ranges[ii][1]]


def plot(title, filename, grid, old, new, shade):
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 6.5), height_ratios=(2, 1),
                                      layout="constrained")
    for label, values, color in ((f"old: CoaddLsf ({oldTag})", old, "0.45"),
                                 (f"new: CombinedLsf ({newTag} data, current code)", new, "C0")):
        lo, med, hi = np.nanpercentile(values, [16, 50, 84], axis=0)
        top.fill_between(grid, lo, hi, color=color, alpha=0.25, lw=0)
        top.plot(grid, med, color=color, lw=1.5, label=label)
    ratio = new/old
    lo, med, hi = np.nanpercentile(ratio, [16, 50, 84], axis=0)
    bottom.fill_between(grid, lo, hi, color="C0", alpha=0.25, lw=0)
    bottom.plot(grid, med, color="C0", lw=1.5)
    bottom.axhline(1.0, color="0.3", lw=0.8)
    for ax in (top, bottom):
        for aa, bb in shade:
            ax.axvspan(aa, bb, color="C1", alpha=0.12, lw=0)
    top.set_ylabel("LSF sigma (nm)")
    top.set_title(title)
    top.legend(loc="upper left", frameon=False)
    top.text(0.99, 0.03, "lines: median; bands: 16-84%; orange: arm overlaps", transform=top.transAxes,
             ha="right", va="bottom", fontsize=8, color="0.3")
    bottom.set_ylabel("new / old")
    bottom.set_xlabel("wavelength (nm)")
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"wrote {filename}")
    for wl in (450, 640, 700, 800, 920, 960, 1100):
        jj = np.argmin(np.abs(grid - wl))
        print(f"   {grid[jj]:6.0f} nm: sigma old {np.nanmedian(old[:, jj]):.4f}"
              f" new {np.nanmedian(new[:, jj]):.4f} nm, ratio {np.nanmedian(ratio[:, jj]):.3f}")


butler = {tag: Butler(REPO, collections=[f"u/rhl/PIPE2D-1993-{tag}-merge", f"u/rhl/PIPE2D-1993-{tag}-coadd"])
          for tag in (oldTag, newTag)}
for visit in (122771, 122783):
    merged = butler[newTag].get("pfsMerged", instrument="PFS", visit=visit)
    lsfOld = butler[oldTag].get("pfsMergedLsf", instrument="PFS", visit=visit)
    lsfNew = butler[newTag].get("pfsMergedLsf", instrument="PFS", visit=visit)
    keys = sorted(rng.choice(sorted(set(lsfOld) & set(lsfNew)), numObjects, replace=False))
    wavelengths = {ff: merged.wavelength[list(merged.fiberId).index(ff)] for ff in keys}
    wl0 = wavelengths[keys[0]]
    grid = np.arange(np.ceil(wl0[0]) + 2, wl0[-1] - 2, step)
    plot(f"pfsMergedLsf, visit {visit} ({numObjects} fibres)",
         os.path.join(outDir, f"lsfWidth-merged-{visit}.png"),
         grid, widths(lsfOld, wavelengths, keys, grid), widths(lsfNew, wavelengths, keys, grid),
         overlaps(lsfNew[keys[0]], wl0))

run = butler[newTag].registry.getCollectionChain(f"u/rhl/PIPE2D-1993-{newTag}-coadd")[0]
refs = list(butler[newTag].registry.queryDatasets("pfsCoaddLsf", collections=run, findFirst=True))
classes = {}  # arms of the inputs --> dict(old=[], new=[], grid, shade)
perRef = 3*numObjects//len(refs) + 1  # Targets sampled per pfsCoadd
for ref in refs:
    coadd = butler[newTag].get("pfsCoadd", ref.dataId)
    lsfOld = butler[oldTag].get("pfsCoaddLsf", ref.dataId)
    lsfNew = butler[newTag].get(ref)
    targets = [tt for tt in lsfNew if lsfNew[tt] is not None and lsfOld.get(tt) is not None]
    targets = [targets[ii] for ii in rng.choice(len(targets), min(len(targets), perRef), replace=False)]
    for tt in targets:
        arms = "".join(aa for aa in "brmn" if aa in set(coadd[tt].observations.arm))
        wavelengths = {tt: coadd[tt].wavelength}
        if arms not in classes:
            wl0 = wavelengths[tt]
            grid = np.arange(np.ceil(wl0[0]) + 2, wl0[-1] - 2, step)
            classes[arms] = dict(old=[], new=[], grid=grid, shade=overlaps(lsfNew[tt], wl0))
        cc = classes[arms]
        if len(cc["old"]) >= numObjects:
            continue
        cc["old"].append(widths(lsfOld, wavelengths, [tt], cc["grid"]))
        cc["new"].append(widths(lsfNew, wavelengths, [tt], cc["grid"]))
for arms, cc in classes.items():
    plot(f"pfsCoaddLsf, weekly_hilo/20260201, targets observed with {arms} ({len(cc['old'])} targets)",
         os.path.join(outDir, f"lsfWidth-coadd-{arms}.png"), cc["grid"], np.concatenate(cc["old"]),
         np.concatenate(cc["new"]), cc["shade"])
