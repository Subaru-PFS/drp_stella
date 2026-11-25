"""Claude: PIPE2D-1993: time and inspect the CombinedLsf from mergeArms on real data (one spectrograph)

Usage: python timeCombinedLsf.py [visit] [spectrograph]
"""
import os
import sys
import time
import pickle
import numpy as np
from lsst.daf.butler import Butler
from pfs.drp.stella.mergeArms import MergeArmsConfig, MergeArmsTask
from pfs.drp.stella.lsf import CoaddLsf

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

visit = int(sys.argv[1]) if len(sys.argv) > 1 else 122771
spectrograph = int(sys.argv[2]) if len(sys.argv) > 2 else 1
butler = Butler(REPO, collections="u/kiyoyabe/weekly_hilo/w.2026.04")
dataId = dict(visit=visit, spectrograph=spectrograph, instrument="PFS")
arms = [aa for aa in "brmn"
        if butler.registry.queryDatasets("pfsArm", arm=aa, **dataId).any(execute=True, exact=False)]
spectra = [butler.get("pfsArm", arm=aa, **dataId) for aa in arms]
lsfs = [butler.get("pfsArm.lsf", arm=aa, **dataId).toLegacy() for aa in arms]
lsfTypes = set(type(ll).__name__ for lsf in lsfs for ll in lsf.values())
print(f"visit {visit} SM{spectrograph} arms {arms}; LSF types {lsfTypes}")
task = MergeArmsTask(config=MergeArmsConfig())
wavelength = task.wavelength.run("m" in arms)
task.normalizeSpectra(spectra, wavelength)
t0 = time.time()
merged = task.mergeSpectra(spectra, wavelength)
t1 = time.time()
lsfDict = task.mergeLsfs(lsfs, spectra, wavelength, merged.weights)
t2 = time.time()
print(f"mergeSpectra {t1 - t0:.1f} s; mergeLsfs {t2 - t1:.1f} s for {len(lsfDict)} fibers")
errors = np.array([inp.indexMap.maxError for lsf in lsfDict.values() for inp in lsf.inputs])
numSamples = np.array([len(inp.indexMap.positions) for lsf in lsfDict.values() for inp in lsf.inputs])
print(f"index maps: samples median {np.median(numSamples):.0f} max {numSamples.max()};"
      f" maxError median {np.median(errors):.1e} max {errors.max():.1e} px")
size = len(pickle.dumps(lsfDict))
print(f"pickle size {size/1e6:.1f} MB ({size/len(lsfDict)/1e3:.1f} kB per fiber)")

ff = sorted(lsfDict)[len(lsfDict)//2]
lsf = lsfDict[ff]
for inp, aa in zip(lsf.inputs, arms):
    print(f"  {aa}: output {inp.indexMap.minIndex}-{inp.indexMap.maxIndex},"
          f" {len(inp.weight.values)} weight bins, median weight {np.median(inp.weight.values):.3g}")
t0 = time.time()
for cc in np.linspace(100, len(wavelength) - 100, 200):
    lsf.computeKernel(cc)
t1 = time.time()
array = np.random.default_rng(1).normal(size=len(wavelength))
lsf.convolve(array)
t2 = time.time()
print(f"computeKernel {1e3*(t1 - t0)/200:.2f} ms; convolve {1e3*(t2 - t1):.0f} ms (fiber {ff})")

# Compare widths with the old construction (warped GaussianLsf, CoaddLsf with equal weights)
ii = list(spectra[0].fiberId).index(ff)
old = CoaddLsf([ll[ff].warp(ss.wavelength[ii], wavelength) for ll, ss in zip(lsfs, spectra)],
               [max(0, np.searchsorted(wavelength, ss.wavelength[ii][0])) for ss in spectra],
               [min(len(wavelength) - 1, np.searchsorted(wavelength, ss.wavelength[ii][-1]))
                for ss in spectra])
print("  wavelength  sigma(new) sigma(old)  [output pixels]")
for wl in (400, 500, 600, 630, 640, 650, 700, 800, 900, 950, 960, 1000, 1100, 1200):
    cc = np.interp(wl, wavelength, np.arange(len(wavelength)))
    try:
        print(f"  {wl:6.0f} {lsf.computeShape1D(cc):10.3f} {old.computeShape1D(cc):10.3f}")
    except Exception as exc:
        print(f"  {wl:6.0f} {type(exc).__name__}")
