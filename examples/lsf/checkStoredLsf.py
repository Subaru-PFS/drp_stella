"""Claude: check that the LSF stored with pfsMerged/pfsCoadd spectra agrees with the deprecated pickles

Usage: python checkStoredLsf.py COLLECTION_PREFIX
(e.g. u/rhl/PIPE2D-1940-comp; reads PREFIX-merge, PREFIX-coadd)
For a sample of spectra and positions, compares the kernel of the stored LSF (spectra.lsf) with the
deprecated pfsMergedLsf/pfsCoaddLsf, and reports the file sizes.
"""
import os
import sys

import numpy as np
from lsst.daf.butler import Butler

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

prefix = sys.argv[1]
rng = np.random.default_rng(12345)
for kind, dataset in (("merge", "pfsMerged"), ("coadd", "pfsCoadd")):
    collection = f"{prefix}-{kind}"
    butler = Butler(REPO, collections=collection)
    run = butler.registry.getCollectionChain(collection)[0]
    worst = 0.0
    numKernels = 0
    for ref in butler.registry.queryDatasets(dataset, collections=run, findFirst=True):
        spectra = butler.get(ref)
        legacy = butler.get(dataset + "Lsf", ref.dataId)
        path = butler.getURI(ref).ospath
        if kind == "merge":
            lsf = spectra.lsf  # drp_stella converts it
            keys = list(spectra.fiberId)
            legacyKeys = keys
        else:
            from pfs.drp.stella.spectraLsf import spectraLsfFromDatamodel
            targets = list(spectra)
            if not targets:
                continue
            wavelength = {tt: spectra[tt].wavelength for tt in targets}
            lsf = spectraLsfFromDatamodel(spectra.lsf, targets, wavelength, len(wavelength[targets[0]]))
            keys = targets
            legacyKeys = {(tt.catId, tt.objId): tt for tt in legacy}
        print(f"{dataset} {dict(ref.dataId.mapping)}: {os.path.getsize(path)/1e6:.1f} MB; "
              f"LSF {type(lsf).__name__} with {len(lsf.keys)} spectra")
        for key in rng.choice(len(keys), min(20, len(keys)), replace=False):
            key = keys[key]
            old = legacy[key] if kind == "merge" else legacy[legacyKeys[(key.catId, key.objId)]]
            for center in rng.uniform(100, lsf.length - 100, 5):
                try:
                    start, values = old.computeWindow(center)
                except Exception:
                    continue
                kernel = lsf.getLegacyLsf(key).computeWindow(center)
                if kernel[0] != start or len(kernel[1]) != len(values):
                    worst = np.inf
                    continue
                worst = max(worst, float(np.max(np.abs(kernel[1] - values))))
                numKernels += 1
    print(f"{dataset}: {numKernels} kernels compared; max |stored - pickled| = {worst:.1e}")
