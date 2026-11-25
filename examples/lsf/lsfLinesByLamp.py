"""Claude: per-line fitted widths (lsfLines) relative to the armLsf model, by visit (lamp)

Usage: python lsfLinesByLamp.py ARM [SPECTROGRAPH...]   (collection u/rhl/PIPE2D-1940-armLsf)
For each visit: number of lines used, median and robust scatter of (line rms - model rms) and
(line sigma / model sigma) where sigma is the Gaussian-equivalent width from getParameters.
"""
import os
import sys

import numpy as np
from lsst.daf.butler import Butler

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

butler = Butler(REPO, collections="u/rhl/PIPE2D-1940-armLsf")
arm = sys.argv[1]
for sp in [int(ss) for ss in sys.argv[2:]] or [1, 2, 3, 4]:
    dataId = dict(instrument="PFS", arm=arm, spectrograph=sp)
    model = butler.get("armLsf", dataId)
    lines = butler.get("lsfLines", dataId)
    # PINNED (16) lines are used
    used = ((np.asarray(lines.flag) & ~16) == 0) & np.isfinite(lines.rms) & np.isfinite(lines.pixel)
    visit = np.asarray(lines.visit)
    for vv in sorted(set(visit[used])):
        sel = used & (visit == vv)
        fiberId = np.asarray(lines.fiberId)[sel]
        row = np.asarray(lines.pixel)[sel]
        drms, ratio = [], []
        for ff in sorted(set(fiberId)):
            this = fiberId == ff
            params = model.getParameters(int(ff), row[this])
            drms.extend(np.asarray(lines.rms)[sel][this] - np.asarray(params["rms"]))
        drms = np.array(drms)
        med = np.median(drms)
        print(f"{arm}{sp} visit {vv}: {sel.sum():6d} lines; line rms - model rms: median {med:+.4f} px, "
              f"scatter {1.4826*np.median(np.abs(drms - med)):.4f} px")
