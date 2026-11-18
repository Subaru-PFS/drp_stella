"""Claude: per-line widths relative to the model, to see whether lamp offsets are per lamp or per line

Usage: python lampLineWidths.py ARM [COLLECTION]   (default u/rhl/PIPE2D-1940-armLsf-lampAvg)

For each distinct line (lamp, wavelength) used in the fit (PFI arcs, all four spectrographs, >= 50 fibres):
median row, median d = line blur rms - model blur rms (bare average-lamp model), its robust error,
the median S/N (flux/fluxErr), and the catalogue description/transition.
Then, per lamp: the median and robust scatter of the per-line medians.
"""
import os
import sys
from collections import defaultdict

import numpy as np
from lsst.daf.butler import Butler

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

LAMPS = {148314: "Ar", 148317: "Xe", 148320: "Kr", 148323: "HgCd", 148326: "Ne"}


def main(arm, collection):
    butler = Butler(REPO, collections=collection)
    perLine = defaultdict(lambda: defaultdict(list))
    for sp in (1, 2, 3, 4):
        dataId = dict(instrument="PFS", arm=arm, spectrograph=sp)
        model = butler.get("armLsf", dataId)
        lines = butler.get("lsfLines", dataId)
        visit = np.asarray(lines.visit)
        used = (((np.asarray(lines.flag) & ~16) == 0) & np.isfinite(lines.rms) & np.isfinite(lines.pixel)
                & np.isin(visit, list(LAMPS)))
        fiberId = np.asarray(lines.fiberId)[used]
        row = np.asarray(lines.pixel)[used]
        modelRms = np.full(used.sum(), np.nan)
        for ff in np.unique(fiberId):
            this = fiberId == ff
            modelRms[this] = np.asarray(model.getParameters(int(ff), row[this])["rms"])
        d = np.asarray(lines.rms)[used] - modelRms
        sn = np.asarray(lines.flux)[used]/np.asarray(lines.fluxErr)[used]
        wl = np.asarray(lines.wavelength)[used]
        desc = np.asarray(lines.description)[used]
        trans = np.asarray(lines.transition)[used]
        for vv, ww, rr, dd, ss, de, tr in zip(visit[used], wl, row, d, sn, desc, trans):
            key = (LAMPS[vv], round(float(ww), 3))
            entry = perLine[key]
            entry["row"].append(rr)
            entry["d"].append(dd)
            entry["sn"].append(ss)
            entry["label"] = f"{de} {tr}"
    print(f"{'lamp':5} {'wavelength':>10} {'row':>5} {'n':>5} {'d':>7} {'err':>6} {'S/N':>6}  description")
    summary = defaultdict(list)
    for (lamp, wl), entry in sorted(perLine.items(), key=lambda kv: np.median(kv[1]["row"])):
        dd = np.asarray(entry["d"])
        if len(dd) < 50:
            continue
        mm = np.median(dd)
        err = 1.4826*np.median(np.abs(dd - mm))/np.sqrt(len(dd))
        summary[lamp].append(mm)
        print(f"{lamp:5} {wl:10.3f} {np.median(entry['row']):5.0f} {len(dd):5d} {mm:+7.3f} {err:6.3f}"
              f" {np.median(entry['sn']):6.0f}  {entry['label']}")
    print("\nper lamp: median and robust scatter of the per-line medians")
    for lamp, mms in summary.items():
        mms = np.asarray(mms)
        mm = np.median(mms)
        scatter = 1.4826*np.median(np.abs(mms - mm))
        print(f"  {lamp:5} {len(mms):3d} lines  median {mm:+.3f}  scatter {scatter:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "u/rhl/PIPE2D-1940-armLsf-lampAvg")
