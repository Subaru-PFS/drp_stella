"""Claude: why do the lamps' line widths differ?  Checks on the lsfLines of a FitLsf run

Usage: python lampDiffChecks.py ARM [COLLECTION]   (default u/rhl/PIPE2D-1940-armLsf-lampAvg)

d = line blur rms - model blur rms (the bare, average-lamp model at the line, with its MTP offset),
for lines used in the fit (flag PINNED only), PFI arcs only.
  1. uniformity: median d per lamp in 4 bins of row (wavelength) and 4 of x (fibre)
  2. locality:   for each line of lamp A, the nearest line of lamp B in the same fibre within DY rows;
                 median of the raw rms difference (B - A): no model involved
  3. brightness: median d per lamp in bins of S/N (flux/fluxErr)
Medians over all four spectrographs, with robust (1.4826 MAD/sqrt(n)) errors.
"""
import os
import sys
from collections import defaultdict

import numpy as np
from lsst.daf.butler import Butler

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

LAMPS = {148314: "Ar", 148317: "Xe", 148320: "Kr", 148323: "HgCd", 148326: "Ne"}
DY = 50
SN_EDGES = [0, 100, 300, 1000, 3000, np.inf]


def med(xx):
    xx = np.asarray(xx)
    xx = xx[np.isfinite(xx)]
    if len(xx) < 10:
        return "      ---     "
    mm = np.median(xx)
    err = 1.4826*np.median(np.abs(xx - mm))/np.sqrt(len(xx))
    return f"{mm:+.3f}±{err:.3f}({len(xx):5d})"


def main(arm, collection):
    butler = Butler(REPO, collections=collection)
    cols = defaultdict(list)
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
        cols["sp"].append(np.full(used.sum(), sp))
        cols["lamp"].append(np.array([LAMPS[vv] for vv in visit[used]]))
        cols["fiberId"].append(fiberId)
        cols["row"].append(row)
        cols["x"].append(np.asarray(lines.x)[used])
        cols["rms"].append(np.asarray(lines.rms)[used])
        cols["d"].append(np.asarray(lines.rms)[used] - modelRms)
        cols["sn"].append(np.asarray(lines.flux)[used]/np.asarray(lines.fluxErr)[used])
    cc = {kk: np.concatenate(vv) for kk, vv in cols.items()}
    lamps = [ll for ll in LAMPS.values() if np.any(cc["lamp"] == ll)]

    print(f"=== {arm}: {len(cc['d'])} lines; overall median d by lamp:")
    for ll in lamps:
        print(f"  {ll:5} {med(cc['d'][cc['lamp'] == ll])}")

    for name, edges in (("row", np.linspace(0, 4176, 5)), ("x", np.linspace(0, 4096, 5))):
        print(f"\n1. median d by lamp in bins of {name}")
        print("  lamp  " + " ".join(f"{name} {lo:4.0f}-{hi:4.0f}      "
                                    for lo, hi in zip(edges[:-1], edges[1:])))
        for ll in lamps:
            sel = cc["lamp"] == ll
            print(f"  {ll:5} " + " ".join(med(cc["d"][sel & (cc[name] >= lo) & (cc[name] < hi)])
                                          for lo, hi in zip(edges[:-1], edges[1:])))

    print("\n2. raw rms difference (column lamp - row lamp), nearest line in the same fibre within "
          f"{DY} rows")
    print("  " + " "*6 + " ".join(f"{ll:^19}" for ll in lamps))
    keyed = {}
    for ll in lamps:
        sel = cc["lamp"] == ll
        keyed[ll] = {}
        for sp, ff, rr, rms in zip(cc["sp"][sel], cc["fiberId"][sel], cc["row"][sel], cc["rms"][sel]):
            keyed[ll].setdefault((sp, ff), []).append((rr, rms))
        keyed[ll] = {kk: np.array(vv) for kk, vv in keyed[ll].items()}
    for aa in lamps:
        out = []
        for bb in lamps:
            if aa == bb:
                out.append(f"{'':19}")
                continue
            diffs = []
            for key, arrA in keyed[aa].items():
                arrB = keyed[bb].get(key)
                if arrB is None:
                    continue
                for rr, rms in arrA:
                    dy = np.abs(arrB[:, 0] - rr)
                    jj = np.argmin(dy)
                    if dy[jj] < DY:
                        diffs.append(arrB[jj, 1] - rms)
            out.append(med(diffs))
        print(f"  {aa:5} " + " ".join(out))

    print("\n3. median d by lamp in bins of S/N (flux/fluxErr)")
    print("  lamp  " + " ".join(f"S/N {lo:4.0f}-{hi:<6.0f}    "
                                for lo, hi in zip(SN_EDGES[:-1], SN_EDGES[1:])))
    for ll in lamps:
        sel = cc["lamp"] == ll
        print(f"  {ll:5} " + " ".join(med(cc["d"][sel & (cc["sn"] >= lo) & (cc["sn"] < hi)])
                                      for lo, hi in zip(SN_EDGES[:-1], SN_EDGES[1:])))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "u/rhl/PIPE2D-1940-armLsf-lampAvg")
