"""Claude: do line widths drift with time?  Same lines in repeated arcs (lsfLines of a FitLsf run)

Usage: python lampRepeats.py ARM COLLECTION VISIT=LAMP [VISIT=LAMP...]

For each lamp, the first listed visit of that lamp is its reference: for every other visit of the lamp,
the median over (spectrograph, fibre, line) of the paired difference in fitted blur rms (no model involved).
Then the local comparison of lamps as in lampDiffChecks.py (nearest line, same fibre, within DY rows),
pooling all visits of each lamp.  Lines used in the fit only (flag PINNED only).
"""
import os
import sys
from collections import defaultdict

import numpy as np
from lsst.daf.butler import Butler

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

DY = 50


def med(xx):
    xx = np.asarray(xx)
    xx = xx[np.isfinite(xx)]
    if len(xx) < 10:
        return "      ---     "
    mm = np.median(xx)
    return f"{mm:+.3f}±{1.4826*np.median(np.abs(xx - mm))/np.sqrt(len(xx)):.3f}({len(xx):5d})"


def main(arm, collection, pairs):
    lamps = {int(vv): ll for vv, ll in (pp.split("=") for pp in pairs)}
    butler = Butler(REPO, collections=collection)
    rms = {}                            # (visit, sp, fiberId, wavelength) -> (row, rms)
    for sp in (1, 2, 3, 4):
        lines = butler.get("lsfLines", instrument="PFS", arm=arm, spectrograph=sp)
        used = ((np.asarray(lines.flag) & ~16) == 0) & np.isfinite(lines.rms) & np.isfinite(lines.pixel)
        for vv, ff, wl, rr, ss in zip(np.asarray(lines.visit)[used], np.asarray(lines.fiberId)[used],
                                      np.asarray(lines.wavelength)[used], np.asarray(lines.pixel)[used],
                                      np.asarray(lines.rms)[used]):
            if int(vv) in lamps:
                rms[(int(vv), sp, int(ff), round(float(wl), 3))] = (rr, ss)
    byVisit = defaultdict(dict)
    for (vv, sp, ff, wl), val in rms.items():
        byVisit[vv][(sp, ff, wl)] = val
    times = {rec.id: rec.timespan.begin.isot[11:19] for rec in butler.registry.queryDimensionRecords(
        "visit", where=f"instrument='PFS' AND visit IN ({', '.join(str(vv) for vv in lamps)})")}

    print(f"=== {arm}: rms minus that of the same line in the lamp's first visit")
    first = {}
    for vv in sorted(lamps):
        ll = lamps[vv]
        if ll not in first:
            first[ll] = vv
            print(f"  {vv} {times.get(vv, '')} {ll:5} reference ({len(byVisit[vv])} lines)")
            continue
        ref = byVisit[first[ll]]
        diffs = [val[1] - ref[key][1] for key, val in byVisit[vv].items() if key in ref]
        print(f"  {vv} {times.get(vv, '')} {ll:5} {med(diffs)}")

    names = list(dict.fromkeys(lamps[vv] for vv in sorted(lamps)))
    keyed = {}
    for ll in names:
        keyed[ll] = defaultdict(list)
        for vv in (vv for vv in lamps if lamps[vv] == ll):
            for (sp, ff, wl), (rr, ss) in byVisit[vv].items():
                keyed[ll][(sp, ff)].append((rr, ss))
        keyed[ll] = {kk: np.array(vv) for kk, vv in keyed[ll].items()}
    print(f"\nraw rms difference (column lamp - row lamp), nearest line in the same fibre within {DY} rows")
    print("  " + " "*6 + " ".join(f"{ll:^19}" for ll in names))
    for aa in names:
        out = []
        for bb in names:
            if aa == bb:
                out.append(f"{'':19}")
                continue
            diffs = []
            for key, arrA in keyed[aa].items():
                arrB = keyed[bb].get(key)
                if arrB is None:
                    continue
                for rr, ss in arrA:
                    dy = np.abs(arrB[:, 0] - rr)
                    jj = np.argmin(dy)
                    if dy[jj] < DY:
                        diffs.append(arrB[jj, 1] - ss)
            out.append(med(diffs))
        print(f"  {aa:5} " + " ".join(out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
