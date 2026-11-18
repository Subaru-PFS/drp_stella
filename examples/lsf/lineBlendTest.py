"""Claude: are the broad arc lines blends?  Per-line excess width vs catalogue neighbours and fit quality

Usage: python lineBlendTest.py ARM [COLLECTION]   (default u/rhl/PIPE2D-1940-armLsf-lampAvg)

For each distinct line used in the fit (PFI arcs, all four spectrographs, >= 50 fibres), medians over
fibres of
  d      line blur rms - model blur rms (bare, average-lamp model)
  peak   the line's fitted peak (as in lsfLines)
  dS     line S - model S (asymmetry)
  chi2   chi^2/ndof of the line's fit
and, from the lamp's full line list (every entry, whatever its status or intensity, including MERGED/DUPLICATE
components), the catalogue neighbours within 1 and 2 pixels: number, and summed intensity / the line's.
Status bits of the line itself: B=BLEND, M=MERGED, C=COMBINED, D=DUPLICATE, b=BROAD.
Then the Spearman rank correlation of d with each of these, over all lines and within each lamp.
"""
import os
import sys
from collections import defaultdict

import numpy as np
import scipy.stats
from lsst.daf.butler import Butler

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

LAMPS = {148314: "Ar", 148317: "Xe", 148320: "Kr", 148323: "HgCd", 148326: "Ne"}
LINELISTS = os.path.join(os.environ["OBS_PFS_DIR"], "pfs", "lineLists")
STATUS = dict(B=2, b=16, M=256, C=512, D=8192)


def readLineList(lamp):
    wl, flux, status = [], [], []
    with open(os.path.join(LINELISTS, f"{lamp}.txt")) as fd:
        for line in fd:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            wl.append(float(fields[0]))
            flux.append(float(fields[1]))
            status.append(int(fields[3]))
    return np.array(wl), np.array(flux), np.array(status)


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
        modelS = np.full(used.sum(), np.nan)
        for ff in np.unique(fiberId):
            this = fiberId == ff
            params = model.getParameters(int(ff), row[this])
            modelRms[this] = np.asarray(params["rms"])
            modelS[this] = np.asarray(params["S"])
        columns = dict(
            row=row,
            d=np.asarray(lines.rms)[used] - modelRms,
            dS=np.asarray(lines.S)[used] - modelS,
            chi2=np.asarray(lines.chi2)[used]/np.maximum(np.asarray(lines.ndof)[used], 1),
            disp=np.asarray(lines.dispersion)[used],
            status=np.asarray(lines.status)[used],
            peak=np.asarray(lines.peak)[used],
        )
        wl = np.asarray(lines.wavelength)[used]
        for ii, (vv, ww) in enumerate(zip(visit[used], wl)):
            entry = perLine[(LAMPS[vv], round(float(ww), 4))]
            for kk, cc in columns.items():
                entry[kk].append(cc[ii])

    catalogues = {}
    rows = []
    for (lamp, wl), entry in perLine.items():
        if len(entry["d"]) < 50:
            continue
        if lamp not in catalogues:
            catalogues[lamp] = readLineList(lamp)
        cWl, cFlux, _ = catalogues[lamp]
        disp = np.median(entry["disp"])
        sep = np.abs(cWl - wl)/disp
        self_ = np.argmin(np.abs(cWl - wl))
        own = cFlux[self_] if cFlux[self_] > 0 else np.nan
        others = np.ones(len(cWl), dtype=bool)
        others[self_] = False
        nearest = np.min(sep[others])
        n1, n2 = np.sum(others & (sep < 1)), np.sum(others & (sep < 2))
        f1 = np.sum(cFlux[others & (sep < 1)])/own
        f2 = np.sum(cFlux[others & (sep < 2)])/own
        status = int(np.bitwise_or.reduce(np.asarray(entry["status"], dtype=int)))
        flags = "".join(kk for kk, bit in STATUS.items() if status & bit) or "-"
        rows.append((lamp, wl, np.median(entry["row"]), len(entry["d"]), np.median(entry["d"]),
                     np.median(entry["dS"]), np.median(entry["chi2"]), nearest, n1, f1, n2, f2, flags,
                     np.median(entry["peak"])))

    rows.sort(key=lambda rr: -rr[4])
    print(f"{'lamp':5} {'wavelength':>10} {'row':>5} {'n':>5} {'d':>7} {'dS':>6} {'chi2':>5}"
          f" {'nearest':>7} {'n<1':>3} {'f<1':>6} {'n<2':>3} {'f<2':>6} {'peak':>7} status")
    for rr in rows:
        print(f"{rr[0]:5} {rr[1]:10.3f} {rr[2]:5.0f} {rr[3]:5d} {rr[4]:+7.3f} {rr[5]:+6.2f} {rr[6]:5.1f}"
              f" {rr[7]:7.2f} {rr[8]:3d} {rr[9]:6.3f} {rr[10]:3d} {rr[11]:6.3f} {rr[13]:7.0f} {rr[12]}")

    names = dict(dS=5, chi2=6, nearest=7, f2=11, peak=13)
    print("\nSpearman rank correlation of d with:   " + " ".join(f"{nn:>13}" for nn in names))
    for lamp in [None] + sorted({rr[0] for rr in rows}):
        sub = [rr for rr in rows if lamp is None or rr[0] == lamp]
        if len(sub) < 4:
            continue
        dd = np.array([rr[4] for rr in sub])
        out = []
        for jj in names.values():
            xx = np.nan_to_num(np.array([rr[jj] for rr in sub], dtype=float), nan=0.0)
            rho, pp = scipy.stats.spearmanr(dd, xx) if np.ptp(xx) > 0 else (np.nan, np.nan)
            out.append(f"{rho:+.2f} (p {pp:.2f})")
        print(f"  {lamp or 'all':5} {len(sub):3d} lines" + " "*20 + " ".join(f"{oo:>13}" for oo in out))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "u/rhl/PIPE2D-1940-armLsf-lampAvg")
