"""Claude: tests of "the lamps fill the telescope pupil differently" with a FitLsf run (armLsf + lsfLines)

Usage: python lampPupilChecks.py [COLLECTION]   (default u/rhl/PIPE2D-1940-armLsf-lampAvg; arms b, r, n)

1. shape: per detector, each lamp's visit offset in rms and in K (and S) about the average lamp; do rms and K
   move together across lamps?  (Spearman over all (detector, lamp) offsets, and per arm)
2. focus: per detector, the spread of the lamp rms offsets (Ne minus the mean of the others, and the rms of
   the offsets) against the detector's median model blur rms and the rms surface's chi^2/ndof
3. fibres: per detector, Ne minus the other lamps in d = line rms - model rms (with the MTP offset), per MTP
   sub-bundle; chi^2/dof of a constant across sub-bundles (errors 1.4826 MAD/sqrt(n) from each side)
"""
import os
import sys

import numpy as np
import scipy.stats
from lsst.daf.butler import Butler

REPO = os.environ.get("PFS_REPO", "/work/datastore")  # Butler repository (or its butler.yaml)

LAMPS = {148314: "Ar", 148317: "Xe", 148320: "Kr", 148323: "HgCd", 148326: "Ne"}


def robust(xx):
    xx = np.asarray(xx)
    xx = xx[np.isfinite(xx)]
    if len(xx) < 5:
        return np.nan, np.nan
    mm = np.median(xx)
    return mm, 1.4826*np.median(np.abs(xx - mm))/np.sqrt(len(xx))


def offsets(surface):
    table = surface.offsets.get("visit", {})
    return {LAMPS[int(vv)]: (val[0] if isinstance(val, tuple) else val)
            for vv, val in table.items() if int(vv) in LAMPS}


def main(collection):
    butler = Butler(REPO, collections=collection)
    shape = []                      # (arm, det, lamp, dRms, dK, dS)
    focus = []
    fibres = []
    for arm in "brn":
        for sp in (1, 2, 3, 4):
            det = f"{arm}{sp}"
            dataId = dict(instrument="PFS", arm=arm, spectrograph=sp)
            model = butler.get("armLsf", dataId)
            data = model.toDatamodel()
            offRms = offsets(data.surfaces["rms"])
            offK = offsets(data.surfaces["K"]) if "K" in data.surfaces else {}
            offS = offsets(data.surfaces["S"]) if "S" in data.surfaces else {}
            for lamp in offRms:
                shape.append((arm, det, lamp, offRms[lamp], offK.get(lamp, np.nan), offS.get(lamp, np.nan)))

            # model blur rms over the detector: every 20th fibre, rows every 200
            fiberIds = sorted(model.fiberGroups.get("mtp", {}))[::20] or None
            rows = np.arange(100, 4100, 200, dtype=float)
            modelRms = np.concatenate([np.asarray(model.getParameters(ff, rows)["rms"]) for ff in fiberIds])
            others = [vv for kk, vv in offRms.items() if kk != "Ne"]
            neMinus = offRms.get("Ne", np.nan) - (np.mean(others) if others else np.nan)
            spread = np.std(list(offRms.values()))
            surf = data.surfaces["rms"]
            focus.append((det, np.median(modelRms), surf.chi2/max(1, surf.ndof), neMinus, spread))

            # per-MTP Ne minus others, from the lines
            lines = butler.get("lsfLines", dataId)
            visit = np.asarray(lines.visit)
            used = (((np.asarray(lines.flag) & ~16) == 0) & np.isfinite(lines.rms) & np.isfinite(lines.pixel)
                    & np.isin(visit, list(LAMPS)))
            fiberId = np.asarray(lines.fiberId)[used]
            row = np.asarray(lines.pixel)[used]
            mRms = np.full(used.sum(), np.nan)
            for ff in np.unique(fiberId):
                this = fiberId == ff
                mRms[this] = np.asarray(model.getParameters(int(ff), row[this])["rms"])
            d = np.asarray(lines.rms)[used] - mRms
            isNe = np.array([LAMPS[vv] == "Ne" for vv in visit[used]])
            mtpOf = model.fiberGroups.get("mtp", {})
            mtp = np.array([mtpOf.get(int(ff), "") for ff in fiberId])
            perMtp = []
            for mm in sorted(set(mtp) - {""}):
                ne, neErr = robust(d[(mtp == mm) & isNe])
                ot, otErr = robust(d[(mtp == mm) & ~isNe])
                if np.isfinite(ne) and np.isfinite(ot):
                    perMtp.append((mm, ne - ot, np.hypot(neErr, otErr)))
            if perMtp:
                vals = np.array([pp[1] for pp in perMtp])
                errs = np.array([pp[2] for pp in perMtp])
                ww = 1/np.maximum(errs, 1e-4)**2
                mean = np.sum(ww*vals)/np.sum(ww)
                chi2 = np.sum(ww*(vals - mean)**2)/max(1, len(vals) - 1)
                fibres.append((det, len(vals), mean, np.std(vals), np.median(errs), chi2,
                               min(perMtp, key=lambda pp: pp[1]), max(perMtp, key=lambda pp: pp[1])))

    print("1. lamp offsets about the average lamp: rms (px), K, S")
    print(f"  {'det':3} " + " ".join(f"{ll:>22}" for ll in LAMPS.values()))
    for det in dict.fromkeys(ss[1] for ss in shape):
        row = {ss[2]: ss for ss in shape if ss[1] == det}
        cells = [f"{row[ll][3]:+6.3f} {row[ll][4]:+6.2f} {row[ll][5]:+6.2f} " if ll in row else f"{'':>22}"
                 for ll in LAMPS.values()]
        print(f"  {det:3} " + " ".join(cells))
    for arm in [None, "b", "r", "n"]:
        sub = [ss for ss in shape if arm is None or ss[0] == arm]
        rr, kk, sS = (np.array([ss[jj] for ss in sub]) for jj in (3, 4, 5))
        good = np.isfinite(rr) & np.isfinite(kk)
        rhoK, pK = scipy.stats.spearmanr(rr[good], kk[good])
        rhoS, pS = scipy.stats.spearmanr(rr[good], sS[good])
        print(f"  Spearman(rms, K) {arm or 'all'}: {rhoK:+.2f} (p {pK:.3f}, n {good.sum()});"
              f"  (rms, S): {rhoS:+.2f} (p {pS:.3f})")

    print("\n2. per detector: median model blur rms, rms surface chi2/ndof, Ne - mean(others), "
          "rms of lamp offsets")
    for ff in focus:
        print(f"  {ff[0]:3} model rms {ff[1]:.3f}  chi2/ndof {ff[2]:5.1f}  Ne-others {ff[3]:+.3f}"
              f"  spread {ff[4]:.3f}")
    for jj, name in ((1, "model rms"), (2, "chi2/ndof")):
        for kk, what in ((3, "|Ne-others|"), (4, "spread")):
            xx = np.array([ff[jj] for ff in focus])
            yy = np.abs(np.array([ff[kk] for ff in focus]))
            rho, pp = scipy.stats.spearmanr(xx, yy)
            print(f"  Spearman({name}, {what}): {rho:+.2f} (p {pp:.3f})")

    print("\n3. Ne - other lamps in (line rms - model rms) per MTP sub-bundle")
    print(f"  {'det':3} {'nMtp':>4} {'mean':>7} {'std':>6} {'medErr':>6} {'chi2/dof':>8}"
          "  min, max sub-bundle")
    for ff in fibres:
        print(f"  {ff[0]:3} {ff[1]:4d} {ff[2]:+7.3f} {ff[3]:6.3f} {ff[4]:6.3f} {ff[5]:8.1f}"
              f"  {ff[6][0]} {ff[6][1]:+.3f}, {ff[7][0]} {ff[7][1]:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "u/rhl/PIPE2D-1940-armLsf-lampAvg")
