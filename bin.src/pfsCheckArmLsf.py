#!/usr/bin/env python
"""Claude: check the LSF of pfsArms (reduceExposure doArmLsf=True) against the lines measured in them

For each pfsArm, using the ``lines`` dataset that reduceExposure measured (arc or sky lines):

- position (dPos, sPos): the kernel's first moment at the catalogue wavelength
  minus the measured centroid (pixels): are the LSF's wavelengths consistent
  with the line positions?  yErr is the median centroid error;
- width: the Gaussian rms of the data over that of the kernel (same window,
  same fitter), for bright, isolated lines;
- skew (dSkew): (first moment - Gaussian centre) of the data minus that of the
  kernel: the LSF's asymmetry.

Statistics are medians and robust (1.4826 MAD) scatters. With ``--breakdown``,
the widths and skews are also broken down by species and wavelength.

Example::

    pfsCheckArmLsf.py /work/datastore u/me/doArmLsf-sky 148258 148259 --breakdown
"""
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from lsst.daf.butler import Butler

from pfs.drp.stella.lsfDiagnostics import checkSpectraLsf, robustStats
from pfs.drp.stella.spectraLsf import BoundArmLsf


def breakdown(lines, minLines=5, minFibers=20):
    """Print the widths and skews by species and wavelength"""
    def show(label, select):
        if select.sum() < minLines:
            return
        width, skew, sn = (lines[name][select].median() for name in ("width", "skew", "sn"))
        print(f"  {label:24} n={select.sum():5d}  width {width:.3f}  dSkew {skew:+.3f}"
              f"  (S/N median {sn:.0f})")

    print("By species:")
    for desc in sorted(set(lines.description)):
        show(desc, lines.description == desc)
    print("By wavelength (nm):")
    edges = np.percentile(lines.wavelength, np.linspace(0, 100, 9))
    for lo, hi in zip(edges[:-1], edges[1:]):
        show(f"{lo:.0f}-{hi:.0f}", (lines.wavelength >= lo) & (lines.wavelength <= hi))
    rounded = np.round(lines.wavelength, 3)
    perLine = lines.groupby(rounded).agg(num=("width", "size"), width=("width", "median"),
                                         skew=("skew", "median"))
    perLine = perLine[perLine.num >= minFibers]
    if len(perLine) > 0:
        width, skew = (perLine[name].median() for name in ("width", "skew"))
        print(f"Median over {len(perLine)} distinct lines (>= {minFibers} each): "
              f"width {width:.3f}, dSkew {skew:+.3f}")


def main():
    parser = ArgumentParser(description="Check the LSF of pfsArms against the lines measured in them")
    parser.add_argument("repo", help="Butler repository")
    parser.add_argument("collection", help="Collection with the pfsArms and their lines")
    parser.add_argument("visits", nargs="+", type=int, help="Visits")
    parser.add_argument("--minSn", type=float, default=100, help="Minimum peak S/N for widths and skews")
    parser.add_argument("--fiberStep", type=int, default=5, help="Measure widths in every n-th fibre")
    parser.add_argument("--breakdown", action="store_true",
                        help="Break the widths and skews down by species and wavelength")
    parser.add_argument("--dump", default=None, help="Write the bright lines' measurements to this CSV file")
    args = parser.parse_args()

    butler = Butler(args.repo, collections=args.collection)
    where = f"instrument='PFS' AND visit IN ({', '.join(str(vv) for vv in args.visits)})"
    refs = sorted(butler.registry.queryDatasets("pfsArm", where=where, findFirst=True),
                  key=lambda ref: (ref.dataId["visit"], ref.dataId["arm"], ref.dataId["spectrograph"]))
    print(f"{'visit':>6} {'det':3} {'lsf':12} {'nPos':>5} {'dPos':>7} {'sPos':>6} {'yErr':>6}"
          f" {'nShape':>6} {'width':>6} {'sWidth':>6} {'dSkew':>7} {'sSkew':>6}")
    tables = []
    for ref in refs:
        dataId = ref.dataId
        det = f"{dataId['arm']}{dataId['spectrograph']}"
        lsf = butler.get(ref.makeComponentRef("lsf"))
        kind = type(lsf).__name__ if lsf is not None else "None"
        if not isinstance(lsf, BoundArmLsf):
            print(f"{dataId['visit']:6d} {det:3} {kind:12} -- no bound arm LSF")
            continue
        result = checkSpectraLsf(butler.get(ref), lsf, butler.get("lines", dataId), minSn=args.minSn,
                                 fiberStep=args.fiberStep)
        mPos, sPos, nPos = robustStats(result.position)
        mWidth, sWidth, nWidth = robustStats(result.width)
        mSkew, sSkew, _ = robustStats(result.skew)
        yErr = np.median(result.positionErr) if len(result.positionErr) > 0 else np.nan
        print(f"{dataId['visit']:6d} {det:3} {kind:12} {nPos:5d} {mPos:+7.3f} {sPos:6.3f} {yErr:6.3f}"
              f" {nWidth:6d} {mWidth:6.3f} {sWidth:6.3f} {mSkew:+7.3f} {sSkew:6.3f}")
        table = result.lines.assign(visit=dataId["visit"], detector=det)
        tables.append(table)

    lines = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if args.dump is not None:
        lines.to_csv(args.dump, index=False)
    if args.breakdown and len(lines) > 0:
        for arm in sorted(set(det[0] for det in lines.detector)):
            print(f"=== arm {arm}")
            breakdown(lines[lines.detector.str.startswith(arm)])


if __name__ == "__main__":
    main()
