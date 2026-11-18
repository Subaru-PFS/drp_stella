#!/usr/bin/env python

from argparse import ArgumentParser
import os

from lsst.daf.butler import Butler
from pfs.drp.stella.referenceLine import ReferenceLineSet
from pfs.drp.stella.lineLists.intensities import measureLampIntensities


def main():
    parser = ArgumentParser(description="Measure a lamp's arc-line intensities (counts/s) from one visit, "
                                        "and write them into a copy of its line list")
    parser.add_argument("lineList", help="Lamp's line list (e.g. .../obs_pfs/pfs/lineLists/Ne_IIS.txt)")
    parser.add_argument("output", help="Line list to write")
    parser.add_argument("--repo", default="/work/datastore", help="Butler repository")
    parser.add_argument("--collection", required=True, nargs="+", help="Collection(s) with the reduced visit")
    parser.add_argument("--visit", type=int, required=True, help="Arc visit")
    parser.add_argument("--engineering", action="store_true",
                        help="Use only the engineering (IIS) fibres (for IIS lamps)")
    parser.add_argument("--integratedSN", type=float, default=None,
                        help="Replace line groups with S/N above this by integrated fluxes "
                             "(e.g. 30 for IIS lamps, whose profile the PFI LSF doesn't describe)")
    parser.add_argument("--previous", default=None,
                        help="Scale unmeasured lines from this list (e.g. the PFI lamp's list)")
    args = parser.parse_args()

    butler = Butler(args.repo, collections=args.collection + ["PFS/raw/pfsConfig", "PFS/raw/sps",
                                                              "PFS/defaults", "PFS/calib"])
    lines = ReferenceLineSet.fromLineList(os.path.abspath(args.lineList))
    previous = ReferenceLineSet.fromLineList(os.path.abspath(args.previous)) if args.previous else None
    newLines, info = measureLampIntensities(butler, args.visit, lines, engineering=args.engineering,
                                            integratedSN=args.integratedSN, previous=previous)
    newLines.writeLineList(args.output)

    result = info["result"]
    print(f"visit {args.visit}: expTime {info['expTime']:.3f} s (mean over pfsArms); "
          f"{int(result.fitted.sum())} lines fitted -> {args.output}")
    for species, scale in info["scales"].items():
        print(f"{species}: unmeasured lines scaled from {args.previous} by {scale:.3g}")
    print("Largest residuals, away from the brightest lines (to find missing lines):")
    print(info["residuals"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
