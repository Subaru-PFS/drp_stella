#!/usr/bin/env python

from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd

from lsst.daf.butler import Butler
from pfs.drp.stella.referenceLine import ReferenceLineSet
from pfs.drp.stella.lineLists.reductions import compareReductions, compareLineLists


def reductions(args):
    old = Butler(args.repo, collections=[args.old] + args.extraCollections)
    new = Butler(args.repo, collections=[args.new] + args.extraCollections)
    tables = []
    for pair in args.visits:
        visit, lineList = pair.split("=")
        lines = ReferenceLineSet.fromLineList(os.path.join(args.lineLists, lineList))
        table = compareReductions(old, new, int(visit), lines, numLines=args.numLines)
        table["lineList"] = lineList
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    if args.output:
        table.to_pickle(args.output)
    perLine = table.groupby(["lineList", "arm", "wavelength"]).agg(
        crFrac=("crFrac", "mean"), ratio=("ratio", "mean"), ratioMedian=("ratioMedian", "mean")).reset_index()
    low = 1 - args.tolerance
    pd.set_option("display.width", 200)
    print(f"Per list and arm: lines, old/new ratio of the mean flux over fibres, lines below {low}")
    print(perLine.groupby(["lineList", "arm"]).agg(
        num=("ratio", "size"), median=("ratio", "median"), min=("ratio", "min"),
        numLow=("ratio", lambda xx: int((xx < low).sum())), crFrac=("crFrac", "mean")).round(3).to_string())
    print(f"\nLines with old/new < {low}")
    print(perLine[perLine.ratio < low].sort_values(["lineList", "ratio"]).round(3).to_string(index=False))


def lists(args):
    for pair in args.pairs:
        old, new = pair.split("=")
        table = compareLineLists(old, new, minSN=args.minSN, tolerance=args.tolerance)
        bright = table[table.bright]
        changed = bright[bright.changed]
        worst = changed.iloc[np.argsort(-np.abs(np.log(changed.ratio.to_numpy())))][:6]
        print(f"{os.path.basename(old)}: {len(table)} measured ({len(bright)} at S/N > {args.minSN}); "
              f"median old/new {table.ratio.median():.4f}; changed by > {args.tolerance:.0%}: "
              f"{table.changed.sum()} ({len(changed)} at S/N > {args.minSN}); worst (wavelength: old/new): "
              + ", ".join(f"{wl:.3f}: {rr:.3f}" for wl, rr in zip(worst.wavelength, worst.ratio)))


def main():
    parser = ArgumentParser(description="Compare two reductions of arc exposures, or the line lists measured "
                                        "from them (PIPE2D-2003, PIPE2D-2004)")
    sub = parser.add_subparsers(dest="command", required=True)
    parse = sub.add_parser("reductions", help="Compare the brightest lines' fluxes between two reductions")
    parse.add_argument("repo", help="Butler repository")
    parse.add_argument("--old", required=True, help="Collection of the first reduction (with crMask)")
    parse.add_argument("--new", required=True, help="Collection of the second reduction")
    parse.add_argument("--extraCollections", nargs="*", default=["PFS/raw/sps"],
                       help="Collections searched after each reduction")
    parse.add_argument("--lineLists", required=True, help="Directory with the line lists")
    parse.add_argument("visits", nargs="+", help="visit=lineList pairs, e.g. 148314=Ar.txt")
    parse.add_argument("--numLines", type=int, default=30, help="Brightest lines per arm")
    parse.add_argument("--tolerance", type=float, default=0.02, help="Report lines lower by more than this")
    parse.add_argument("--output", help="Pickle the per-spectrograph table here")
    parse.set_defaults(function=reductions)
    parse = sub.add_parser("lists", help="Compare two versions of line lists")
    parse.add_argument("pairs", nargs="+", help="old=new pairs of line-list paths")
    parse.add_argument("--minSN", type=float, default=30, help="S/N above which a line is bright")
    parse.add_argument("--tolerance", type=float, default=0.02, help="Fractional change that counts")
    parse.set_defaults(function=lists)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
