#!/usr/bin/env python

from argparse import ArgumentParser
import os

from pfs.drp.stella.referenceLine import ReferenceLineSet
from pfs.drp.stella.lineLists.broad import flagBroadFamilies


def main():
    parser = ArgumentParser(description="Set BROAD on every member of the given resonance families, and "
                                        "clear it on the other lines with known transitions (PIPE2D-1945)")
    parser.add_argument("lineList", help="Line list to update (e.g. .../obs_pfs/pfs/lineLists/Kr.txt)")
    parser.add_argument("families", nargs="*",
                        help="Families to flag: names (1s2, 1s4, 6p3P1, 5p1P1, 5p3P1) or lower-level tokens "
                             "(e.g. '5s_2[1/2]*_1'); none clears BROAD on all lines with known transitions")
    parser.add_argument("--output", default=None,
                        help="Line list to write (default: update lineList in place)")
    parser.add_argument("--dry-run", action="store_true", help="Only report the changes")
    args = parser.parse_args()

    path = os.path.abspath(args.lineList)
    newLines, counts = flagBroadFamilies(ReferenceLineSet.fromLineList(path), args.families)
    print(f"{os.path.basename(path)}: BROAD before {counts['before']}, after {counts['after']} "
          f"(set {counts['set']}, cleared {counts['cleared']})")
    if counts["cleared"]:
        print("  cleared:", " ".join(f"{wl:.3f}" for wl in counts["clearedWavelengths"]))
    if not args.dry_run:
        newLines.writeLineList(args.output or path)
        print("  ->", args.output or path)


if __name__ == "__main__":
    main()
