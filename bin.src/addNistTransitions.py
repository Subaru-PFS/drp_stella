#!/usr/bin/env python

from argparse import ArgumentParser
import os

from pfs.drp.stella.referenceLine import ReferenceLineSet
from pfs.drp.stella.lineLists.nist import loadNist, addTransitions


def main():
    parser = ArgumentParser(description="Fill in the transitions of a line list from NIST ASD dumps "
                                        "(see fetchNistLines.py); lines already identified are left alone")
    parser.add_argument("lineList", help="Line list to update")
    parser.add_argument("nist", nargs="+", help="NIST lines dumps (nist_<El>[<stage>].dat)")
    parser.add_argument("--output", default=None,
                        help="Line list to write (default: update lineList in place)")
    args = parser.parse_args()

    path = os.path.abspath(args.lineList)
    newLines, counts = addTransitions(ReferenceLineSet.fromLineList(path), loadNist(args.nist))
    newLines.writeLineList(args.output or path)
    print(", ".join(f"{k} {v}" for k, v in counts.items()), "->", args.output or path)


if __name__ == "__main__":
    main()
