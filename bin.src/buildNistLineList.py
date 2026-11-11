#!/usr/bin/env python

from argparse import ArgumentParser

from pfs.drp.stella.lineLists.nist import buildLineList


def main():
    parser = ArgumentParser(description="Build a line list for one species from a NIST ASD lines dump, "
                                        "merging components that PFS can't resolve")
    parser.add_argument("nist", help="NIST lines dump for one species (see fetchNistLines.py)")
    parser.add_argument("description", help="Species for the description column, e.g. HeI")
    parser.add_argument("output", help="Line list to write")
    parser.add_argument("--minFrac", type=float, default=0.0,
                        help="Keep lines with NIST intensity >= this fraction of the strongest")
    parser.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"), default=(370.0, 1270.0),
                        help="Wavelength range (nm)")
    parser.add_argument("--gap", type=float, default=0.06, help="Merge components closer than this (nm)")
    args = parser.parse_args()

    lines = buildLineList(args.nist, args.description, minFrac=args.minFrac, lamMin=args.range[0],
                          lamMax=args.range[1], gap=args.gap)
    lines.writeLineList(args.output)
    print(f"{len(lines)} {args.description} lines -> {args.output}")


if __name__ == "__main__":
    main()
