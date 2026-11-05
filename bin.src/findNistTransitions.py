#!/usr/bin/env python

from argparse import ArgumentParser

from pfs.drp.stella.lineLists.nist import readLevels, findTransitions


def main():
    parser = ArgumentParser(description="List allowed transitions between NIST energy levels with Ritz "
                                        "(vacuum) wavelengths near the given wavelengths: to identify lines "
                                        "missing from NIST's line list")
    parser.add_argument("levels", help="NIST energy-levels dump (fetchNistLines.py --levels)")
    parser.add_argument("wavelength", type=float, nargs="+", help="Observed (vacuum) wavelength(s), nm")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Search +- this (nm)")
    args = parser.parse_args()

    levels = readLevels(args.levels)
    for wavelength in args.wavelength:
        print(f"{wavelength}:")
        print(findTransitions(levels, wavelength, args.tolerance).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
