#!/usr/bin/env python

from argparse import ArgumentParser
import os

from pfs.drp.stella.referenceLine import ReferenceLineSet
from pfs.drp.stella.lineLists.nist import ROMAN, parseSpecies, speciesInLines, fetchNistLines, fetchNistLevels


def main():
    parser = ArgumentParser(description="Download NIST ASD lines (vacuum wavelengths) for the species in "
                                        "line lists, or for given species, one species per query; files are "
                                        "named nist_<El>[<stage>].dat (and nist_levels_<El>[<stage>].dat)")
    parser.add_argument("lineLists", nargs="*", help="Line lists whose species and wavelength range to fetch")
    parser.add_argument("--species", nargs="*", default=[], help="Species to fetch, e.g. CdI HgI")
    parser.add_argument("--range", nargs=2, type=float, metavar=("MIN", "MAX"), default=None,
                        help="Wavelength range (nm); default: the line lists' range, padded")
    parser.add_argument("--levels", action="store_true", help="Also fetch the energy levels")
    parser.add_argument("--outdir", default=".", help="Directory for the dumps")
    args = parser.parse_args()

    species = {parseSpecies(s) for s in args.species} - {None}
    lamMin, lamMax = (args.range if args.range else (1e9, -1e9))
    for path in args.lineLists:
        sp, lo, hi = speciesInLines(ReferenceLineSet.fromLineList(os.path.abspath(path)))
        species |= sp
        if not args.range:
            lamMin, lamMax = min(lamMin, lo - 5), max(lamMax, hi + 20)
    if not species:
        parser.error("no species given or found")
    if lamMin > lamMax:
        lamMin, lamMax = 370.0, 1270.0
    roman = {v: k for k, v in ROMAN.items()}
    for element, stage in sorted(species):
        suffix = f"{element}{'' if stage == 1 else roman[stage]}"
        path = fetchNistLines(element, stage, lamMin, lamMax, os.path.join(args.outdir, f"nist_{suffix}.dat"))
        print(f"{element} {roman[stage]}: {lamMin:.1f}-{lamMax:.1f} nm -> {path}")
        if args.levels:
            levels = os.path.join(args.outdir, f"nist_levels_{suffix}.dat")
            print("   levels ->", fetchNistLevels(element, stage, levels))


if __name__ == "__main__":
    main()
