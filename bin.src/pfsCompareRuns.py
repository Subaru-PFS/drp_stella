#!/usr/bin/env python
"""Claude: compare the spectra written by two pipeline runs (a regression check)

To check a change, run the same tasks on the same data with the reference
code and with the new code, into different collections; e.g. for each of
the two checkouts::

    setup -k -r CHECKOUT
    export PYTHONHASHSEED=0       # so that any iteration over a set is reproducible
    pipetask run -b REPO -i INPUT -o u/me/TAG-merge -d "visit IN (...)" \\
        -p "$DRP_STELLA_DIR/pipelines/reduceExposure.yaml#mergeArms"
    pipetask run -b REPO -i INPUT -o u/me/TAG-coadd -d "combination='...'" \\
        -p "$DRP_STELLA_DIR/pipelines/science.yaml#coaddSpectra"

and then compare the two runs::

    pfsCompareRuns.py REPO u/me/ref-merge u/me/new-merge --datasets pfsMerged
    pfsCompareRuns.py REPO u/me/ref-coadd u/me/new-coadd --datasets pfsCoadd --noise
"""
from argparse import ArgumentParser

from pfs.drp.stella.compareRuns import compareRuns


def main():
    parser = ArgumentParser(description="Compare the spectra written by two pipeline runs")
    parser.add_argument("repo", help="Butler repository")
    parser.add_argument("reference", help="Collection of the reference run")
    parser.add_argument("new", help="Collection of the new run")
    parser.add_argument("--datasets", nargs="+", default=["pfsMerged", "pfsCoadd"],
                        help="Dataset types to compare (sets of fibre or target spectra)")
    parser.add_argument("--where", default="", help="Data query restricting the datasets")
    parser.add_argument("--noise", action="store_true",
                        help="Report the changes in units of the noise (unmasked pixels)")
    parser.add_argument("--detail", action="store_true",
                        help="Report the changes in each kind of array, with the worst pixels")
    args = parser.parse_args()

    for comparison in compareRuns(args.repo, args.reference, args.new, args.datasets, args.where,
                                  detail=args.detail):
        print("\n".join(comparison.summaryLines()))
        if args.noise:
            print("\n".join(comparison.noiseLines()))
        if args.detail:
            print("\n".join(comparison.detailLines()))


if __name__ == "__main__":
    main()
