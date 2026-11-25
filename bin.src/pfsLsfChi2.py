#!/usr/bin/env python
"""Claude: where are the chi^2/ndof of an LSF fit's per-line fits large?

For each detector, the used lines in lsfLines: the median chi^2/ndof overall and
for bright lines, binned in detector x, y and S/N, per visit (lamp), and the
worst fibres. With ``--shape``, the bright lines' shape parameters binned in x
or y.

Example::

    pfsLsfChi2.py /work/datastore u/me/armLsf --detectors b3 r4 b1 --shape y
"""
from argparse import ArgumentParser

from lsst.daf.butler import Butler

from pfs.drp.stella.lsfDiagnostics import chi2Breakdown, parameterBreakdown, selectUsedLines


def main():
    parser = ArgumentParser(description="Where the per-line LSF fits' chi^2/ndof are large")
    parser.add_argument("repo", help="Butler repository")
    parser.add_argument("collection", help="Collection with lsfLines")
    parser.add_argument("--detectors", nargs="+", default=None,
                        help="Detectors, e.g. b3 r4 (default: all with lsfLines)")
    parser.add_argument("--brightSn", type=float, default=300, help="Flux S/N of the bright lines")
    parser.add_argument("--shape", choices=("x", "y"), default=None,
                        help="Also bin the bright lines' shape parameters along this axis")
    args = parser.parse_args()

    butler = Butler(args.repo, collections=args.collection)
    if args.detectors is None:
        refs = butler.registry.queryDatasets("lsfLines", findFirst=True)
        detectors = sorted({(ref.dataId["arm"], ref.dataId["spectrograph"]) for ref in refs})
    else:
        detectors = [(det[0], int(det[1:])) for det in args.detectors]

    for arm, spectrograph in detectors:
        lines = butler.get("lsfLines", instrument="PFS", arm=arm, spectrograph=spectrograph)
        used = selectUsedLines(lines)
        bright = selectUsedLines(used, args.brightSn)
        length = 4096 if arm == "n" else 4176   # rows of the H4RG or CCDs
        result = chi2Breakdown(used, length=length)
        brightMedian = (bright.chi2/bright.ndof).median()
        print(f"=== {arm}{spectrograph}: {result['num']} lines, median chi2/ndof {result['median']:.2f}; "
              f"S/N > {args.brightSn:g}: {len(bright)} lines, median {brightMedian:.2f}")
        for name in ("x", "y", "sn"):
            bins = [f"{start:.0f}:{median:.2f}({num})" for start, median, num in result[name]]
            print(f"  by {name}: " + "  ".join(bins))
        print("  by visit/lamp: " + "  ".join(f"{visit}/{desc}:{median:.2f}({num})"
                                              for visit, desc, median, num in result["visit"]))
        fiber = result["fiber"]
        print(f"  per-fibre medians: 10% {fiber.quantile(0.1):.2f}, 50% {fiber.median():.2f}, "
              f"90% {fiber.quantile(0.9):.2f}; worst: "
              + ", ".join(f"{ff}:{vv:.1f}" for ff, vv in fiber[-8:].items()))
        if args.shape is not None and len(bright) > 0:
            extent = 4096 if args.shape == "x" else length
            print(f"  bright lines binned in {args.shape}:")
            print(parameterBreakdown(bright, args.shape, extent).to_string(index=False, float_format="%.4g"))


if __name__ == "__main__":
    main()
