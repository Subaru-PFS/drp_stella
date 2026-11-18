#!/usr/bin/env python
"""Claude: residuals of bright arc lines from a fitted armLsf, in units of each line's peak

For each detector, the used, bright lines in lsfLines are refit with the model's
kernel at their fitted centres (only an amplitude and a constant are free), and
the residuals are divided by the fitted peak. Per lamp, it prints:

- maxAbs: the median over lines of the largest absolute residual (includes the noise);
- noise: the median over lines of the largest standard deviation, over the peak;
- stack: the largest absolute value of the stacked residual profile (the
  systematic misfit), and its offset (pixels) from the line centre.

The collections must include the RUN collection that FitLsfTask wrote: in a
CALIBRATION collection the model can only be looked up with an exposure's date.

Example::

    pfsLsfResiduals.py /work/datastore --collections u/me/armLsf u/me/arcs PFS/defaults --detectors b2 r2
"""
from argparse import ArgumentParser

from lsst.daf.butler import Butler

from pfs.drp.stella.lsfDiagnostics import lineResiduals, selectUsedLines, summarizeResiduals


def main():
    parser = ArgumentParser(description="Residuals of bright arc lines from a fitted armLsf")
    parser.add_argument("repo", help="Butler repository")
    parser.add_argument("--collections", nargs="+", required=True,
                        help="Collections with armLsf, lsfLines and the pfsArms of the arcs")
    parser.add_argument("--detectors", nargs="+", default=None,
                        help="Detectors, e.g. b2 r2 (default: all with an armLsf)")
    parser.add_argument("--minSn", type=float, default=300, help="Minimum flux S/N of the lines")
    parser.add_argument("--maxLines", type=int, default=400, help="Maximum number of lines per detector")
    parser.add_argument("--halfWidth", type=float, default=12, help="Kernel half-width (pixels)")
    parser.add_argument("--perVisit", action="store_true",
                        help="Include each visit's (lamp's) offsets, rather than using the average lamp")
    parser.add_argument("--datasetType", default="armLsf_calib",
                        help="Dataset type of the LSF model ('armLsf' for collections written before it was "
                             "a calibration)")
    args = parser.parse_args()

    butler = Butler(args.repo, collections=args.collections)
    if args.detectors is None:
        refs = butler.registry.queryDatasets(args.datasetType, findFirst=True)
        detectors = sorted({(ref.dataId["arm"], ref.dataId["spectrograph"]) for ref in refs})
    else:
        detectors = [(det[0], int(det[1:])) for det in args.detectors]

    print(f"{'det':3} {'lamp':6} {'n':>4} {'maxAbs':>7} {'noise':>7} {'stack':>7} {'at':>6}")
    for arm, spectrograph in detectors:
        dataId = dict(instrument="PFS", arm=arm, spectrograph=spectrograph)
        model = butler.get(args.datasetType, dataId)
        lines = selectUsedLines(butler.get("lsfLines", dataId), args.minSn)
        if len(lines) > args.maxLines:
            lines = lines.sample(args.maxLines, random_state=1)

        def getSpectra(visit):
            return butler.get("pfsArm", dataId, visit=visit)

        residuals = lineResiduals(model, lines, getSpectra, args.halfWidth, args.perVisit)
        for lamp, ss in summarizeResiduals(residuals, halfWidth=args.halfWidth).items():
            print(f"{arm}{spectrograph:<2d} {lamp:6} {ss['num']:4d} {ss['maxAbs']:7.4f} {ss['noise']:7.4f} "
                  f"{ss['worst']:+7.4f} {ss['worstOffset']:+6.2f}")


if __name__ == "__main__":
    main()
