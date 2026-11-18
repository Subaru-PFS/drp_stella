#!/usr/bin/env python
"""Claude: the widths of the Gaussians equivalent to a fitted armLsf

At a central fibre and a few rows of each detector, prints the rms (pixels) of:

- fwhm: the FWHM of the pixel-integrated kernel, over 2.3548;
- fit: the unweighted least-squares Gaussian (``computeWidth(method="fit")``):
  right for unresolved lines;
- fit(s): the same for a Gaussian line of intrinsic rms ``s`` pixels, with
  ``s`` removed in quadrature;
- moment: the second moment within +/- 12 pixels, halo included
  (``method="moment"``): right for broad Gaussian lines;
- core: the second moment within +/- 4 pixels.

The collection must be the RUN collection that FitLsfTask wrote: in a
CALIBRATION collection the model can only be looked up with an exposure's date.

Example::

    pfsLsfWidths.py /work/datastore u/me/armLsf --detectors b2 r2 n2
"""
from argparse import ArgumentParser

from lsst.daf.butler import Butler

from pfs.drp.stella.lsfDiagnostics import gaussianEquivalents


def main():
    parser = ArgumentParser(description="Widths of the Gaussians equivalent to a fitted armLsf")
    parser.add_argument("repo", help="Butler repository")
    parser.add_argument("collection", help="Collection with the armLsf")
    parser.add_argument("--detectors", nargs="+", default=["b2", "r2", "n2"], help="Detectors, e.g. b2 r2")
    parser.add_argument("--rows", nargs="+", type=float, default=[300, 1100, 2000, 2900, 3800],
                        help="Rows (pixels)")
    parser.add_argument("--fiberId", type=int, default=None,
                        help="Fibre (default: the central science fibre)")
    parser.add_argument("--intrinsic", nargs="+", type=float, default=[1, 2, 4],
                        help="Intrinsic rms widths (pixels) of resolved lines")
    parser.add_argument("--datasetType", default="armLsf_calib",
                        help="Dataset type of the LSF model ('armLsf' for collections written before it was "
                             "a calibration)")
    args = parser.parse_args()

    butler = Butler(args.repo, collections=args.collection)
    header = None
    for det in args.detectors:
        model = butler.get(args.datasetType, instrument="PFS", arm=det[0], spectrograph=int(det[1:]))
        fiberId = args.fiberId
        if fiberId is None:
            # The central science fibre: the engineering fibres' LSFs are narrower
            science = sorted(set(model.fiberId) - set(model.engineeringFiberId))
            fiberId = int(science[len(science)//2])
        for row in args.rows:
            widths = gaussianEquivalents(model, fiberId, row, args.intrinsic)
            if header is None:
                header = list(widths)
                print(f"{'det':3} {'fiberId':>7} {'row':>5} " + " ".join(f"{nn:>7}" for nn in header))
            print(f"{det:3} {fiberId:7d} {row:5.0f} " + " ".join(f"{widths[nn]:7.3f}" for nn in header))


if __name__ == "__main__":
    main()
