#!/usr/bin/env python
"""Claude: fit the LSF of one detector with FitLsfTask, without pipetask

The same fit as ``pipetask run -p $DRP_STELLA_DIR/pipelines/lsf.yaml`` for one
detector, but reading the inputs with the butler and calling ``FitLsfTask.run``
directly, so that it can be run in a notebook and the config changed freely.
The outputs are written to files rather than to the butler.

The inputs are arc exposures reduced with ``reduceExposure``: their
``pfsArm``, the ``detectorMap`` used to extract each, and the raw header (for
the lamps). Any mixture of lamps can be used; IIS arcs (lamps ``*_eng``) are
used only for the engineering fibres.

Usage::

    python fitLsfOneDetector.py /work/datastore u/rhl/shaka-arc,PFS/raw/sps,PFS/defaults,PFS/calib \\
        r 1 148314 148317 148320 148326 --fiberStep 10 --nWorkers 8
"""
import argparse
import logging

import numpy as np
from lsst.daf.butler import Butler

from pfs.drp.stella.lsfDiagnostics import selectUsedLines
from pfs.drp.stella.pipelines.fitLsf import FitLsfTask


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="Butler repository")
    parser.add_argument("collections", help="Input collections (comma-separated), with the reduced arcs")
    parser.add_argument("arm", help="Arm (b, r, m, n)")
    parser.add_argument("spectrograph", type=int, help="Spectrograph (1-4)")
    parser.add_argument("visits", type=int, nargs="+", help="Arc visits")
    parser.add_argument("--fiberStep", type=int, default=1, help="Fit every fiberStep-th fibre")
    parser.add_argument("--nWorkers", type=int, default=1, help="Processes for the per-line fits")
    parser.add_argument("--output", default="lsf", help="Prefix for the output files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    # Read the inputs: one pfsArm, detectorMap and header per visit
    butler = Butler(args.repo, collections=args.collections.split(","))
    dataIds = [dict(instrument="PFS", visit=visit, arm=args.arm, spectrograph=args.spectrograph)
               for visit in args.visits]
    pfsArmList = [butler.get("pfsArm", dataId) for dataId in dataIds]
    detectorMapList = [butler.get("detectorMap", dataId) for dataId in dataIds]
    metadataList = [butler.get("raw.metadata", dataId) for dataId in dataIds]

    # Configure and run the task; every field of FitLsfConfig is documented in its doc string
    config = FitLsfTask.ConfigClass()
    config.fiberStep = args.fiberStep
    config.nWorkers = args.nWorkers
    task = FitLsfTask(config=config)
    result = task.run(args.arm, args.spectrograph, args.visits, pfsArmList, detectorMapList, metadataList)
    armLsf, lsfLines = result.armLsf, result.lsfLines

    # Write the outputs: armLsf can be read back with ArmLsf.readFits
    detector = f"{args.arm}{args.spectrograph}"
    armLsf.writeFits(f"{args.output}-armLsf-{detector}.fits")
    lsfLines.writeFits(f"{args.output}-lsfLines-{detector}.fits")

    # Use the model: the parameters along the central fibre, and the LSF profile of a line at its centre
    fiberId = int(armLsf.fiberId[len(armLsf.fiberId)//2])
    rows = np.linspace(200, armLsf.length - 200, 5)
    params = armLsf.getParameters(np.full(len(rows), fiberId), rows)
    print(params["row", "radius", "rms", "S", "K", "sigma"])
    profile = armLsf.computeProfile(fiberId, 0.5*armLsf.length)
    print(f"LSF profile of fibre {fiberId} at row {0.5*armLsf.length:.0f}: {np.round(profile.values, 4)}")

    df = lsfLines.toDataFrame()
    used = selectUsedLines(df)      # the lines that entered the smooth fits
    print(f"{len(used)} of {len(df)} line fits used; median chi2/dof {np.median(used.chi2/used.ndof):.2f}")


if __name__ == "__main__":
    main()
