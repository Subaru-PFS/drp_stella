#!/usr/bin/env python

"""Adjust a detectorMap

This provides a simple harness for the `AdjustDetectorMapTask` to be run from
the command line, allowing for simple debugging and testing away from the full
pipeline framework.
"""

from __future__ import annotations

import argparse
import logging

from lsst.afw.image import VisitInfo

from pfs.drp.stella.arcLine import ArcLineSet
from pfs.drp.stella.DetectorMapContinued import DetectorMap
from pfs.drp.stella.adjustDetectorMap import AdjustDetectorMapTask


def adjustDetectorMap(
    detectorMap: DetectorMap,
    lines: ArcLineSet,
    arm: str,
    configFile: str | None = None,
) -> DetectorMap:
    """Adjust a detector map using arc lines

    Parameters
    ----------
    detectorMap : `DetectorMap`
        The detector map to adjust.
    lines : `ArcLineSet`
        The arc lines to use for the adjustment.
    arm : `str`
        The arm to adjust.
    configFile : `str`, optional
        Path to a configuration file. If provided, it will override the default
        configuration.

    Returns
    -------
    adjustedDetectorMap : `DetectorMap`
        The adjusted detector map.
    """
    logger = logging.getLogger()
    logger.getChild("adjustDetectorMap").setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    config = AdjustDetectorMapTask.ConfigClass()
    if configFile:
        config.load(configFile)

    task = AdjustDetectorMapTask(config=config, log=logger)
    return task.run(
        detectorMap=detectorMap,
        lines=lines,
        arm=arm,
        visitInfo=VisitInfo(),
    )


def main():
    parser = argparse.ArgumentParser(description="Adjust a detector map")
    parser.add_argument("detectorMap", help="Path to the detector map FITS file")
    parser.add_argument("lines", help="Path to the arc lines FITS file")
    parser.add_argument("arm", help="The arm to adjust")
    parser.add_argument("--config", help="Path to the config file", default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output", help="Path to the output FITS file", default=None)
    parser.add_argument("--pdb", action="store_true", help="Drop into pdb on exception")
    args = parser.parse_args()

    if args.debug:
        import debug  # noqa: enable LSST debugging

    detectorMap = DetectorMap.readFits(args.detectorMap)
    lines = ArcLineSet.readFits(args.lines)

    if args.pdb:
        import pdb
        import sys

        def info(type, value, tb):
            if hasattr(sys, "ps1") or not sys.stderr.isatty():
                sys.__excepthook__(type, value, tb)
            else:
                import traceback

                traceback.print_exception(type, value, tb)
                print()
                pdb.pm()

        sys.excepthook = info

    adjusted = adjustDetectorMap(
        detectorMap=detectorMap,
        lines=lines,
        arm=args.arm,
        configFile=args.config,
    )

    if args.output:
        adjusted.writeFits(args.output)


if __name__ == "__main__":
    main()
