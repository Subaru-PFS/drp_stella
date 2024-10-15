#!/usr/bin/env python
from argparse import ArgumentParser

import logging
import astropy.time

from lsst.daf.butler import Timespan
from pfs.drp.stella.gen3 import certifyDetectorMaps


def main():
    """Run certifyDetectorMaps"""
    parser = ArgumentParser(description="Certify detector maps")
    parser.add_argument("repo", help="URI for datastore repository")
    parser.add_argument("collection", help="Collection containing detectorMaps to certify")
    parser.add_argument("target", help="Target collection for certified detectorMaps")
    parser.add_argument(
        "--instrument", default="PFS", help="Instrument name or fully-qualified instrument class"
    )
    parser.add_argument("--datasetType", default="detectorMap_candidate", help="Dataset type to certify")
    parser.add_argument("--begin-date", help="ISO-8601 date (TAI) this calibration should start being used.")
    parser.add_argument("--end-date", help="ISO-8601 date (TAI) this calibration should stop being used.")
    parser.add_argument(
        "--transfer",
        default="copy",
        choices=("auto", "move", "copy", "direct", "split", "link", "hardlink", "relsymlink", "symlink"),
        help="Mode for transferring files into the datastore",
    )

    args = parser.parse_args()

    timespan = Timespan(
        begin=astropy.time.Time(args.begin_date, scale="tai") if args.begin_date is not None else None,
        end=astropy.time.Time(args.end_date, scale="tai") if args.end_date is not None else None,
    )

    log = logging.getLogger("certifyDetectorMaps")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    return certifyDetectorMaps(
        args.repo,
        args.instrument,
        args.datasetType,
        args.collection,
        args.target,
        timespan,
        args.transfer,
    )


if __name__ == "__main__":
    main()
