#!/usr/bin/env python

from argparse import ArgumentParser
from pfs.drp.stella.gen3 import certifyCalibrations


def main():
    parser = ArgumentParser()
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("inputCollection", help="Input collection with data to certify")
    parser.add_argument("outputCollection", help="Output calibration collection")
    parser.add_argument("datasetType", help="Dataset type to decertify")
    parser.add_argument("--begin-date", default=None, help="Beginning date")
    parser.add_argument("--end-date", default=None, help="Ending date")
    parser.add_argument(
        "--search-all-inputs",
        default=False,
        action="store_true",
        help=("Search all children of the inputCollection if it is a CHAINED collection, "
              "instead of just the most recent one."),
    )
    parser.add_argument("--id", nargs="*", action="append", help="KEY=VALUE pairs of dataId")

    args = parser.parse_args()

    dataIds = None
    if args.id:
        dataIds = [{key: value for key, value in [item.split("=") for item in ident]} for ident in args.id]

    return certifyCalibrations(
        args.repo,
        args.inputCollection,
        args.outputCollection,
        args.datasetType,
        args.begin_date,
        args.end_date,
        args.search_all_inputs,
        dataIds,
    )


if __name__ == "__main__":
    main()
