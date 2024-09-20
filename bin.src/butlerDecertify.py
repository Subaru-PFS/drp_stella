#!/usr/bin/env python

from argparse import ArgumentParser
from pfs.drp.stella.gen3 import decertifyCalibrations


def main():
    parser = ArgumentParser()
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("collection", help="Collection to decertify")
    parser.add_argument("datasetType", help="Dataset type to decertify")
    parser.add_argument("--begin-date", default=None, help="Beginning date")
    parser.add_argument("--end-date", default=None, help="Ending date")
    parser.add_argument("--id", nargs="*", action="append", help="KEY=VALUE pairs of dataId")

    args = parser.parse_args()

    dataIds = None
    if args.id:
        dataIds = [{key: value for key, value in [item.split("=") for item in ident]} for ident in args.id]

    return decertifyCalibrations(
        args.repo, args.collection, args.datasetType, args.begin_date, args.end_date, dataIds
    )


if __name__ == "__main__":
    main()
