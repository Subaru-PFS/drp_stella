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

    args = parser.parse_args()
    return decertifyCalibrations(
        args.repo, args.collection, args.datasetType, args.begin_date, args.end_date
    )


if __name__ == "__main__":
    main()
