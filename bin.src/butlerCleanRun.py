#!/usr/bin/env python

from argparse import ArgumentParser
from pfs.drp.stella.gen3 import cleanRun


def main():
    parser = ArgumentParser(description="Remove datasets from collections")
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("collection", help="Glob for collections to clean")
    parser.add_argument("datasetType", nargs="+", help="Dataset types to clean")
    parser.add_argument("--id", nargs="*", action="append", help="KEY=VALUE pairs of dataId")

    args = parser.parse_args()

    dataIds = None
    if args.id:
        dataIds = [{key: value for key, value in [item.split("=") for item in ident]} for ident in args.id]

    return cleanRun(args.repo, args.collection, args.datasetType, dataIds)


if __name__ == "__main__":
    main()
