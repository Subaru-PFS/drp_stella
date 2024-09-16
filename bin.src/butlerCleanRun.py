#!/usr/bin/env python

from argparse import ArgumentParser
from pfs.drp.stella.gen3 import cleanRun


def main():
    parser = ArgumentParser(description="Remove datasets from collections")
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("collection", help="Glob for collections to clean")
    parser.add_argument("datasetType", nargs="+", help="Dataset types to clean")

    args = parser.parse_args()
    return cleanRun(args.repo, args.collection, args.datasetType)


if __name__ == "__main__":
    main()
