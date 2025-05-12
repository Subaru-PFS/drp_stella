#!/usr/bin/env python

from argparse import ArgumentParser

from pfs.drp.stella.gen3 import defineCombination


def main():
    """Command-line front-end to defineCombination"""
    parser = ArgumentParser(description="Define a combination of visits")
    parser.add_argument("repo", help="URI for datastore repository")
    parser.add_argument("instrument", help="Instrument name or fully-qualified instrument class")
    parser.add_argument("name", help="Name of the combination")
    parser.add_argument("--collections", default="PFS/raw/pfsConfig", help="Collections with pfsConfig")
    parser.add_argument("--where", "-d", default=None, help="WHERE clause for selecting visits")
    parser.add_argument(
        "--max-group-size", type=int, default=2000, help="Maximum number of objects in a group"
    )
    parser.add_argument(
        "--update", default=False, action="store_true", help="Update an existing registry"
    )
    parser.add_argument("visits", nargs="*", type=int, help="Visit IDs to select")

    args = parser.parse_args()
    return defineCombination(
        args.repo,
        args.instrument,
        args.name,
        collections=args.collections,
        where=args.where,
        visitList=args.visits,
        maxGroupSize=args.max_group_size,
        update=args.update,
    )


if __name__ == "__main__":
    main()
