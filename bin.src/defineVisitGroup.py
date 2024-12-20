#!/usr/bin/env python

from argparse import ArgumentParser

from pfs.drp.stella.gen3 import createVisitGroups


def main():
    """Command-line front-end to defineVisitGroup"""
    parser = ArgumentParser(description="Define a group of visits")
    parser.add_argument("repo", help="URI for datastore repository")
    parser.add_argument("instrument", help="Instrument name or fully-qualified instrument class")
    parser.add_argument("--where", "-d", default=None, help="WHERE clause for selecting exposures")
    parser.add_argument(
        "--update", default=False, action="store_true", help="Update an existing registry"
    )
    parser.add_argument(
        "--force", default=False, action="store_true", help="Force this selection of visits as a group?"
    )
    parser.add_argument("visits", nargs="*", type=int, help="Exposure IDs to select")

    args = parser.parse_args()
    return createVisitGroups(
        args.repo, args.instrument, args.where, args.visits, forceAll=args.force, update=args.update
    )


if __name__ == "__main__":
    main()
