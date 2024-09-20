#!/usr/bin/env python

from argparse import ArgumentParser

from pfs.drp.stella.gen3 import defineCombination


def main():
    """Command-line front-end to defineCombination"""
    parser = ArgumentParser(description="Define a combination of exposures")
    parser.add_argument("repo", help="URI for datastore repository")
    parser.add_argument("instrument", help="Instrument name or fully-qualified instrument class")
    parser.add_argument("name", help="Name of the combination")
    parser.add_argument("--where", "-d", default=None, help="WHERE clause for selecting exposures")
    parser.add_argument(
        "--update", default=False, action="store_true", help="Update an existing registry"
    )
    parser.add_argument("exposures", nargs="*", type=int, help="Exposure IDs to select")

    args = parser.parse_args()
    return defineCombination(
        args.repo, args.instrument, args.name, args.where, args.exposures, update=args.update
    )


if __name__ == "__main__":
    main()
