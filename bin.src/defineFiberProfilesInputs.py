#!/usr/bin/env python

from argparse import ArgumentParser

from pfs.drp.stella.gen3 import defineFiberProfilesInputs
from pfs.drp.stella.utils.visits import parseIntegerList

def main():
    """Command-line front-end to defineFiberProfilesInputs"""
    parser = ArgumentParser(description="Define a fiberProfiles run")
    parser.add_argument("repo", help="URI for datastore repository")
    parser.add_argument("instrument", help="Instrument name or fully-qualified instrument class")
    parser.add_argument("name", help="Name of the run")
    parser.add_argument(
        "--bright",
        action="append",
        default=[],
        help="Bright group exposures, e.g., 123..134:2^234..245")
    parser.add_argument(
        "--dark",
        action="append",
        default=[],
        help="Dark group exposures, e.g., 123..134:2^234..245")
    parser.add_argument("--update", action="store_true", help="Allow updating existing run?")

    args = parser.parse_args()

    bright = [parseIntegerList(vv) for vv in args.bright]
    dark = [parseIntegerList(vv) for vv in args.dark]
    return defineFiberProfilesInputs(args.repo, args.instrument, args.name, bright, dark, update=args.update)


if __name__ == "__main__":
    main()
