#!/usr/bin/env python

from argparse import ArgumentParser
from pfs.drp.stella.gen3 import ingestPfsConfig


def main():
    """Run ingestPfsConfig"""
    parser = ArgumentParser(description="Ingest a pfsConfig file into a gen3 datastore")
    parser.add_argument("repo", help="URI for datastore repository")
    parser.add_argument("instrument", help="Instrument name or fully-qualified instrument class")
    parser.add_argument("run", help="Run into which to ingest files")
    parser.add_argument("path", nargs="+", help="Path/glob for pfsConfig files")
    parser.add_argument(
        "--transfer",
        default="auto",
        choices=("auto", "move", "copy", "direct", "split", "link", "hardlink", "relsymlink", "symlink"),
        help="Mode for transferring files into the datastore",
    )
    parser.add_argument(
        "--no-update", default=False, action="store_true", help="Don't allow updating of the registry"
    )

    args = parser.parse_args()
    ingestPfsConfig(args.repo, args.instrument, args.run, args.path, args.transfer, not args.no_update)


if __name__ == "__main__":
    main()
