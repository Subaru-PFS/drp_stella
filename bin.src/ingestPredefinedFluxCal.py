#!/usr/bin/env python

from argparse import ArgumentParser
import typing

from pfs.drp.stella.gen3 import ingestPredefinedFluxCal


def main() -> None:
    """Ingest a predefined fluxCal file into a gen3 datastore."""
    parser = ArgumentParser(description=typing.cast(str, main.__doc__))
    parser.add_argument(
        "repo",
        help="URI for datastore repository",
    )
    parser.add_argument(
        "run",
        help="Run into which to ingest files",
    )
    parser.add_argument(
        "path",
        metavar="fluxcal",
        help="Path to a fluxCal FITS file.",
    )
    parser.add_argument(
        "--instrument",
        default="PFS",
        help="Instrument name or fully-qualified instrument class (default: 'PFS')",
    )
    parser.add_argument(
        "--transfer",
        default="auto",
        choices=("auto", "move", "copy", "direct", "split", "link", "hardlink", "relsymlink", "symlink"),
        help="Mode for transferring files into the datastore",
    )

    args = parser.parse_args()
    ingestPredefinedFluxCal(args.repo, args.instrument, args.run, args.path, args.transfer)


if __name__ == "__main__":
    main()
