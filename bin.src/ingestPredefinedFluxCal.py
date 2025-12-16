#!/usr/bin/env python3
import argparse
import typing

from pfs.drp.stella.gen3 import ingestPredefinedFluxCal


def main() -> None:
    """Ingest a predefined fluxCal.

    This program registers a single fluxCal many times
    with various visit numbers
    so that the pipeline will use it to calibrate all these visits.
    """
    parser = argparse.ArgumentParser(
        description=typing.cast(str, main.__doc__),
    )
    parser.add_argument(
        "fluxCal",
        help="""
        Path to a predefined fluxCal
        """,
    )
    parser.add_argument(
        "-b",
        "--butler-config",
        metavar="URI",
        required=True,
        help="""
        [Required] URI for datastore repository
        """,
    )
    parser.add_argument(
        "-d",
        "--data-query",
        metavar="SQL",
        required=True,
        help="""
        [Required] Query string to choose visits
        to which to apply the predefined fluxCal.
        """,
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="COLLECTION[,...]",
        required=True,
        help="""
        [Required] Comma-separated names of the collection(s)
        to search for pfsConfig files.
        """,
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="RUN",
        required=True,
        help="""
        [Required] Run collection to which to register the specified fluxCal.
        """,
    )
    parser.add_argument(
        "--transfer",
        default="auto",
        choices=("auto", "move", "copy", "direct", "split", "link", "hardlink", "relsymlink", "symlink"),
        help="""
        Mode for transferring files into the datastore
        """,
    )
    args = parser.parse_args()

    collections = [x.strip() for x in args.input.split(",")]
    collections = [x for x in collections if x]
    if not collections:
        raise RuntimeError("--input collection must not be empty.")

    ingestPredefinedFluxCal(
        args.butler_config,
        args.fluxCal,
        collections,
        args.data_query,
        args.output,
        transfer=args.transfer,
    )


if __name__ == "__main__":
    main()
