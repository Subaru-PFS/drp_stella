#!/usr/bin/env python

from argparse import ArgumentParser
import os
import typing

from lsst.daf.butler import Butler, DatasetRef, DatasetType, DimensionGroup, FileDataset
from lsst.pipe.base import Instrument
from lsst.resources import ResourcePath

from pfs.drp.stella.utils.logging import getLogger


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


def ingestPredefinedFluxCal(
    repo: str,
    instrument: str,
    run: str,
    path: str,
    transfer: str | None = "auto",
) -> None:
    """Ingest a predefined fluxCal into the datastore.

    Parameters
    ----------
    repo : `str`
        URI string of the Butler repo to use.
    instrument : `str`
        Instrument name or fully-qualified class name as a string.
    run : `str`
        The run in which the files should be ingested.
    path : `str`
        Path to the fluxCal FITS file to ingest.
    transfer : `str`, optional
        Transfer mode to use for ingest. If not `None`, must be one of 'auto',
        'move', 'copy', 'direct', 'split', 'hardlink', 'relsymlink' or
        'symlink'.
    """
    log = getLogger("pfs.ingestPredefinedFluxCal")

    if not os.path.isfile(path):
        raise RuntimeError(f"Not a file: '{path}'")

    butler = Butler.from_config(repo, run=run)

    datasetType = DatasetType(
        name="fluxCal_predefined",
        dimensions=DimensionGroup(universe=butler.registry.dimensions, names=["instrument"]),
        storageClass="FocalPlaneFunction",
    )
    dataId = {"instrument": Instrument.from_string(instrument, butler.registry).getName()}

    dataset = FileDataset(
        path=ResourcePath(path, root=os.getcwd(), forceAbsolute=True),
        refs=[DatasetRef(datasetType, dataId, run)],
    )

    log.info("Ingesting %s...", path)

    butler.registry.registerDatasetType(datasetType)
    butler.ingest(dataset, transfer=transfer)


if __name__ == "__main__":
    main()
