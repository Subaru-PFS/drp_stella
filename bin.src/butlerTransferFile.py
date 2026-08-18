#!/usr/bin/env python

from argparse import ArgumentParser
from pfs.drp.stella.gen3 import transferButlerFiles


def main():
    parser = ArgumentParser(description="Transfer dataset files from a butler repo to a local directory")
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("collections", help="Collections to search")
    parser.add_argument(
        "items",
        nargs="+",
        help=(
            "Dataset type names and KEY=VALUE dataId pairs, e.g.: "
            "pfsArm detectorMap visit=12345 arm=r spectrograph=1"
        ),
    )
    parser.add_argument("--directory", default=".", help="Directory to transfer files to")
    parser.add_argument(
        "--transfer",
        default="symlink",
        help=(
            "Transfer mode to use, as accepted by Butler.retrieveArtifacts "
            "(e.g., 'auto', 'copy', 'link', 'symlink', 'hardlink', 'relsymlink')"
        ),
    )
    args = parser.parse_args()

    datasetTypes = [item for item in args.items if "=" not in item]
    dataId = {key: value for key, value in (item.split("=", 1) for item in args.items if "=" in item)}
    if not datasetTypes:
        parser.error("At least one dataset type must be specified")

    transferButlerFiles(
        args.repo,
        args.collections,
        datasetTypes,
        dataId=dataId,
        directory=args.directory,
        transfer=args.transfer,
    )


if __name__ == "__main__":
    main()
