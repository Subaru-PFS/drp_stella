#!/usr/bin/env python

from argparse import ArgumentParser
from pfs.drp.stella.gen3 import parseDataIds, pullButlerCache, pullButlerCalibCache


def main():
    parser = ArgumentParser(
        description=(
            "Pull datasets from a remote butler repo into a local cache repo, "
            "for offline pipetask development. See connectRemoteButler.sh for setting up "
            "access to the remote repo."
        )
    )
    parser.add_argument("source", help="URI/config of the source (remote-view) repo")
    parser.add_argument("target", help="Path to the local cache repo")
    parser.add_argument("collections", nargs="+", help="Collections to search in the source repo")
    parser.add_argument(
        "--dataset-type",
        dest="datasetTypes",
        action="append",
        default=[],
        help="Dataset type to pull (may be repeated; default: all)",
    )
    parser.add_argument(
        "--id",
        nargs="*",
        action="append",
        help="KEY=VALUE pairs of dataId selecting datasets to pull (may be repeated for multiple dataIds)",
    )
    parser.add_argument("--where", default="", help="SQL WHERE clause to restrict the pull")
    parser.add_argument(
        "--transfer",
        default="copy",
        help="Transfer mode to use for file artifacts (e.g., 'auto', 'copy', 'link', 'symlink', 'hardlink')",
    )
    parser.add_argument(
        "--calib",
        action="store_true",
        help="Treat 'collections' as CALIBRATION collections, preserving certification timespans",
    )
    parser.add_argument(
        "--clobber",
        action="store_true",
        help="Clobber datasets that already exist in the target repo (--calib only)",
    )
    args = parser.parse_args()
    dataIds = parseDataIds(args.id)

    if args.calib:
        if dataIds:
            parser.error("--id is not supported with --calib")
        pullButlerCalibCache(
            args.source,
            args.target,
            args.collections,
            datasetTypes=args.datasetTypes,
            transfer=args.transfer,
            clobber=args.clobber,
        )
    else:
        num = pullButlerCache(
            args.source,
            args.target,
            args.collections,
            datasetTypes=args.datasetTypes,
            dataIds=dataIds,
            where=args.where,
            transfer=args.transfer,
        )
        print(f"Number of datasets pulled: {num}")


if __name__ == "__main__":
    main()
