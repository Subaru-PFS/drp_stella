#!/usr/bin/env python

from argparse import ArgumentParser
from pfs.drp.stella.gen3 import butlerTransfer, butlerTransferCalib, parseDataIds


def main():
    parser = ArgumentParser(
        description=(
            "Pull datasets from a remote butler repo into a local repo, "
            "for offline pipetask development. See connectRemoteButler.sh for setting up "
            "access to the remote repo."
        )
    )
    parser.add_argument("source", help="URI/config of the source (remote-view) repo")
    parser.add_argument("target", help="Path to the target (local) repo")
    parser.add_argument(
        "collections",
        nargs="+",
        help="Collections to search in the source repo (comma-separated lists are also accepted)",
    )
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
    parser.add_argument(
        "--resolve-calib",
        action="store_true",
        help=(
            "Resolve and pull just the calibration dataset of each --dataset-type that is valid "
            "(per its CALIBRATION collection certification timespan) for each --id, instead of "
            "pulling every match -- the same per-exposure calib lookup pipetask itself performs, "
            "without needing a pipeline or quantum graph. Requires --id and --dataset-type; "
            "not supported with --where or --calib."
        ),
    )
    args = parser.parse_args()
    dataIds = parseDataIds(args.id)
    collections = [cc for item in args.collections for cc in item.split(",")]

    if args.calib:
        if dataIds:
            parser.error("--id is not supported with --calib")
        if args.resolve_calib:
            parser.error("--resolve-calib is not supported with --calib")
        butlerTransferCalib(
            args.source,
            args.target,
            collections,
            datasetTypes=args.datasetTypes,
            transfer=args.transfer,
            clobber=args.clobber,
        )
    else:
        if args.resolve_calib:
            if not dataIds:
                parser.error("--resolve-calib requires --id")
            if args.where:
                parser.error("--resolve-calib is not supported with --where")
        num = butlerTransfer(
            args.source,
            args.target,
            collections,
            datasetTypes=args.datasetTypes,
            dataIds=dataIds,
            where=args.where,
            transfer=args.transfer,
            resolveCalib=args.resolve_calib,
        )
        print(f"Number of datasets pulled: {num}")


if __name__ == "__main__":
    main()
