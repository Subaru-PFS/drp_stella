#!/usr/bin/env python

from argparse import ArgumentParser
from pfs.drp.stella.gen3 import butlerTransferForGraph


def main():
    parser = ArgumentParser(
        description=(
            "Pull exactly the overall input datasets of a quantum graph from a remote butler repo "
            "into a local repo, for offline pipetask development. Build the graph first with "
            "'pipetask qgraph' against the remote-view repo, then run 'pipetask run -g <graph> ...' "
            "against the local repo afterwards, reusing the same graph. "
            "See connectRemoteButler.sh for setting up access to the remote repo."
        )
    )
    parser.add_argument("source", help="URI/config of the source (remote-view) repo")
    parser.add_argument("target", help="Path to the target (local) repo")
    parser.add_argument("graph", help="Path to a quantum graph file produced by 'pipetask qgraph'")
    parser.add_argument(
        "--transfer",
        default="copy",
        help="Transfer mode to use for file artifacts (e.g., 'auto', 'copy', 'link', 'symlink', 'hardlink')",
    )
    args = parser.parse_args()

    num = butlerTransferForGraph(args.source, args.target, args.graph, transfer=args.transfer)
    print(f"Number of datasets pulled: {num}")


if __name__ == "__main__":
    main()
