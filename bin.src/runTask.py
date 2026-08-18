#!/usr/bin/env python

"""Run a Task on local files

This provides a generic harness for running any `lsst.pipe.base.Task` (or
`PipelineTask`) from the command line, on files residing on the local
filesystem, allowing for simple debugging and testing away from the full
pipeline framework and without any butler.
"""

from __future__ import annotations

import argparse

from pfs.drp.stella.harness import runTask


def main():
    parser = argparse.ArgumentParser(description="Run a Task on local files")
    parser.add_argument("task", help="Fully-qualified dotted path to the Task to run")
    parser.add_argument(
        "data",
        nargs="*",
        help="Data to pass to the task's run method, as name:type=value (e.g., spectrum:PfsArm=file.fits)",
    )
    parser.add_argument("--config", help="Path to a config override file", default=None)
    parser.add_argument(
        "--extra",
        help=(
            "Path to a python file that may add or modify entries in the data passed to the task's run "
            "method, e.g. for values not read from a file. Executed with a 'data' dict bound in its "
            "namespace: assign into data[name] to supply an extra run keyword argument."
        ),
        default=None,
    )
    parser.add_argument(
        "--log-level",
        action="append",
        default=[],
        dest="logLevels",
        metavar="[name=]LEVEL",
        help="Set a log level; may be used multiple times. If 'name=' is omitted, sets the root level.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--pdb", action="store_true", help="Drop into pdb on exception")
    args = parser.parse_args()

    if args.debug:
        import debug  # noqa: enable LSST debugging

    if args.pdb:
        import pdb
        import sys

        def info(type, value, tb):
            if hasattr(sys, "ps1") or not sys.stderr.isatty():
                sys.__excepthook__(type, value, tb)
            else:
                import traceback

                traceback.print_exception(type, value, tb)
                print()
                pdb.pm()

        sys.excepthook = info

    runTask(
        args.task,
        args.data,
        configFile=args.config,
        logLevels=args.logLevels,
        extraFile=args.extra,
    )


if __name__ == "__main__":
    main()
