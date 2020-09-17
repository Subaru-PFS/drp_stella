#!/usr/bin/env python
from argparse import ArgumentParser
from pfs.drp.stella import FiberProfileSet
from lsst.log import Log

parser = ArgumentParser(description="Combine multiple FiberProfileSets")
parser.add_argument("output", help="Output filename")
parser.add_argument("input", nargs="+", help="Input filenames")
args = parser.parse_args()
log = Log()

profiles = [FiberProfileSet.readFits(fn) for fn in args.input]
combined = FiberProfileSet.fromCombination(*profiles)
combined.writeFits(args.output)
log.info("Wrote %d traces to %s", len(combined), args.output)
