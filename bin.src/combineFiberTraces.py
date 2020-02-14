#!/usr/bin/env python
from argparse import ArgumentParser
from pfs.drp.stella import FiberTraceSet
from lsst.log import Log

parser = ArgumentParser(description="Combine multiple FiberTraceSets")
parser.add_argument("output", help="Output filename")
parser.add_argument("input", nargs="+", help="Input filenames")
args = parser.parse_args()
log = Log()

fiberTraces = [FiberTraceSet.readFits(fn) for fn in args.input]
combined = FiberTraceSet.fromCombination(*fiberTraces)
combined.writeFits(args.output)
log.info("Wrote %d traces to %s", len(combined), args.output)
