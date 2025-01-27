#!/usr/bin/env python

from argparse import ArgumentParser
from astropy.time import Time
from lsst.daf.butler import Timespan
from pfs.drp.stella.gen3 import findAssociations, timespanFromDayObs


def timeToString(time: Time | None) -> str:
    """Convert a Time object to a string, allowing for `None`"""
    return time.isot if time else "None"


def main():
    parser = ArgumentParser()
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("collections", help="Collections to search")
    parser.add_argument("datasetType", help="Dataset type to search")
    parser.add_argument("--id", nargs="+", help="KEY=VALUE pairs of dataId")
    parser.add_argument("--day-obs", help="Day of observation to limit query")
    parser.add_argument("--time", help="Time to limit query")
    parser.add_argument("--time-begin", help="Timespan begin to limit query")
    parser.add_argument("--time-end", help="Timespan end to limit query")
    args = parser.parse_args()

    dataId = None
    if args.id:
        dataId = {key: value for key, value in [item.split("=") for item in args.id]}

    timespan = None
    if args.day_obs:
        if timespan is not None:
            raise RuntimeError("May only specify one of --day-obs, --time, --time-begin/--time-end")
        timespan = timespanFromDayObs(int(args.day_obs))
    if args.time:
        if timespan is not None:
            raise RuntimeError("May only specify one of --day-obs, --time, --time-begin/--time-end")
        timespan = Time(args.time)
    if args.time_begin or args.time_end:
        if timespan is not None:
            raise RuntimeError("May only specify one of --day-obs, --time, --time-begin/--time-end")
        timespan = Timespan(args.time_begin, args.time_end)

    results = findAssociations(args.repo, args.collections, args.datasetType, dataId, time=timespan)
    for rr in results:
        string = f"dataId={rr.ref.dataId} run={rr.ref.run} collection={rr.collection}"
        if hasattr(rr, "timespan"):
            string += f" begin={timeToString(rr.timespan.begin)} end={timeToString(rr.timespan.end)}"
        print(string)


if __name__ == "__main__":
    main()
