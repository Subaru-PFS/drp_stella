#!/usr/bin/env python

from argparse import ArgumentParser

from pfs.drp.stella.lineLists.lamps import scanLampHeaders, lampMismatches


def main():
    parser = ArgumentParser(description="List the lamps lit in each visit according to the raw headers "
                                        "(getLamps), with the registry's lamps alongside; the registry is "
                                        "wrong for some visits, so select lamp exposures with this table")
    parser.add_argument("output", help="Table to write (.pkl, or .csv)")
    parser.add_argument("--repo", default="/work/datastore", help="Butler repository")
    parser.add_argument("--collection", default="PFS/raw/sps", help="Collection with the raws")
    parser.add_argument("--since", type=int, default=0, help="Only visits with day_obs >= this (YYYYMMDD)")
    parser.add_argument("--visits", type=int, nargs="+", default=None, help="Only these visits")
    parser.add_argument("--nProcesses", type=int, default=64, help="Number of processes")
    parser.add_argument("--mismatches", default=None,
                        help="Also write the visits whose registry lamps disagree with the headers (.csv)")
    args = parser.parse_args()

    scan = scanLampHeaders(args.repo, args.collection, since=args.since, visits=args.visits,
                           nProcesses=args.nProcesses, log=print)
    if args.output.endswith(".csv"):
        scan.to_csv(args.output, index=False)
    else:
        scan.to_pickle(args.output)
    mismatches = lampMismatches(scan)
    errors = scan.header.str.startswith("ERROR").sum()
    print(f"{len(scan)} visits ({errors} unreadable headers); {len(mismatches)} registry/header mismatches "
          f"-> {args.output}")
    if args.mismatches:
        mismatches.to_csv(args.mismatches, index=False)
        print("mismatches ->", args.mismatches)


if __name__ == "__main__":
    main()
