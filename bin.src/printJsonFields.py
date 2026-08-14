#!/usr/bin/env python3
"""
Read a file of JSON-lines (one JSON object per line) and print out a
chosen list of fields from each line, separated by a delimiter.

Usage:
    printJsonFields.py [-f FIELD [FIELD ...]] [-d DELIM] [file ...]

Examples:
    # Just print the "message" field (default) for every line in log.jsonl
    ./printJsonFields.py log.json

    # Print asctime and message, tab-separated
    ./printJsonFields.py -f asctime message log.json

    # Read from stdin
    cat log.json | ./printJsonFields.py -f levelname message
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Extract fields from JSON log files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="JSON file(s) to read. Reads stdin if omitted.",
    )
    parser.add_argument(
        "-f", "--fields",
        nargs="+",
        default=["message"],
        help="List of fields to print (default: message).",
    )
    parser.add_argument(
        "-d", "--delim",
        default="|",
        help="Delimiter to place between fields (default: pipe).",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Silently skip lines that aren't valid JSON or are missing a field, "
             "instead of printing a warning to stderr.",
    )
    args = parser.parse_args()

    # Use stdin if no files were given
    sources = args.files if args.files else ["-"]

    for source in sources:
        fd = sys.stdin if source == "-" else open(source, "r")
        try:
            for lineno, line in enumerate(fd, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if not args.skip_errors:
                        print(f"{source}:{lineno}: invalid JSON, skipping",
                              file=sys.stderr)
                    continue

                values = []
                missing = False
                for field in args.fields:
                    if field in record:
                        values.append(str(record[field]))
                    else:
                        missing = True
                        values.append("")

                if missing and not args.skip_errors:
                    print(f"{source}:{lineno}: missing field(s), "
                          f"printing what's available", file=sys.stderr)

                print(args.delim.join(values))
        finally:
            if fd is not sys.stdin:
                fd.close()


if __name__ == "__main__":
    main()
