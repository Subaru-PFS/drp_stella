#!/usr/bin/env python3
"""Parse fitDetectorMap pipeline log output into structured tables and plots.

The script extracts statistics that are already logged at INFO level by
``FitDistortedDetectorMapTask`` and organises them into readable tables and
optional CSV files / figures.

Structured per-wavelength CSV files (``wavelengthStats_N.csv``) can also be
read by this script if ``--qa-dir`` is supplied; these are written by the task
when ``DETECTORMAP_PLOT_DIR`` is set and contain mean/RMS residuals per
reference-line wavelength that are **not** available from the log alone.

Usage examples
--------------
Parse a single log file and print tables::

    parse_detectormap_log.py task.log

Parse multiple log files, write CSVs to a directory, and produce plots::

    parse_detectormap_log.py run1.log run2.log --output-dir ./qa --plot

Read per-wavelength CSVs from the plot directory and plot the blue-bias::

    parse_detectormap_log.py task.log --qa-dir $DETECTORMAP_PLOT_DIR --plot

Pipe log output in directly::

    cat task.log | parse_detectormap_log.py -
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Regex patterns for log lines
# ---------------------------------------------------------------------------

# Common header prefix: LEVEL TIMESTAMP LOGGER (CONTEXT)(SOURCE:LINE) - MESSAGE
_LOG_PREFIX = re.compile(
    r"^(?P<level>\w+)\s+"
    r"(?P<timestamp>\S+)\s+"
    r"(?P<logger>\S+)\s+"
    r"\((?P<context>[^)]*)\)"
    r"\([^)]+\)\s+-\s+"
    r"(?P<message>.+)$"
)

# dataId extracted from context like "fitDetectorMap:{instrument: 'PFS', arm: 'b', spectrograph: 1}"
_DATAID_RE = re.compile(
    r"arm:\s*'?(?P<arm>[a-z])'?.*?spectrograph:\s*(?P<spectrograph>\d+)"
)

# Key=value pairs: chi2=X dof=X xRMS=X yRMS=X xSoften=X ySoften=X
_KV_RE = re.compile(r"(?P<key>\w+)=(?P<value>-?[\d.]+(?:e[+-]?\d+)?)")

# "from N lines ..." or "from N/M lines ..."
_FROM_RE = re.compile(r"from\s+(?P<n>\d+)(?:/(?P<denom>\d+))?\s+lines?")

# "from N fibers ..."
_FROM_FIBERS_RE = re.compile(r"from\s+(?P<n>\d+)\s+fibers?")

# Species counts like "(CdI: 2113, HgI: 4300, Trace: 52018)"
_COUNTS_RE = re.compile(r"\((?P<counts>[^)]+)\)")

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_kv(text: str) -> dict:
    """Extract all key=float_value pairs from *text*."""
    return {m.group("key"): float(m.group("value")) for m in _KV_RE.finditer(text)}


def _parse_counts(text: str) -> dict:
    """Parse species count string like 'CdI: 2113, HgI: 4300'."""
    m = _COUNTS_RE.search(text)
    if not m:
        return {}
    counts = {}
    for part in m.group("counts").split(","):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                counts[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return counts


def _extract_dataid(context: str) -> dict:
    """Return dict with arm and spectrograph from the log context field."""
    m = _DATAID_RE.search(context)
    if m:
        return {"arm": m.group("arm"), "spectrograph": int(m.group("spectrograph"))}
    return {}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


class RunStats:
    """Collected statistics for one fitDetectorMap run (one dataId)."""

    def __init__(self, arm: str, spectrograph: int):
        self.arm = arm
        self.spectrograph = spectrograph
        self.label = f"{arm}{spectrograph}"

        # fit summary rows — list of dicts with keys: stage, chi2, xRMS, etc.
        self.fit_summary: List[dict] = []

        # per-species stats: list of dicts with keys: description + stats
        self.species_stats: List[dict] = []

        # per-fiber stats: list of dicts
        self.fiber_stats: List[dict] = []

        # per-wavelength stats (from log DEBUG lines, if present)
        self.wl_stats: List[dict] = []

    def add_fit_summary(self, stage: str, kv: dict, n: int, denom: Optional[int] = None):
        row = {"stage": stage, "n": n}
        if denom is not None:
            row["n_total"] = denom
        row.update(kv)
        self.fit_summary.append(row)

    def add_species(self, description: str, kv: dict, n: int):
        row = {"description": description, "n": n}
        row.update(kv)
        self.species_stats.append(row)

    def add_fiber(self, fiber_id: int, kv: dict, n: int):
        row = {"fiberId": fiber_id, "n": n}
        row.update({k: v for k, v in kv.items() if k not in ("fiberId", "n")})
        self.fiber_stats.append(row)

    def add_wavelength(self, wavelength: float, description: str, kv: dict, n: int):
        row = {"wavelength": wavelength, "description": description, "n": n}
        row.update({k: v for k, v in kv.items() if k not in ("wavelength", "n")})
        self.wl_stats.append(row)


# ---------------------------------------------------------------------------
# Log file parser
# ---------------------------------------------------------------------------


def parse_log(lines: List[str]) -> Dict[str, RunStats]:
    """Parse log lines and return a dict keyed by arm+spectrograph label."""
    runs: Dict[str, RunStats] = {}
    current: Optional[RunStats] = None

    for raw in lines:
        line = raw.rstrip()
        m = _LOG_PREFIX.match(line)
        if not m:
            continue

        context = m.group("context")
        msg = m.group("message")

        dataid = _extract_dataid(context)
        if dataid:
            label = f"{dataid['arm']}{dataid['spectrograph']}"
            if label not in runs:
                runs[label] = RunStats(dataid["arm"], dataid["spectrograph"])
            current = runs[label]

        if current is None:
            continue

        kv = _parse_kv(msg)

        # ----- overall fit summary lines ----- (longer strings checked first)
        for stage in ("Softened fit quality from reserved lines", "Fit quality from reserved lines",
                      "Softened fit", "Final fit"):
            if msg.startswith(stage + ":"):
                fm = _FROM_FIBERS_RE.search(msg) or _FROM_RE.search(msg)
                n = int(fm.group("n")) if fm else 0
                denom = int(fm.group("denom")) if fm and "denom" in fm.groupdict() and fm.group("denom") else None
                current.add_fit_summary(stage, kv, n, denom)
                break

        if msg.startswith("Final result:"):
            fm = _FROM_RE.search(msg)
            n = int(fm.group("n")) if fm else 0
            current.add_fit_summary("Final result (measureQuality)", kv, n)

        # ----- per-species stats -----
        # "Stats for CdI: chi2=..." — description has no '=' or digit at start
        m2 = re.match(r"^Stats for ([A-Za-z][^:=\d][^:]*):", msg)
        if m2 and "fiberId" not in msg and "wavelength" not in msg:
            description = m2.group(1).strip()
            fm = _FROM_RE.search(msg)
            n = int(fm.group("n")) if fm else 0
            current.add_species(description, kv, n)

        # ----- per-fiber stats -----
        mf = re.match(r"^Stats for fiberId=(\d+):", msg)
        if mf:
            fm = _FROM_RE.search(msg)
            n = int(fm.group("n")) if fm else 0
            current.add_fiber(int(mf.group(1)), kv, n)

        # ----- per-wavelength stats (DEBUG) -----
        mw = re.match(r"^Stats for wavelength=([\d.]+)\s+\(([^)]+)\):", msg)
        if mw:
            fm = _FROM_FIBERS_RE.search(msg)
            n = int(fm.group("n")) if fm else 0
            current.add_wavelength(float(mw.group(1)), mw.group(2).strip(), kv, n)

    return runs


# ---------------------------------------------------------------------------
# QA CSV reader (from _saveQaStats)
# ---------------------------------------------------------------------------


def read_qa_csv_dir(qa_dir: str) -> Dict[str, List[dict]]:
    """Read all wavelengthStats_N.csv files from a qa directory tree.

    Returns a dict keyed by subdirectory name (arm+spec-visit label).
    """
    result: Dict[str, List[dict]] = {}
    for path in sorted(glob.glob(os.path.join(qa_dir, "**", "wavelengthStats_*.csv"), recursive=True)):
        subdir = os.path.basename(os.path.dirname(path))
        rows: List[dict] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                typed = {k: _try_numeric(v) for k, v in row.items()}
                rows.append(typed)
        if rows:
            key = subdir
            result.setdefault(key, []).extend(rows)
    return result


def _try_numeric(val: str):
    """Convert string to int or float if possible."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def _fmt(val, width=10, decimals=4):
    if isinstance(val, float):
        return f"{val:>{width}.{decimals}f}"
    return f"{str(val):>{width}}"


def print_fit_summary(run: RunStats):
    if not run.fit_summary:
        return
    print(f"\n=== {run.label}: Fit summary ===")
    cols = ["stage", "n", "chi2", "xRMS", "yRMS", "xSoften", "ySoften"]
    header = f"{'Stage':<45} {'N':>8} {'chi2':>12} {'xRMS':>8} {'yRMS':>8} {'xSoft':>8} {'ySoft':>8}"
    print(header)
    print("-" * len(header))
    for row in run.fit_summary:
        stage = row.get("stage", "")[:45]
        n = row.get("n", "")
        chi2 = row.get("chi2", float("nan"))
        xrms = row.get("xRMS", float("nan"))
        yrms = row.get("yRMS", float("nan"))
        xsof = row.get("xSoften", float("nan"))
        ysof = row.get("ySoften", float("nan"))
        print(f"{stage:<45} {n:>8} {chi2:>12.1f} {xrms:>8.4f} {yrms:>8.4f} {xsof:>8.4f} {ysof:>8.4f}")


def print_species_stats(run: RunStats):
    if not run.species_stats:
        return
    print(f"\n=== {run.label}: Per-species statistics ===")
    header = f"{'Species':<12} {'N':>8} {'chi2':>12} {'xRMS':>8} {'yRMS':>8} {'xSoft':>8} {'ySoft':>8}"
    print(header)
    print("-" * len(header))
    seen = set()
    for row in run.species_stats:
        key = (row["description"], row["n"])
        if key in seen:
            continue
        seen.add(key)
        print(f"{row['description']:<12} {row['n']:>8} "
              f"{row.get('chi2', float('nan')):>12.1f} "
              f"{row.get('xRMS', float('nan')):>8.4f} "
              f"{row.get('yRMS', float('nan')):>8.4f} "
              f"{row.get('xSoften', float('nan')):>8.4f} "
              f"{row.get('ySoften', float('nan')):>8.4f}")


def print_fiber_stats(run: RunStats):
    if not run.fiber_stats:
        return
    print(f"\n=== {run.label}: Per-fiber statistics (sampled) ===")
    header = f"{'FiberID':>8} {'N':>6} {'xRMS':>8} {'yRMS':>8} {'xSoft':>8} {'ySoft':>8}"
    print(header)
    print("-" * len(header))
    for row in run.fiber_stats:
        print(f"{int(row['fiberId']):>8} {row['n']:>6} "
              f"{row.get('xRMS', float('nan')):>8.4f} "
              f"{row.get('yRMS', float('nan')):>8.4f} "
              f"{row.get('xSoften', float('nan')):>8.4f} "
              f"{row.get('ySoften', float('nan')):>8.4f}")


def print_wavelength_stats(run: RunStats):
    if not run.wl_stats:
        return
    print(f"\n=== {run.label}: Per-wavelength statistics (DEBUG log) ===")
    header = f"{'Wavelength':>12} {'Desc':<8} {'N':>6} {'xRMS':>8} {'yRMS':>8} {'xSoft':>8}"
    print(header)
    print("-" * len(header))
    for row in sorted(run.wl_stats, key=lambda r: r["wavelength"]):
        print(f"{row['wavelength']:>12.4f} {row['description']:<8} {row['n']:>6} "
              f"{row.get('xRMS', float('nan')):>8.4f} "
              f"{row.get('yRMS', float('nan')):>8.4f} "
              f"{row.get('xSoften', float('nan')):>8.4f}")


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def write_csv(rows: List[dict], path: str):
    if not rows:
        return
    keys = list(rows[0].keys())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_runs(runs: Dict[str, RunStats], qa_data: Dict[str, List[dict]], output_dir: Optional[str]):
    try:
        import matplotlib
        matplotlib.use("Agg" if output_dir else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- x_mean vs wavelength from per-wavelength CSV data ---
    if qa_data:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for idx, (label, rows) in enumerate(qa_data.items()):
            rows_sorted = sorted(rows, key=lambda r: r.get("wavelength", 0))
            wl = [r["wavelength"] for r in rows_sorted]
            x_mean = [r.get("x_mean", float("nan")) for r in rows_sorted]
            x_rms = [r.get("x_rms", float("nan")) for r in rows_sorted]
            col = colors[idx % len(colors)]
            axes[0].plot(wl, x_mean, "o-", color=col, label=label, markersize=4)
            axes[1].plot(wl, x_rms, "o-", color=col, label=label, markersize=4)

        axes[0].axhline(0, color="k", lw=0.5, ls="--")
        axes[0].set_ylabel("Mean x residual (pix)")
        axes[0].set_title("Mean x residual vs wavelength — blue bias check")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].axhline(0, color="k", lw=0.5, ls="--")
        axes[1].set_ylabel("Robust RMS x residual (pix)")
        axes[1].set_xlabel("Wavelength (nm)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if output_dir:
            path = os.path.join(output_dir, "x_residual_vs_wavelength.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Wrote {path}")
        else:
            plt.show()
        plt.close(fig)

    # --- xRMS vs fiberId from log data ---
    fiber_data = {lbl: run.fiber_stats for lbl, run in runs.items() if run.fiber_stats}
    if fiber_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, (label, rows) in enumerate(fiber_data.items()):
            fids = [r["fiberId"] for r in rows]
            xrms = [r.get("xRMS", float("nan")) for r in rows]
            col = colors[idx % len(colors)]
            ax.plot(fids, xrms, "o", color=col, label=label, markersize=4)
        ax.set_xlabel("Fiber ID")
        ax.set_ylabel("x RMS (pix)")
        ax.set_title("x RMS vs fiber ID")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_dir:
            path = os.path.join(output_dir, "xRMS_vs_fiberId.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Wrote {path}")
        else:
            plt.show()
        plt.close(fig)

    # --- xRMS vs wavelength from DEBUG log data ---
    wl_data = {lbl: run.wl_stats for lbl, run in runs.items() if run.wl_stats}
    if wl_data:
        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, (label, rows) in enumerate(wl_data.items()):
            rows_sorted = sorted(rows, key=lambda r: r["wavelength"])
            wl = [r["wavelength"] for r in rows_sorted]
            xrms = [r.get("xRMS", float("nan")) for r in rows_sorted]
            col = colors[idx % len(colors)]
            ax.plot(wl, xrms, "o-", color=col, label=label, markersize=5)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("x RMS (pix)")
        ax.set_title("x RMS vs wavelength (from DEBUG log)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if output_dir:
            path = os.path.join(output_dir, "xRMS_vs_wavelength.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Wrote {path}")
        else:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "logfiles",
        nargs="*",
        default=["-"],
        metavar="LOGFILE",
        help="Log files to parse. Use '-' or omit for stdin.",
    )
    parser.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        help="Directory to write CSV files and plots.",
    )
    parser.add_argument(
        "--qa-dir",
        metavar="DIR",
        help="Root directory containing wavelengthStats_N.csv files written by the task "
             "(i.e. the value of DETECTORMAP_PLOT_DIR). Used for the blue-bias plot.",
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Produce diagnostic plots.",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Suppress printed tables.",
    )
    args = parser.parse_args()

    # Read log lines from files or stdin
    all_lines: List[str] = []
    for path in args.logfiles:
        if path == "-":
            all_lines.extend(sys.stdin.readlines())
        else:
            with open(path) as f:
                all_lines.extend(f.readlines())

    runs = parse_log(all_lines)
    if not runs:
        print("No fitDetectorMap log entries found.", file=sys.stderr)
        sys.exit(1)

    # Print tables
    if not args.no_tables:
        for run in runs.values():
            print_fit_summary(run)
            print_species_stats(run)
            print_fiber_stats(run)
            print_wavelength_stats(run)

    # Write CSVs
    if args.output_dir:
        for label, run in runs.items():
            if run.fit_summary:
                write_csv(run.fit_summary, os.path.join(args.output_dir, f"{label}_fit_summary.csv"))
            if run.species_stats:
                write_csv(run.species_stats, os.path.join(args.output_dir, f"{label}_species_stats.csv"))
            if run.fiber_stats:
                write_csv(run.fiber_stats, os.path.join(args.output_dir, f"{label}_fiber_stats.csv"))
            if run.wl_stats:
                write_csv(run.wl_stats, os.path.join(args.output_dir, f"{label}_wavelength_stats.csv"))

    # Read QA CSVs from plot directory
    qa_data: Dict[str, List[dict]] = {}
    if args.qa_dir:
        qa_data = read_qa_csv_dir(args.qa_dir)
        if not qa_data:
            print(f"No wavelengthStats_*.csv files found under {args.qa_dir}", file=sys.stderr)

    # Plots
    if args.plot:
        plot_runs(runs, qa_data, args.output_dir)


if __name__ == "__main__":
    main()
