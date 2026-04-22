#!/usr/bin/env python3
"""Parse fitDetectorMap pipeline log output into structured tables and plots.

The script extracts statistics logged at INFO level by
``FitDistortedDetectorMapTask`` and organises them into readable tables and
an aggregated CSV file.

Per-wavelength QA statistics are emitted as ``QA_WL`` lines at INFO level on
every run (no special environment variables required).  Use ``--output-dir``
to write a single ``wavelength_qa.csv`` that aggregates all detectors and
visits from any number of log files.

Usage examples
--------------
Parse a single log file and print tables::

    parse_detectormap_log.py task.log

Parse multiple log files and write CSVs to a directory::

    parse_detectormap_log.py b1.log b2.log b3.log b4.log --output-dir ./qa

Parse and plot the blue-bias (x_mean vs wavelength)::

    parse_detectormap_log.py *.log --output-dir ./qa --plot

Pipe log output in directly::

    cat task.log | parse_detectormap_log.py -
"""

import argparse
import csv
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

# dataId extracted from context like "{instrument: 'PFS', arm: 'b', spectrograph: 1, visit: 121319}"
_DATAID_RE = re.compile(
    r"arm:\s*'?(?P<arm>[a-z])'?.*?spectrograph:\s*(?P<spectrograph>\d+)"
    r"(?:.*?visit:\s*(?P<visit>\d+))?"
)

# Key=value pairs: chi2=X dof=X xRMS=X yRMS=X — also matches nan/inf
_KV_RE = re.compile(r"(?P<key>\w+)=(?P<value>-?(?:[\d.]+(?:e[+-]?\d+)?|nan|inf))")

# "from N lines ..." or "from N/M lines ..."
_FROM_RE = re.compile(r"from\s+(?P<n>\d+)(?:/(?P<denom>\d+))?\s+lines?")

# "from N fibers ..."
_FROM_FIBERS_RE = re.compile(r"from\s+(?P<n>\d+)\s+fibers?")

# "Species counts like "(CdI: 2113, HgI: 4300, Trace: 52018)"
_COUNTS_RE = re.compile(r"\((?P<counts>[^)]+)\)")

# Structured per-wavelength QA line emitted by _logQaStats
_QA_WL_RE = re.compile(
    r"^QA_WL\s+wl=(?P<wl>[\d.]+)\s+desc=(?P<desc>\S+)\s+"
    r"n_good=(?P<n_good>\d+)\s+n_used=(?P<n_used>\d+)\s+n_reserved=(?P<n_reserved>\d+)\s+"
    r"x_mean=(?P<x_mean>-?(?:[\d.]+(?:e[+-]?\d+)?|nan))\s+"
    r"x_rms=(?P<x_rms>(?:[\d.]+(?:e[+-]?\d+)?|nan))\s+"
    r"y_mean=(?P<y_mean>-?(?:[\d.]+(?:e[+-]?\d+)?|nan))\s+"
    r"y_rms=(?P<y_rms>(?:[\d.]+(?:e[+-]?\d+)?|nan))"
)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_kv(text: str) -> dict:
    """Extract all key=float_value pairs from *text* (handles nan/inf)."""
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
    """Return dict with arm, spectrograph, and visit from the log context field."""
    m = _DATAID_RE.search(context)
    if m:
        result = {"arm": m.group("arm"), "spectrograph": int(m.group("spectrograph"))}
        if m.group("visit") is not None:
            result["visit"] = int(m.group("visit"))
        return result
    return {}


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
# Data containers
# ---------------------------------------------------------------------------


class RunStats:
    """Collected statistics for one fitDetectorMap run (one dataId)."""

    def __init__(self, arm: str, spectrograph: int, visit: int = 0):
        self.arm = arm
        self.spectrograph = spectrograph
        self.visit = visit
        self.label = f"{arm}{spectrograph}-v{visit}" if visit else f"{arm}{spectrograph}"

        self.fit_summary: List[dict] = []
        self.species_stats: List[dict] = []
        self.fiber_stats: List[dict] = []
        self.wl_stats: List[dict] = []  # from legacy DEBUG log lines
        self.wl_qa: List[dict] = []     # from QA_WL INFO lines

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

    def add_wl_qa(self, wl: float, desc: str, n_good: int, n_used: int, n_reserved: int,
                  x_mean: float, x_rms: float, y_mean: float, y_rms: float):
        self.wl_qa.append({
            "arm": self.arm,
            "spectrograph": self.spectrograph,
            "visit": self.visit,
            "wavelength": wl,
            "description": desc,
            "n_good": n_good,
            "n_used": n_used,
            "n_reserved": n_reserved,
            "x_mean": x_mean,
            "x_rms": x_rms,
            "y_mean": y_mean,
            "y_rms": y_rms,
        })


# ---------------------------------------------------------------------------
# Log file parser
# ---------------------------------------------------------------------------


def parse_log(lines: List[str]) -> Dict[str, RunStats]:
    """Parse log lines and return a dict keyed by arm+spec+visit label."""
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
            arm = dataid["arm"]
            spec = dataid["spectrograph"]
            visit = dataid.get("visit", 0)
            label = f"{arm}{spec}-v{visit}" if visit else f"{arm}{spec}"
            if label not in runs:
                runs[label] = RunStats(arm, spec, visit)
            current = runs[label]

        if current is None:
            continue

        kv = _parse_kv(msg)

        # ----- QA_WL structured line -----
        mqa = _QA_WL_RE.match(msg)
        if mqa:
            current.add_wl_qa(
                float(mqa.group("wl")), mqa.group("desc"),
                int(mqa.group("n_good")), int(mqa.group("n_used")), int(mqa.group("n_reserved")),
                float(mqa.group("x_mean")), float(mqa.group("x_rms")),
                float(mqa.group("y_mean")), float(mqa.group("y_rms")),
            )
            continue

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

        # ----- per-wavelength stats (legacy DEBUG) -----
        mw = re.match(r"^Stats for wavelength=([\d.]+)\s+\(([^)]+)\):", msg)
        if mw:
            fm = _FROM_FIBERS_RE.search(msg)
            n = int(fm.group("n")) if fm else 0
            current.add_wavelength(float(mw.group(1)), mw.group(2).strip(), kv, n)

    return runs


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


def print_wavelength_qa(run: RunStats):
    rows = run.wl_qa or run.wl_stats
    if not rows:
        return
    print(f"\n=== {run.label}: Per-wavelength QA ===")
    header = f"{'Wavelength':>12} {'Desc':<8} {'N_good':>7} {'N_used':>7} {'x_mean':>8} {'x_rms':>8} {'y_mean':>8} {'y_rms':>8}"
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: r.get("wavelength", 0)):
        ng = row.get("n_good", row.get("n", ""))
        nu = row.get("n_used", "")
        print(f"{row.get('wavelength', 0):>12.4f} {row.get('description', ''):8} {ng!s:>7} {nu!s:>7} "
              f"{row.get('x_mean', float('nan')):>8.4f} "
              f"{row.get('x_rms', float('nan')):>8.4f} "
              f"{row.get('y_mean', float('nan')):>8.4f} "
              f"{row.get('y_rms', float('nan')):>8.4f}")


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


def write_aggregated_wl_csv(runs: Dict[str, "RunStats"], path: str):
    """Write a single wavelength_qa.csv aggregating all runs."""
    all_rows = []
    for run in runs.values():
        all_rows.extend(run.wl_qa)
    write_csv(all_rows, path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_runs(runs: Dict[str, "RunStats"], output_dir: Optional[str]):
    try:
        import matplotlib
        matplotlib.use("Agg" if output_dir else "TkAgg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- x_mean and y_mean vs wavelength from QA_WL log lines ---
    wl_data = {lbl: run.wl_qa for lbl, run in runs.items() if run.wl_qa}
    if not wl_data:
        # Fall back to legacy DEBUG stats if available
        wl_data = {lbl: run.wl_stats for lbl, run in runs.items() if run.wl_stats}

    if wl_data:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for idx, (label, rows) in enumerate(wl_data.items()):
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

    if not args.no_tables:
        for run in runs.values():
            print_fit_summary(run)
            print_species_stats(run)
            print_fiber_stats(run)
            print_wavelength_qa(run)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # Aggregated wavelength QA CSV (all detectors/visits in one file)
        wl_rows = [row for run in runs.values() for row in run.wl_qa]
        if wl_rows:
            write_csv(wl_rows, os.path.join(args.output_dir, "wavelength_qa.csv"))
        # Per-label fit summary and species CSVs
        for label, run in runs.items():
            if run.fit_summary:
                write_csv(run.fit_summary, os.path.join(args.output_dir, f"{label}_fit_summary.csv"))
            if run.species_stats:
                write_csv(run.species_stats, os.path.join(args.output_dir, f"{label}_species_stats.csv"))

    if args.plot:
        plot_runs(runs, args.output_dir)


if __name__ == "__main__":
    main()
