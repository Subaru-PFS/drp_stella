#!/usr/bin/env bash
# Wrapper around `pipetask run --debug` that injects a temporary debug.py
# into PYTHONPATH, enabling lsstDebug diagnostic plots from
# FitDistortedDetectorMapTask without committing a debug.py to the repository.
#
# Usage
# -----
#   run_fitDetectorMap_debug.sh [pipetask run options...]
#
# Examples
# --------
# Interactive plots (requires a display):
#
#   run_fitDetectorMap_debug.sh \
#       -p "$DRP_STELLA_DIR/pipelines/detectorMap.yaml" \
#       -b /path/to/repo -i input/collection -o output/collection \
#       --instrument lsst.obs.pfs.PfsSimulator \
#       -d "instrument=PFS AND arm='b' AND spectrograph=1 AND visit=12345"
#
# Save plots to files (no display needed, safe over SSH / in batch):
#
#   DETECTORMAP_PLOT_DIR=/my/plots \
#   run_fitDetectorMap_debug.sh ...same args...
#
# Environment variables
# ---------------------
# DETECTORMAP_PLOT_DIR
#   Directory to write PNG files into.  If unset, plots are shown
#   interactively via plt.show().
#
# DETECTORMAP_DEBUG_FLAGS
#   Comma-separated list of flags to enable.  Defaults to all flags:
#   baseResiduals,slitOffsets,lineQa,plot,distortion,finalResiduals,wlResid

set -euo pipefail

if [ -z "${DRP_STELLA_DIR:-}" ]; then
    echo "ERROR: DRP_STELLA_DIR is not set. Source the LSST stack and setup drp_stella first." >&2
    exit 1
fi

_IMPL="$DRP_STELLA_DIR/bin.src/debug_detectormap.py"
if [ ! -f "$_IMPL" ]; then
    echo "ERROR: Expected implementation file not found: $_IMPL" >&2
    exit 1
fi

# Create a temporary directory and register cleanup on any exit.
_TMPDIR=$(mktemp -d)
trap 'rm -rf "$_TMPDIR"' EXIT

# Write a minimal debug.py that delegates to the real implementation file.
# Using importlib preserves the module's global scope correctly so that the
# lsstDebug.Info = DebugInfo assignment takes effect in the right namespace.
cat > "$_TMPDIR/debug.py" << PYEOF
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location("_debug_detectormap",
            _os.path.join(_os.environ["DRP_STELLA_DIR"], "bin.src", "debug_detectormap.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PYEOF

# Prepend the temp dir so our debug.py is found before anything else.
export PYTHONPATH="$_TMPDIR${PYTHONPATH:+:$PYTHONPATH}"

exec pipetask run --debug "$@"
