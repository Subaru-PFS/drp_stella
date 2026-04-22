"""lsstDebug override for DetectorMap fitting diagnostics.

Setup
-----
``pipetask run --debug`` looks for a file named ``debug.py`` on
``PYTHONPATH``.  The ``python/`` subdirectory of this package is already
on ``PYTHONPATH``, so the simplest setup is a one-time symlink there
(``python/debug.py`` is listed in ``.gitignore`` so it won't be committed)::

    ln -s $DRP_STELLA_DIR/docs/debug_detectormap.py $DRP_STELLA_DIR/python/debug.py

Then run with ``--debug``:

    pipetask run --debug \
        -p "$DRP_STELLA_DIR/pipelines/detectorMap.yaml" \
        -b /path/to/repo -i input/collection -o output/collection \
        ...

Save plots to files (no display needed, safe for batch/SSH)::

    DETECTORMAP_PLOT_DIR=/my/output/plots \
    pipetask run --debug ...

    # Plots land in /my/output/plots/ as:
    #   baseResiduals.png   -- quiver + fiberId/wavelength heatmap before fit
    #   slitOffsets.png     -- per-fiber spatial and spectral slit offsets
    #   lineQa.png          -- centroid error vs fit RMS scatter
    #   model.png           -- 2x3 panel: residuals, positions, offset heatmaps
    #   distortion.png      -- x/y distortion field (model vs observed)
    #   residuals[_0-2].png -- summary + heatmap (+ trace) residual figures
    #   wlResiduals.png     -- wavelength residuals vs row for sampled fibers

Environment variables
---------------------
DETECTORMAP_PLOT_DIR : str, optional
    Directory to write PNG files into.  If unset, plots are shown
    interactively via ``plt.show()``.

DETECTORMAP_DEBUG_FLAGS : str, optional
    Comma-separated list of flags to enable.  Defaults to all flags:
    ``baseResiduals,slitOffsets,lineQa,plot,distortion,finalResiduals,wlResid``
"""

import os
import lsstDebug

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

_PLOT_DIR = os.environ.get("DETECTORMAP_PLOT_DIR", "")

_DEFAULT_FLAGS = {
    "baseResiduals",   # quiver + heatmap of residuals w.r.t. base detectorMap
    "slitOffsets",     # per-fiber spatial/spectral slit offset values
    "lineQa",          # centroid error vs fit RMS
    "plot",            # 2x3 model plot (residuals + position + heatmap)
    "distortion",      # distortion field
    "finalResiduals",  # residuals + heatmap after the final fit iteration
    "wlResid",         # wavelength residuals vs row
}

_raw = os.environ.get("DETECTORMAP_DEBUG_FLAGS", "")
_ENABLED_FLAGS = set(_raw.split(",")) if _raw else _DEFAULT_FLAGS

# ---------------------------------------------------------------------------
# lsstDebug override
# ---------------------------------------------------------------------------

_TARGET_MODULE = "pfs.drp.stella.fitDistortedDetectorMap"


def DebugInfo(name):
    """Return an lsstDebug Info object, with debug flags set for the
    DetectorMap fitting task."""
    # getInfo == the original lsstDebug.Info; calling it here instead of
    # DebugInfo avoids infinite recursion.
    di = lsstDebug.getInfo(name)

    if name == _TARGET_MODULE:
        for flag in _ENABLED_FLAGS:
            setattr(di, flag, True)

        # plotDir: a truthy string causes _showOrSavePlot to save files
        # instead of calling plt.show().  An empty string / False leaves
        # behaviour interactive.
        if _PLOT_DIR:
            di.plotDir = _PLOT_DIR

        # Optional layout tweaks for the wavelength-residuals grid.
        # di.wlResidRows = 3
        # di.wlResidCols = 2

    return di


lsstDebug.Info = DebugInfo
