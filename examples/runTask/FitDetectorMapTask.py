"""runTask.py template for FitDetectorMapTask

Copy this file, review and edit the values in the "Edit these" block, and run:

    runTask.py pfs.drp.stella.fitDetectorMap.FitDetectorMapTask \\
        lines:ArcLineSet=detectorMapLines.fits [base:DetectorMap=base.fits] \\
        --extra FitDetectorMapTask.py --output detectorMap:detectorMap.fits \\
        [--config overrides.py] [--log-level fitDetectorMap=DEBUG]

This supplies the arguments to ``FitDetectorMapTask.run`` that can't be read
from a file: the data identifier, the detector bounding box and the visit
information. If no ``base`` detectorMap is provided on the command line, the
task reads the one pointed to in its configuration.

See README.md in this directory for how to use runTask.py and its templates.
"""

from lsst.afw.image import VisitInfo
from lsst.geom import Box2I, Extent2I, Point2I

from pfs.drp.stella.referenceLine import ReferenceLineStatus

# Declarations, used by the unit tests to check that this template is up to date.
TASK = "pfs.drp.stella.fitDetectorMap.FitDetectorMapTask"
INPUTS = {"lines": "ArcLineSet", "base": "DetectorMap"}  # name:type=filename on the command line

# ---- Edit these ----
ARM = "b"  # Spectrograph arm: b, r, n, m
SPECTROGRAPH = 1  # Spectrograph module: 1-4
VISIT = 12345  # Visit number, used only for logging and the output metadata
DIMENSIONS = (4096, 4176)  # Detector dimensions (pixels); ignored if a base detectorMap is provided
RESET_STATUS = True  # Clear the line status flags set by a previous fitDetectorMap run?
DROP_TRACES = False  # Drop trace measurements (e.g., if there are too many for your machine)?
# --------------------

base = data.get("base")
data["dataId"] = dict(visit=VISIT, arm=ARM, spectrograph=SPECTROGRAPH)
data["bbox"] = base.bbox if base is not None else Box2I(Point2I(0, 0), Extent2I(*DIMENSIONS))
data["visitInfo"] = base.visitInfo if base is not None and base.visitInfo is not None else VisitInfo()

lines = data["lines"]
if RESET_STATUS:
    lines.status[:] &= ~(ReferenceLineStatus.DETECTORMAP_USED | ReferenceLineStatus.DETECTORMAP_RESERVED)
if DROP_TRACES:
    lines = lines[lines.description != "Trace"]
data["lines"] = lines
