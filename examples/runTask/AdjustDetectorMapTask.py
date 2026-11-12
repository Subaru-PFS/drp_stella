"""runTask.py template for AdjustDetectorMapTask

Copy this file, review and edit the values in the "Edit these" block, and run:

    runTask.py pfs.drp.stella.adjustDetectorMap.AdjustDetectorMapTask \\
        detectorMap:DetectorMap=detectorMap.fits lines:ArcLineSet=lines.fits \\
        --extra AdjustDetectorMapTask.py --output detectorMap:adjusted.fits \\
        [--config overrides.py] [--log-level adjustDetectorMap=DEBUG]

This supplies the arguments to ``AdjustDetectorMapTask.run`` that can't be
read from a file: the arm and the visit information.

See README.md in this directory for how to use runTask.py and its templates.
"""

from lsst.afw.image import VisitInfo

from pfs.drp.stella.referenceLine import ReferenceLineStatus

# Declarations, used by the unit tests to check that this template is up to date.
TASK = "pfs.drp.stella.adjustDetectorMap.AdjustDetectorMapTask"
INPUTS = {"detectorMap": "DetectorMap", "lines": "ArcLineSet"}  # name:type=filename on the command line

# ---- Edit these ----
ARM = "b"  # Spectrograph arm: b, r, n, m
SEED = 0  # Seed for the random selection of reserved lines (the pipeline uses the visit number)
RESET_STATUS = True  # Clear the line status flags set by a previous fit or adjustment?
DROP_TRACES = False  # Drop trace measurements (e.g., if there are too many for your machine)?
# --------------------

detectorMap = data["detectorMap"]
data["arm"] = ARM
data["visitInfo"] = detectorMap.visitInfo if detectorMap.visitInfo is not None else VisitInfo()
data["seed"] = SEED

lines = data["lines"]
if RESET_STATUS:
    lines.status[:] &= ~(ReferenceLineStatus.DETECTORMAP_USED | ReferenceLineStatus.DETECTORMAP_RESERVED)
if DROP_TRACES:
    lines = lines[lines.description != "Trace"]
data["lines"] = lines
