"""runTask.py template for FitDetectorMapCombinedTask

Copy this file, review and edit the values in the "Edit these" block, and run:

    runTask.py pfs.drp.stella.pipelines.fitDetectorMapCombined.FitDetectorMapCombinedTask \\
        arcLines:ArcLineSet=lines1.fits \\
        --extra FitDetectorMapCombinedTask.py --output detectorMap:detectorMap.fits \\
        [--config overrides.py] [--log-level fitDetectorMapCombined=DEBUG]

``FitDetectorMapCombinedTask.run`` takes a list of line measurements (one per
exposure, arcs and twilights), which the command line can't express: the file
on the command line is the first, and more can be listed in ``MORE_LINES``.
This also supplies the data identifier, the visit information, the metadata
and the detector bounding box.

See README.md in this directory for how to use runTask.py and its templates.
"""

from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertyList
from lsst.geom import Box2I, Extent2I, Point2I

from pfs.drp.stella.harness import readData

# Declarations, used by the unit tests to check that this template is up to date.
TASK = "pfs.drp.stella.pipelines.fitDetectorMapCombined.FitDetectorMapCombinedTask"
INPUTS = {"arcLines": "ArcLineSet"}  # name:type=filename on the command line

# ---- Edit these ----
ARM = "b"  # Spectrograph arm: b, r, n, m
SPECTROGRAPH = 1  # Spectrograph module: 1-4
VISIT = 12345  # Visit number, used only for logging and the output metadata
DIMENSIONS = (4096, 4176)  # Detector dimensions (pixels)
MORE_LINES = []  # Filenames of additional line measurements (ArcLineSet), e.g., from other exposures
# --------------------

data["arcLines"] = [data["arcLines"]] + [readData("ArcLineSet", filename) for filename in MORE_LINES]
data["dataId"] = dict(visit=VISIT, arm=ARM, spectrograph=SPECTROGRAPH)
data["visitInfo"] = VisitInfo(id=VISIT)
data["metadata"] = PropertyList()
data["bbox"] = Box2I(Point2I(0, 0), Extent2I(*DIMENSIONS))
