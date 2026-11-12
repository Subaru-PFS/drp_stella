"""runTask.py template for CentroidSolarTask

Copy this file, review and edit the values in the "Edit these" block, and run:

    runTask.py pfs.drp.stella.centroidSolar.CentroidSolarTask \\
        pfsArm:PfsArm=pfsArm.fits pfsConfig:PfsConfig=pfsConfig.fits \\
        detectorMap:DetectorMap=detectorMap.fits \\
        --extra CentroidSolarTask.py --output lines:solarLines.fits \\
        [--config overrides.py] [--log-level centroidSolar=DEBUG] [--debug]

This supplies the arguments to ``CentroidSolarTask.run`` that can't be read
from a file: the line-spread function, and the visit information (for the
telescope altitude, which sets the atmospheric transmission).

See README.md in this directory for how to use runTask.py and its templates.
"""

from lsst.afw.image import VisitInfo
from lsst.geom import SpherePoint, degrees

from pfs.drp.stella.centroidSolar import defaultLsf
from pfs.drp.stella.reduceExposure import ReduceExposureConfig

# Declarations, used by the unit tests to check that this template is up to date.
TASK = "pfs.drp.stella.centroidSolar.CentroidSolarTask"
INPUTS = {"pfsArm": "PfsArm", "pfsConfig": "PfsConfig", "detectorMap": "DetectorMap"}

# ---- Edit these ----
LSF_SIGMA = None  # Gaussian sigma (nm) of the LSF; None for the pipeline's default for the arm
# --------------------

pfsArm = data["pfsArm"]
header = pfsArm.metadata
data["visitInfo"] = VisitInfo(
    boresightAzAlt=SpherePoint(header["AZIMUTH"] * degrees, header["ALTITUDE"] * degrees)
)

arm = pfsArm.identity.arm
widths = dict(ReduceExposureConfig().gaussianLsfWidth)
if LSF_SIGMA is not None:
    widths[arm] = LSF_SIGMA
data["lsf"] = defaultLsf(arm, pfsArm.fiberId, data["detectorMap"], widths)
