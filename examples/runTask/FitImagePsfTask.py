"""runTask.py template for FitImagePsfTask

Copy this file, review and edit the values in the "Edit these" block, and run:

    runTask.py pfs.drp.stella.fitImagePsf.FitImagePsfTask \\
        detectorMap:DetectorMap=detectorMap.fits \\
        --extra FitImagePsfTask.py --output psf:psf.fits \\
        [--config overrides.py] [--log-level fitImagePsf=DEBUG]

``FitImagePsfTask.run`` takes lists of exposures and pfsConfigs (one per
visit), which the command line can't express, so they are listed here. This
also supplies the arm and spectrograph.

See README.md in this directory for how to use runTask.py and its templates.
"""

from pfs.drp.stella.harness import readData

# Declarations, used by the unit tests to check that this template is up to date.
TASK = "pfs.drp.stella.fitImagePsf.FitImagePsfTask"
INPUTS = {"detectorMap": "DetectorMap"}  # name:type=filename on the command line

# ---- Edit these ----
ARM = "r"  # Spectrograph arm: b, r, n, m
SPECTROGRAPH = 1  # Spectrograph module: 1-4
EXPOSURES = ["postISRCCD.fits"]  # Filenames of the exposures (ISR-corrected), one per visit
PFS_CONFIGS = ["pfsConfig.fits"]  # Filenames of the pfsConfig for each exposure, in the same order
# --------------------

if len(EXPOSURES) != len(PFS_CONFIGS):
    raise RuntimeError(f"Number of exposures ({len(EXPOSURES)}) and pfsConfigs ({len(PFS_CONFIGS)}) differ")
data["exposures"] = [readData("Exposure", filename) for filename in EXPOSURES]
data["pfsConfigs"] = [readData("PfsConfig", filename) for filename in PFS_CONFIGS]
data["arm"] = ARM
data["spectrograph"] = SPECTROGRAPH
