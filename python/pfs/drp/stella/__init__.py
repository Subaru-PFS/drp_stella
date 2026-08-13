try:
    from .version import *
except ImportError:
    print("WARNING: unable to import version.py in drp_stella; did you build with scons?")
    __version__ = "unknown"

import pfs.drp.stella.astropyFix  # noqa: monkey-patch astropy's lazyproperty to avoid deadlocks
import pfs.drp.stella.pickleUtils  # noqa: register pickle functions
from .datamodel import *
from .SpectrumContinued import *
from .SpectrumSetContinued import *
from .FiberTrace import *
from .FiberTraceContinued import *
from .FiberTraceSet import *
from .FiberTraceSetContinued import *
from .DetectorMapContinued import *
from .SplinedDetectorMapContinued import *
from .utils import *
from .spline import *
from .SpectralPsf import *
from .SpectralPsfContinued import *
from .NevenPsfContinued import *
from .lsf import *
from .buildFiberProfiles import *
from .fiberProfile import *
from .fiberProfileSet import *
from .fitPolynomial import *
from .slitOffsets import *
from .referenceLine import *
from .arcLine import *
from .centroidImage import *
from .DistortionContinued import *
from .PolynomialDistortionContinued import *
from .MosaicPolynomialDistortionContinued import *
from .LayeredDetectorMapContinued import *
from .OpticalModelDetectorMapContinued import *

from lsst.afw.image import Mask
for plane in (
    "BAD_FLAT", "FIBERTRACE", "BAD_FIBERTRACE", "BAD_SKY", "BAD_FLUXCAL", "IPC", "REFLINE", "BAD_FIBERNORMS",
    "UNMASKEDNAN", "BAD_PFI_CORRECTION"
):
    Mask.addMaskPlane(plane)

# afw hands a plane the lowest free bit when it is first claimed, so the
# name-to-bit map depends on the order planes are claimed in the process.
# `SpectrumSet.toPfsArm` freezes that map into the pfsArm it writes: `Spectrum.mask`
# is an afw Mask, and the datamodel `MaskHelper` is a verbatim copy of its plane
# dictionary. `MaskHelper` then compares those numbers between files without
# remapping them, so two processes that numbered their planes differently write
# pfsArms that will not merge.
#
# Reading a postISRCCD is not enough to pick up the numbering it was written with:
# `conformMaskPlanes` remaps any plane the reading process does not already know
# into its own next free bit, taking the unknown names in alphabetical order. Only
# a process that claimed the planes before it opened the file keeps the file's
# numbering.
#
# Hence claiming obs_pfs's planes here rather than where they are used: this package
# is where `toPfsArm` lives, so importing it is the one thing every process that can
# write a pfsArm has in common.
from lsst.obs.pfs.maskPlanes import addObsPfsMaskPlanes  # noqa: E402
addObsPfsMaskPlanes()
