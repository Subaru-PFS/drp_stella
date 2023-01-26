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
from .GlobalDetectorModel import *
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
from .DifferentialDetectorMapContinued import *
from .DistortedDetectorMapContinued import *
from .DetectorDistortion import *
from .referenceLine import *
from .arcLine import *
from .centroidImage import *
from .DoubleDetectorMapContinued import *
from .DoubleDistortion import *
from .DistortionContinued import *
from .DoubleDistortionContinued import *
from .MultipleDistortionsDetectorMapContinued import *
from .RotScaleDistortionContinued import *

from lsst.afw.image import Mask
for plane in ("BAD_FLAT", "FIBERTRACE", "BAD_FIBERTRACE", "BAD_SKY", "BAD_FLUXCAL", "IPC", "REFLINE"):
    Mask.addMaskPlane(plane)
