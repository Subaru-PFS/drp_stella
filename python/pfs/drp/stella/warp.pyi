from typing import overload
import numpy as np
from lsst.afw.image import MaskedImageF
from lsst.afw.image.pixel import SinglePixelF
from lsst.afw.math import WarpingControl
from pfs.drp.stella.DetectorMapContinued import DetectorMap

@overload
def warpFiber(
    image: MaskedImageF,
    detectorMap: DetectorMap,
    fiberId: int,
    halfWidth: int,
    warpingKernelName: str = "lanczos3",
) -> MaskedImageF: ...
@overload
def warpFiber(
    image: MaskedImageF,
    detectorMap: DetectorMap,
    fiberId: int,
    halfWidth: int,
    warpingControl: WarpingControl,
    pad: SinglePixelF = SinglePixelF(np.nan),
) -> MaskedImageF: ...
@overload
def warpImage(
    fromImage: MaskedImageF,
    fromDetectorMap: DetectorMap,
    toDetectorMap: DetectorMap,
    warpingKernelName: str = "lanczos3",
    numWavelengthKnots: int = 75,
) -> MaskedImageF: ...
@overload
def warpImage(
    fromImage: MaskedImageF,
    fromDetectorMap: DetectorMap,
    toDetectorMap: DetectorMap,
    warpingControl: WarpingControl,
    pad: SinglePixelF = SinglePixelF(np.nan),
    numWavelengthKnots: int = 75,
) -> MaskedImageF: ...
