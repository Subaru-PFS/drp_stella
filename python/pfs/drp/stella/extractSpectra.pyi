from lsst.afw.image import Image, MaskedImage
from pfs.drp.stella import FiberTraceSet

def extractSpectra(
    image: MaskedImage,
    fiberTraces: FiberTraceSet,
    badBitMask: int = 0,
    kernelHalfWidth: int = 2,
    kernelOrder: int = 3,
    xBackgroundSize: int = 100,
    yBackgroundSize: int = 100,
    minFracMask: float = 0.3,
    minFracImage: float = 0.4,
) -> tuple[SpectrumSet, Image]: ...
