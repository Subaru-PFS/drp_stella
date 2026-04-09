from typing import overload
from lsst.afw.image import Image, MaskedImage
from pfs.drp.stella import FiberTrace, FiberTraceSet, SpectrumSet

class FiberKernel:
    def __init__(self, kernelHalfWidth: int, kernelOrder: int, xBackgroundSize: int, yBackgroundSize: int) -> None: ...
    @overload
    def __call__(self, fiberTrace: FiberTrace) -> FiberTrace: ...
    @overload
    def __call__(self, fiberTraces: FiberTraceSet) -> FiberTraceSet: ...


def fitFiberKernel(
    image: MaskedImage,
    fiberTraces: FiberTraceSet,
    spectra: SpectrumSet,
    badBitMask: int = 0,
    kernelHalfWidth: int = 2,
    kernelOrder: int = 3,
    xBackgroundSize: int = 500,
    yBackgroundSize: int = 500,
) -> tuple[FiberKernel, Image]: ...
