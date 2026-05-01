from typing import overload
import numpy as np
from lsst.geom import Point2D, Extent2I, Box2I
from lsst.afw.image import Image, MaskedImage
from pfs.drp.stella import FiberTrace, FiberTraceSet, SpectrumSet
from pfs.drp.stella.math import NormalizedPolynomial2D

class BaseKernel:
    @property
    def halfWidth(self) -> int: ...
    @property
    def numParams(self) -> int: ...
    @property
    def coefficients(self) -> list[float]: ...
    def getHalfWidth(self) -> int: ...
    def getNumParams(self) -> int: ...
    def getCoefficients(self) -> list[float]: ...
    @overload
    def __call__(self, fiberTrace: FiberTrace, bbox: Box2I) -> FiberTrace: ...
    @overload
    def __call__(self, fiberTraces: FiberTraceSet, bbox: Box2I) -> FiberTraceSet: ...
    @overload
    def makeOffsetImages(self, dims: Extent2I) -> np.ndarray: ...
    @overload
    def makeOffsetImages(self, width: int, height: int) -> np.ndarray: ...

class FiberKernel(BaseKernel):
    def __init__(
        self,
        dims: Extent2I,
        kernelHalfWidth: int,
        xKernelNum: int,
        yKernelNum: int,
        coefficients: np.ndarray,
    ) -> None: ...
    @overload
    def evaluate(self, x: float, y: float) -> np.ndarray: ...
    @overload
    def evaluate(self, xy: Point2D) -> np.ndarray: ...

@overload
def fitFiberKernel(
    image: MaskedImage,
    fiberTraces: FiberTraceSet,
    badBitMask: int = 0,
    kernelHalfWidth: int = 2,
    xKernelNum: int = 7,
    yKernelNum: int = 7,
    rows: np.ndarray | None = None,
    maxIter: int = 20,
    andersonDepth: int = 5,
    fluxTol: float = 1.0e-3,
    lsqThreshold: float = 1.0e-16,
) -> FiberKernel: ...
@overload
def fitImageKernel(
    source: MaskedImage,
    target: MaskedImage,
    badBitMask: int = 0,
    kernelHalfWidth: int = 2,
    xKernelNum: int = 7,
    yKernelNum: int = 7,
    xBackgroundNum: int = 9,
    yBackgroundNum: int = 9,
    rows: np.ndarray | None = None,
    lsqThreshold: float = 1.0e-16,
) -> FiberKernel: ...
