from typing import Optional, Tuple

import numpy as np
import matplotlib.figure
from lsst.geom import Box2I
from lsst.afw.image import MaskedImage

class KernelSolution:
    bbox: Box2I
    kernelHalfWidth: int
    backgroundOrder: int
    kernel: np.ndarray
    background: np.ndarray
    rejected: np.ndarray
    success: bool
    numPixels: int
    numFit: int
    numRejected: int
    numIter: int
    chi2: float
    rms: float
    def getNumParams(self) -> int: ...
    @property
    def numParams(self) -> int: ...
    def getKernelSum(self) -> float: ...
    @property
    def kernelSum(self) -> float: ...
    def getDegreesOfFreedom(self) -> int: ...
    @property
    def degreesOfFreedom(self) -> int: ...
    def getReducedChi2(self) -> float: ...
    @property
    def reducedChi2(self) -> float: ...
    def __repr__(self) -> str: ...

class AlardLuptonResult:
    difference: MaskedImage
    solutions: list[KernelSolution]
    numRegionsX: int
    numRegionsY: int
    kernelHalfWidth: int
    def getSolutionAt(self, x: int, y: int) -> KernelSolution | None: ...
    def getChi2(self) -> float: ...
    @property
    def chi2(self) -> float: ...
    def getNumFit(self) -> int: ...
    @property
    def numFit(self) -> int: ...
    def getNumRejected(self) -> int: ...
    @property
    def numRejected(self) -> int: ...
    def plotSpatialKernel(
        self,
        *,
        doNormalize: bool = ...,
        vmin: Optional[float] = ...,
        vmax: Optional[float] = ...,
        percentile: float = ...,
        symmetric: bool = ...,
        cmap: str = ...,
        colorbar: Optional[str] = ...,
        markCenter: bool = ...,
        annotate: bool = ...,
        figsize: Optional[Tuple[float, float]] = ...,
        fig: Optional[matplotlib.figure.Figure] = ...,
        axes: Optional[np.ndarray] = ...,
    ) -> Tuple[matplotlib.figure.Figure, np.ndarray]: ...
    def plotPsfMatchResult(
        self,
        source: MaskedImage,
        target: MaskedImage,
        *,
        vmin: Optional[float] = ...,
        vmax: Optional[float] = ...,
        stretchAlgorithm: str = ...,
        percentile: float = ...,
        symmetric: bool = ...,
        zscaleSamples: int = ...,
        zscaleContrast: float = ...,
        diffVmin: Optional[float] = ...,
        diffVmax: Optional[float] = ...,
        diffPercentile: float = ...,
        diffSymmetric: bool = ...,
        cmap: str = ...,
        diffCmap: str = ...,
        titles: Tuple[str, str, str, str] = ...,
        showRegions: bool = ...,
        showRejected: bool = ...,
        figsize: Optional[Tuple[float, float]] = ...,
        fig: Optional[matplotlib.figure.Figure] = ...,
        axes: Optional[np.ndarray] = ...,
    ) -> Tuple[matplotlib.figure.Figure, np.ndarray]: ...

def fitAlardLuptonKernel(
    source: MaskedImage,
    target: MaskedImage,
    kernelHalfWidth: int = 10,
    numRegionsX: int = 1,
    numRegionsY: int = 1,
    backgroundOrder: int = 1,
    badBitMask: int = 0,
    rejIter: int = 2,
    rejThresh: float = 3.0,
    lsqThreshold: float = 1.0e-6,
) -> AlardLuptonResult: ...
