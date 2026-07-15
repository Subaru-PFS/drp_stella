from typing import List, Optional, Tuple, overload
import numpy as np
from lsst.afw.image import ImageF, MaskedImageF
from lsst.geom import Box2I, Extent2I

from .Spectrum import Spectrum

fiberMaskPlane: str


class FiberTrace:
    @overload
    def __init__(self, maskedImage: MaskedImageF, fiberTraceId: int = 0): ...
    @overload
    def __init__(self, fiberTrace: "FiberTrace", deep: bool = False): ...
    def getTrace(self) -> MaskedImageF: ...
    @property
    def trace(self) -> MaskedImageF: ...
    def getFiberId(self) -> int: ...
    def setFiberId(self, fiberId: int) -> None: ...
    @property
    def fiberId(self) -> int: ...
    @overload
    def constructImage(self, spectrum: Spectrum) -> ImageF: ...
    @overload
    def constructImage(self, spectrum: Spectrum, bbox: Box2I) -> ImageF: ...
    @overload
    def constructImage(self, image: ImageF, spectrum: Spectrum) -> None: ...
    @overload
    def constructImage(self, image: ImageF, flux: np.ndarray) -> None: ...
    @staticmethod
    def fromProfile(
        fiberId: int,
        dims: Extent2I,
        radius: int,
        oversample: float,
        rows: np.ndarray,
        profiles: np.ndarray,
        good: np.ndarray,
        positions: List[Tuple[int, np.ndarray]],
        norm: Optional[np.ndarray] = None,
    ) -> "FiberTrace": ...
    @staticmethod
    def boxcar(
        fiberId: int,
        dims: Extent2I,
        radius: float,
        centers: np.ndarray,
        norm: Optional[np.ndarray] = None,
    ) -> "FiberTrace": ...
    def extractAperture(self, image: MaskedImageF, badBitmask: int) -> Spectrum: ...
