from types import SimpleNamespace
from typing import Dict, Iterator, Optional, Tuple, Union, overload

import numpy as np
from lsst.afw.image import ImageF
from lsst.geom import Box2I
from matplotlib.pyplot import Axes, Figure
from pfs.datamodel import PfsArm

from .FiberTraceSetContinued import FiberTraceSet
from .SpectrumContinued import Spectrum

__all__ = ("SpectrumSet",)

class SpectrumSet:
    @overload
    def __init__(self, length: int): ...
    @overload
    def __init__(self, nSpectra: int, length: int): ...
    def size(self) -> int: ...
    def reserve(self, int): ...
    def add(self, spectrum: Spectrum): ...
    def getLength(self) -> int: ...
    def getAllFiberIds(self) -> np.ndarray: ...
    def getAllFluxes(self) -> np.ndarray: ...
    def getAllWavelengths(self) -> np.ndarray: ...
    def getAllMasks(self) -> np.ndarray: ...
    def getAllCovariances(self) -> np.ndarray: ...
    def getAllBackgrounds(self) -> np.ndarray: ...
    def getAllNormalizations(self) -> np.ndarray: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Spectrum: ...
    def __setitem__(self, index: int, spectrum: Spectrum): ...
    def __iter__(self) -> Iterator["SpectrumSet"]: ...
    def toPfsArm(self, dataId: Dict[str, Union[str, int, float]]) -> PfsArm: ...
    @classmethod
    def fromPfsArm(cls, pfsArm: PfsArm) -> "SpectrumSet": ...
    @classmethod
    def _parsePath(
        cls, path: str, hdu: Optional[int] = None, flags: Optional[int] = None
    ) -> SimpleNamespace: ...
    def writeFits(self, filename: str): ...
    @classmethod
    def readFits(cls, *args, **kwargs) -> "SpectrumSet": ...
    def makeImage(self, box: Box2I, fiberTraces: FiberTraceSet) -> ImageF: ...
    def plot(self, numRows: int = 3, filename: str = None) -> Tuple[Figure, Axes]: ...
