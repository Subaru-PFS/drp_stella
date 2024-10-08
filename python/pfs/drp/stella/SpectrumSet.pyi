from typing import Any, Dict, Iterator, List, Optional, Tuple, overload

import numpy as np
from lsst.afw.image import Image
from lsst.daf.base import PropertySet
from lsst.geom import Box2I
from matplotlib import Axes, Figure

from .datamodel import PfsArm
from .FiberTraceSetContinued import FiberTraceSet
from .Spectrum import Spectrum

class SpectrumSet:
    @overload
    def __init__(self, length: int): ...
    @overload
    def __init__(self, nSpectra: int, length: int): ...
    def size(self) -> int: ...
    def reserve(self, num: int): ...
    def add(self, spectrum: Spectrum): ...
    def getLength(self) -> int: ...
    def getAllFiberIds(self) -> np.ndarray: ...
    def getAllFluxes(self) -> np.ndarray: ...
    def getAllWavelengths(self) -> np.ndarray: ...
    def getAllMasks(self) -> np.ndarray: ...
    def getAllVariances(self) -> np.ndarray: ...
    def getAllNormalizations(self) -> np.ndarray: ...
    def getAllNotes(self) -> List[PropertySet]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Spectrum: ...
    def __setitem__(self, index: int, spectrum: Spectrum): ...
    def __iter__(self) -> Iterator[Spectrum]: ...  # auto-provided by python from len and getitem
    def toPfsArm(self, dataId: Dict[str, Any]) -> PfsArm: ...
    @classmethod
    def fromPfsArm(self, pfsArm: PfsArm) -> "SpectrumSet": ...
    def writeFits(self, filename: str): ...
    @classmethod
    def readFits(self, path: str) -> "SpectrumSet": ...
    def makeImage(self, box: Box2I, fiberTraces: FiberTraceSet, useSky: bool = False) -> Image: ...
    def plot(self, numRows: int = 3, filename: Optional[str] = None) -> Tuple[Figure, Axes]: ...
