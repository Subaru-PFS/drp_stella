from typing import Dict, List, overload

import numpy as np
from lsst.afw.geom import Span
from lsst.afw.image import MaskedImageF
from .DetectorMap import DetectorMap

class TracePeak:
    def __init__(
        self, row: int, low: int, peak: float, high: int, peakErr: float, flux: float, fluxErr: float
    ): ...
    span: Span
    peak: float
    peakErr: float
    flux: float
    fluxErr: float
    row: int
    low: int
    high: int
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

@overload
def findTracePeaks(image: MaskedImageF, threshold: float, badBitMask: int = 0) -> List[List[TracePeak]]: ...
@overload
def findTracePeaks(
    image: MaskedImageF,
    detectorMap: DetectorMap,
    threshold: float,
    radius: float,
    badBitMask: int = 0,
    fiberId: np.ndarray = np.array([], dtype=float),
) -> Dict[int, List[TracePeak]]: ...
def centroidPeak(
    peak: TracePeak,
    image: MaskedImageF,
    psfSigma: float,
    badBitMask: int = 0,
    extent: float = 3.0,
    ampAst4: float = 1.33,
) -> None: ...
def medianFilterColumns(image: np.ndarray, mask: np.ndarray, halfHeight: int = 35) -> np.ndarray: ...
