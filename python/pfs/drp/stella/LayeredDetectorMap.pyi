from typing import Iterable, List, Optional, overload
import numpy as np

from lsst.geom import AffineTransform, Box2I
from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertySet

from .DetectorMap import DetectorMap
from .SplinedDetectorMapContinued import SplinedDetectorMap
from .Distortion import Distortion

class LayeredDetectorMap(DetectorMap):
    @overload
    def __init__(
        self,
        bbox: Box2I,
        spatial: np.ndarray,
        spectral: np.ndarray,
        base: SplinedDetectorMap,
        distortions: Iterable[Distortion],
        rightCcd: AffineTransform,
        visitInfo: Optional[VisitInfo] = None,
        metadata: Optional[PropertySet] = None,
        samplingFactor: float = 10.0,
    ): ...
    @overload
    def __init__(
        self,
        bbox: Box2I,
        spatial: np.ndarray,
        spectral: np.ndarray,
        base: SplinedDetectorMap,
        distortions: Iterable[Distortion],
        rightCcd: np.ndarray,
        visitInfo: Optional[VisitInfo] = None,
        metadata: Optional[PropertySet] = None,
        samplingFactor: float = 10.0,
    ): ...
    def getBase(self) -> SplinedDetectorMap: ...
    def getDistortions(self) -> List[Distortion]: ...
    def getRightCcd(self) -> AffineTransform: ...
    def getRightCcdParameters(self) -> np.ndarray: ...
    @property
    def base(self) -> SplinedDetectorMap: ...
    @property
    def distortions(self) -> List[Distortion]: ...
    @property
    def rightCcd(self) -> AffineTransform: ...
