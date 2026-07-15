from typing import List, Optional
import numpy as np
from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertySet
from lsst.geom import Box2I

from .DetectorMap import DetectorMap
from .spline import SplineD


class SplinedDetectorMap(DetectorMap):
    def __init__(
        self,
        bbox: Box2I,
        fiberId: np.ndarray,
        centerKnots: List[np.ndarray],
        centerValues: List[np.ndarray],
        wavelengthKnots: List[np.ndarray],
        wavelengthValues: List[np.ndarray],
        spatialOffsets: Optional[np.ndarray] = None,
        spectralOffsets: Optional[np.ndarray] = None,
        visitInfo: Optional[VisitInfo] = None,
        metadata: Optional[PropertySet] = None,
    ): ...
    def getXCenterSpline(self, fiberId: int) -> SplineD: ...
    def getWavelengthSpline(self, fiberId: int) -> SplineD: ...
    def setXCenter(self, fiberId: int, knots: np.ndarray, xCenter: np.ndarray) -> None: ...
    def setWavelength(self, fiberId: int, knots: np.ndarray, wavelength: np.ndarray) -> None: ...
