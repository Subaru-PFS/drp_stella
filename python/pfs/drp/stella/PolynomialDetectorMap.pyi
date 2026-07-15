from typing import Optional
from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertySet

from .DetectorMap import DetectorMap
from .PolynomialDistortion import PolynomialDistortion
from .SplinedDetectorMapContinued import SplinedDetectorMap


class PolynomialDetectorMap(DetectorMap):
    def __init__(
        self,
        base: SplinedDetectorMap,
        distortion: PolynomialDistortion,
        visitInfo: Optional[VisitInfo] = None,
        metadata: Optional[PropertySet] = None,
        samplingFactor: float = 50.0,
    ): ...
    def getBase(self) -> SplinedDetectorMap: ...
    def getDistortion(self) -> PolynomialDistortion: ...
    @property
    def base(self) -> SplinedDetectorMap: ...
    @property
    def distortion(self) -> PolynomialDistortion: ...
