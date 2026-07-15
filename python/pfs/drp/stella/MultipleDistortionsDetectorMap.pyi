from typing import Iterable, List, Optional
from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertySet

from .DetectorMap import DetectorMap
from .Distortion import Distortion
from .SplinedDetectorMapContinued import SplinedDetectorMap


class MultipleDistortionsDetectorMap(DetectorMap):
    def __init__(
        self,
        base: SplinedDetectorMap,
        distortions: Iterable[Distortion],
        visitInfo: Optional[VisitInfo] = None,
        metadata: Optional[PropertySet] = None,
        samplingFactor: float = 50.0,
    ): ...
    def getBase(self) -> SplinedDetectorMap: ...
    def getDistortions(self) -> List[Distortion]: ...
    @property
    def base(self) -> SplinedDetectorMap: ...
    @property
    def distortions(self) -> List[Distortion]: ...
