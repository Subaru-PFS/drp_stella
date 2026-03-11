from typing import Optional

import numpy as np

from lsst.geom import Box2I
from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertySet

from .DetectorMap import DetectorMap
from .OpticalModel import SlitModel, OpticsModel, DetectorModel
from .math import SplineD

class _OpticalModelDataCoordinate:
    WAVELENGTH = 1
    SLIT_SPATIAL = 2
    SLIT_SPECTRAL = 3
    DETECTOR_X = 4
    DETECTOR_Y = 5
    PIXELS_P = 6
    PIXELS_Q = 7
    ROW = 7
    COL = 6

class OpticalModelData:
    def __init__(
        self,
        wavelength: np.ndarray,
        slit: np.ndarray,
        detector: np.ndarray,
        pixels: np.ndarray,
    ): ...
    wavelength: np.ndarray
    slit: np.ndarray
    detector: np.ndarray
    pixels: np.ndarray
    Coordinate = _OpticalModelDataCoordinate
    def getArray(self, coord: Coordinate) -> np.ndarray: ...
    def getSpline(self, x: Coordinate, y: Coordinate) -> SplineD: ...

class OpticalModelDetectorMap(DetectorMap):
    def __init__(
        self,
        bbox: Box2I,
        slitModel: SlitModel,
        opticsModel: OpticsModel,
        detectorModel: DetectorModel,
        visitInfo: Optional[VisitInfo] = None,
        metadata: Optional[PropertySet] = None,
    ): ...
    def getSlitModel(self) -> SlitModel: ...
    def getOpticsModel(self) -> OpticsModel: ...
    def getDetectorModel(self) -> DetectorModel: ...
    @property
    def slitModel(self) -> SlitModel: ...
    @property
    def opticsModel(self) -> OpticsModel: ...
    @property
    def detectorModel(self) -> DetectorModel: ...
    def getXDetectorSpline(self, fiberId: int) -> SplineD: ...
    def getYDetectorSpline(self, fiberId: int) -> SplineD: ...
    def getWavelengthSpline(self, fiberId: int) -> SplineD: ...
