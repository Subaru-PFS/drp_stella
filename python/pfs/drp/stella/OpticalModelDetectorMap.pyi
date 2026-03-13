from typing import Optional, overload

import numpy as np

from lsst.geom import Box2I
from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertySet

from .DetectorMap import DetectorMap
from .OpticalModel import SlitModel, OpticsModel, DetectorModel
from .math import SplineD

class _OpticalModelDetectorMapCoordinate:
    WAVELENGTH = 1
    SLIT_SPATIAL = 2
    SLIT_SPECTRAL = 3
    DETECTOR_X = 4
    DETECTOR_Y = 5
    PIXELS_P = 6
    PIXELS_Q = 7
    ROW = 7
    COL = 6

class OpticalModelDetectorMapData:
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
    def getArray(self, coord: OpticalModelDetectorMap.Coordinate) -> np.ndarray: ...
    def getSpline(
        self, x: OpticalModelDetectorMap.Coordinate, y: OpticalModelDetectorMap.Coordinate
    ) -> SplineD: ...

class OpticalModelDetectorMap(DetectorMap):
    Coordinate = _OpticalModelDetectorMapCoordinate
    def __init__(
        self,
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
    def getData(self, fiberId: int) -> OpticalModelDetectorMapData: ...
    def getSpline(
        self,
        fiberId: int,
        coordFrom: Coordinate,
        coordTo: Coordinate,
        value: float,
    ) -> SplineD: ...
    @overload
    def calculate(
        self,
        fiberId: int,
        coordFrom: Coordinate,
        coordTo: Coordinate,
        value: float,
    ) -> float: ...
    @overload
    def calculate(
        self,
        fiberId: np.ndarray,
        coordFrom: Coordinate,
        coordTo: Coordinate,
        value: np.ndarray,
    ) -> np.ndarray: ...
