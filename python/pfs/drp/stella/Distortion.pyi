from typing import overload, Tuple

import numpy as np

from lsst.geom import Point2D
import pfs.datamodel

__all__ = ("Distortion",)

class Distortion:
    def clone(self) -> "Distortion": ...
    @classmethod
    def register(cls, SubClass: type): ...
    @classmethod
    def fromDatamodel(cls, distortion: pfs.datamodel.Distortion) -> "Distortion": ...
    @classmethod
    def readFits(cls, pathName: str) -> "Distortion": ...
    def writeFits(self, pathName: str): ...
    @classmethod
    def fromBytes(cls, string: bytes) -> "Distortion": ...
    def toBytes(self) -> bytes: ...
    @overload
    def __call__(self, xy: Point2D) -> Point2D: ...
    @overload
    def __call__(self, x: float, y: float) -> Point2D: ...
    @overload
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...
    @overload
    def __call__(self, xy: np.ndarray) -> np.ndarray: ...
    def calculateChi2(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        xErr: np.ndarray,
        yErr: np.ndarray,
        good: np.ndarray = None,
        sysErr: float = 0.0,
    ) -> Tuple[float, int]: ...
