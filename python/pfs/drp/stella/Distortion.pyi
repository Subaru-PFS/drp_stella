from typing import overload, Tuple

import numpy as np
import astropy.io.fits

from lsst.geom import Point2D
import pfs.datamodel

__all__ = ("Distortion",)

class Distortion:
    @classmethod
    def register(cls, SubClass: type): ...
    @classmethod
    def fromDatamodel(cls, distortion: pfs.datamodel.Distortion) -> "Distortion": ...
    @classmethod
    def fromFits(cls, fits: astropy.io.fits.HDUList) -> "Distortion": ...
    def toFits(self): ...
    @classmethod
    def readFits(cls, pathName: str) -> "Distortion": ...
    @classmethod
    def fromBytes(cls, string: str) -> "Distortion": ...
    def writeFits(self, pathName: str): ...
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
