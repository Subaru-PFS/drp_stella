from typing import Optional, overload
import numpy as np
from lsst.geom import Box2D

from .Distortion import Distortion
from .math import NormalizedPolynomial2D


class PolynomialDistortion(Distortion):
    @overload
    def __init__(self, distortionOrder: int, range: Box2D, coeff: np.ndarray): ...
    @overload
    def __init__(
        self, distortionOrder: int, range: Box2D, xCoeff: np.ndarray, yCoeff: np.ndarray
    ): ...
    @overload
    def __init__(self, xOrder: int, yOrder: int, range: Box2D, coeff: np.ndarray): ...
    @overload
    def __init__(
        self, xOrder: int, yOrder: int, range: Box2D, xCoeff: np.ndarray, yCoeff: np.ndarray
    ): ...
    @overload
    @staticmethod
    def fit(
        xOrder: int,
        yOrder: int,
        range: Box2D,
        x: np.ndarray,
        y: np.ndarray,
        xMeas: np.ndarray,
        yMeas: np.ndarray,
        xErr: np.ndarray,
        yErr: np.ndarray,
        isLine: np.ndarray,
        threshold: float = 1.0e-6,
        forced: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
    ) -> "PolynomialDistortion": ...
    @overload
    @staticmethod
    def fit(
        order: int,
        range: Box2D,
        x: np.ndarray,
        y: np.ndarray,
        xMeas: np.ndarray,
        yMeas: np.ndarray,
        xErr: np.ndarray,
        yErr: np.ndarray,
        isLine: np.ndarray,
        threshold: float = 1.0e-6,
        forced: Optional[np.ndarray] = None,
        params: Optional[np.ndarray] = None,
    ) -> "PolynomialDistortion": ...
    @overload
    @staticmethod
    def getNumParametersForOrder(xOrder: int, yOrder: int) -> int: ...
    @overload
    @staticmethod
    def getNumParametersForOrder(order: int) -> int: ...
    def getOrder(self) -> int: ...
    def getXOrder(self) -> int: ...
    def getYOrder(self) -> int: ...
    def getRange(self) -> Box2D: ...
    def getCoefficients(self) -> np.ndarray: ...
    @staticmethod
    def getNumDistortionForOrder(order: int) -> int: ...
    def getXNumDistortion(self) -> int: ...
    def getYNumDistortion(self) -> int: ...
    def getXCoefficients(self) -> np.ndarray: ...
    def getYCoefficients(self) -> np.ndarray: ...
    def getXPoly(self) -> NormalizedPolynomial2D: ...
    def getYPoly(self) -> NormalizedPolynomial2D: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
