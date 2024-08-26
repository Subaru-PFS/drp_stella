from typing import overload

import numpy as np

class _InterpolationTypes:
    NOTAKNOT: int = 0
    NATURAL: int = 1

class _ExtrapolationTypes:
    ALL: int = 0
    SINGLE: int = 1
    NONE: int = 2

class SplineD:
    InterpolationTypes = _InterpolationTypes
    ExtrapolationTypes = _ExtrapolationTypes
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        interpolationType: int = _InterpolationTypes.NOTAKNOT,
        extrapolationType: int = _ExtrapolationTypes.ALL,
    ): ...
    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: np.ndarray) -> np.ndarray: ...
    def getX(self) -> np.ndarray: ...
    def getY(self) -> np.ndarray: ...
    def getInterpolationType(self) -> int: ...
    @property
    def interpolationType(self) -> int: ...
    def getExtrapolationType(self) -> int: ...
    @property
    def extrapolationType(self) -> int: ...
