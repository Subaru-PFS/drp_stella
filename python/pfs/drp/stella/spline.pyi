from typing import overload

import numpy as np

class _InterpolationTypes:
    NOTAKNOT: int = 0
    NATURAL: int = 1

class SplineD:
    InterpolationTypes = _InterpolationTypes
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        type: int = _InterpolationTypes.NOTAKNOT,
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
