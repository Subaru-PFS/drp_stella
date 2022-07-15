from configparser import Interpolation
from typing import Literal, Union, overload
import numpy as np

class SplineD:
    class InterpolationTypes:
        pass
    NOTAKNOT: InterpolationTypes
    NATURAL: InterpolationTypes
    interpolationType: InterpolationTypes

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        type: InterpolationTypes = NOTAKNOT,
    ): ...
    @overload
    def __call__(self, x: float) -> float: ...
    @overload
    def __call__(self, x: np.ndarray) -> np.ndarray: ...
    def getX(self) -> np.ndarray: ...
    def getY(self) -> np.ndarray: ...
    def getInterpolationType(self) -> InterpolationTypes: ...
