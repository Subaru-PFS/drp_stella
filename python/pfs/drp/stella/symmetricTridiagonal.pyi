from typing import Optional

import numpy as np

class SymmetricTridiagonalWorkspace:
    def __init__(self): ...
    def reset(self, num: int): ...
    longArray1: np.ndarray
    longArray2: np.ndarray
    shortArray: np.ndarray

def solveSymmetricTridiagonal(
    diagonal: np.ndarray,
    offDiag: np.ndarray,
    answer: np.ndarray,
    workspace: Optional[SymmetricTridiagonalWorkspace] = None,
) -> np.ndarray: ...
def invertSymmetricTridiagonal(
    diagonal: np.ndarray,
    offDiag: np.ndarray,
    workspace: Optional[SymmetricTridiagonalWorkspace] = None,
) -> np.ndarray: ...
