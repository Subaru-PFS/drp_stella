from typing import Tuple, overload

import numpy as np
from lsst.afw.image import Mask
from matplotlib.pyplot import Axes, Figure
from numpy.typing import ArrayLike
from pfs.datamodel import PfsFiberArray

__all__ = ("Spectrum",)

class Spectrum:
    @overload
    def __init__(self, length: int, fiberId: int = 0): ...
    @overload
    def __init__(
        self,
        flux: np.ndarray,
        mask: Mask,
        background: np.ndarray,
        norm: np.ndarray,
        covariance: np.ndarray,
        wavelength: np.ndarray,
        fiberId: int = 0,
    ): ...

    flux: np.ndarray
    spectrum: np.ndarray
    background: np.ndarray
    norm: np.ndarray
    variance: np.ndarray
    covariance: np.ndarray
    wavelength: np.ndarray
    mask: Mask
    normFlux: np.ndarray
    fiberId: int

    def getNumPixels(self) -> int: ...
    def __len__(self) -> int: ...
    def isWavelengthSet(self) -> bool: ...
    def plot(
        self, numRows: int = 3, doBackground: bool = False, filename: str = None
    ) -> Tuple[Figure, Axes]: ...
    def wavelengthToPixels(self, wavelength: ArrayLike) -> ArrayLike: ...
    def pixelsToWavelength(self, pixels: ArrayLike) -> ArrayLike: ...
    def toPfsFiberArray(self) -> PfsFiberArray: ...
