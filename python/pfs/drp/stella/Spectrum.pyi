from typing import overload
import numpy as np
from lsst.afw.image import Mask
from lsst.daf.base import PropertySet

class Spectrum:
    @overload
    def __init__(self, length: int, fiberId: int = 0): ...
    @overload
    def __init__(
        self,
        flux: np.ndarray,
        mask: np.ndarray,
        norm: np.ndarray,
        covariance: np.ndarray,
        wavelength: np.ndarray,
        fiberId: int = 0,
    ): ...
    def getFlux(self) -> np.ndarray: ...
    def setFlux(self, flux: np.ndarray): ...
    @property
    def flux(self) -> np.ndarray: ...
    def getNorm(self) -> np.ndarray: ...
    def setNorm(self, norm: np.ndarray): ...
    @property
    def norm(self) -> np.ndarray: ...
    def getVariance(self) -> np.ndarray: ...
    def setVariance(self, variance: np.ndarray): ...
    @property
    def variance(self) -> np.ndarray: ...
    def getWavelength(self) -> np.ndarray: ...
    def setWavelength(self, wavelength: np.ndarray): ...
    @property
    def wavelength(self) -> np.ndarray: ...
    def getMask(self) -> Mask: ...
    def setMask(self, mask: Mask): ...
    @property
    def mask(self) -> Mask: ...
    def getFiberId(self) -> np.ndarray: ...
    def setFiberId(self, fiberid: np.ndarray): ...
    @property
    def fiberid(self) -> np.ndarray: ...
    def getNotes(self) -> PropertySet: ...
    @property
    def notes(self) -> PropertySet: ...
    def getNumPixels(self) -> int: ...
    def __len__(self) -> int: ...
    def isWavelengthSet(self) -> bool: ...
    def getNormFlux(self) -> np.ndarray: ...
    @property
    def normFlux(self) -> np.ndarray: ...
