from typing import List
import numpy as np
from lsst.geom import Extent2I

from .DetectorMap import DetectorMap
from .SpectralPsf import OversampledPsf, SpectralPsf


class NevenPsf(SpectralPsf, OversampledPsf):
    def __init__(
        self,
        detectorMap: DetectorMap,
        fiberId: np.ndarray,
        wavelength: np.ndarray,
        images: List[np.ndarray],
        oversampleFactor: int,
        targetSize: Extent2I,
    ): ...
    def size(self) -> int: ...
    def __len__(self) -> int: ...
    def getFiberId(self) -> np.ndarray: ...
    def getWavelength(self) -> np.ndarray: ...
    def getImages(self) -> List[np.ndarray]: ...
