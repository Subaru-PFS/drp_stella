from typing import List, Tuple
import numpy as np
from lsst.afw.image import MaskedImageF


def fitSwathProfiles(
    images: List[MaskedImageF],
    centers: List[np.ndarray],
    spectra: List[np.ndarray],
    fiberId: np.ndarray,
    yMin: int,
    yMax: int,
    badBitMask: int,
    oversample: int,
    radius: int,
    rejIter: int = 1,
    rejThresh: float = 4.0,
    matrixTol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]: ...
def fitAmplitudes(
    image: MaskedImageF,
    centers: np.ndarray,
    sigma: float,
    badBitMask: int = 0,
    maxSigma: float = 4.0,
) -> np.ndarray: ...
def calculateSwathProfile(
    values: np.ndarray,
    mask: np.ndarray,
    rejIter: int = 1,
    rejThresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]: ...
