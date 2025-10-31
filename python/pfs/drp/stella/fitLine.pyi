from typing import overload
import numpy as np
from pfs.drp.stella.SpectrumContinued import Spectrum

class FitLineResult:
    rms: float
    isValid: bool
    num: int
    amplitude: float
    center: float
    rmsSize: float
    bg0: float
    bg1: float
    amplitudeErr: float
    centerErr: float
    rmsSizeErr: float
    bg0Err: float
    bg1Err: float

@overload
def fitLine(
    spectrum: Spectrum,
    mask: int,
    peakPosition: float,
    rmsSize: float,
    badBitMask: int,
    fittingHalfSize: int = 0,
) -> FitLineResult: ...
@overload
def fitLine(
    flux: np.ndarray,
    mask: np.ndarray,
    peakPosition: float,
    rmsSize: float,
    badBitMask: int,
    fittingHalfSize: int = 0,
) -> FitLineResult: ...
