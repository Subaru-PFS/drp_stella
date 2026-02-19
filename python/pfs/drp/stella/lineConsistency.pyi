import numpy as np
from lsst.afw.math import StatisticsControl

class ConsistencyResult:
    fiberId: np.ndarray
    wavelength: np.ndarray
    x: np.ndarray
    y: np.ndarray
    xErr: np.ndarray
    yErr: np.ndarray
    flux: np.ndarray
    fluxErr: np.ndarray

def checkLineConsistency(
    fiberId: np.ndarray,
    wavelength: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xErr: np.ndarray,
    yErr: np.ndarray,
    flux: np.ndarray,
    fluxErr: np.ndarray,
    control: StatisticsControl = StatisticsControl(),
) -> ConstistencyResult: ...
def checkTraceConsistency(
    fiberId: np.ndarray,
    wavelength: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xErr: np.ndarray,
    flux: np.ndarray,
    fluxErr: np.ndarray,
    control: StatisticsControl = StatisticsControl(),
) -> ConsistencyResult: ...
