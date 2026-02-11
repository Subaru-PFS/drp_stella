import numpy as np

def checkLineConsistency(
    fiberId: np.ndarray,
    wavelength: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xErr: np.ndarray,
    yErr: np.ndarray,
    threshold: float = 3.0,
) -> np.ndarray: ...
def checkTraceConsistency(
    fiberId: np.ndarray, xx: np.ndarray, yy: np.ndarray, xErr: np.ndarray, threshold: float = 3.0
) -> np.ndarray: ...
