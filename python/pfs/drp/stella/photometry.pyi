from typing import Optional
import numpy as np
from lsst.afw.detection import Psf
from lsst.afw.image import MaskedImageF
from lsst.afw.table import BaseCatalog

from .DetectorMap import DetectorMap


def photometer(
    image: MaskedImageF,
    fiberId: np.ndarray,
    wavelength: np.ndarray,
    psf: Psf,
    detectorMap: DetectorMap,
    badBitMask: int = 0,
    positions: Optional[np.ndarray] = None,
) -> BaseCatalog: ...
