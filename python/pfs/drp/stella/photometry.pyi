from typing import Optional
import numpy as np
from lsst.afw.image import MaskedImageF
from lsst.afw.table import BaseCatalog

from .SpectralPsf import SpectralPsf


def photometer(
    image: MaskedImageF,
    fiberId: np.ndarray,
    wavelength: np.ndarray,
    psf: SpectralPsf,
    badBitMask: int = 0,
    positions: Optional[np.ndarray] = None,
) -> BaseCatalog: ...
