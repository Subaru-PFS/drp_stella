from functools import lru_cache
from typing import overload, Optional

import numpy as np
from numpy.typing import ArrayLike
import pfs.datamodel.drp

from .pfsFiberArray import PfsFiberArray, PfsSimpleSpectrum
from .pfsFiberArraySet import PfsFiberArraySet
from ..math import NormalizedPolynomial1D

__all__ = ("PfsArm", "PfsMerged", "PfsReference", "PfsSingle", "PfsObject", "PfsFiberNorms")


class PfsArm(pfs.datamodel.drp.PfsArm, PfsFiberArraySet):
    _ylabel = "electrons/spectral pixel"
    pass


class PfsMerged(pfs.datamodel.drp.PfsMerged, PfsFiberArraySet):
    _ylabel = "electrons/nm"
    pass


class PfsReference(pfs.datamodel.drp.PfsReference, PfsSimpleSpectrum):
    _ylabel = "nJy"
    pass


class PfsSingle(pfs.datamodel.drp.PfsSingle, PfsFiberArray):
    _ylabel = "nJy"
    pass


class PfsObject(pfs.datamodel.drp.PfsObject, PfsFiberArray):
    _ylabel = "nJy"
    pass


class PfsFiberNorms(pfs.datamodel.PfsFiberNorms):
    @overload
    def calculate(self, fiberId: int) -> np.ndarray:
        ...

    def calculate(self, fiberId: int, rows: Optional[ArrayLike] = None) -> ArrayLike:
        """Calculate the normalization for a fiber

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.

        norm : `numpy.ndarray` of `float`
            Normalization for each spectral pixel.
        """
        poly = self.getPolynomial(fiberId)
        if rows is None:
            rows = np.arange(self.height, dtype=float)
        return poly(rows)

    @lru_cache(maxsize=1000)
    def getPolynomial(self, fiberId: int) -> NormalizedPolynomial1D:
        """Return the polynomial for a fiber

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.

        Returns
        -------
        poly : `NormalizedPolynomial1D`
            Normalized polynomial for the fiber.
        """
        coeff = self[fiberId]
        return NormalizedPolynomial1D(coeff, 0, self.height)

    def getMetadata(self):
        """Return metadata

        Required for recordCalibInputs.
        """
        return self.metadata
