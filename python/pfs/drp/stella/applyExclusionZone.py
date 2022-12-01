from typing import Union

import numpy as np

from .referenceLine import ReferenceLineSet, ReferenceLineStatus
from .arcLine import ArcLineSet

__all__ = ("getExclusionZone", "applyExclusionZone",)


def getExclusionZone(
    wavelength: np.ndarray,
    exclusionRadius: float
) -> np.ndarray:
    """Get a boolean array indicating which lines violate the exclusion zone

    A line cannot have another line within ``exclusionRadius``.

    Parameters
    ----------
    wavelenth : `numpy.ndarray`
        Line wavelengths (nm).
    exclusionRadius : `float`
        Radius in wavelength (nm) to apply around lines.

    Returns
    -------
    excluded : `np.ndarray` of `bool`
        Boolean array indicating which lines violate the exclusion zone.
    """
    excluded = np.zeros_like(wavelength, dtype=bool)
    if exclusionRadius <= 0:
        # No exclusion zone to apply
        return excluded
    for wl in wavelength:
        distance = wavelength - wl
        excluded |= (np.abs(distance) < exclusionRadius) & (distance != 0)
    return excluded


def applyExclusionZone(lines: Union[ReferenceLineSet, ArcLineSet], exclusionRadius: float,
                       status: ReferenceLineStatus = ReferenceLineStatus.BLEND):
    """Apply an exclusion zone around each line

    A line cannot have another line within ``exclusionRadius``.

    The line list is modified in-place.

    Parameters
    ----------
    lines : `ReferenceLineSet` or `ArcLineSet`
        Line list.
    exclusionRadius : `float`
        Radius in wavelength (nm) to apply around lines.
    status : `ReferenceLineStatus`
        Status to apply to lines that fall within the exclusion zone.
    """
    if exclusionRadius <= 0:
        # No exclusion zone to apply
        return
    excluded = getExclusionZone(lines.wavelength, exclusionRadius)
    lines.status[excluded] |= status
