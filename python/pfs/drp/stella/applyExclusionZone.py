from typing import Union

import numpy as np

from .referenceLine import ReferenceLineSet, ReferenceLineStatus
from .arcLine import ArcLineSet

__all__ = ("applyExclusionZone",)


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
    wavelength = lines.wavelength
    reject = np.zeros(len(lines), dtype=bool)
    for ll in lines:
        distance = wavelength - ll.wavelength
        reject |= (np.abs(distance) < exclusionRadius) & (distance != 0)
    for ll, rej in zip(lines, reject):
        if rej:
            ll.status |= status
