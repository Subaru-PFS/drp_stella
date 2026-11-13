from typing import Optional, Union

import numpy as np

from .referenceLine import ReferenceLineSet, ReferenceLineStatus
from .arcLine import ArcLineSet
from .maskNearby import maskNearby

__all__ = ("getExclusionZone", "applyExclusionZone",)


def maskNearbyRelative(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    exclusionRadius: float,
    minRelativeIntensity: float,
) -> np.ndarray:
    """Mask lines that have a line of comparable intensity nearby

    A line is masked if there is another line within ``exclusionRadius`` that
    is not much fainter than it: a neighbor that is fainter than
    ``minRelativeIntensity`` times the line's intensity is treated as not
    visible, and doesn't cause the line to be masked (though the neighbor
    itself is masked, because of the brighter line). Lines with the same
    wavelength don't mask each other (as for `maskNearby`).

    A non-finite intensity is treated as unknown: such a line masks, and is
    masked by, all its neighbors.

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`
        Line wavelengths (nm).
    intensity : `numpy.ndarray` of `float`
        Line intensities.
    exclusionRadius : `float`
        Radius in wavelength (nm) to apply around lines.
    minRelativeIntensity : `float`
        Minimum intensity of a neighbor, relative to the line, for the neighbor
        to cause the line to be masked.

    Returns
    -------
    mask : `numpy.ndarray` of `bool`
        Boolean array indicating which lines are masked.
    """
    order = np.argsort(wavelength, kind="stable")
    sortedWavelength = wavelength[order]
    sortedIntensity = intensity[order]
    low = np.searchsorted(sortedWavelength, sortedWavelength - exclusionRadius, side="right")
    high = np.searchsorted(sortedWavelength, sortedWavelength + exclusionRadius, side="left")
    mask = np.zeros(wavelength.size, dtype=bool)
    for ii in np.flatnonzero(high - low > 1):
        neighbors = np.arange(low[ii], high[ii])
        neighbors = neighbors[sortedWavelength[neighbors] != sortedWavelength[ii]]
        if neighbors.size == 0:
            continue
        # Comparisons with NaN are False, so unknown intensities count as comparable
        faint = sortedIntensity[neighbors] < minRelativeIntensity*sortedIntensity[ii]
        mask[ii] = not np.all(faint)
    result = np.zeros_like(mask)
    result[order] = mask
    return result


def getExclusionZone(
    wavelength: np.ndarray,
    exclusionRadius: float,
    status: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    minRelativeIntensity: float = 0.0,
) -> np.ndarray:
    """Get a boolean array indicating which lines violate the exclusion zone

    A line cannot have another line within ``exclusionRadius`` unless that line
    is flagged ``NOT_VISIBLE`` (lines that PFS doesn't typically detect) or
    ``PROTECTED`` (lines that we want to keep).

    If ``intensity`` is provided and ``minRelativeIntensity`` is positive, a
    line that is fainter than ``minRelativeIntensity`` times the intensity of a
    line within the exclusion zone is treated as not visible for the purpose
    of excluding that brighter line: the faint line violates the exclusion
    zone, but the bright line does not (unless there is another line of
    comparable intensity nearby).

    Parameters
    ----------
    wavelength : `numpy.ndarray`
        Line wavelengths (nm).
    exclusionRadius : `float`
        Radius in wavelength (nm) to apply around lines.
    status : `numpy.ndarray`
        `ReferenceLineStatus`-equivalent integer for each line.
    intensity : `numpy.ndarray`, optional
        Line intensities.
    minRelativeIntensity : `float`, optional
        Minimum intensity, relative to a line, of another line within the
        exclusion zone for that other line to cause the line to be excluded.
        Only used if ``intensity`` is provided.

    Returns
    -------
    excluded : `np.ndarray` of `bool`
        Boolean array indicating which lines violate the exclusion zone.
    """
    excluded = np.zeros_like(wavelength, dtype=bool)
    if exclusionRadius <= 0:
        # No exclusion zone to apply
        return excluded

    visible = (status & ReferenceLineStatus.NOT_VISIBLE) == 0
    if intensity is not None and minRelativeIntensity > 0:
        excluded[visible] = maskNearbyRelative(
            np.asarray(wavelength, dtype=float)[visible],
            np.asarray(intensity, dtype=float)[visible],
            exclusionRadius,
            minRelativeIntensity,
        )
    else:
        excluded[visible] = maskNearby(wavelength[visible], exclusionRadius)
    unprotected = (status & ReferenceLineStatus.PROTECTED) == 0
    return excluded & unprotected


def applyExclusionZone(
    lines: Union[ReferenceLineSet, ArcLineSet],
    exclusionRadius: float,
    status: ReferenceLineStatus = ReferenceLineStatus.BLEND,
    minRelativeIntensity: float = 0.0,
):
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
    minRelativeIntensity : `float`, optional
        Minimum intensity, relative to a line, of another line within the
        exclusion zone for that other line to cause the line to be excluded;
        see `getExclusionZone`. Only used for a `ReferenceLineSet`, which has
        line-list intensities.
    """
    if exclusionRadius <= 0:
        # No exclusion zone to apply
        return
    intensity = lines.intensity if isinstance(lines, ReferenceLineSet) else None
    excluded = getExclusionZone(
        lines.wavelength, exclusionRadius, lines.status, intensity, minRelativeIntensity
    )
    lines.status[excluded] |= status
