from dataclasses import dataclass
from deprecated import deprecated
from typing import Any, Dict, List

import numpy as np

from pfs.datamodel import Identity

from .table import TableBase
from .referenceLine import ReferenceLineSet, ReferenceLine, ReferenceLineStatus
from .datamodel.pfsFiberArraySet import PfsFiberArraySet

__all__ = ("ArcLine", "ArcLineSet")


@dataclass
class ArcLine:
    """Data for a single reference line

    Parameters
    ----------
    fiberId : `int`
        Fiber identifier.
    wavelength : `float`
        Reference line wavelength (nm).
    x, y : `float`
        Measured position.
    xErr, yErr : `float`
        Error in measured position.
    intensity : `float`
        Measured intensity (arbitrary units).
    intensityErr : `float`
        Error in measured intensity (arbitrary units).
    flag : `bool`
        Measurement flag (``True`` indicates an error in measurement).
    status : `int`
        Bitmask indicating the quality of the reference line.
    description : `str`
        Line description (e.g., ionic species)
    """
    fiberId: np.int32
    wavelength: float
    x: float
    y: float
    xErr: float
    yErr: float
    intensity: float
    intensityErr: float
    flag: bool
    status: np.int32
    description: str


class ArcLineSet(TableBase):
    fitsExtName = "ARCLINES"
    RowClass = ArcLine
    damdver = 1

    # Column types.
    # The columns are set up by TableBase.__init_subclass__.
    fiberId: np.ndarray
    wavelength: np.ndarray
    x: np.ndarray
    y: np.ndarray
    xErr: np.ndarray
    yErr: np.ndarray
    intensity: np.ndarray
    intensityErr: np.ndarray
    flag: np.ndarray
    status: np.ndarray
    description: np.ndarray

    @property  # type: ignore [misc]
    @deprecated(reason="use the 'rows' attribute instead of 'lines'")
    def lines(self):
        """Return array of lines

        Included for backwards compatibility.
        """
        return self.rows

    def extractReferenceLines(self, fiberId: int = None) -> ReferenceLineSet:
        """Generate a list of reference lines

        Parameters
        ----------
        fiberId : `int`, optional
            Use lines from this fiber exclusively. Otherwise, we'll average the
            intensities of lines with the same wavelength and description.

        Returns
        -------
        refLines : `pfs.drp.stella.ReferenceLineSet`
            Reference lines.
        """
        if fiberId is not None:
            select = self.fiberId == fiberId
            return ReferenceLineSet.fromColumns(
                description=self.description[select],
                wavelength=self.wavelength[select],
                intensity=self.intensity[select],
                status=self.status[select],
            )
        rows: List[ReferenceLine] = []
        unique = set(zip(self.wavelength, self.description, self.status))
        for wavelength, description, status in sorted(unique):
            select = ((self.description == description) & (self.wavelength == wavelength) &
                      (self.status == status) & np.isfinite(self.intensity))

            intensity = np.average(self.intensity[select]) if np.any(select) else np.nan
            rows.append(ReferenceLine(description, wavelength, intensity, status))
        return ReferenceLineSet.fromRows(rows)

    def applyExclusionZone(self, exclusionRadius: float,
                           status: ReferenceLineStatus = ReferenceLineStatus.BLEND
                           ):
        """Apply an exclusion zone around each line

        A line cannot have another line within ``exclusionRadius``.

        The line list is modified in-place.

        Parameters
        ----------
        exclusionRadius : `float`
            Radius in wavelength (nm) to apply around lines.
        status : `ReferenceLineStatus`
            Status to apply to lines that fall within the exclusion zone.
        """
        from .applyExclusionZone import applyExclusionZone
        return applyExclusionZone(self, exclusionRadius, status)

    def asPfsFiberArraySet(self, identity: Identity = None) -> PfsFiberArraySet:
        """Represent as a PfsFiberArraySet

        This can be useful when fitting models of line intensities.

        Parameters
        ----------
        identity : `Identity`
            Identity to give the output `PfsFiberArraySet`.

        Returns
        -------
        spectra : `PfsFiberArraySet`
            Lines represented as a `PfsFiberArraySet`.
        """
        fiberId = np.array(sorted(set(self.fiberId)), dtype=int)
        numFibers = fiberId.size
        wlSet = np.array(sorted(set(self.wavelength)), dtype=float)
        numWavelength = wlSet.size
        if identity is None:
            identity = Identity(-1)
        flags = ReferenceLineStatus.getMasks()
        metadata: Dict[str, Any] = {}

        wavelength = np.vstack([wlSet]*numFibers)
        flux = np.full((numFibers, numWavelength), np.nan, dtype=float)
        mask = np.full((numFibers, numWavelength), flags["REJECTED"], dtype=np.int32)
        covar = np.full((numFibers, 3, numWavelength), np.nan, dtype=float)
        variance = covar[:, 0, :]
        sky = np.zeros_like(flux)
        norm = np.ones_like(flux)

        wlLookup = {wl: ii for ii, wl in enumerate(wlSet)}
        fiberLookup = {ff: ii for ii, ff in enumerate(fiberId)}
        wlIndices = np.array([wlLookup[ll.wavelength] for ii, ll in enumerate(self)])
        fiberIndices = np.array([fiberLookup[ll.fiberId] for ii, ll in enumerate(self)])
        flux[fiberIndices, wlIndices] = self.intensity
        variance[fiberIndices, wlIndices] = self.intensityErr**2
        mask[fiberIndices, wlIndices] = self.status

        return PfsFiberArraySet(identity, fiberId, wavelength, flux, mask, sky, norm, covar, flags, metadata)
