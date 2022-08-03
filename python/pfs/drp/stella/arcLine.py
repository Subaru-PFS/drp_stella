from deprecated import deprecated
from typing import Any, Dict
import numpy as np

from pfs.datamodel import Identity
from pfs.datamodel.drp import LineMeasurements

from .table import Table
from .referenceLine import ReferenceLineSet, ReferenceLine, ReferenceLineStatus
from .datamodel.pfsFiberArraySet import PfsFiberArraySet

__all__ = ("ArcLine", "ArcLineSet")


class ArcLineSet(Table):
    DamdClass = LineMeasurements

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
        refLines = ReferenceLineSet.empty()
        if fiberId is not None:
            select = self.fiberId == fiberId
            for args in zip(self.description[select], self.wavelength[select], self.flux[select],
                            self.status[select],
                            self.transition[select],
                            self.source[select]):
                refLines.append(*args)
        else:
            unique = set(zip(self.wavelength, self.description, self.status,
                             self.transition, self.source))
            for wavelength, description, status, transition, source in sorted(unique):
                select = ((self.description == description) & (self.wavelength == wavelength) &
                          (self.status == status) & np.isfinite(self.flux))

                intensity = np.average(self.flux[select]) if np.any(select) else np.nan
                refLines.append(
                    ReferenceLine(
                        description=description,
                        wavelength=wavelength,
                        intensity=intensity,
                        status=status,
                        transition=transition,
                        source=source
                    )
                )
        return refLines

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
        flux[fiberIndices, wlIndices] = self.flux
        variance[fiberIndices, wlIndices] = self.flux**2
        mask[fiberIndices, wlIndices] = self.status

        return PfsFiberArraySet(identity, fiberId, wavelength, flux, mask, sky, norm, covar, flags, metadata)

    @property  # type: ignore
    @deprecated(reason="The 'intensity' attribute has been replaced by 'flux'")
    def intensity(self):
        return self.flux

    @property  # type: ignore
    @deprecated(reason="The 'intensityErr' attribute has been replaced by 'fluxErr'")
    def intensityErr(self):
        return self.fluxErr


ArcLine = ArcLineSet.RowClass
