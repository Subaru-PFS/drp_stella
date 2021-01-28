import numpy as np

from lsst.afw.image import stripVisitInfoKeywords, setVisitInfoMetadata, VisitInfo
from lsst.daf.base import PropertyList

from pfs.datamodel import PfsFiberProfiles
from .fiberProfile import FiberProfile
from .FiberTraceSetContinued import FiberTraceSet
from .spline import SplineD

__all__ = ("FiberProfileSet",)


class FiberProfileSet:
    """A group of `FiberProfile`s, indexed by fiberId

    Parameters
    ----------
    fiberProfiles : `dict` mapping `int` to `pfs.drp.stella.FiberProfile`
        The fiber profiles, indexed by fiber identifier.
    visitInfo : `lsst.afw.image.VisitInfo`, optional
        Parameters characterising the visit.
    metadata : `lsst.daf.base.PropertyList`
        Keyword-value metadata, used for the header.
    """
    def __init__(self, fiberProfiles, visitInfo=None, metadata=None):
        self.fiberProfiles = fiberProfiles
        self.visitInfo = visitInfo
        self.metadata = metadata

    @classmethod
    def makeEmpty(cls, visitInfo=None, metadata=None):
        """Construct an empty `FiberProfileSet`

        Parameters
        ----------
        visitInfo : `lsst.afw.image.VisitInfo`, optional
            Parameters characterising the visit.
        metadata : `lsst.daf.base.PropertyList`
            Keyword-value metadata, used for the header.

        Returns
        -------
        self : `FiberProfileSet`
            Empty set of fiber profiles.
        """
        return cls({}, visitInfo, metadata)

    @classmethod
    def fromCombination(cls, *profiles):
        """Combine multiple `FiberProfileSet`s into a new one

        Parameters
        ----------
        *profiles : `FiberProfileSet`
            Sets of fiber profiles.
        """
        if len(profiles) == 0:
            raise RuntimeError("No profiles provided")
        self = cls.makeEmpty(profiles[0].visitInfo, profiles[0].metadata)
        for ii, pp in enumerate(profiles):
            have = set(self.fiberId)
            new = set(pp.fiberId)
            if not have.isdisjoint(new):
                raise RuntimeError(f"Duplicate fiberIds for input {ii}: {have.intersection(new)}")
            self.update(pp)
        return self

    @property
    def fiberId(self):
        """Return the fiber identifiers

        The list is sorted, for consistency of the order.
        """
        return np.array(sorted(self.fiberProfiles.keys()))

    def __len__(self):
        return len(self.fiberProfiles)

    def __getitem__(self, fiberId):
        return self.fiberProfiles[fiberId]

    def __setitem__(self, fiberId, fiberProfile):
        self.fiberProfiles[fiberId] = fiberProfile

    def __iter__(self):
        return iter(sorted(self.fiberProfiles.keys()))

    def __contains__(self, fiberId):
        return fiberId in self.fiberProfiles

    def update(self, other):
        """Update the profiles from another set of profiles

        The set of profiles is extended (any existing fiberIds are overwritten),
        but the ``visitInfo`` and ``metadata`` are not.

        Parameters
        ----------
        other : `pfs.drp.stella.FiberProfileSet`
            Set of fiber profiles to include.
        """
        self.fiberProfiles.update(other.fiberProfiles)

    def getMetadata(self):
        """Return the metadata

        This interface is included to duck-type an `lsst.afw.image.Exposure`.
        """
        return self.metadata

    def getVisitInfo(self):
        """Return the visitInfo

        This interface is included to duck-type an `lsst.afw.image.Exposure`.
        """
        return self.visitInfo

    def makeFiberTracesFromDetectorMap(self, detectorMap):
        """Construct fiber traces using the detectorMap

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y. This is used to provide the
            xCenter of the trace as a function of detector row.

        Returns
        -------
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        """
        rows = np.arange(detectorMap.bbox.getHeight(), dtype=float)
        centers = {fiberId: SplineD(rows, detectorMap.getXCenter(fiberId)) for fiberId in self}
        return self.makeFiberTraces(detectorMap.bbox.getDimensions(), centers)

    def makeFiberTraces(self, dimensions, centers):
        """Construct fiber traces

        Parameters
        ----------
        dimensions : `lsst.geom.Extent2I`
            Dimensions of the image.
        centers : `dict` mapping `int` to callable
            Callables, indexed by fiberId, that provide the center of the trace
            as a function of row for each fiber.

        Returns
        -------
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        """
        traces = FiberTraceSet(len(self), self.metadata)
        for fiberId in self:
            traces.add(self[fiberId].makeFiberTrace(dimensions, centers[fiberId], fiberId))
        return traces

    def extractSpectra(self, maskedImage, detectorMap, badBitMask=0):
        """Extract spectra from an image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image containing spectra.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        badBitMask : `int`
            Bitmask indicating bad pixels.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        traces = self.makeFiberTracesFromDetectorMap(detectorMap)
        return traces.extractSpectra(maskedImage, badBitMask)

    @classmethod
    def fromPfsFiberProfiles(cls, pfsFiberProfiles):
        """Convert from a `pfs.datamodel.PfsFiberProfiles`

        Essentially, this involves reformatting the list of profile arrays to
        a `dict` of `pfs.drp.stella.FiberProfile` objects indexed by
        ``fiberId``.

        Parameters
        ----------
        pfsFiberProfiles : `pfs.datamodel.PfsFiberProfiles`
            Datamodel version of fiber profiles to convert.

        Returns
        -------
        self : cls
            Fiber profiles in ``drp_stella`` format.
        """
        numFibers = len(pfsFiberProfiles)
        profiles = {}
        for ii in range(numFibers):
            fiberId = pfsFiberProfiles.fiberId[ii]
            profiles[fiberId] = FiberProfile(
                pfsFiberProfiles.radius[ii], pfsFiberProfiles.oversample[ii],
                pfsFiberProfiles.rows[ii], pfsFiberProfiles.profiles[ii],
                pfsFiberProfiles.norm[ii] if pfsFiberProfiles.norm[ii].size > 0 else None)

        metadata = PropertyList()
        for key, value in pfsFiberProfiles.metadata.items():
            metadata.set(key, value)

        visitInfo = VisitInfo(metadata)
        stripVisitInfoKeywords(metadata)

        return cls(profiles, visitInfo, metadata)

    def toPfsFiberProfiles(self, identity):
        """Convert to a `pfs.datamodel.PfsFiberProfiles`

        Essentially, this involves reformatting the `dict` of
        `pfs.drp.stella.FiberProfile` objects to a list of profile arrays.

        Parameters
        ----------
        identity : `pfs.datamodel.CalibIdentity`
            Identity of the calib data.

        Returns
        -------
        pfsFiberProfiles : `pfs.datamodel.PfsFiberProfiles`
            Fiber profiles in ``datamodel`` format.
        """
        radius = [self[fiberId].radius for fiberId in self]
        oversample = [self[fiberId].oversample for fiberId in self]
        rows = [self[fiberId].rows for fiberId in self]
        profiles = [self[fiberId].profiles for fiberId in self]
        norm = [self[fiberId].norm if self[fiberId].norm is not None else [] for fiberId in self]

        metadata = self.metadata.deepCopy()
        setVisitInfoMetadata(metadata, self.visitInfo)

        return PfsFiberProfiles(identity, self.fiberId, radius, oversample, rows, profiles, norm,
                                metadata.toDict())

    @classmethod
    def readFits(cls, filename):
        """Read from a FITS file

        Parameters
        ----------
        filename : `str`
            Name of FITS file.

        Returns
        -------
        self : `FiberProfileSet`
            Fiber profiles read from FITS file.
        """
        profiles = PfsFiberProfiles.readFits(filename)
        return cls.fromPfsFiberProfiles(profiles)

    def writeFits(self, filename):
        """Write to a FITS file

        Parameters
        ----------
        filename : `str`
            Name of FITS file.
        """
        identity = PfsFiberProfiles.parseFilename(filename)
        profiles = self.toPfsFiberProfiles(identity)
        profiles.writeFits(filename)
