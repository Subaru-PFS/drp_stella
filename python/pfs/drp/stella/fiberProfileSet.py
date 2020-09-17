import io
import numpy as np
import astropy.io.fits

import lsst.afw.fits

from .fiberProfile import FiberProfile
from .FiberTraceSetContinued import FiberTraceSet
from .spline import SplineF

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
        rows = np.arange(detectorMap.bbox.getHeight(), dtype=np.float32)
        centers = {fiberId: SplineF(rows, detectorMap.getXCenter(fiberId)) for fiberId in self}
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
        fits = astropy.io.fits.open(filename)

        hdu = fits["FIBERS"]
        fiberId1 = hdu.data["fiberId"]
        radius = hdu.data["radius"]
        oversample = hdu.data["oversample"]
        norm = hdu.data["norm"]

        hdu = fits["PROFILES"]
        fiberId2 = hdu.data["fiberId"]
        rows = hdu.data["rows"]
        profiles = hdu.data["profiles"]
        masks = hdu.data["masks"]

        numFibers = len(fiberId1)
        fiberProfiles = {}
        for ii in range(numFibers):
            fiberId = fiberId1[ii]
            select = fiberId2 == fiberId
            prof = np.ma.masked_array(np.array(profiles[select].tolist()),
                                      mask=(np.array(masks[select].tolist(), dtype=bool) if
                                            masks[select].size > 0 else None))
            fiberProfiles[fiberId] = FiberProfile(radius[ii], oversample[ii], rows[select], prof,
                                                  norm[ii] if norm[ii].size > 0 else None)

        # Read the primary header with lsst.afw.fits
        # This requires writing the FITS file into memory and reading it from there
        buffer = io.BytesIO()
        fits.writeto(buffer)
        ss = buffer.getvalue()
        size = len(ss)
        ff = lsst.afw.fits.MemFileManager(size)
        ff.setData(ss, size)
        metadata = ff.readMetadata(0)

        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(fiberProfiles, visitInfo, metadata)

    def writeFits(self, filename):
        """Write to a FITS file

        Parameters
        ----------
        filename : `str`
            Name of FITS file.
        """
        radius = [self[fiberId].radius for fiberId in self]
        oversample = [self[fiberId].oversample for fiberId in self]
        rows = [self[fiberId].rows for fiberId in self]
        norm = [self[fiberId].norm if self[fiberId].norm is not None else [] for fiberId in self]

        date = self.getVisitInfo().getDate()
        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)
        header = astropy.io.fits.Header()
        for key in metadata.names():
            header[key] = metadata.get(key)

        header["INHERIT"] = True
        header["OBSTYPE"] = "fiberProfiles"
        header["HIERARCH calibDate"] = date.toPython(date.UTC).strftime("%Y-%m-%d")

        fibersHdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column("fiberId", format="J", array=self.fiberId),
            astropy.io.fits.Column("radius", format="J", array=radius),
            astropy.io.fits.Column("oversample", format="E", array=oversample),
            astropy.io.fits.Column("norm", format="PE()", array=norm),
        ], header=header.copy(), name="FIBERS")

        numProfiles = sum(len(self[fiberId].rows) for fiberId in self)
        fiberId = np.zeros(numProfiles, dtype=int)
        rows = np.zeros(numProfiles, dtype=float)
        profiles = []
        masks = []
        start = 0
        for ff in self:
            prof = self[ff]
            num = len(prof.rows)
            fiberId[start:start + num] = ff
            rows[start:start + num] = prof.rows
            for ii in range(num):
                profiles.append(prof.profiles[ii])
                if isinstance(prof.profiles, np.ma.MaskedArray):
                    masks.append(prof.profiles.mask[ii])
                else:
                    masks.append([])
            start += num

        profilesHdu = astropy.io.fits.BinTableHDU.from_columns([
            astropy.io.fits.Column("fiberId", format="J", array=fiberId),
            astropy.io.fits.Column("rows", format="E", array=rows),
            astropy.io.fits.Column("profiles", format="PE()", array=profiles),
            astropy.io.fits.Column("masks", format="PL()", array=masks),
        ], header=header.copy(), name="PROFILES")

        fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(header=header.copy()),
                                        fibersHdu, profilesHdu])
        with open(filename, "wb") as fd:
            fits.writeto(fd)
