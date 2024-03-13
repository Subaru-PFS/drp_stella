from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

from lsst.afw.image import stripVisitInfoKeywords, setVisitInfoMetadata, VisitInfo, MaskedImageF
from lsst.daf.base import PropertyList

from pfs.datamodel import PfsFiberProfiles, CalibIdentity
from .fiberProfile import FiberProfile
from .FiberTraceSetContinued import FiberTrace, FiberTraceSet
from .spline import SplineD
from .profile import fitSwathProfiles

if TYPE_CHECKING:
    import matplotlib

__all__ = ("FiberProfileSet",)


class FiberProfileSet:
    """A group of `FiberProfile`s, indexed by fiberId

    Parameters
    ----------
    fiberProfiles : `dict` mapping `int` to `pfs.drp.stella.FiberProfile`
        The fiber profiles, indexed by fiber identifier.
    identity : `pfs.datamodel.CalibIdentity`
        Identifying information for the calibration.
    visitInfo : `lsst.afw.image.VisitInfo`, optional
        Parameters characterising the visit.
    metadata : `lsst.daf.base.PropertyList`, optional
        Keyword-value metadata, used for the header.
    """
    def __init__(self, fiberProfiles, identity, visitInfo=None, metadata=None):
        self.fiberProfiles = fiberProfiles
        self.identity = identity
        self.visitInfo = visitInfo
        if metadata is None:
            metadata = PropertyList()
        self.metadata = metadata

    @property
    def hash(self):
        """Return a hash of the fiber profiles"""
        return self.toPfsFiberProfiles().hash

    @classmethod
    def makeEmpty(cls, identity, visitInfo=None, metadata=None):
        """Construct an empty `FiberProfileSet`

        Parameters
        ----------
        identity : `pfs.datamodel.CalibIdentity`
            Identifying information for the calibration.
        visitInfo : `lsst.afw.image.VisitInfo`, optional
            Parameters characterising the visit.
        metadata : `lsst.daf.base.PropertyList`
            Keyword-value metadata, used for the header.

        Returns
        -------
        self : `FiberProfileSet`
            Empty set of fiber profiles.
        """
        return cls({}, identity, visitInfo, metadata)

    @classmethod
    def fromCombination(cls, *profiles):
        """Combine multiple `FiberProfileSet`s into a new one

        The ``identity``, ``visitInfo`` and ``metadata`` from the first of
        the input profiles will be used for the combination.

        Parameters
        ----------
        *profiles : `FiberProfileSet`
            Sets of fiber profiles.
        """
        if len(profiles) == 0:
            raise RuntimeError("No profiles provided")
        self = cls.makeEmpty(profiles[0].identity, profiles[0].visitInfo, profiles[0].metadata)
        for ii, pp in enumerate(profiles):
            have = set(self.fiberId)
            new = set(pp.fiberId)
            if not have.isdisjoint(new):
                raise RuntimeError(f"Duplicate fiberIds for input {ii}: {have.intersection(new)}")
            self.update(pp)
        return self

    @classmethod
    def fromImages(
        cls,
        identity: CalibIdentity,
        imageList: Iterable[MaskedImageF],
        fiberId: Any,
        centerList: Iterable[Any],
        normList: Iterable[Any],
        radius: int,
        oversample: int,
        swathSize: float,
        rejIter: int = 1,
        rejThresh: float = 4.0,
        matrixTol: float = 1.0e-4,
        maskPlanes: Optional[Iterable[str]] = None,
        visitInfo: Optional[VisitInfo] = None,
        metadata: Optional[PropertyList] = None,
    ) -> "FiberProfileSet":
        """Construct a `FiberProfileSet` from measuring images

        The profile for each row is constructed in an oversampled pixel space,
        relative the known center for each row. We combine the values within
        each swath to produce an average profile.

        Parameters
        ----------
        identity : `pfs.datamodel.CalibIdentity`
            Identifying information for the calibration.
        imageList : iterable of `lsst.afw.image.MaskedImageF`
            Images from which to measure the fiber profiles.
        fiberId : `numpy.ndarray` of `int`, shape ``(Nfibers,)``
            Fiber identifiers.
        centerList : iterable of `numpy.ndarray` of `float` with shape ``(Nfibers, Nrows)``
            Arrays that provide the center of the trace as a function of
            row for each fiber in each image.
        normList : iterable of `numpy.ndarray` of `float` with shape ``(Nfibers, Nrows)``
            Arrays that provide the flux of the trace (the normalisation) as a
            function of row for each fiber in each image.
        radius : `int`
            Distance either side (i.e., a half-width) of the center the profile
            is measured for.
        oversample : `int`
            Oversample factor for the profile.
        swathSize : `float`
            Desired size of swath, in number of rows. The actual swath size used
            will be slightly different, to fit the total number of rows.
        rejIter : `int`
            Number of rejection iterations when combining profiles in a swath.
        rejThresh : `float`
            Rejection threshold (sigma) when combining profiles in a swath.
        matrixTol : `float`
            Tolerance for matrix inversion.
        maskPlanes : iterable of `str`
            Mask planes to ignore.
        visitInfo : `lsst.afw.image.VisitInfo`, optional
            Parameters characterising the visit.
        metadata : `lsst.daf.base.PropertyList`, optional
            Keyword-value metadata, used for the header.

        Returns
        -------
        self : `FiberProfileSet`
            Measured fiber profiles.
        """
        # Check consistent image dimensions
        numImages = len(imageList)
        numFibers = fiberId.size
        if numImages == 0:
            raise RuntimeError("No images")
        dims = imageList[0].getDimensions()
        for image in imageList[1:]:
            if dims != image.getDimensions():
                raise RuntimeError(f"Dimension mismatch: {dims} vs {image.getDimensions()}")
        width, height = dims
        badBitmask = imageList[0].mask.getPlaneBitMask(maskPlanes) if maskPlanes is not None else 0

        # Check consistency between imageList, fiberId, centerList
        if len(centerList) != numImages:
            raise RuntimeError(f"Mismatch between number of images ({numImages}) and number of centers "
                               f"{len(centerList)}")
        if len(normList) != numImages:
            raise RuntimeError(f"Mismatch between number of images ({numImages}) and number of spectra "
                               f"{len(normList)}")
        centersShape = (numFibers, height)
        for ii, (centers, norm) in enumerate(zip(centerList, normList)):
            if centers.shape != centersShape:
                raise RuntimeError(f"Bad shape for centers {ii}: {centersShape} vs {centers.shape}")
            if norm.shape != centersShape:
                raise RuntimeError(f"Bad shape for norm {ii}: {centersShape} vs {norm.shape}")

        # Ensure fiberId, centers are sorted with fibers in order from left to right
        indices = np.argsort(centerList[0][:, height//2])
        if not np.all(indices[1:] > indices[:-1]):  # monotonic increasing
            fiberId = fiberId[indices]
            centerList = [centers[indices] for centers in centerList]
            normList = [norm[indices] for norm in normList]

        # Interleave swaths by half, so there's twice as many as you would expect if they didn't interleave.
        # Minimum of four bounds produces two swaths, so we can interpolate.
        numSwaths = max(4, int(np.ceil(2*height/swathSize)))
        bounds = np.linspace(0, height - 1, numSwaths, dtype=int)

        profileList = defaultdict(list)  # List of profiles as a function of fiberId
        yProfile = []  # List of mean row
        for yMin, yMax in zip(bounds[:-2], bounds[2:]):
            profiles, masks = fitSwathProfiles(imageList, centerList, normList, fiberId.astype(np.int32),
                                               yMin, yMax, badBitmask, oversample, radius, rejIter, rejThresh,
                                               matrixTol)
            yProfile.append(0.5*(yMin + yMax))  # XXX this doesn't account for masked rows
            for ff, pp, mm in zip(fiberId, profiles, masks):
                pp = np.ma.MaskedArray(pp, mm)
                profileList[ff].append(pp/np.ma.sum(pp, axis=0))

        return cls({ff: FiberProfile(radius, oversample, np.array(yProfile),
                                     np.ma.masked_array(profileList[ff])) for
                    ff in fiberId}, identity, visitInfo, metadata)

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

    def makeFiberTracesFromDetectorMap(self, detectorMap, boxcarWidth=0, ignoreFiberId=[]):
        """Construct fiber traces using the detectorMap

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y. This is used to provide the
            xCenter of the trace as a function of detector row.
        boxcarWidth: `int`
            Width of boxcar extraction; use fiberProfiles if <= 0
        ignoreFiberId: `list` of `int`
            List (actually iterable) of fiberIds to ignore

        Returns
        -------
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        """
        ignoreFiberId = set(ignoreFiberId)

        if boxcarWidth <= 0:
            rows = np.arange(detectorMap.bbox.getHeight(), dtype=float)
            centers = {fiberId: SplineD(rows, detectorMap.getXCenter(fiberId)) for
                       fiberId in self if fiberId not in ignoreFiberId}
            return self.makeFiberTraces(detectorMap.bbox.getDimensions(), centers)
        else:
            dims = detectorMap.getBBox().getDimensions()

            fiberTraces = FiberTraceSet(len(detectorMap))
            norm = None
            for fiberId in detectorMap.fiberId:
                if fiberId in self.fiberId and fiberId not in ignoreFiberId:
                    centers = detectorMap.getXCenter(fiberId)

                    if norm is None:
                        norm = np.full_like(centers, boxcarWidth)
                    ft = FiberTrace.boxcar(fiberId, dims, boxcarWidth/2, centers, norm=norm)

                    fiberTraces.add(ft)

            return fiberTraces

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
            if fiberId in centers:
                traces.add(self[fiberId].makeFiberTrace(dimensions, centers[fiberId], fiberId))
        return traces

    def extractSpectra(self, maskedImage, detectorMap, badBitMask=0, minFracMask=0.3):
        """Extract spectra from an image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image containing spectra.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        badBitMask : `int`
            Bitmask indicating bad pixels.
        minFracMask : `float`
            Minimum pixel fraction of profile to accumulate mask.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        traces = self.makeFiberTracesFromDetectorMap(detectorMap)
        return traces.extractSpectra(maskedImage, badBitMask, minFracMask)

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

        return cls(profiles, pfsFiberProfiles.identity, visitInfo, metadata)

    def toPfsFiberProfiles(self):
        """Convert to a `pfs.datamodel.PfsFiberProfiles`

        Essentially, this involves reformatting the `dict` of
        `pfs.drp.stella.FiberProfile` objects to a list of profile arrays.

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
        if self.visitInfo is not None:
            setVisitInfoMetadata(metadata, self.visitInfo)

        return PfsFiberProfiles(self.identity, self.fiberId, radius, oversample, rows, profiles, norm,
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
        self.toPfsFiberProfiles().writeFits(filename)

    def plot(
        self,
        rows: int = 10,
        cols: int = 10,
        *,
        fontsize: int = 4,
        show: bool = True,
        fiberId: Optional[List[int]] = None,
    ) -> Dict[int, Tuple["matplotlib.Figure", "matplotlib.Axes"]]:
        """Plot the fiber profiles

        Parameters
        ----------
        rows, cols : `int`, optional
            Number of rows and columns to use.
        fontsize : `int`, optional
            Font size to use for fiberId label.
        show : `bool`, optional
            Show the plots?
        fiberId : list of `int`, optional
            Fiber identifiers to plot. If ``None``, plot all.

        Returns
        -------
        figAxes : `dict`[`fiberId`, (`matplotlib.Figure`, `matplotlib.Axes`)]
            Figures and axes indexed by fiberId.
        """
        import matplotlib.pyplot as plt

        figAxes: Dict[int, Tuple["matplotlib.Figure", "matplotlib.Axes"]] = {}

        if not fiberId:
            fiberId = list(sorted(self.fiberProfiles.keys(), reverse=True))
        while fiberId:
            fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=False)
            fig.subplots_adjust(hspace=0, wspace=0)
            for ff, ax in zip(fiberId, axes.flatten()):
                self.fiberProfiles[ff].plotInAxes(ax)
                ax.text(0.05, 0.95, f"fiberId={ff}", fontsize=fontsize, horizontalalignment="left",
                        verticalalignment="top", transform=ax.transAxes)
                figAxes[ff] = (fig, ax)
            for ax in axes.flatten()[len(fiberId):]:
                ax.axis("off")
            fiberId = fiberId[rows*cols:]

        if show:
            plt.show()
        return figAxes

    def plotHistograms(
        self,
        numBins=50,
        show=False,
        centroidRange=(-0.2, 0.2),
        widthRange=(0.5, 4.0),
        minRange=(-0.01, 0.01),
        maxRange=(0.05, 0.3),
        skewRange=(-1.0, 1.0),
        r90Range=(1, 5),
        wingFracRange=(0, 0.2),
        variationRange=(0, 0.01),
        fiberGridSize=100,
    ):
        """Plot histograms of statistics about the fiber profiles

        Parameters
        ----------
        numBins : `int`, optional
            Number of bins to use in the histograms.
        show : `bool`
            Show the plots?
        centroidRange : `tuple` of `float`, optional
            Minimum and maximum centroid values to plot.
        widthRange : `tuple` of `float`, optional
            Minimum and maximum width values to plot.
        minRange : `tuple` of `float`, optional
            Minimum and maximum minimum values to plot.
        maxRange : `tuple` of `float`, optional
            Minimum and maximum maximum values to plot.
        skewRange : `tuple` of `float`, optional
            Minimum and maximum skew values to plot.
        r90Range : `tuple` of `float`, optional
            Minimum and maximum r90 values to plot.
        wingFracRange : `tuple` of `float`, optional
            Minimum and maximum wingFrac values to plot.
        variationRange : `tuple` of `float`, optional
            Minimum and maximum variation values to plot.

        Returns
        -------
        figAxes : `list`
            List of tuples of `matplotlib.Figure` and arrays of
            `matplotlib.Axes`.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from .fitDistortedDetectorMap import addColorbar

        stats = {fiberId: self[fiberId].calculateStatistics() for fiberId in self}
        fiberGridSize = min(fiberGridSize, len(stats))

        fiberId = np.concatenate([np.full_like(self[ff].rows, ff, dtype=int) for ff in stats])
        rows = np.concatenate([self[ff].rows for ff in stats])
        numRows = max(len(self[ff].rows) for ff in stats)

        centroids = np.concatenate([stats[fiberId].centroid for fiberId in stats])
        widths = np.concatenate([stats[fiberId].width for fiberId in stats])
        minimums = np.concatenate([stats[fiberId].min for fiberId in stats])
        maximums = np.concatenate([stats[fiberId].max for fiberId in stats])
        skew = np.concatenate([stats[fiberId].skew for fiberId in stats])
        r90 = np.concatenate([stats[fiberId].r90 for fiberId in stats])
        wingFrac = np.concatenate([stats[fiberId].wingFrac for fiberId in stats])
        variation = np.concatenate([stats[fiberId].variation for fiberId in stats])

        figAxes = []
        cmap = plt.get_cmap("viridis")
        for name, values, valueRange, description in (
            ("centroid", centroids, centroidRange, "Profile centroid calculated using quartic correction"),
            ("width", widths, widthRange, "Square root of the profile second moment"),
            ("min", minimums, minRange, "Minimum value of the profile"),
            ("max", maximums, maxRange, "Maximum value of the profile"),
            ("skew", skew, skewRange, "Third moment of the profile, normalised by the second moment"),
            ("r90", r90, r90Range, "Half-width enclosing 90% of the profile flux"),
            ("wingFrac", wingFrac, wingFracRange, "Fraction of the profile flux in the wings"),
            ("variation", variation, variationRange, "RMS variation within the profile"),
        ):
            fig, axes = plt.subplots(ncols=2)
            axes[0].hist(values, bins=np.linspace(*valueRange, numBins))
            axes[0].semilogy()
            axes[0].set_xlabel(name)
            axes[0].set_ylabel("Count")

            norm = Normalize(vmin=valueRange[0], vmax=valueRange[1])
            axes[1].hexbin(rows, fiberId, values.T, gridsize=(numRows, fiberGridSize), cmap=cmap, norm=norm)

            axes[1].set_xlabel("row")
            axes[1].set_ylabel("fiberId")
            addColorbar(fig, axes[1], cmap, norm, name)

            fig.suptitle(description)
            fig.tight_layout()
            figAxes.append((fig, axes))

        if show:
            plt.show()

        return figAxes

    def average(self, fiberId: Optional[Iterable[int]] = None) -> "FiberProfile":
        """Return the average of the profiles

        Parameters
        ----------
        fiberId : iterable of `int`, optional
            Fiber identifiers to include. If ``None``, include all.

        Returns
        -------
        average : `FiberProfile`
            Average profile.
        """
        if fiberId is None:
            fiberId = self.fiberId

        radius = set(self[ff].radius for ff in fiberId)
        if len(radius) != 1:
            raise RuntimeError(f"Multiple radii: {radius}")
        oversample = set(self[ff].oversample for ff in fiberId)
        if len(oversample) != 1:
            raise RuntimeError(f"Multiple oversamples: {oversample}")
        shape = self[fiberId[0]].profiles.shape
        for ff in fiberId[1:]:
            if self[ff].profiles.shape != shape:
                raise RuntimeError(f"Profile shape mismatch: {shape} vs {self[ff].profiles.shape}")

        profiles = np.ma.masked_invalid([self[ff].profiles for ff in fiberId]).mean(axis=0)
        return FiberProfile(radius.pop(), oversample.pop(), self[fiberId[0]].rows, profiles)

    def replaceFibers(self, replaceFibers: Iterable[int], nearest: int = 2):
        """Replace profiles of certain fibers

        Some fibers have broken cobras which do not allow them to be moved
        behind their black spot, and hence it can be difficult to measure their
        profiles (especially if they are next to another fiber with a broken
        cobra, since we never get a measurement of one fiber's profile without
        the other's). Here, we replace the profiles of these fibers with the
        average of their nearest neighbors.

        The normalisations of the fibers are NOT replaced; they should be
        measured from data using the new profiles.

        Parameters
        ----------
        replaceFibers : iterable of `int`
            Fiber identifiers for which to replace the profiles. May include
            fibers not present in the set.
        nearest : `int`, optional
            Number of nearest neighbors to use when replacing the profiles.
        """
        badFibers = set(replaceFibers) & set(self.fiberId)
        if not badFibers:
            return
        goodFibers = np.array(list(set(self.fiberId) - set(replaceFibers)))
        for fiberId in badFibers:
            indices = np.argpartition(np.abs(goodFibers - fiberId), np.arange(nearest))[:nearest]
            neighbors = goodFibers[indices]
            self[fiberId] = self.average(neighbors)
