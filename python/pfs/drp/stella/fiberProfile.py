import numpy as np
import scipy

from lsst.geom import Box2I, Point2I
from lsst.afw.image import MaskedImageF
from pfs.drp.stella.spline import SplineF
from pfs.drp.stella.FiberTraceContinued import FiberTrace
from pfs.drp.stella.profile import calculateSwathProfile

import lsstDebug

__all__ = ("FiberProfile",)


class FiberProfile:
    """A profile of the fiber in the spatial dimension

    The fiber profile is represented as a series of empirical (i.e., pixellated)
    oversampled measurements for a number of "swaths" (a contiguous set of
    rows).

    Parameters
    ----------
    radius : `int`
        Distance either side (i.e., a half-width) of the center the profile is
        measured for.
    oversample : `float`
        Oversample factor for the profile.
    rows : array_like of `float`, shape ``(N,)``
        Average row value for the swath.
    profiles : array_like of `float`, shape ``(N, M)``
        Profiles for each swath, each of width
        ``M = 2*(radius + 1)*oversample + 1``.
    norm : array_like of `float`, optional
        Normalisation for each spectral pixel.
    """
    def __init__(self, radius, oversample, rows, profiles, norm=None):
        self.radius = radius
        self.oversample = oversample
        self.rows = rows
        self.profiles = profiles
        profileSize = 2*(radius + 1)*oversample + 1
        profileCenter = (radius + 1)*oversample
        self.index = (np.arange(profileSize, dtype=int) - profileCenter)/self.oversample
        self.norm = norm

    def __reduce__(self):
        """Pickling"""
        return self.__class__, (self.radius, self.oversample, self.rows, self.profiles, self.norm)

    @classmethod
    def fromImage(cls, maskedImage, centerFunc, radius, oversample, swathSize,
                  rejIter=2, rejThresh=3.0, masks=None):
        """Construct a `FiberProfile` from measuring an image

        This uses the same algorithm as ``findAndTraceApertures()``.

        The profile for each row is constructed in an oversampled pixel space,
        relative the known center for each row. We combine the values within
        each swath to produce an average profile.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImageF`
            Image from which to measure the fiber profile.
        centerFunc : callable
            A callable that provides the center of the trace as a function of
            row.
        radius : `int`
            Distance either side (i.e., a half-width) of the center the profile
            is measured for.
        oversample : `float`
            Oversample factor for the profile.
        swathSize : `float`
            Desired size of swath, in number of rows. The actual swath size used
            will be slightly different, to fit the total number of rows.
        rejIter : `int`
            Number of rejection iterations when combining profiles in a swath.
        rejThresh : `float`
            Rejection threshold (sigma) when combining profiles in a swath.
        masks : iterable of `str`
            Mask planes to ignore.

        Returns
        -------
        self : `FiberProfile`
            Measured fiber profile.
        """
        height = maskedImage.getHeight()
        width = maskedImage.getWidth()
        badBitmask = maskedImage.mask.getPlaneBitMask(masks) if masks is not None else 0

        profileSize = 2*(radius + 1)*oversample + 1  # radius+1 gives us a buffer on either side
        profileCenter = (radius + 1)*oversample

        # Interleave swaths by half, so there's twice as many as you would expect if they didn't interleave.
        # Minimum of four bounds produces two swaths, so we can interpolate.
        numSwaths = max(4, int(np.ceil(2*height/swathSize)))
        bounds = np.linspace(0, height - 1, numSwaths, dtype=int)

        # Combine profiles within each swath
        profileList = []
        yProfile = []
        for yMin, yMax in zip(bounds[:-2], bounds[2:]):
            yNum = yMax - yMin + 1
            swathImage = np.zeros((yNum, profileSize), dtype=float)
            swathMask = np.ones_like(swathImage, dtype=bool)
            columns = np.arange(-radius, radius + 1, dtype=int)
            rows = np.arange(yMin, yMax + 1, dtype=int)
            xCenter = centerFunc(rows)

            xx, yy = np.meshgrid(columns, rows)
            xFrom = xx + np.floor(xCenter + 0.5).astype(np.int)[:, np.newaxis]
            select = (xFrom >= 0) & (xFrom < width)
            values = maskedImage.image.array[yy[select], xFrom[select]]
            bad = (maskedImage.mask.array[yy[select], xFrom[select]] & badBitmask) != 0
            xTo = np.rint((xFrom - xCenter[:, np.newaxis])*oversample).astype(int) + profileCenter

            yTo = yy - yMin
            swathImage[yTo[select], xTo[select]] = values
            swathMask[yTo[select], xTo[select]] = bad
            swath = np.ma.array(swathImage, mask=swathMask)
            swath /= swath.sum(axis=1)[:, np.newaxis]

            profileData, profileMask = calculateSwathProfile(swath.data, swath.mask, rejIter, rejThresh)
            profile = np.ma.masked_array(profileData, mask=profileMask)

            if lsstDebug.Info(__name__).plotSamples:
                import matplotlib.pyplot as plt
                xProf = (np.arange(profileSize) - profileCenter)/oversample
                for array in swath:
                    plt.plot(xProf[~array.mask], array.compressed(), 'k.')
                    if np.any(array.mask):
                        plt.plot(xProf[array.mask], array.data[array.mask], 'r.')
                plt.plot(xProf[~profile.mask], profile.compressed(), 'b.')
                plt.show()

            allBad = np.logical_and.reduce(swath.mask, axis=1)
            yAverage = np.mean(np.arange(yMin, yMax + 1, dtype=int)[~allBad])

            if not np.all(allBad):
                profileList.append(profile)
                yProfile.append(yAverage)

        if len(profileList) == 0:
            raise RuntimeError("No profiles found")

        return cls(radius, oversample, np.array(yProfile), np.ma.masked_array(profileList))

    def plot(self, show=True, annotate=True):
        """Plot a fiber profile

        Produces a set of subplots on a single figure, showing the measured
        profile for each swath.

        Parameters
        ----------
        show : `bool`
            Show the plot?
        annotate : `bool`
            Annotate each swath with the mean and stdev; and draw a Gaussian?

        Returns
        -------
        fig : `matplotlib.pyplot.Figure`
            Figure containing the plot
        axes : `list` of `list` of `matplotlib.pyplot.Axes`
            Grid of axes (from ``matplotlib.pyplot.subplots``). Some may be
            blank.
        """
        import matplotlib.pyplot as plt
        numCols = int(np.ceil(np.sqrt(len(self.profiles))))
        numRows = int(np.ceil(len(self.profiles)/numCols))
        fig, axes = plt.subplots(numRows, numCols, sharex=True, sharey=True, squeeze=False)
        fig.subplots_adjust(hspace=0, wspace=0)

        for prof, yProf, ax in zip(self.profiles, self.rows, sum(axes.tolist(), [])):
            mean = np.sum(prof*self.index)/np.sum(prof)
            rms = np.sqrt(np.sum(prof*(self.index - mean)**2/np.sum(prof)))

            text = r"$\bar{y} = %d$" % yProf
            if annotate:
                text += "\n" + "\n".join((
                    r"$\mu = %.2f$" % mean,
                    r"$\sigma = %.1f$" % rms,
                ))

            ax.plot(self.index, prof, 'k.')
            if annotate:
                ax.plot(self.index, np.exp(-0.5*((self.index - mean)/rms)**2)/rms/np.sqrt(2*np.pi), 'r-')
            ax.axvline(0.0, ls=":", color="blue")
            ax.axhline(0.0, ls=":", color="black")
            ax.text(0.05, 0.95, text, fontsize=6, horizontalalignment="left",
                    verticalalignment="top", transform=ax.transAxes)

        if show:
            plt.show()
        return fig, axes

    def makeFiberTraceFromDetectorMap(self, detectorMap, fiberId):
        """Make a FiberTrace object

        We put the profile into an image, which is used for the FiberTrace.
        The position of the trace as a function of row comes from the
        detectorMap's xCenter.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to/from detector x,y.
        fiberId : `int`
            Fiber identifier.

        Returns
        -------
        fiberTrace : `pfs.drp.stella.FiberTrace`
            A pixellated version of the profile, at a fixed trace position.
        """
        dimensions = detectorMap.bbox.getDimensions()
        rows = np.arange(dimensions.getHeight(), dtype=np.float32)
        xCenter = detectorMap.getXCenter(fiberId)
        centerFunc = SplineF(rows, xCenter)
        return self.makeFiberTrace(dimensions, centerFunc)

    def makeFiberTrace(self, dimensions, centerFunc, fiberId):
        """Make a FiberTrace object

        We put the profile into an image, which is used for the FiberTrace.

        Parameters
        ----------
        dimensions : `lsst.geom.Extent2I`
            Dimensions of the image.
        centerFunc : callable
            A callable that provides the center of the trace as a function of
            row.
        fiberId : `int`
            Fiber identifier.

        Returns
        -------
        fiberTrace : `pfs.drp.stella.FiberTrace`
            A pixellated version of the profile, at a fixed trace position.
        """
        width, height = dimensions
        rows = np.arange(height, dtype=int)
        xCen = centerFunc(rows)
        xMin = max(0, int(np.min(xCen)) - self.radius)
        xMax = min(width, int(np.ceil(np.max(xCen))) + self.radius)
        xx = np.arange(xMin, xMax + 1, dtype=int)
        xImg = xx - xMin
        box = Box2I(Point2I(xMin, 0), Point2I(xMax, height - 1))
        image = MaskedImageF(box)

        # Interpolate in y: find the two closest swaths, and determine the weighting to do the interpolation
        indices = np.arange(len(self.rows), dtype=int)
        nextIndex = scipy.interpolate.interp1d(self.rows, indices, kind="next",
                                               fill_value="extrapolate")(rows).astype(int)
        prevIndex = scipy.interpolate.interp1d(self.rows, indices, kind="previous",
                                               fill_value="extrapolate")(rows).astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            # nextIndex == prevIndex if we're extrapolating
            nextWeight = np.where(nextIndex == prevIndex, 0.5,
                                  (rows - self.rows[prevIndex])/(self.rows[nextIndex] - self.rows[prevIndex]))
        prevWeight = 1.0 - nextWeight

        # Interpolate in x: spline the profile for each swath, and combine with the appropriate weighting
        xProf = self.index.astype(np.float32)
        profiles = self.profiles.astype(np.float32)
        good = ~self.profiles.mask
        splines = [SplineF(xProf[gg], prof.data[gg], SplineF.NATURAL) for
                   prof, gg in zip(profiles, good)]
        xLow = np.array([xProf[gg][0] for gg in good])
        xHigh = np.array([xProf[gg][-1] for gg in good])

        def getProfile(xx, index):
            """Generate a profile, interpolating in the spatial dimension

            Parameters
            ----------
            xx : array_like, shape ``(N,)``
                Positions in the spatial dimension.
            index : `int`
                Index of the profile to use (depends on position in the spectral
                dimension).

            Returns
            -------
            values : `numpy.ndarray`, shape ``(N,)``
                Interpolated values of ``profile[index]`` at positions ``xx``.
            """
            result = np.zeros_like(xx)
            inBounds = (xx >= xLow[index]) & (xx <= xHigh[index])
            result[inBounds] = splines[index](xx[inBounds])
            return result

        for yy in rows:
            xRel = (xx - xCen[yy]).astype(np.float32)  # x position on image relative to center of trace
            nextProfile = getProfile(xRel, nextIndex[yy])
            prevProfile = getProfile(xRel, prevIndex[yy])
            image.image.array[yy, xImg] = nextProfile*nextWeight[yy] + prevProfile*prevWeight[yy]

        # Set normalisation to what is desired
        if self.norm is not None:
            scale = (self.norm/image.image.array.sum(axis=1))[:, np.newaxis]
            image.image.array *= scale
            image.variance.array *= scale**2

        # Deselect bad pixels and bad rows
        ftBitMask = 2**image.mask.addMaskPlane("FIBERTRACE")
        norm = image.image.array.sum(axis=1)
        good = (image.image.array > 0) | (norm[:, np.newaxis] > 0)
        image.mask.array[:] = np.where(good, ftBitMask, 0)

        trace = FiberTrace(image)
        trace.fiberId = fiberId

        return trace

    def extractSpectrum(self, maskedImage, detectorMap, fiberId):
        """Extract a single spectrum from an image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image containing the spectrum.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to/from detector x,y.
        fiberId : `int`
            Fiber identifier.

        Returns
        -------
        spectrum : `pfs.drp.stella.Spectrum`
            Extracted spectrum.
        """
        trace = self.makeFiberTrace(detectorMap, fiberId)
        return trace.extractSpectrum(maskedImage)
