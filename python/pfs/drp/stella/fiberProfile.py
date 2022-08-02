import numpy as np

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
    norm : array_like of `np.float32`, optional
        Normalisation for each spectral pixel.
    """
    def __init__(self, radius, oversample, rows, profiles, norm=None):
        self.radius = radius
        self.oversample = oversample
        self.rows = rows
        self.profiles = profiles
        profileSize = 2*int((radius + 1)*oversample + 0.5) + 1
        profileCenter = int((radius + 1)*oversample + 0.5)
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
            xFrom = xx + np.floor(xCenter + 0.5).astype(np.int32)[:, np.newaxis]
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
            if np.sum(~profileMask) < 3:
                # Not enough points to form a spline
                continue
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
            if not np.all(allBad):
                yAverage = np.mean(np.arange(yMin, yMax + 1, dtype=int)[~allBad])
            else:
                yAverage = np.nan

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

    def __makeFiberTrace(self, dimensions, xCenter, fiberId):
        """Make a FiberTrace object

        Helper function for makeFiberTrace/makeFiberTraceFromDetectorMap to get the numpy types right
        """
        return FiberTrace.fromProfile(fiberId, dimensions, self.radius, self.oversample, self.rows,
                                      self.profiles.data, ~self.profiles.mask, xCenter, self.norm)

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
        xCenter = detectorMap.getXCenter(fiberId)

        return self.__makeFiberTrace(dimensions, xCenter, fiberId)

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
        rows = np.arange(dimensions.getY(), dtype=float)
        xCenter = centerFunc(rows)
        return self.__makeFiberTrace(dimensions, xCenter, fiberId)

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
        trace = self.makeFiberTraceFromDetectorMap(detectorMap, fiberId)
        return trace.extractSpectrum(maskedImage)
