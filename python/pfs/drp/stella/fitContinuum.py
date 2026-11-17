from contextlib import contextmanager
import numpy as np

import lsstDebug

from lsst.pex.config import Config, Field, ChoiceField, ListField
from lsst.pipe.base import Task, Struct
from lsst.afw.math import stringToInterpStyle, makeInterpolate
from pfs.datamodel.pfsFiberArraySet import PfsFiberArraySet
from pfs.drp.stella import Spectrum, SpectrumSet
from pfs.drp.stella.maskLines import maskLines
from .interpolate import interpolateIndices
from .referenceLine import ReferenceLineSet

from typing import Tuple

__all__ = ("FitContinuumConfig", "FitContinuumTask", "FitContinuumError")


class FitContinuumError(RuntimeError):
    """Error when fitting continuum"""

    pass


class FitContinuumConfig(Config):
    """Configuration for SubtractContinuumTask"""

    fitType = ChoiceField(
        dtype=str,
        default="AKIMA_SPLINE",
        doc="Functional form for fit",
        allowed={
            "NATURAL_SPLINE": "Natural spline",
            "CUBIC_SPLINE": "Cubic spline",
            "AKIMA_SPLINE": "Akima spline",
        },
    )
    numKnots = Field(dtype=int, default=30, doc="Number of spline knots")
    knotPosition = ChoiceField(
        dtype=str,
        default="median",
        doc="Position of each spline knot within its bin",
        allowed={
            "median": "Median position of the good pixels in the bin",
            "fixed": "Nominal position of the knot, regardless of masking",
        },
    )
    iterations = Field(dtype=int, default=3, doc="Number of fitting iterations")
    rejection = Field(dtype=float, default=3.0, doc="Rejection threshold (standard deviations)")
    doMaskLines = Field(dtype=bool, default=True, doc="Mask reference lines before fitting?")
    maskLineRadius = Field(dtype=int, default=5, doc="Number of pixels either side of reference line to mask")
    mask = ListField(
        dtype=str, default=["BAD", "CR", "NO_DATA", "SUSPECT", "BAD_FLAT"], doc="Mask planes to ignore"
    )
    telluricCenter = ListField(
        # A: 758.2 to 773.0 nm
        # B: 686.5 to 697.5 nm
        dtype=float, default=[765.6, 692.0], doc="Center of telluric absorption bands (nm)"
    )
    telluricWidth = ListField(
        dtype=float, default=[8.0, 6.0], doc="Half-width of telluric absorption bands (nm)"
    )

    def validate(self):
        """Validate configuration"""
        super().validate()
        if len(self.telluricCenter) != len(self.telluricWidth):
            raise RuntimeError("telluricCenter and telluricWidth must have the same length")


class FitContinuumTask(Task):
    """Subtract continuum from spectra

    Debug settings:
    - ``plot`` (`bool`): activate plotting
    - ``plotAll`` (`bool`): plot all data (even rejected)? Otherwise plot
        binned data.
    - ``plotBins`` (`int`): number of bins if not ``plotAll`` (default 1000).
    """

    ConfigClass = FitContinuumConfig
    _DefaultName = "fitContinuum"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitType = stringToInterpStyle(self.config.fitType)

    def run(
        self, spectra: SpectrumSet | PfsFiberArraySet, lines: ReferenceLineSet | None = None
    ) -> SpectrumSet:
        """Fit spectrum continua

        Fit the continuum for each spectrum.

        If the spectra have wavelengths, the spline knots are placed at the
        same wavelengths for all spectra (``numKnots`` knots evenly spaced
        over the full wavelength range of the spectra), so that the continua
        are consistent from spectrum to spectrum.

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`/`pfs.datamodel.PfsFiberArraySet`
            Set of spectra to which to fit continua.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.

        Returns
        -------
        continuum : `np.ndarray` of `float`
            Continuum fit for each input spectrum.
        """
        if isinstance(spectra, PfsFiberArraySet):
            spectra = SpectrumSet.fromPfsArm(spectra)

        knotWavelengths = self.getKnotWavelengths(spectra)
        continua = np.full((len(spectra), spectra.getLength()), np.nan, dtype=np.float32)
        for ii, spec in enumerate(spectra):
            knots = None
            if knotWavelengths is not None and spec.isWavelengthSet():
                knots = wavelengthToPixels(spec.wavelength, knotWavelengths)
            try:
                continua[ii] = self.fitContinuum(spec, lines, knots=knots)
            except FitContinuumError:
                continue
        return continua

    def getKnotWavelengths(self, spectra: SpectrumSet) -> np.ndarray | None:
        """Get wavelengths of spline knots common to a set of spectra

        The knots are evenly spaced over the full wavelength range of the
        spectra, at the centers of ``numKnots`` bins.

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Spectra for which to determine knots.

        Returns
        -------
        knotWavelengths : `numpy.ndarray` of `float`, or `None`
            Wavelengths of knots (nm), or `None` if no spectrum has
            wavelengths.
        """
        minWavelength = np.inf
        maxWavelength = -np.inf
        for spec in spectra:
            if not spec.isWavelengthSet():
                continue
            wavelength = spec.wavelength
            wavelength = wavelength[np.isfinite(wavelength)]
            if wavelength.size == 0:
                continue
            minWavelength = min(minWavelength, wavelength.min())
            maxWavelength = max(maxWavelength, wavelength.max())
        if not np.isfinite(minWavelength) or not np.isfinite(maxWavelength) or maxWavelength <= minWavelength:
            return None
        binWidth = (maxWavelength - minWavelength) / self.config.numKnots
        return minWavelength + (np.arange(self.config.numKnots) + 0.5) * binWidth

    def fitContinuum(self, spectrum, lines=None, *, knots=None):
        """Fit continuum to the spectrum

        Uses ``lsst.afw.math.Interpolate`` to fit, and performs iterative
        rejection. Optionally masks identified reference lines.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum to fit.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.
        knots : array_like of `float`, optional
            Nominal positions of the spline knots, in pixels. Each knot's bin
            extends halfway to its neighbours (and to the ends of the
            spectrum for the first and last knots). If not provided, we use
            ``numKnots`` bins of equal size.

        Raises
        ------
        FitContinuumError
            If we had no good values.

        Returns
        -------
        continuum : `numpy.ndarray`
            Array of continuum fit.
        """
        if hasattr(spectrum, "normFlux"):
            flux = spectrum.normFlux
        else:
            flux = spectrum.flux
            if hasattr(flux, "norm"):
                flux /= flux.norm  # Normalize the flux to 1.0
        good = np.isfinite(flux)
        if self.config.doMaskLines and lines and spectrum.isWavelengthSet():
            good &= ~maskLines(spectrum.wavelength, lines.wavelength, self.config.maskLineRadius)
        if self.config.telluricCenter and spectrum.isWavelengthSet():
            for center, width in zip(self.config.telluricCenter, self.config.telluricWidth):
                good &= np.abs(spectrum.wavelength - center) >= width
        good &= (spectrum.mask.array[0] & spectrum.mask.getPlaneBitMask(self.config.mask)) == 0
        if not np.any(good):
            raise FitContinuumError("No good values when fitting continuum")
        keep = np.ones_like(good, dtype=bool)
        for ii in range(self.config.iterations):
            use = good & keep
            fit = self._fitContinuumImpl(flux, use, knots)
            diff = flux - fit
            lq, uq = np.percentile(diff[use], [25.0, 75.0])
            stdev = 0.741 * (uq - lq)
            with np.errstate(invalid="ignore"):
                keep = np.isfinite(diff) & (np.abs(diff) <= self.config.rejection * stdev)
        return self._fitContinuumImpl(flux, good & keep, knots)

    def _fitContinuumImpl(self, values, good, knots=None):
        """Implementation of the business part of fitting

        Parameters
        ----------
        values : `numpy.ndarray`, floating-point
            Spectrum array to fit.
        good : `numpy.ndarray`, boolean
            Boolean array indicating which points are good.
        knots : `numpy.ndarray` of `float`, optional
            Nominal positions of the spline knots, in pixels.

        Raises
        ------
        FitContinuumError
            If we had no good knots.

        Returns
        -------
        fit : `numpy.ndarray`, floating-point
            Fit array.
        """
        indices = np.arange(len(values), dtype=values.dtype)
        fixed = self.config.knotPosition == "fixed"
        if knots is None and not fixed:
            xKnots, binned = binData(indices, values, good, self.config.numKnots)
        else:
            if knots is None:
                # Centers of bins of equal size
                knots = (np.arange(self.config.numKnots) + 0.5) * len(values) / self.config.numKnots - 0.5
            xKnots, binned = binDataByKnots(indices, values, good, knots, fixed)
        use = np.isfinite(xKnots) & np.isfinite(binned)
        if not np.any(use):
            raise FitContinuumError("No finite knots when fitting continuum")

        try:
            interp = makeInterpolate(xKnots[use], binned[use], self.fitType)
        except Exception as e:
            msg = "Fitting continuum: " + ", ".join(e.args)
            self.log.warn(msg)
            raise FitContinuumError(msg)

        fit = np.array(interp.interpolate(indices)).astype(values.dtype)

        if lsstDebug.Info(__name__).plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()

            if lsstDebug.Info(__name__).plotAll:
                # Show good points as black, rejected points as red, but with a continuous line
                # https://matplotlib.org/gallery/lines_bars_and_markers/multicolored_line.html
                import matplotlib
                from matplotlib.collections import LineCollection

                cmap, norm = matplotlib.colors.from_levels_and_colors([0.0, 0.5, 2.0], ["red", "black"])
                points = np.array([indices, values]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lines = LineCollection(segments, cmap=cmap, norm=norm)
                lines.set_array(good.astype(int))
                ax.add_collection(lines)
            else:
                # Plot binned data
                xBinned, yBinned = binData(indices, values, good, lsstDebug.Info(__name__).plotBins or 1000)
                ax.plot(xBinned, yBinned, "k-")

            ax.plot(indices, fit, "b-")
            ax.plot(xKnots, binned, "bo")
            ax.set_ylim(0.7 * fit.min(), 1.3 * fit.max())
            plt.show()

        return fit

    def wrapArray(self, array, fiberId):
        """Wrap array in a ``Spectrum``

        Parameters
        ----------
        array : `numpy.ndarray`, 1D float
            Array to wrap.
        fiberId : `int`
            Fiber identifier for spectrum.

        Returns
        -------
        result : `pfs.drp.stella.Spectrum`
            Spectrum with array.
        """
        result = Spectrum(len(array), fiberId)
        result.flux[:] = array
        return result

    def subtractContinuum(self, maskedImage, fiberTraces, detectorMap=None, lines=None):
        """Subtract continuum from an image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image containing 2D spectra.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Location and profile of the 2D spectra.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional.
            Mapping of fiberId,wavelength to x,y.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        continua : `np.ndarray` of `float`
            Continuum fit for each input spectrum.
        continuumImage : `lsst.afw.image.Image`
            Image containing continua.
        """
        badBitMask = maskedImage.mask.getPlaneBitMask(self.config.mask)
        spectra = fiberTraces.extractSpectra(maskedImage, badBitMask)
        if detectorMap is not None:
            for ss in spectra:
                ss.setWavelength(detectorMap.getWavelength(ss.fiberId))
        continua = self.run(spectra, lines)
        continuumSpectra = SpectrumSet(spectra.getLength())
        for cc, ff in zip(continua, spectra.getAllFiberIds()):
            continuumSpectra.add(self.wrapArray(cc, ff))
        continuumImage = continuumSpectra.makeImage(maskedImage.getBBox(), fiberTraces)
        maskedImage -= continuumImage
        bad = ~np.isfinite(continuumImage.array)
        maskedImage.mask.array[bad] |= maskedImage.mask.getPlaneBitMask("NO_DATA")
        return Struct(spectra=spectra, continua=continua, continuumImage=continuumImage)

    @contextmanager
    def subtractionContext(self, maskedImage, fiberTraces, detectorMap=None, lines=None):
        """Context manager for temporarily subtracting continuum

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image containing 2D spectra.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Location and profile of the 2D spectra.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional.
            Mapping of fiberId,wavelength to x,y.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.

        Yields
        ------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        continua : `np.ndarray` of `float`
            Continuum fit for each input spectrum.
        continuumImage : `lsst.afw.image.Image`
            Image containing continua.
        """
        results = self.subtractContinuum(maskedImage, fiberTraces, detectorMap, lines)
        try:
            yield results
        finally:
            maskedImage += results.continuumImage


def binData(xx: np.ndarray, yy: np.ndarray, good: np.ndarray, numBins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Bin arrays

    Parameters
    ----------
    xx, yy : `numpy.ndarray`
        Arrays to bin.
    good : `numpy.ndarray`, boolean
        Boolean array indicating which points are good.
    numBins : `int`
        Number of bins.

    Returns
    -------
    xBinned, yBinned : `numpy.ndarray`
        Binned data.
    """
    bad = ~good
    xMasked = np.copy(xx)
    yMasked = np.copy(yy)
    xMasked[bad] = np.nan
    yMasked[bad] = np.nan

    lenInput = len(xx)
    lenWide = (lenInput + (numBins - 1)) // numBins * numBins
    numExtra = lenWide - lenInput
    select = np.ones(shape=(lenWide,), dtype=bool)
    if numExtra > 0:
        indexExtra = np.around(
            (lenWide - 1) / numExtra * (np.arange(numExtra).astype(np.float32) + 0.5)
        ).astype(int)
        select[indexExtra] = False

    xWide = np.full(shape=(lenWide,), dtype=xMasked.dtype, fill_value=np.nan)
    yWide = np.full(shape=(lenWide,), dtype=yMasked.dtype, fill_value=np.nan)
    xWide[select] = xMasked
    yWide[select] = yMasked

    xBinned = _nanmedianRows(xWide.reshape((numBins, -1)))
    yBinned = _nanmedianRows(yWide.reshape((numBins, -1)))

    return xBinned, yBinned


def binDataByKnots(
    xx: np.ndarray, yy: np.ndarray, good: np.ndarray, knots: np.ndarray, fixed: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin arrays around nominated knots

    Each knot's bin extends halfway to its neighbours; the first and last
    bins extend to the ends of the arrays.

    Parameters
    ----------
    xx, yy : `numpy.ndarray`
        Arrays to bin. ``xx`` must be sorted.
    good : `numpy.ndarray`, boolean
        Boolean array indicating which points are good.
    knots : array_like of `float`
        Nominal positions of the knots, in the same units as ``xx``.
    fixed : `bool`, optional
        Use the nominal knot positions for the binned ``xx``? Otherwise, use
        the median of the good ``xx`` values in each bin.

    Returns
    -------
    xBinned, yBinned : `numpy.ndarray`
        Binned data. Bins with no good points have NaN values.
    """
    knots = np.sort(np.asarray(knots, dtype=float))
    numBins = len(knots)
    edges = 0.5*(knots[:-1] + knots[1:])
    binIndex = np.searchsorted(edges, xx, side="right")
    yBinned = _binnedMedians(binIndex, np.where(good, yy, np.nan), numBins)
    if fixed:
        xBinned = knots.astype(xx.dtype)
        xBinned[np.isnan(yBinned)] = np.nan
    else:
        xBinned = _binnedMedians(binIndex, np.where(good, xx, np.nan), numBins)
    return xBinned, yBinned


def _binnedMedians(binIndex: np.ndarray, values: np.ndarray, numBins: int) -> np.ndarray:
    """Median of values in each bin, ignoring NaNs

    Parameters
    ----------
    binIndex : `numpy.ndarray` of `int`, shape ``(N,)``
        Index of the bin for each value, in the range ``[0, numBins)``.
    values : `numpy.ndarray`, floating-point, shape ``(N,)``
        Values; NaNs are ignored.
    numBins : `int`
        Number of bins.

    Returns
    -------
    median : `numpy.ndarray`, floating-point, shape ``(numBins,)``
        Median of the values in each bin; NaN for bins with no values.
    """
    select = ~np.isnan(values)
    index = binIndex[select]
    values = values[select]
    order = np.lexsort((values, index))
    index = index[order]
    values = values[order]
    counts = np.bincount(index, minlength=numBins)
    starts = np.cumsum(counts) - counts
    median = np.full(numBins, np.nan, dtype=values.dtype)
    have = counts > 0
    low = starts[have] + (counts[have] - 1) // 2
    high = starts[have] + counts[have] // 2
    median[have] = 0.5*(values[low] + values[high])
    return median


def wavelengthToPixels(wavelength: np.ndarray, knotWavelengths: np.ndarray) -> np.ndarray:
    """Convert knot wavelengths to pixel positions in a spectrum

    Knots outside the wavelength range of the spectrum are dropped.

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`
        Wavelength of each pixel in the spectrum (increasing).
    knotWavelengths : `numpy.ndarray` of `float`
        Wavelengths of the knots.

    Returns
    -------
    knots : `numpy.ndarray` of `float`
        Positions of the knots, in pixels.
    """
    knotWavelengths = np.asarray(knotWavelengths, dtype=float)
    finite = np.isfinite(wavelength)
    if np.all(finite):
        knots = interpolateIndices(wavelength, knotWavelengths)
    else:
        # Interpolate over the finite pixels only, then map back to pixels
        pixels = np.flatnonzero(finite)
        indices = interpolateIndices(wavelength[finite], knotWavelengths)
        knots = np.interp(indices, np.arange(len(pixels)), pixels.astype(float))
    return knots[np.isfinite(knots)]


def _nanmedianRows(array: np.ndarray) -> np.ndarray:
    """Median of each row of a 2D array, ignoring NaNs

    This is equivalent to ``np.nanmedian(array, axis=1)``, but much faster for
    arrays with many short rows (for which numpy falls back to a slow
    masked-array implementation). Rows with no non-NaN values produce NaN.

    Parameters
    ----------
    array : `numpy.ndarray`, floating-point, shape ``(M, N)``
        Array for which to calculate the median of each row.

    Returns
    -------
    median : `numpy.ndarray`, floating-point, shape ``(M,)``
        Median of each row.
    """
    numRows = array.shape[0]
    rows = np.arange(numRows)
    ordered = np.sort(array, axis=1)  # NaNs sort to the end
    count = np.count_nonzero(~np.isnan(ordered), axis=1)
    low = np.maximum((count - 1) // 2, 0)
    high = np.maximum(count // 2, 0)
    median = 0.5*(ordered[rows, low] + ordered[rows, high])
    median[count == 0] = np.nan
    return median
