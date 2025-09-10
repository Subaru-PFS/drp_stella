from contextlib import contextmanager
import warnings
import numpy as np

import lsstDebug

from lsst.pex.config import Config, Field, ChoiceField, ListField
from lsst.pipe.base import Task, Struct
from lsst.afw.math import stringToInterpStyle, makeInterpolate
from pfs.datamodel.pfsFiberArraySet import PfsFiberArraySet
from pfs.drp.stella import Spectrum, SpectrumSet
from pfs.drp.stella.maskLines import maskLines
from .referenceLine import ReferenceLineSet
from .utils import robustRms

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
    iterations = Field(dtype=int, default=3, doc="Number of fitting iterations")
    rejection = Field(dtype=float, default=3.0, doc="Rejection threshold (standard deviations)")
    doMaskLines = Field(dtype=bool, default=True, doc="Mask reference lines before fitting?")
    maskLineRadius = Field(dtype=int, default=5, doc="Number of pixels either side of reference line to mask")
    mask = ListField(
        dtype=str, default=["BAD", "CR", "NO_DATA", "SUSPECT", "BAD_FLAT"], doc="Mask planes to ignore"
    )


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

    config: FitContinuumConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitType = stringToInterpStyle(self.config.fitType)

    def run(
        self, spectra: SpectrumSet | PfsFiberArraySet, lines: ReferenceLineSet | None = None
    ) -> SpectrumSet:
        """Fit spectrum continua

        Fit the continuum for each spectrum.

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

        continua = np.full((len(spectra), spectra.getLength()), np.nan, dtype=np.float32)
        for ii, spec in enumerate(spectra):
            try:
                continua[ii] = self.fitContinuum(spec, lines)
            except FitContinuumError:
                continue
        return continua

    def fitContinuum(self, spectrum, lines=None):
        """Fit continuum to the spectrum

        Uses ``lsst.afw.math.Interpolate`` to fit, and performs iterative
        rejection. Optionally masks identified reference lines.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum` or `pfs.datamodel.PfsFiberArray`
            Spectrum to fit.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.

        Raises
        ------
        FitContinuumError
            If we had no good values.

        Returns
        -------
        continuum : `numpy.ndarray`
            Array of continuum fit.
        """
        haveWavelength = False
        if isinstance(spectrum, Spectrum):
            flux = spectrum.normFlux
            mask = spectrum.mask.array[0]
            maskVal = spectrum.mask.getPlaneBitMask(self.config.mask)
            haveWavelength = spectrum.isWavelengthSet()
        else:  # PfsFiberArray or PfsFiberArraySet
            flux = spectrum.flux
            if hasattr(flux, "norm"):  # PfsFiberArraySet
                flux /= flux.norm  # Normalize the flux to 1.0
            mask = spectrum.mask
            maskVal = spectrum.flags.get(*self.config.mask)
            haveWavelength = True

        bad = ~np.isfinite(flux)
        if self.config.doMaskLines and lines and haveWavelength:
            bad |= maskLines(spectrum.wavelength, lines.wavelength, self.config.maskLineRadius)

        bad |= (mask & maskVal) != 0
        return self.fitArray(flux, bad)

    def fitArray(self, array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fit continuum to an array

        Uses ``lsst.afw.math.Interpolate`` to fit, and performs iterative
        rejection.

        Parameters
        ----------
        array : `numpy.ndarray` of floating-point
            Array to fit.
        mask : `numpy.ndarray` of boolean
            Mask array; non-zero values are considered bad.

        Raises
        ------
        FitContinuumError
            If we had no good values.

        Returns
        -------
        fit : `numpy.ndarray`, floating-point
            Fit array.
        """
        if np.all(mask):
            raise FitContinuumError("No good values when fitting continuum")
        good = ~mask
        keep = np.ones_like(mask, dtype=bool)
        for ii in range(self.config.iterations):
            use = good & keep
            fit = self._fitContinuumImpl(array, use)
            diff = array - fit
            stdev = robustRms(diff[use])
            with np.errstate(invalid="ignore"):
                keep = np.isfinite(diff) & (np.abs(diff) <= self.config.rejection * stdev)
        return self._fitContinuumImpl(array, good & keep)

    def _fitContinuumImpl(self, values, good):
        """Implementation of the business part of fitting

        Parameters
        ----------
        values : `numpy.ndarray`, floating-point
            Spectrum array to fit.
        good : `numpy.ndarray`, boolean
            Boolean array indicating which points are good.

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
        knots, binned = binData(indices, values, good, self.config.numKnots)
        use = np.isfinite(knots) & np.isfinite(binned)
        if not np.any(use):
            raise FitContinuumError("No finite knots when fitting continuum")

        try:
            interp = makeInterpolate(knots[use], binned[use], self.fitType)
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
            ax.plot(knots, binned, "bo")
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
        result.flux = array
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

    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
        xBinned = np.nanmedian(xWide.reshape((numBins, -1)), axis=(1,))
        yBinned = np.nanmedian(yWide.reshape((numBins, -1)), axis=(1,))

    return xBinned, yBinned
