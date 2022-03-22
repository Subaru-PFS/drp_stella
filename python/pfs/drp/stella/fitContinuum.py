from contextlib import contextmanager
import numpy as np

import lsstDebug

from lsst.pex.config import Config, Field, ChoiceField, ListField
from lsst.pipe.base import Task, Struct
from lsst.afw.math import stringToInterpStyle, makeInterpolate
from pfs.drp.stella import Spectrum, SpectrumSet
from pfs.drp.stella.maskLines import maskLines

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
    mask = ListField(dtype=str, default=["BAD", "CR", "NO_DATA", "BAD_FLAT"], doc="Mask planes to ignore")


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

    def run(self, spectra, lines=None):
        """Fit spectrum continua

        Fit the continuum for each spectrum.

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Set of spectra to which to fit continua.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.

        Returns
        -------
        continuum : `pfs.drp.stella.SpectrumSet`
            Continuum fit for each input spectrum.
        """
        continua = SpectrumSet(spectra.getLength())
        for spec in spectra:
            try:
                result = self.fitContinuum(spec, lines)
            except FitContinuumError:
                continue
            continuum = self.wrapArray(result, spec.fiberId)
            continua.add(continuum)
        return continua

    def fitContinuum(self, spectrum, lines=None):
        """Fit continuum to the spectrum

        Uses ``lsst.afw.math.Interpolate`` to fit, and performs iterative
        rejection. Optionally masks identified reference lines.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
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
        flux = spectrum.normFlux
        good = np.isfinite(flux)
        if self.config.doMaskLines and lines and spectrum.isWavelengthSet():
            good &= ~maskLines(spectrum.wavelength, lines.wavelength, self.config.maskLineRadius)
        good &= (spectrum.mask.array[0] & spectrum.mask.getPlaneBitMask(self.config.mask)) == 0
        if not np.any(good):
            raise FitContinuumError("No good values when fitting continuum")
        keep = np.ones_like(good, dtype=bool)
        for ii in range(self.config.iterations):
            use = good & keep
            fit = self._fitContinuumImpl(flux, use)
            diff = flux - fit
            lq, uq = np.percentile(diff[use], [25.0, 75.0])
            stdev = 0.741*(uq - lq)
            with np.errstate(invalid='ignore'):
                keep = np.isfinite(diff) & (np.abs(diff) <= self.config.rejection*stdev)
        return self._fitContinuumImpl(flux, good & keep)

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
        indices = np.arange(len(values), dtype=float)
        knots, binned = binData(indices, values, good, self.config.numKnots)
        use = np.isfinite(knots) & np.isfinite(binned)
        if not np.any(use):
            raise FitContinuumError("No finite knots when fitting continuum")
        interp = makeInterpolate(knots[use], binned[use], self.fitType)
        fit = np.array(interp.interpolate(indices))

        if lsstDebug.Info(__name__).plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

            if lsstDebug.Info(__name__).plotAll:
                # Show good points as black, rejected points as red, but with a continuous line
                # https://matplotlib.org/gallery/lines_bars_and_markers/multicolored_line.html
                import matplotlib
                from matplotlib.collections import LineCollection
                cmap, norm = matplotlib.colors.from_levels_and_colors([0.0, 0.5, 2.0], ['red', 'black'])
                points = np.array([indices, values]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lines = LineCollection(segments, cmap=cmap, norm=norm)
                lines.set_array(good.astype(int))
                ax.add_collection(lines)
            else:
                # Plot binned data
                xBinned, yBinned = binData(indices, values, good, lsstDebug.Info(__name__).plotBins or 1000)
                ax.plot(xBinned, yBinned, 'k-')

            ax.plot(indices, fit, 'b-')
            ax.plot(knots, binned, 'bo')
            ax.set_ylim(0.7*fit.min(), 1.3*fit.max())
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
        result.spectrum = array
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
        continua : `pfs.drp.stella.SpectrumSet`
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
        continuumImage = continua.makeImage(maskedImage.getBBox(), fiberTraces)
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
        continua : `pfs.drp.stella.SpectrumSet`
            Continuum fit for each input spectrum.
        continuumImage : `lsst.afw.image.Image`
            Image containing continua.
        """
        results = self.subtractContinuum(maskedImage, fiberTraces, detectorMap, lines)
        try:
            yield results
        finally:
            maskedImage += results.continuumImage


def binData(xx, yy, good, numBins):
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
    edges = (np.linspace(0, len(xx), numBins + 1) + 0.5).astype(int)
    xBinned = np.empty(numBins)
    yBinned = np.empty(numBins)
    for ii, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        select = good[low:high]
        xBinned[ii] = np.median(xx[low:high][select]) if np.any(select) else np.nan
        yBinned[ii] = np.median(yy[low:high][select]) if np.any(select) else np.nan
    return xBinned, yBinned
