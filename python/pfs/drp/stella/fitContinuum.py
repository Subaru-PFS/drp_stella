import numpy as np

import lsstDebug

from lsst.pex.config import Config, Field, ChoiceField
from lsst.pipe.base import Task, Struct
from lsst.afw.math import stringToInterpStyle, makeInterpolate
from pfs.drp.stella import Spectrum, SpectrumSet

__all__ = ["FitContinuumConfig", "FitContinuumTask"]


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


class FitContinuumTask(Task):
    """Subtract continuum from spectra

    Debug settings:
    - ``plot`` (`bool`): activate plotting
    - ``plotAll`` (`bool`): plot all data (even rejected)? Otherwise plot
        binned data.
    - ``plotBins`` (`int`): number of bins if not ``plotAll`` (default 1000).
    """
    ConfigClass = FitContinuumConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitType = stringToInterpStyle(self.config.fitType)

    def run(self, spectra):
        """Fit spectrum continua

        Fit the continuum for each spectrum.

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Set of spectra to which to fit continua.

        Returns
        -------
        continuum : `pfs.drp.stella.SpectrumSet`
            Continuum fit for each input spectrum.
        """
        continua = SpectrumSet(spectra.getLength())
        for spec in spectra:
            continuum = self.wrapArray(self.fitContinuum(spec), spec.fiberId)
            continua.add(continuum)
        return continua

    def fitContinuum(self, spectrum):
        """Fit continuum to the spectrum

        Uses ``lsst.afw.math.Interpolate`` to fit, and performs iterative
        rejection. Optionally masks identified reference lines.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum to fit.

        Returns
        -------
        continuum : `numpy.ndarray`
            Array of continuum fit.
        """
        num = spectrum.getNumPixels()
        if self.config.doMaskLines:
            good = self.maskLines(spectrum)
        else:
            good = np.ones(num, dtype=bool)
        oldGood = good
        for ii in range(self.config.iterations):
            fit = self._fitContinuumImpl(spectrum.spectrum, good)
            diff = spectrum.spectrum - fit
            lq, uq = np.percentile(diff[good], [25.0, 75.0])
            stdev = 0.741*(uq - lq)
            good = np.isfinite(diff) & (np.abs(diff) <= self.config.rejection*stdev)
            if np.all(good == oldGood):
                break
            oldGood = good
        return self._fitContinuumImpl(spectrum.spectrum, good).astype(np.float32)

    def _fitContinuumImpl(self, values, good):
        """Implementation of the business part of fitting

        Parameters
        ----------
        values : `numpy.ndarray`, floating-point
            Spectrum array to fit.
        good : `numpy.ndarray`, boolean
            Boolean array indicating which points are good.

        Returns
        -------
        fit : `numpy.ndarray`, floating-point
            Fit array.
        """
        indices = np.arange(len(values), dtype=np.float)
        knots, binned = binData(indices, values, good, self.config.numKnots)
        interp = makeInterpolate(knots, binned, self.fitType)
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

    def maskLines(self, spectrum):
        """Mask reference lines in the spectrum

        The list of reference lines that have already been identified
        in the spectrum is used.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum to mask.

        Returns
        -------
        good : boolean `numpy.ndarray`
            Array indiciating which values are good (i.e., do not
            contain a line).
        """
        wavelength = spectrum.wavelength
        delta = np.diff(wavelength)
        assert np.all(delta >= 0) or np.all(delta <= 0), "Monotonic"
        del delta
        num = len(wavelength)
        good = np.ones_like(wavelength, dtype=bool)
        lines = [ref.wavelength for ref in spectrum.referenceLines]
        indices = np.interp(lines, wavelength, np.arange(num, dtype=int), left=np.nan, right=np.nan)
        for ii in indices:
            if np.isnan(ii):
                continue
            index = int(ii + 0.5)
            low = max(0, index - self.config.maskLineRadius)
            high = min(num - 1, index + self.config.maskLineRadius)
            good[low:high] = False
        return good


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
        xBinned[ii] = np.median(xx[low:high][select])
        yBinned[ii] = np.median(yy[low:high][select])
    return xBinned, yBinned
