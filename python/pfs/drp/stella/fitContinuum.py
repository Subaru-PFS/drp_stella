from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

import lsstDebug

from lsst.pex.config import Config, Field, ChoiceField, ListField
from lsst.pipe.base import Task, Struct
from lsst.afw.math import stringToInterpStyle, makeInterpolate
from pfs.datamodel import Observations, PfsFiberArraySet, PfsFiberArray
from .SpectrumContinued import Spectrum
from .SpectrumSetContinued import SpectrumSet
from .maskLines import maskLines
from .referenceLine import ReferenceLineSet

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

    def run(
        self,
        spectra: Union[SpectrumSet, PfsFiberArraySet],
        refLines: Optional[ReferenceLineSet] = None,
    ) -> Dict[int, np.ndarray]:
        """Fit spectrum continua

        Fit the continuum for each spectrum.

        Parameters
        ----------
        spectra : `SpectrumSet` or `PfsFiberArraySet`
            Set of spectra to which to fit continua.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.

        Returns
        -------
        continuum : `dict` mapping `int` to `numpy.ndarray`
            Measured continuum for each input spectrum, indexed by fiberId.
        """
        if isinstance(spectra, SpectrumSet):
            fiberId = spectra.getAllFiberIds()
            spectra = spectra.toPfsArm(dict(visit=-1, arm="x", spectrograph=0))
        else:
            fiberId = spectra.fiberId

        assert isinstance(spectra, PfsFiberArraySet)
        edges = self.calculateEdges(spectra.wavelength)
        continuum: Dict[int, np.ndarray] = {}
        empty1d = np.array([])
        empty2d = np.array([[], []]).T
        for ii in range(spectra.numSpectra):
            norm = spectra.norm[ii]
            spectrum = PfsFiberArray(
                None,
                Observations(empty1d, [], empty1d, empty1d, empty1d, empty2d, empty2d),
                spectra.wavelength[ii],
                spectra.flux[ii]/norm,
                spectra.mask[ii],
                spectra.sky[ii]/norm,
                spectra.covar[ii]/norm**2,
                np.array([[]]),
                spectra.flags,
            )
            try:
                continuum[fiberId[ii]] = self.fitContinuum(spectrum, refLines=refLines, edges=edges)*norm
            except FitContinuumError:
                continue
        return continuum

    def fitContinuum(
        self,
        spectrum: Union[Spectrum, PfsFiberArray],
        refLines: Optional[ReferenceLineSet] = None,
        edges: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Fit continuum to a single spectrum

        Uses ``lsst.afw.math.Interpolate`` to fit, and performs iterative
        rejection. Optionally masks identified reference lines.

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum to fit.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.
        edges : array_like, optional
            Wavelengths (nm) of bin boundaries. If not provided, the
            ``numKnots`` configuration parameter will be used to construct a
            linear sampling of bins.

        Raises
        ------
        FitContinuumError
            If we had no good values.

        Returns
        -------
        continuum : `numpy.ndarray`
            Array of continuum fit.
        """
        if isinstance(spectrum, Spectrum):
            spectrum = spectrum.toPfsFiberArray()
        length = len(spectrum)
        if edges is not None:
            # Convert wavelength knots to pixels
            edges = np.array(interp1d(spectrum.wavelength, np.arange(length), bounds_error=False)(edges))
        else:
            # Generate knots in pixels
            edges = np.linspace(0, length, self.config.numKnots + 1)

        flux = spectrum.flux
        good = np.isfinite(flux)
        if self.config.doMaskLines and refLines and np.all(np.isfinite(spectrum.wavelength)):
            good &= ~maskLines(spectrum.wavelength, refLines.wavelength, self.config.maskLineRadius)
        good &= (spectrum.mask & spectrum.flags.get(*self.config.mask)) == 0
        if not np.any(good):
            raise FitContinuumError("No good values when fitting continuum")
        keep = np.ones_like(good, dtype=bool)
        for ii in range(self.config.iterations):
            use = good & keep
            fit = self._fitContinuumImpl(flux, use, edges)
            diff = flux - fit
            lq, uq = np.percentile(diff[use], [25.0, 75.0])
            stdev = 0.741*(uq - lq)
            with np.errstate(invalid='ignore'):
                keep = np.isfinite(diff) & (np.abs(diff) <= self.config.rejection*stdev)
        return self._fitContinuumImpl(flux, good & keep, edges)

    def _fitContinuumImpl(self, values: np.ndarray, good: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Implementation of the business part of fitting

        Parameters
        ----------
        values : `numpy.ndarray`, floating-point
            Spectrum array to fit.
        good : `numpy.ndarray`, boolean
            Boolean array indicating which points are good.
        edges : `numpy.ndarray`, floating-point
            Indices of bin boundaries.

        Raises
        ------
        FitContinuumError
            If we had no good knots.

        Returns
        -------
        fit : `numpy.ndarray`, floating-point
            Fit array.
        """
        centers, binned = binData(values, good, edges)
        use = np.isfinite(centers) & np.isfinite(binned)
        if not np.any(use):
            raise FitContinuumError("No finite knots when fitting continuum")
        interp = makeInterpolate(centers[use], binned[use], self.fitType)
        indices = np.arange(len(values), dtype=values.dtype)
        fit = np.array(interp.interpolate(indices)).astype(values.dtype)

        if lsstDebug.Info(__name__).plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()

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

            ax.plot(indices, fit, 'b-')
            ax.plot(centers, binned, 'bo')
            plotMin = values[good].min()
            plotMax = values[good].max()
            buffer = 0.1*(plotMax - plotMin)
            ax.set_ylim(plotMin - buffer, plotMax + buffer)
            plt.show()

        return fit

    def calculateEdges(self, wavelength: np.ndarray) -> np.ndarray:
        """Calculate bin boundaries in wavelength space

        Provides ``numKnots + 1``, to allow for ``numKnots`` knots.

        Parameters
        ----------
        wavelength : `np.ndarray``
            Wavelength array.

        Returns
        -------
        edges : `np.ndarray`
            Bin boundaries in wavelength (nm).
        """
        return np.linspace(wavelength.min(), wavelength.max(), self.config.numKnots + 1, True)

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
        continuumImage = fiberTraces.makeImage(maskedImage.getBBox(), continua)
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


def binData(
    values: np.ndarray, good: np.ndarray, edges: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin arrays

    This differs from ``numpy.histogram`` in that we take a median over each
    bin.

    Parameters
    ----------
    values : `numpy.ndarray`
        Array to bin.
    good : `numpy.ndarray` of `bool`
        Boolean array indicating which values are good.
    edges : `numpy.ndarray`
        Indices of bins boundaries.

    Returns
    -------
    centers : `numpy.ndarray`
        Centers of the bins.
    binned : `numpy.ndarray`
        Median within each bin.
    """
    centers = 0.5*(edges[:-1] + edges[1:])
    binned = np.empty_like(centers)
    edges = (edges + 0.5).astype(int)
    for ii, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        select = good[low:high]
        binned[ii] = np.median(values[low:high][select]) if np.any(select) else np.nan
    return centers, binned.astype(yy.dtype)
