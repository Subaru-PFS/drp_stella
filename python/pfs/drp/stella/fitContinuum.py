import itertools
import os
from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Tuple, Union

import astropy.io.fits
import lsstDebug
import numpy as np
from lsst.afw.image import VisitInfo
from lsst.afw.math import makeInterpolate, stringToInterpStyle
from lsst.pex.config import ChoiceField, Config, Field, ListField
from lsst.pipe.base import Struct, Task
from lsst.utils import getPackageDir
from numpy.typing import ArrayLike
from pfs.datamodel import Observations, PfsFiberArray, PfsFiberArraySet
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from .interpolate import interpolateFlux
from .maskLines import maskLines
from .math import NormalizedPolynomial1D
from .referenceLine import ReferenceLineSet
from .SpectrumContinued import Spectrum
from .SpectrumSetContinued import SpectrumSet

__all__ = (
    "FitContinuumError",
    "BaseFitContinuumConfig",
    "BaseFitContinuumTask",
    "FitSplineContinuumConfig",
    "FitSplineContinuumTask",
    "AtmosphericTransmission",
    "FitModelContinuumConfig",
    "FitModelContinuumTask",
)


class FitContinuumError(RuntimeError):
    """Error when fitting continuum"""

    pass


class BaseFitContinuumConfig(Config):
    """Configuration for BaseFitContinuumTask"""

    iterations = Field(dtype=int, default=3, doc="Number of fitting iterations")
    rejection = Field(dtype=float, default=3.0, doc="Rejection threshold (standard deviations)")
    doMaskLines = Field(dtype=bool, default=True, doc="Mask reference lines before fitting?")
    maskLineRadius = Field(dtype=int, default=5, doc="Number of pixels either side of reference line to mask")
    mask = ListField(dtype=str, default=["BAD", "CR", "NO_DATA", "BAD_FLAT"], doc="Mask planes to ignore")


class BaseFitContinuumTask(Task):
    """Base class for Task to subtract continuum from spectra

    Debug settings:
    - ``plot`` (`bool`): activate plotting
    - ``plotAll`` (`bool`): plot all data (even rejected)? Otherwise plot
        binned data.
    - ``plotBins`` (`int`): number of bins if not ``plotAll`` (default 1000).
    """

    ConfigClass = BaseFitContinuumConfig
    _DefaultName = "fitContinuum"

    def run(
        self,
        spectra: Union[SpectrumSet, PfsFiberArraySet],
        refLines: Optional[ReferenceLineSet] = None,
        visitInfo: Optional[VisitInfo] = None,
    ) -> Dict[int, np.ndarray]:
        """Fit spectrum continua

        Fit the continuum for each spectrum.

        Parameters
        ----------
        spectra : `SpectrumSet` or `PfsFiberArraySet`
            Set of spectra to which to fit continua.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.
        visitInfo : `VisitInfo`, optional
            Structured visit metadata.

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
        parameters = self.extractParameters(spectra, visitInfo)
        continuum: Dict[int, np.ndarray] = {}
        empty1d = np.array([])
        empty2d = np.array([[], []]).T
        for ii in range(spectra.numSpectra):
            norm = spectra.norm[ii]
            spectrum = PfsFiberArray(
                None,
                Observations(empty1d, [], empty1d, empty1d, empty1d, empty2d, empty2d),
                spectra.wavelength[ii],
                spectra.flux[ii] / norm,
                spectra.mask[ii],
                spectra.sky[ii] / norm,
                spectra.covar[ii] / norm**2,
                np.array([[]]),
                spectra.flags,
            )
            try:
                continuum[fiberId[ii]] = self.fitContinuum(spectrum, refLines, parameters) * norm
            except FitContinuumError:
                continue
        return continuum

    def runSingle(
        self,
        spectrum: Union[Spectrum, PfsFiberArray],
        refLines: Optional[ReferenceLineSet] = None,
        visitInfo: Optional[VisitInfo] = None,
    ) -> np.ndarray:
        """Fit spectrum continua

        Fit the continuum for each spectrum.

        Parameters
        ----------
        spectrum : `Spectrum` or `PfsFiberArray`
            Spectrum to which to fit continua.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.
        visitInfo : `VisitInfo`, optional
            Structured visit metadata.

        Returns
        -------
        continuum : `dict` mapping `int` to `numpy.ndarray`
            Measured continuum for each input spectrum, indexed by fiberId.
        """
        if isinstance(spectrum, Spectrum):
            spectrum = spectrum.toPfsFiberArray()
        parameters = self.extractParameters(spectrum, visitInfo)
        return self.fitContinuum(spectrum, refLines, parameters)

    def extractParameters(
        self,
        spectra: Union[PfsFiberArraySet, PfsFiberArray],
        visitInfo: Optional[VisitInfo] = None,
    ) -> Optional[Struct]:
        """Extract parameters in preparation for fitting

        These parameters are specific to the algorithm adopted by subclasses.

        Parameters
        ----------
        spectra : `PfsFiberArraySet` or `PfsFiberArray`
            Spectra (or spectrum) to be fit.
        visitInfo : `VisitInfo`, optional
            Structured visit metadata.

        Returns
        -------
        parameters : `Struct` or `None`
            Parameters used in fitting.
        """
        return None

    def fitContinuum(
        self,
        spectrum: PfsFiberArray,
        refLines: Optional[ReferenceLineSet] = None,
        parameters: Optional[Struct] = None,
    ) -> np.ndarray:
        """Fit continuum to a single spectrum

        Uses ``lsst.afw.math.Interpolate`` to fit, and performs iterative
        rejection. Optionally masks identified reference lines.

        Parameters
        ----------
        spectrum : `PfsFiberArray`
            Spectrum to fit.
        lines : `pfs.drp.stella.ReferenceLineSet`, optional
            Reference lines to mask.
        parameters : `Struct` or `None`
            Parameters used in fitting. Some subclasses require them.

        Raises
        ------
        FitContinuumError
            If we had no good values.

        Returns
        -------
        continuum : `numpy.ndarray`
            Array of continuum fit.
        """
        good = np.isfinite(spectrum.flux)
        if self.config.doMaskLines and refLines and np.all(np.isfinite(spectrum.wavelength)):
            good &= ~maskLines(spectrum.wavelength, refLines.wavelength, self.config.maskLineRadius)
        good &= (spectrum.mask & spectrum.flags.get(*self.config.mask)) == 0
        if not np.any(good):
            raise FitContinuumError("No good values when fitting continuum")
        keep = np.ones_like(good, dtype=bool)
        for ii in range(self.config.iterations):
            use = good & keep
            fit = self._fitContinuumImpl(spectrum, use, parameters)
            if lsstDebug.Info(__name__).plot:
                self.plotFit(spectrum, use, fit)

            diff = spectrum.flux - fit
            lq, uq = np.percentile(diff[use], [25.0, 75.0])
            stdev = 0.741 * (uq - lq)
            with np.errstate(invalid="ignore"):
                keep = np.isfinite(diff) & (np.abs(diff) <= self.config.rejection * stdev)
        fit = self._fitContinuumImpl(spectrum, good & keep, parameters)
        if lsstDebug.Info(__name__).plot:
            self.plotFit(spectrum, use, fit)
        return fit

    def plotFit(self, spectrum: PfsFiberArray, good: np.ndarray, fit: np.ndarray):
        """Plot the fit

        Parameters
        ----------
        spectrum : `PfsFiberArray`
            Spectrum being fit.
        good : `np.ndarray` of `bool`
            Boolean array indicating which points are good.
        fit : `np.ndarray` of `float`
            Fit array.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Show good points as black, rejected points as red, but with a continuous line
        # https://matplotlib.org/gallery/lines_bars_and_markers/multicolored_line.html
        import matplotlib
        from matplotlib.collections import LineCollection

        cmap, norm = matplotlib.colors.from_levels_and_colors([0.0, 0.5, 2.0], ["red", "black"])
        points = np.array([spectrum.wavelength, spectrum.flux]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lines = LineCollection(segments, cmap=cmap, norm=norm)
        lines.set_array(good.astype(int))
        ax.add_collection(lines)

        ax.plot(spectrum.wavelength, fit, "b-")
        plotMin = spectrum.flux[good].min()
        plotMax = spectrum.flux[good].max()
        buffer = 0.1 * (plotMax - plotMin)
        ax.set_ylim(plotMin - buffer, plotMax + buffer)
        plt.show()

    def _fitContinuumImpl(
        self,
        spectrum: PfsFiberArray,
        good: np.ndarray,
        parameters: Optional[Struct],
    ) -> np.ndarray:
        """Implementation of the business part of fitting

        Parameters
        ----------
        spectrum : `PfsFiberArray`
            Spectrum to fit.
        good : `numpy.ndarray`, boolean
            Boolean array indicating which points are good.
        parameters : `Struct` or `None`
            Parameters used in fitting. Some subclasses require them.

        Raises
        ------
        FitContinuumError
            If there is no good data.

        Returns
        -------
        fit : `numpy.ndarray`, floating-point
            Fit array.
        """
        raise NotImplementedError("Subclasses must implement")

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


class FitSplineContinuumConfig(BaseFitContinuumConfig):
    """Configuration for FitSplineContinuumTask"""

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


class FitSplineContinuumTask(BaseFitContinuumTask):
    """Subtract continuum from spectra

    Debug settings:
    - ``plot`` (`bool`): activate plotting
    - ``plotAll`` (`bool`): plot all data (even rejected)? Otherwise plot
        binned data.
    - ``plotBins`` (`int`): number of bins if not ``plotAll`` (default 1000).
    """

    ConfigClass = FitSplineContinuumConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitType = stringToInterpStyle(self.config.fitType)

    def extractParameters(
        self,
        spectra: Union[PfsFiberArraySet, PfsFiberArray],
        visitInfo: Optional[VisitInfo] = None,
    ) -> Optional[Struct]:
        """Extract parameters in preparation for fitting

        We calculate bin boundaries in wavelength space, to ensure all fibers
        use the same knots.

        Provides ``numKnots + 1``, to allow for ``numKnots`` knots.

        Parameters
        ----------
        spectra : `PfsFiberArraySet` or `PfsFiberArray`
            Spectra (or spectrum) to be fit.
        visitInfo : `VisitInfo`, optional
            Structured visit metadata.

        Returns
        -------
        parameters : `Struct` or `None`
            Parameters used in fitting.
        """
        edges = None
        if np.all(np.isfinite(spectra.wavelength) & (spectra.wavelength != 0.0)):
            minWavelength = spectra.wavelength.min()
            maxWavelength = spectra.wavelength.max()
            edges = np.linspace(minWavelength, maxWavelength, self.config.numKnots + 1, True, dtype=float)
        return Struct(edges=edges)

    def _fitContinuumImpl(
        self,
        spectrum: PfsFiberArray,
        good: np.ndarray,
        parameters: Optional[Struct],
    ) -> np.ndarray:
        """Implementation of the business part of fitting

        Parameters
        ----------
        spectrum : `PfsFiberArray`
            Spectrum to fit.
        good : `numpy.ndarray`, boolean
            Boolean array indicating which points are good.
        parameters : `Struct` or `None`
            Parameters used in fitting. Some subclasses require them.

        Raises
        ------
        FitContinuumError
            If we had no good knots.

        Returns
        -------
        fit : `numpy.ndarray`, floating-point
            Fit array.
        """
        length = spectrum.length
        if parameters is not None and parameters.edges is not None:
            # Convert wavelength knots to pixels
            edges = interp1d(spectrum.wavelength, np.arange(length), bounds_error=False)(parameters.edges)
        else:
            # Generate knots in pixels
            edges = np.linspace(0, length, self.config.numKnots + 1)

        centers, binned = binData(spectrum.flux, good, edges)
        use = np.isfinite(centers) & np.isfinite(binned)
        if not np.any(use):
            raise FitContinuumError("No finite knots when fitting continuum")
        interp = makeInterpolate(centers[use], binned[use], self.fitType)
        indices = np.arange(length, dtype=spectrum.flux.dtype)
        return np.array(interp.interpolate(indices)).astype(spectrum.flux.dtype)

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


def binData(values: np.ndarray, good: np.ndarray, edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    centers = 0.5 * (edges[:-1] + edges[1:])
    binned = np.empty_like(centers)
    edges = (edges + 0.5).astype(int)
    for ii, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        select = good[low:high]
        binned[ii] = np.median(values[low:high][select]) if np.any(select) else np.nan
    return centers, binned.astype(yy.dtype)


class AtmosphericTransmissionInterpolator:
    """Object for interpolating the atmospheric transmission in PWV

    When fitting a spectrum, we know the zenith distance and wavelength
    sampling, and we're fitting for the precipitable water vapor (PWV). This
    object provides an efficient interpolation of the atmospheric model for
    a requested PWV.

    Parameters
    ----------
    pwv : `np.ndarray`
        Precipitable water vapor (mm) values.
    transmission: `np.ndarray`
        Corresponding transmission (as a function of wavelength) values. Any
        interpolation in wavelength should already have been done.
    """

    def __init__(self, pwv: np.ndarray, transmission: np.ndarray):
        if transmission.shape[0] != pwv.size:
            raise RuntimeError(
                f"Size mismatch between pwv ({pwv.size}) and transmission ({transmission.shape[0]})"
            )
        indices = np.argsort(pwv)
        self.pwv: np.ndarray = pwv[indices]
        self.transmission: np.ndarray = transmission[indices]

    def __call__(self, pwv: float) -> np.ndarray:
        """Interpolate the transmission spectra for the PWV value

        Parameters
        ----------
        pwv : `float`
            Precipitable water vapor (mm) value at which to interpolate.

        Returns
        -------
        transmission : `np.ndarray`
            Transmission spectrum for the input PWV.
        """
        if pwv <= 0:
            return np.zeros_like(self.transmission[0])
        index = min(int(np.searchsorted(self.pwv, pwv)), self.pwv.size - 2)
        pwvLow: float = self.pwv[index]
        pwvHigh: float = self.pwv[index + 1]
        highWeight = (pwv - pwvLow) / (pwvHigh - pwvLow)
        lowWeight = 1.0 - highWeight
        return lowWeight * self.transmission[index] + highWeight * self.transmission[index + 1]


class AtmosphericTransmission:
    """Model of atmospheric transmission

    The model includes both zenith distance (ZD) and precipitable water vapor
    (PWV).

    Parameters
    ----------
    wavelength : `np.ndarray` of `float`, shape ``(W,)``
        Wavelength sampling.
    zd : `np.ndarray` of `float`, shape ``(Z,)``
        Zenith distance values; repeated values allowed.
    pwv : `np.ndarray` of `float`, shape ``(P,)``
        Precipitable water vapor (mm) values; repeated values allowed.
    transmission : `dict` mapping (`float`,`float`) to `np.ndarray`
        Transmission spectra, indexed by a tuple of zenith distance value and
        PWV value. Each transmission spectrum should have the same length as
        the ``wavelength`` array.
    """

    def __init__(
        self,
        wavelength: np.ndarray,
        zd: Iterable[float],
        pwv: Iterable[float],
        transmission: Dict[Tuple[float, float], np.ndarray],
    ):
        self.wavelength = wavelength
        self.zd = np.array(sorted(set(zd)))
        self.pwv = np.array(sorted(set(pwv)))
        self.transmission = transmission
        for key in itertools.product(self.zd, self.pwv):
            if key not in self.transmission:
                raise RuntimeError(f"Grid point ZD,PWV={key} not present in data")
            if self.transmission[key].shape != self.wavelength.shape:
                raise RuntimeError(f"Shape of transmission for ZD,PWV={key} doesn't match")

    @classmethod
    def fromFits(cls, filename: str) -> "AtmosphericTransmission":
        """Construct from FITS file

        Parameters
        ----------
        filename : `str`
            Filename of atmospheric model FITS file.

        Returns
        -------
        self : `AtmosphericTransmission`
            Model constructed from FITS file.
        """
        with astropy.io.fits.open(filename) as fits:
            wavelength = fits["WAVELENGTH"].data
            zd = fits["TRANSMISSION"].data["zd"]
            pwv = fits["TRANSMISSION"].data["pwv"]
            transmission = fits["TRANSMISSION"].data["transmission"]
        return cls(
            wavelength=wavelength,
            zd=zd,
            pwv=pwv,
            transmission={(zz, pp): tt for zz, pp, tt in zip(zd, pwv, transmission)},
        )

    def makeInterpolator(self, zd: float, wavelength: ArrayLike) -> AtmosphericTransmissionInterpolator:
        """Construct an interpolator

        When fitting an atmospheric model to data, we know the zenith distance
        at which the data was obtained and the wavelength sampling, and we're
        fitting for the precipitable water vapor (PWV). The fitting process will
        evaluate the model for many different PWV values, so that needs to be as
        efficient as possible. This method does the interpolation in zenith
        distance and wavelength up front, and provides an interpolator that
        operates solely on the PWV value.

        Parameters
        ----------
        zd : `float`
            Zenith distance (degrees) at which to interpolate.
        wavelength : array_like
            Wavelength array for interpolation.

        Returns
        -------
        interpolator : `AtmosphericTransmissionInterpolator`
            Object that will perform interpolation in PWV.
        """
        wavelength = np.asarray(wavelength)
        index = min(int(np.searchsorted(self.zd, zd)), self.zd.size - 2)
        zdLow = self.zd[index]
        zdHigh = self.zd[index + 1]
        highWeight = (zd - zdLow) / (zdHigh - zdLow)
        lowWeight = 1.0 - highWeight
        transmission = np.full((self.pwv.size, wavelength.size), np.nan, dtype=float)
        for ii, pwv in enumerate(self.pwv):
            low = self.transmission[zdLow, pwv]
            high = self.transmission[zdHigh, pwv]
            pwvTransmission = low * lowWeight + high * highWeight
            transmission[ii] = interpolateFlux(
                self.wavelength, pwvTransmission, wavelength, fill=np.nan, jacobian=False
            )
        return AtmosphericTransmissionInterpolator(self.pwv, transmission)

    def __call__(self, zd: float, pwv: float, wavelength: ArrayLike) -> np.ndarray:
        """Evaluate the atmospheric transmission

        This method provides a full-featured interpolation of the model. For
        faster individual interpolations when varying only PWV (e.g., when
        fitting a spectrum of known zenith distance and wavelength sampling),
        use the interpolator provided by the ``makeInterpolator`` method.

        Parameters
        ----------
        zd: `float`
            Zenith distance, in degrees.
        pwv : `float`
            Precipitable water vapour, in mm.
        wavelength : array_like
            Wavelength array for which to provide corresponding transmission.

        Returns
        -------
        result : `np.ndarray` of `float`
            Transmission for the provided wavelengths.
        """
        return self.makeInterpolator(zd, wavelength)(pwv)


class FitModelContinuumConfig(BaseFitContinuumConfig):
    model = Field(dtype=str, doc="Filename of model, in PfsFiberArray FITS format")  # Note: no default
    transmission = Field(
        dtype=str,
        default=os.path.join(getPackageDir("drp_pfs_data"), "atmosphere", "pfs_atmosphere.fits"),
        doc="Filename of atmospheric transmission model",
    )
    order = Field(dtype=int, default=10, doc="Order of the multiplicative scaling polynomial in wavelength")
    guessPwv = Field(dtype=float, default=1.5, doc="Starting guess for PWV (mm)")


class FitModelContinuumTask(BaseFitContinuumTask):
    """Fit an external model to the continuum

    The continuum we fit is provided by an external model of flux as a function
    of wavelength, scaled by a polynomial in wavelength, and atmospheric
    absorption by water vapor applied. The fit model parameters are the
    polynomial coefficients and the precipitable water vapor (PWV).
    """

    ConfigClass = FitModelContinuumConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = PfsFiberArray.readFits(self.config.model)
        self.transmission = AtmosphericTransmission(self.config.transmission)

    def extractParameters(
        self,
        spectra: Union[PfsFiberArraySet, PfsFiberArray],
        visitInfo: Optional[VisitInfo] = None,
    ) -> Optional[Struct]:
        """Extract parameters in preparation for fitting

        We calculate bin boundaries in wavelength space, to ensure all fibers
        use the same knots.

        Provides ``numKnots + 1``, to allow for ``numKnots`` knots.

        Parameters
        ----------
        spectra : `PfsFiberArraySet` or `PfsFiberArray`
            Spectra (or spectrum) to be fit.
        visitInfo : `VisitInfo`, optional
            Structured visit metadata.

        Returns
        -------
        parameters : `Struct` or `None`
            Parameters used in fitting.
        """
        if visitInfo is None:
            raise RuntimeError("visitInfo is required for FitModelContinuumTask")
        return Struct(zd=visitInfo.getBoresightAzAlt().getLatitude())

    def _fitContinuumImpl(
        self,
        spectrum: PfsFiberArray,
        good: np.ndarray,
        parameters: Optional[Struct],
    ) -> np.ndarray:
        """Implementation of the business part of fitting

        Parameters
        ----------
        spectrum : `PfsFiberArray`
            Spectrum to fit.
        good : `numpy.ndarray`, boolean
            Boolean array indicating which points are good.
        parameters : `Struct` or `None`
            Parameters used in fitting. Some subclasses require them.

        Raises
        ------
        FitContinuumError
            If we had no good knots.

        Returns
        -------
        fit : `numpy.ndarray`, floating-point
            Fit array.
        """
        assert parameters is not None
        transmission = self.transmission.makeInterpolator(parameters.zd, spectrum.wavelength)
        numTransmission = 1
        numPoly = self.config.order + 1
        numParams = numPoly + numTransmission
        length = spectrum.length
        indices = np.arange(length, dtype=float)
        flux = spectrum.flux
        invError = 1.0 / np.sqrt(spectrum.variance)

        # Interpolate model to match spectrum wavelength sampling
        model = interpolateFlux(self.model.wavelength, self.model.flux, spectrum.wavelength, fill=np.nan)
        select = good & np.isfinite(model)

        # Get initial guess
        guess = np.zeros(numParams, dtype=float)
        guess[-2] = np.median((model / flux)[select])  # Constant term
        guess[-1] = self.config.guessPwv

        # Define model
        def function(params: np.ndarray) -> np.ndarray:
            """Evaluate the flux from the model

            Parameters
            ----------
            params : `np.ndarray`
                Fitting parameters.

            Returns
            -------
            flux : `np.ndarray`
                Flux from the model.
            """
            poly = NormalizedPolynomial1D(params[:-1], 0.0, length)
            pwv = params[-1]
            return model * poly(indices) * transmission(pwv)

        def residuals(params: np.ndarray) -> np.ndarray:
            """Calculate the error-normalised residuals

            Parameters
            ----------
            params : `np.ndarray`
                Fitting parameters.

            Returns
            -------
            residuals : `np.ndarray`
                Residuals from the fit, normalised by the errors.
            """
            return ((flux - function(params)) * invError)[select]

        # Non-linear least-squares fit
        result = least_squares(residuals, guess, method="lm")
        if not result.success:
            raise RuntimeError(f"Failed to fit continuum: {result.message}")
        return model(result.x)
