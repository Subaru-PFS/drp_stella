import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union

import lsstDebug
import numpy as np
from lsst.afw.image import VisitInfo
from lsst.afw.math import makeInterpolate, stringToInterpStyle
from lsst.pex.config import ChoiceField, Config, Field, ListField
from lsst.pipe.base import Struct, Task
from lsst.utils import getPackageDir
from pfs.datamodel import Observations
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from .atmosphere import AtmosphericTransmission
from .datamodel import PfsFiberArray, PfsFiberArraySet, PfsSimpleSpectrum
from .interpolate import interpolateFlux
from .lsf import Lsf, LsfDict
from .maskLines import maskLines
from .referenceLine import ReferenceLineSet
from .SpectrumContinued import Spectrum
from .SpectrumSetContinued import SpectrumSet
from .spline import SplineD

__all__ = (
    "FitContinuumError",
    "BaseFitContinuumConfig",
    "BaseFitContinuumTask",
    "FitSplineContinuumConfig",
    "FitSplineContinuumTask",
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
    maskLineRadius = Field(dtype=int, default=2, doc="Number of pixels either side of reference line to mask")
    mask = ListField(dtype=str, default=["BAD", "CR", "NO_DATA", "BAD_FLAT"], doc="Mask planes to ignore")


class BaseFitContinuumTask(Task):
    """Base class for Task to subtract continuum from spectra

    Debug settings:
    - ``plot`` (`bool`): activate plotting
    """

    ConfigClass = BaseFitContinuumConfig
    _DefaultName = "fitContinuum"

    def run(
        self,
        spectra: Union[SpectrumSet, PfsFiberArraySet],
        refLines: Optional[ReferenceLineSet] = None,
        visitInfo: Optional[VisitInfo] = None,
        lsf: Optional[LsfDict] = None,
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
        lsfDict : `LsfDict`, optional
            Line-spread functions, indexed by fiberId.

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

        empty1d = np.array([])
        empty2d = np.array([[], []]).T
        spectrumList: List[PfsFiberArray] = []
        for ii in range(spectra.numSpectra):
            spectrumList.append(
                # Deliberately ignoring the normalisation: we want to subtract the flux
                PfsFiberArray(
                    None,
                    Observations(empty1d, [], empty1d, empty1d, empty1d, empty2d, empty2d),
                    spectra.wavelength[ii],
                    spectra.flux[ii],
                    spectra.mask[ii],
                    spectra.sky[ii],
                    spectra.covar[ii],
                    np.array([[]]),
                    spectra.flags,
                )
            )

        parameters = self.extractParameters(spectrumList, visitInfo)
        continuum: Dict[int, np.ndarray] = {}
        for ff, spectrum, params in zip(fiberId, spectrumList, parameters):
            try:
                continuum[ff] = self.fitContinuum(
                    spectrum,
                    refLines,
                    params,
                    lsf.get(ff, None) if lsf is not None else None,
                )
            except FitContinuumError:
                continue
        return continuum

    def runSingle(
        self,
        spectrum: Union[Spectrum, PfsFiberArray],
        refLines: Optional[ReferenceLineSet] = None,
        visitInfo: Optional[VisitInfo] = None,
        lsf: Optional[Lsf] = None,
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
        lsf : `Lsf`, optional
            Line-spread function.

        Returns
        -------
        continuum : `dict` mapping `int` to `numpy.ndarray`
            Measured continuum for each input spectrum, indexed by fiberId.
        """
        if isinstance(spectrum, Spectrum):
            spectrum = spectrum.toPfsFiberArray()
        parameters = self.extractParameters([spectrum], visitInfo)
        assert len(parameters) == 1
        return self.fitContinuum(spectrum, refLines, parameters[0], lsf)

    def extractParameters(
        self,
        spectra: List[PfsFiberArray],
        visitInfo: Optional[VisitInfo] = None,
    ) -> List[Struct]:
        """Extract parameters in preparation for fitting

        These parameters are specific to the algorithm adopted by subclasses.

        Parameters
        ----------
        spectra : `list` of `PfsFiberArray`
            Spectra to be fit.
        visitInfo : `VisitInfo`, optional
            Structured visit metadata.

        Returns
        -------
        parameters : `list` of `Struct`
            Parameters used in fitting for each spectrum.
        """
        return [Struct() for _ in spectra]

    def fitContinuum(
        self,
        spectrum: PfsFiberArray,
        refLines: Optional[ReferenceLineSet] = None,
        parameters: Optional[Struct] = None,
        lsf: Optional[Lsf] = None,
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
            fit = self._fitContinuumImpl(spectrum, use, parameters, lsf)
            use &= np.isfinite(fit)
            if lsstDebug.Info(__name__).plot:
                self.plotFit(spectrum, use, fit)
            resid = spectrum.flux - fit
            lq, uq = np.percentile(resid[use], [25.0, 75.0])
            stdev = 0.741 * (uq - lq)
            with np.errstate(invalid="ignore"):
                diff = resid/np.sqrt(spectrum.variance + stdev**2)
                keep = np.isfinite(diff) & (np.abs(diff) <= self.config.rejection * stdev)

        fit = self._fitContinuumImpl(spectrum, good & keep, parameters, lsf)
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
        lsf: Optional[Lsf] = None,
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
        lsf : `Lsf`, optional
            Line-spread function.

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

    def subtractContinuum(self, maskedImage, fiberTraces, detectorMap=None, lines=None, visitInfo=None,
                          lsf=None):
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
        visitInfo : `lsst.afw.image.VisitInfo`, optional
            Structured visit metadata.
        lsf : `pfs.drp.stella.Lsf`, optional
            Line-spread function.

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
        continua = self.run(spectra, lines, visitInfo, lsf)
        continuumImage = fiberTraces.makeImage(maskedImage.getBBox(), continua)
        maskedImage -= continuumImage
        bad = ~np.isfinite(continuumImage.array)
        maskedImage.mask.array[bad] |= maskedImage.mask.getPlaneBitMask("NO_DATA")
        return Struct(spectra=spectra, continua=continua, continuumImage=continuumImage)

    @contextmanager
    def subtractionContext(self, maskedImage, fiberTraces, detectorMap=None, lines=None, visitInfo=None,
                           lsf=None):
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
        visitInfo : `lsst.afw.image.VisitInfo`, optional
            Structured visit metadata.
        lsf : `pfs.drp.stella.Lsf`, optional
            Line-spread function.

        Yields
        ------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        continua : `pfs.drp.stella.SpectrumSet`
            Continuum fit for each input spectrum.
        continuumImage : `lsst.afw.image.Image`
            Image containing continua.
        """
        results = self.subtractContinuum(maskedImage, fiberTraces, detectorMap, lines, visitInfo, lsf)
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
    """

    ConfigClass = FitSplineContinuumConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fitType = stringToInterpStyle(self.config.fitType)

    def extractParameters(
        self,
        spectra: List[PfsFiberArray],
        visitInfo: Optional[VisitInfo] = None,
    ) -> List[Struct]:
        """Extract parameters in preparation for fitting

        We calculate bin boundaries in wavelength space, to ensure all fibers
        use the same knots.

        Provides ``numKnots + 1``, to allow for ``numKnots`` knots.

        Parameters
        ----------
        spectra : `list` of `PfsFiberArray`
            Spectra to be fit.
        visitInfo : `VisitInfo`, optional
            Structured visit metadata.

        Returns
        -------
        parameters : `list` of `Struct`
            Parameters used in fitting for each spectrum.
        """
        minWavelength = spectra[0].wavelength.min()
        maxWavelength = spectra[0].wavelength.max()
        if np.isfinite(minWavelength) and minWavelength != 0.0:
            for ss in spectra[1:]:
                minWavelength = min(minWavelength, ss.wavelength.min())
                maxWavelength = max(maxWavelength, ss.wavelength.max())
            edges = np.linspace(minWavelength, maxWavelength, self.config.numKnots + 1, True, dtype=float)
        else:
            edges = None
        return [Struct(edges=edges) for _ in spectra]

    def _fitContinuumImpl(
        self,
        spectrum: PfsFiberArray,
        good: np.ndarray,
        parameters: Optional[Struct],
        lsf: Optional[Lsf] = None,
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
        lsf : `Lsf`, optional
            Line-spread function.

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
    return centers, binned.astype(values.dtype)


class FitModelContinuumConfig(BaseFitContinuumConfig):
    model = Field(dtype=str, doc="Filename of model, in PfsFiberArray FITS format")  # Note: no default
    transmission = Field(
        dtype=str,
        default=os.path.join(getPackageDir("drp_pfs_data"), "atmosphere", "pfs_atmosphere.fits"),
        doc="Filename of atmospheric transmission model",
    )
    numKnots = Field(dtype=int, default=30, doc="Number of knots for multiplicative spline")
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
        self.model = PfsSimpleSpectrum.readFits(self.config.model)
        self.transmission = AtmosphericTransmission.fromFits(self.config.transmission)

    def extractParameters(
        self,
        spectra: List[PfsFiberArray],
        visitInfo: Optional[VisitInfo] = None,
    ) -> List[Struct]:
        """Extract parameters in preparation for fitting

        We calculate bin boundaries in wavelength space, to ensure all fibers
        use the same knots.

        Provides ``numKnots + 1``, to allow for ``numKnots`` knots.

        Parameters
        ----------
        spectra : `list` of `PfsFiberArray`
            Spectra to be fit.
        visitInfo : `VisitInfo`, optional
            Structured visit metadata.

        Returns
        -------
        parameters : `list` of `Struct`
            Parameters used in fitting for each spectrum.
        """
        if visitInfo is None:
            raise RuntimeError("visitInfo is required for FitModelContinuumTask")
        zd = 90 - visitInfo.getBoresightAzAlt().getLatitude().asDegrees()
        return [Struct(zd=zd)]*len(spectra)

    def _fitContinuumImpl(
        self,
        spectrum: PfsFiberArray,
        good: np.ndarray,
        parameters: Optional[Struct],
        lsf: Optional[Lsf] = None,
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
        lsf : `Lsf`, optional
            Line-spread function.

        Raises
        ------
        FitContinuumError
            If we failed to fit.

        Returns
        -------
        fit : `numpy.ndarray`, floating-point
            Fit array.
        """
        assert parameters is not None
        if lsf is None:
            raise RuntimeError("lsf is required for FitModelContinuumTask")
        transmission = self.transmission.makeInterpolator(parameters.zd, spectrum.wavelength, lsf=lsf)
        numTransmission = 1
        numParams = self.config.numKnots + numTransmission
        length = spectrum.length
        knots = np.linspace(0.0, length, self.config.numKnots)
        indices = np.arange(length, dtype=float)
        flux = spectrum.flux
        with np.errstate(divide="ignore", invalid="ignore"):
            invError = 1.0 / np.sqrt(spectrum.variance)

        # Interpolate model to match spectrum wavelength sampling
        model = interpolateFlux(self.model.wavelength, self.model.flux, spectrum.wavelength, fill=np.nan)
        select = good & np.isfinite(model) & (model != 0)

        if not np.any(select):
            raise FitContinuumError("No good points")

        # Get initial guess
        pwvIndex = -1  # Index of PWV
        guess = np.zeros(numParams, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):  # Model can be zero; we catch it after the math
            guess[:pwvIndex] = np.median((flux / model / transmission(self.config.guessPwv))[select])
        guess[pwvIndex] = self.config.guessPwv  # PWV
        scale = np.full_like(guess, 1.0)
        scale[:pwvIndex] = 0.1*np.abs(guess[:pwvIndex])
        scale[pwvIndex] = 0.1

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
            values = params[:-1]
            pwv = params[-1]
            spline = SplineD(knots, values, SplineD.NATURAL)
            return model * spline(indices) * transmission(pwv)

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
            with np.errstate(invalid="ignore"):
                return ((flux - function(params)) * invError)[select]

        # Non-linear least-squares fit
        result = least_squares(residuals, guess, x_scale=scale, method="lm")
        if not result.success:
            raise FitContinuumError(f"Failed to fit continuum: {result.message}")
        return function(result.x)
