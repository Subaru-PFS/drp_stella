from collections import defaultdict
import contextlib
import dataclasses
import logging
import math
import re
import warnings

from astropy import constants as const
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import minimize

import lsstDebug
from lsst.pipe.base import (
    ArgumentParser,
    PipelineTask,
    PipelineTaskConfig,
    QuantumContext,
    Struct,
)
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

import lsst.daf.persistence
from lsst.pex.config import Field, ChoiceField, ListField, ConfigurableField

from pfs.datamodel import FiberStatus, PfsConfig, Target, TargetType
from pfs.datamodel.pfsFluxReference import PfsFluxReference

from .datamodel import PfsArm, PfsFiberArray, PfsMerged, PfsSimpleSpectrum, PfsSingle
from .datamodel.pfsTargetSpectra import PfsCalibratedSpectra
from .fitFocalPlane import FitFocalPlaneConfig, FitFocalPlaneTask
from .fitPfsFluxReference import removeBadFluxes
from .fitReference import FilterCurve
from .fluxCalibrate import fluxCalibrate, FluxCalibrateConnections
from .focalPlaneFunction import ConstantFocalPlaneFunction, FluxCalib
from .gen3 import readDatasetRefs
from .lsf import warpLsf, Lsf, LsfDict
from .subtractSky1d import subtractSky1d
from .utils import getPfsVersions
from .utils import debugging
from .utils.polynomialND import NormalizedPolynomialND
from .FluxTableTask import FluxTableTask

from collections.abc import Callable, Generator, Iterable
from typing import Literal, overload

__all__ = ["FitFluxCalConfig", "FitFluxCalTask"]


class Photometerer:
    """This class photometers spectra to get broadband fluxes."""

    def __init__(self) -> None:
        self._filterCurves: dict[str, FilterCurve] = {}

    def getFilterCurve(self, filterName: str) -> FilterCurve:
        """Get the transmission curve of ``filterName``

        Calling this method is equivalent to calling the constructor of
        `FilterCurve`. The only difference is that this method returns
        a cached instance if it is available.

        Parameters
        ----------
        filterName : `str`
            Filter name.

        Returns
        -------
        filterCurve : `FilterCurve`
            Transmission curve.
        """
        instance = self._filterCurves.get(filterName)
        if instance is None:
            instance = self._filterCurves[filterName] = FilterCurve(filterName)
        return instance

    @overload
    def __call__(self, spectrum: PfsSimpleSpectrum, filterName: str) -> float:
        ...

    @overload
    def __call__(
        self, spectrum: PfsSimpleSpectrum, filterName: str, doComputeError: Literal[False]
    ) -> float:
        ...

    @overload
    def __call__(
        self, spectrum: PfsFiberArray, filterName: str, doComputeError: Literal[True]
    ) -> tuple[float, float]:
        ...

    def __call__(self, spectrum, filterName, doComputeError=False):
        """Get a broadband flux by integrating ``spectrum``

        Parameters
        ----------
        spectrum : `PfsSimpleSpectrum`
            Flux-calibrated spectrum.
        filterName : `str`
            Filter name.
        doComputeError : `bool`
            Whether to compute an error bar (standard deviation).
            If ``doComputeError=True``, ``spectrum`` must be of
            `pfs.datamodel.PfsFiberArray` type.

        Returns
        -------
        photometry : `float`
            Broadband flux.
        error : `float`
            Error of ``photometry`` (Returned only if ``doComputeError=True``).
        """
        if filterName == "bp_gaia":
            # Because the short-wavelength tail of Bp filter curve is not
            # covered by PFS, we must corrected it with HSC's fluxes.
            # This formula looks very different from the one (relation among
            # magnitudes) found in the comments of PIPE2D-1596, but they are
            # equivalent in fact.
            x = math.log(
                self.getFilterCurve("g_hsc").photometer(spectrum, doComputeError=False)
                / self.getFilterCurve("r2_hsc").photometer(spectrum, doComputeError=False)
            )
            corr = math.exp(
                -0.0918003282594816
                + x
                * (
                    +0.03244092
                    + x
                    * (
                        -0.0441896155367245
                        + x * (-0.0354497452767988 + x * (+0.0415418421342531 + x * (0.0312321866067972)))
                    )
                )
            )
            if doComputeError:
                photo, error = self.getFilterCurve(filterName).photometer(
                    spectrum, doComputeError=doComputeError
                )
                return photo * corr, error * corr
            else:
                photo = self.getFilterCurve(filterName).photometer(spectrum, doComputeError=doComputeError)
                return photo * corr

        elif filterName == "g_gaia":
            # Because the short-wavelength tail of G filter curve is not
            # covered by PFS, we must multiply `photo` by a constant ~ 1.
            corr = 0.9845
            if doComputeError:
                photo, error = self.getFilterCurve(filterName).photometer(
                    spectrum, doComputeError=doComputeError
                )
                return photo * corr, error * corr
            else:
                photo = self.getFilterCurve(filterName).photometer(spectrum, doComputeError=doComputeError)
                return photo * corr
        else:
            return self.getFilterCurve(filterName).photometer(spectrum, doComputeError=doComputeError)


class MinimizationMonitor:
    """Callback function for scipy.optimize.minimize().

    Parameters
    ----------
    objective : `Callable` [[`np.ndarray`], `float`]
        Objective function
    tol : `float`
        Tolerance. If stddev of objective's return values are less than
        this much, relatively, then ``__call__()`` will raise StopIteration.
        Disabled if not positive.
    windowSize : `int`
        Number of samples to use for calculating moving average.
    log : `logging.Logger`, optional
        Logger.
    """

    objective: Callable[[np.ndarray], float]
    tol: float
    windowSize: int
    log: logging.Logger | None

    fun: float
    """The smallest objective value ever seen"""

    x: np.ndarray
    """The parameter of the objective when it is ``minFun``"""

    window: np.ndarray
    """Recent objective values"""

    nCalls: int
    """Number of calls ever made to ``self.__call__()``."""

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        *,
        tol: float = -1,
        windowSize: int = 10,
        log: logging.Logger | None = None,
    ) -> None:
        self.objective = objective
        self.tol = tol
        self.windowSize = windowSize
        self.log = log
        self.reset()

    def reset(self) -> None:
        """Reset state"""
        self.fun = np.inf
        self.x = np.zeros(shape=())
        self.window = np.zeros(shape=(self.windowSize,), dtype=float)
        self.nCalls = 0

    def __call__(self, xk) -> None:
        """
        Print ``self.objective(xk)`` to the logger if necessary,
        and raises StopIteration if necessary.

        Parameters
        ----------
        xk : `numpy.ndarray`
            Parameter at which to evaluate objective.

        Raises
        ------
        StopIteration
            Raised if minimization should stop.
        """
        fun = self.objective(xk)
        if fun < self.fun:
            self.fun = fun
            self.x = np.copy(xk)
            if self.log is not None:
                self.log.debug("smallest objective ever found: %s", fun)

        self.nCalls += 1
        self.window[:-1] = self.window[1:]
        self.window[-1] = fun

        if not (self.tol > 0) or self.nCalls < self.windowSize:
            return

        mean = np.mean(self.window)
        variance = np.var(self.window, ddof=1)
        if variance < (self.tol * mean) ** 2:
            if self.log is not None:
                self.log.debug("Minimization stops because var(objective) is small enough.")
            raise StopIteration()


@dataclasses.dataclass
class PhotometryPair:
    """Broadband photometry pair: a truth value (observed value) and a model
    value to be fitted to the truth value.

    Parameters
    ----------
    truth : `float`
        True flux.
    model : `float`
        Model flux to be fitted to the truth.
    truthError : `float`
        Error, such that ``(truth - model)**2 / (truthError**2 + modelError**2)``
        will be chi^2.
    modelError : `float`
        Error, such that ``(truth - model)**2 / (truthError**2 + modelError**2)``
        will be chi^2.
    filterName : `str`
        Filter name.
    """

    truth: float
    model: float
    truthError: float
    modelError: float
    filterName: str


class BroadbandFluxChi2:
    """Chi^2 computed by comparing a spectrum with broadband fluxes.

    Parameters
    ----------
    pfsConfig : `PfsConfig`
        PFS fiber configuration.
    pfsMerged : `PfsMerged`
        Merged spectra.
    broadbandFluxType : {"fiber", "psf", "total"}
        Type of broadband flux to use.
    badMask : `list` [`str`]
        Mask planes for bad pixels.
    softenBroadbandFluxErr: `float`
        Soften broadband flux errors: err**2 -> err**2 + (soften*flux)**2
    smoothFilterWidth : `float`
        Width (nm) of smoothing filter.
        (A copy of) ``pfsMerged`` will be made smooth with this filter.
        Disabled if it is zero or negative.
    minIntegrandWavelength: float,
        Minimum wavelength in the spectra to integrate.
    maxIntegrandWavelength: float,
        Maximum wavelength in the spectra to integrate.
    log : `logging.Logger`, optional
        Logger.
    """

    def __init__(
        self,
        pfsConfig: PfsConfig,
        pfsMerged: PfsMerged,
        broadbandFluxType: Literal["fiber", "psf", "total"],
        badMask: list[str],
        softenBroadbandFluxErr: float,
        smoothFilterWidth: float,
        minIntegrandWavelength: float,
        maxIntegrandWavelength: float,
        log: logging.Logger | None,
    ) -> None:
        self.log = log
        self.badMask = badMask
        self.softenBroadbandFluxErr = softenBroadbandFluxErr

        self.obsSpectra: dict[int, PfsSingle] = {
            fiberId: pfsMerged.extractFiber(PfsSingle, pfsConfig, fiberId) for fiberId in pfsConfig.fiberId
        }

        if broadbandFluxType == "fiber":
            bbFluxList = pfsConfig.fiberFlux
            bbFluxErrList = pfsConfig.fiberFluxErr
        elif broadbandFluxType == "psf":
            bbFluxList = pfsConfig.psfFlux
            bbFluxErrList = pfsConfig.psfFluxErr
        elif broadbandFluxType == "total":
            bbFluxList = pfsConfig.totalFlux
            bbFluxErrList = pfsConfig.totalFluxErr
        else:
            raise ValueError(f"`broadbandFluxType` must be one of fiber|psf|total. ('{broadbandFluxType}')")

        self.arms: dict[int, str] = getExistentArms(pfsMerged)

        self.bbFlux: dict[int, list[tuple[float, float, str]]] = {
            fiberId: list(zip(bbFlux, bbFluxErr, filterNames))
            for fiberId, bbFlux, bbFluxErr, filterNames in zip(
                pfsConfig.fiberId, bbFluxList, bbFluxErrList, pfsConfig.filterNames
            )
        }

        for fiberId in pfsConfig.fiberId:
            spectrum = self.obsSpectra[fiberId]
            wavelenPerPix = np.nanmedian(spectrum.wavelength[1:] - spectrum.wavelength[:-1])
            filterWidthInPix = 2 * int(round(smoothFilterWidth / (2 * wavelenPerPix) - 0.5)) + 1
            if filterWidthInPix > 1:
                # We modify only `flux` member, and we don't touch `covar`.
                # It won't be a problem because we are only interested in the
                # integral of `flux` and the integral's error bar.
                # The integral's theoretical statistical error won't change
                # much when we apply a median filter to `flux` here.
                # Furthermore, if we were to correct the variance layer here,
                # the computed error bar would be very bad because of loss of
                # off-diagonal covariances.
                spectrum.flux[:] = median_filter(
                    spectrum.flux,
                    size=filterWidthInPix,
                    mode="reflect",
                )

                spectrum.flux[
                    (spectrum.wavelength < minIntegrandWavelength)
                    | (maxIntegrandWavelength < spectrum.wavelength)
                ] = np.nan

        self.fiberIdToPhotometries: dict[int, list[PhotometryPair]] = {}
        self.photometer = Photometerer()

    def __call__(self, fiberId: np.ndarray, fluxCalib: np.ndarray, *, l1=False, save=False) -> float:
        """Compute chi^2.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            List of fiber IDs.
        fluxCalib : `numpy.ndarray` of `float`, shape ``(N, M)``
            Flux calibration vector, such that (observed flux) / (fluxCalib)
            will be a calibrated flux.
        l1 : `bool`
            If True, the return value is not chi^2 but an L1 loss.
        save : `bool`
            Save to ``self`` the results of photometering ``spectra/fluxCalib``.
            The saved results will be used when you call ``self.rescaleFluxCalib()``
            later.

        Returns
        -------
        chi2 : float
            Chi^2.
        """
        soften = self.softenBroadbandFluxErr
        lossFunc = self._getLossFunction(l1=l1)
        chi2 = 0.0
        fiberIdToPhotometries: dict[int, list[PhotometryPair]] = {}

        for fId, calib in zip(fiberId, fluxCalib):
            spectrum = self.obsSpectra[fId]
            calibrated = PfsSingle(
                spectrum.target,
                spectrum.observations,
                spectrum.wavelength,
                np.copy(spectrum.flux),
                np.copy(spectrum.mask),
                np.copy(spectrum.sky),
                np.copy(spectrum.covar),
                np.copy(spectrum.covar2),
                spectrum.flags,
            )
            isGood = (
                np.isfinite(calibrated.flux)
                & np.isfinite(calibrated.variance)
                & np.isfinite(calib)
                & (calib != 0)
            )
            calibrated.wavelength = calibrated.wavelength[isGood]
            calibrated.flux = calibrated.flux[isGood]
            calibrated.mask = calibrated.mask[isGood]
            calibrated.sky = calibrated.sky[isGood]
            calibrated.covar = calibrated.covar[:, isGood]
            # (TODO: I don't know what to do with `spectrum.covar2`)
            calib = calib[isGood]

            calibrated /= calib
            photometries: list[PhotometryPair] = []

            for bbFlux, bbFluxErr, filterName in self.bbFlux[fId]:
                if np.isfinite(bbFlux) and bbFluxErr > 0:
                    photometry, photoError = self.photometer(calibrated, filterName, doComputeError=True)
                    relativeErr = (bbFlux - photometry) / math.hypot(bbFluxErr, photoError, soften * bbFlux)
                    chi2 += lossFunc(relativeErr)
                    if save:
                        photometries.append(
                            PhotometryPair(
                                truth=bbFlux,
                                model=photometry,
                                truthError=bbFluxErr,
                                modelError=photoError,
                                filterName=filterName,
                            )
                        )

            if save:
                fiberIdToPhotometries[fId] = photometries

        if save:
            self.fiberIdToPhotometries = fiberIdToPhotometries

        return chi2

    def rescaleFluxCalib(self, fiberId: np.ndarray, scales: np.ndarray, *, l1=False) -> float:
        """Recompute chi^2, rescaling by ``scales`` the ``fluxCalib`` argument
        of the last call to ``self.__call__(..., save=True)``

        Note that model fluxes saved by ``self.__call__(..., save=True)``
        are _divided_ (as opposed to "multiplied") by ``scales`` because
        ``fluxCalib`` is to divide observed spectra.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            List of fiber IDs.
        scales : `numpy.ndarray` of `float`, shape ``(N,)``
            Scales by which to multiply ``fluxCalib``.
        l1 : `bool`
            If True, the return value is not chi^2 but an L1 loss.

        Returns
        -------
        chi2 : float
            Chi^2.
        """
        soften = self.softenBroadbandFluxErr
        lossFunc = self._getLossFunction(l1=l1)
        chi2 = 0.0

        for fId, scale in zip(fiberId, scales):
            for pair in self.fiberIdToPhotometries[fId]:
                relativeErr = (pair.truth - pair.model / scale) / math.hypot(
                    pair.truthError, pair.modelError / scale, soften * pair.truth
                )
                chi2 += lossFunc(relativeErr)

        return chi2

    def rescaleFluxCalibEx(
        self, fiberId: np.ndarray, scales: np.ndarray, wavelengths: np.ndarray, *, l1=False
    ) -> float:
        """Recompute chi^2, rescaling by ``scales`` the ``fluxCalib`` argument
        of the last call to ``self.__call__(..., save=True)``

        The rescaling is not mathematically exact, but is sufficiently accurate
        for preliminary steps of minimization.

        Note that model fluxes saved by ``self.__call__(..., save=True)``
        are _divided_ (as opposed to "multiplied") by ``scales`` because
        ``fluxCalib`` is to divide observed spectra.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            List of fiber IDs.
        scales : `numpy.ndarray` of `float`, shape ``(N,M)``
            Scales by which to multiply ``fluxCalib``.
        wavelengths : `numpy.ndarray` of `float`, shape ``(N,M)``
            wavelengths corresponding to ``scales``
        l1 : `bool`
            If True, the return value is not chi^2 but an L1 loss.

        Returns
        -------
        chi2 : float
            Chi^2.
        """
        soften = self.softenBroadbandFluxErr
        lossFunc = self._getLossFunction(l1=l1)
        chi2 = 0.0

        for fId, scale, wavelength in zip(fiberId, scales, wavelengths):
            spectrum = self.obsSpectra[fId]
            nSamples = len(scale)
            scaleArray = PfsSingle(
                spectrum.target,
                spectrum.observations,
                wavelength,
                scale,
                np.zeros(shape=(nSamples,)),
                np.zeros(shape=(nSamples,)),
                np.ones(
                    shape=(
                        3,
                        nSamples,
                    )
                ),
                np.ones_like(spectrum.covar2),
                spectrum.flags,
            )

            for pair in self.fiberIdToPhotometries[fId]:
                s = self.photometer(scaleArray, pair.filterName)
                relativeErr = (pair.truth - pair.model / s) / math.hypot(
                    pair.truthError, pair.modelError / s, soften * pair.truth
                )
                chi2 += lossFunc(relativeErr)

        return chi2

    @contextlib.contextmanager
    def temporarilyCalibrateFlux(
        self, fiberId: np.ndarray, fluxCalib: np.ndarray
    ) -> Generator[None, None, None]:
        """Divide observed spectra of flux standards by a calibration vector
        temporarily.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            List of fiber IDs.
        fluxCalib : `numpy.ndarray` of `float`, shape ``(N, M)``
            Flux calibration vector, such that (observed flux) / (fluxCalib)
            will be a calibrated flux.
        """
        originalObsSpectra = dict(self.obsSpectra)
        originalBBFlux = dict(self.bbFlux)

        for fId, calib in zip(fiberId, fluxCalib):
            bbFlux = list(self.bbFlux[fId])

            spectrum = self.obsSpectra[fId]
            spectrum = PfsSingle(
                spectrum.target,
                spectrum.observations,
                spectrum.wavelength,
                np.copy(spectrum.flux),
                np.copy(spectrum.mask),
                np.copy(spectrum.sky),
                np.copy(spectrum.covar),
                np.copy(spectrum.covar2),
                spectrum.flags,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spectrum /= calib

            self._addressAbsentArms(spectrum, bbFlux, self.arms[fId])

            self.bbFlux[fId] = bbFlux
            self.obsSpectra[fId] = spectrum

        try:
            yield
        finally:
            self.bbFlux = originalBBFlux
            self.obsSpectra = originalObsSpectra

    def _addressAbsentArms(
        self, spectrum: PfsSingle, bbFlux: list[tuple[float, float, str]], arms: str
    ) -> None:
        """Address the problem arising from absent arms
        (see PIPE2D-1427 and PIPE2D-1596).

        If an arm is absent, spectrum in its wavelength range is not available.
        Because we compare broadband fluxes with integrations of spectra,
        sampling points absent from the integration ranges lead to erroneous
        chi^2.

        To address this problem, we throw away broadband fluxes if the bands
        are hardly covered by the observed spectra. In addition, we interpolate
        or extrapolate the observed spectra so that they will cover entire
        bands of broadband filters.

        Parameters
        ----------
        spectrum : `PfsSingle`
            Spectrum. Must be flux-calibrated, at least roughly.
        bbFlux : `list` [`tuple` [`float`, `float`, `str`]]
            Broadband fluxes. Each element of the list is a tuple of
            ``(flux, fluxErr, filterName)``
        arms : `str`
            Existent arms. "brn" and "bmn" for example.
        """
        if not bbFlux:
            return

        suffixes = list(
            set(re.search(r"[^_]*\Z", filterName).group() for flux, fluxErr, filterName in bbFlux)
        )
        if len(suffixes) != 1:
            if self.log:
                self.log.warning(
                    "Flux standard is discarded"
                    " because broadband photometries are from multiple instruments: %s",
                    spectrum.getIdentity(),
                )
            bbFlux[:] = []
            return

        (instrument,) = suffixes

        if instrument in ["hsc", "ps1"]:
            if arms == "brn":
                # No tweak is required.
                return

            if arms == "rn":
                bbFlux[:] = [
                    (flux, fluxErr, filterName)
                    for flux, fluxErr, filterName in bbFlux
                    if filterName.startswith(("i", "z", "y"))
                ]
                return

            if arms == "br":
                bbFlux[:] = [
                    (flux, fluxErr, filterName)
                    for flux, fluxErr, filterName in bbFlux
                    if filterName.startswith(("g", "r", "i", "z"))
                ]
                return

            if arms == "bmn":
                interpolateLinearly(spectrum, (590, 630), (720, 760), self.badMask, mode="interpolate")
                interpolateLinearly(spectrum, (835, 875), (970, 1010), self.badMask, mode="interpolate")
                return

            if arms == "mn":
                bbFlux[:] = [
                    (flux, fluxErr, filterName)
                    for flux, fluxErr, filterName in bbFlux
                    if filterName.startswith(("i", "z", "y"))
                ]
                interpolateLinearly(spectrum, (720, 740), (740, 760), self.badMask, mode="extrapolate-left")
                interpolateLinearly(spectrum, (835, 875), (970, 1010), self.badMask, mode="interpolate")
                return

            if arms == "bm":
                bbFlux[:] = [
                    (flux, fluxErr, filterName)
                    for flux, fluxErr, filterName in bbFlux
                    if filterName.startswith(("g", "r", "i"))
                ]
                interpolateLinearly(spectrum, (590, 630), (720, 760), self.badMask, mode="interpolate")
                return

        if instrument in ["gaia"]:
            if arms == "brn":
                # No tweak is required.
                return

            if arms == "rn":
                bbFlux[:] = [
                    (flux, fluxErr, filterName)
                    for flux, fluxErr, filterName in bbFlux
                    if filterName.startswith("rp")
                ]
                interpolateLinearly(spectrum, (650, 670), (670, 690), self.badMask, mode="extrapolate-left")
                return

            if arms == "br":
                interpolateLinearly(spectrum, (900, 920), (920, 940), self.badMask, mode="extrapolate-right")
                return

            if arms == "bmn":
                interpolateLinearly(spectrum, (590, 630), (720, 760), self.badMask, mode="interpolate")
                interpolateLinearly(spectrum, (835, 875), (970, 1010), self.badMask, mode="interpolate")
                return

            if arms == "mn":
                bbFlux[:] = []  # We ignore this flux standard completely.
                return

            if arms == "bm":
                bbFlux[:] = [
                    (flux, fluxErr, filterName)
                    for flux, fluxErr, filterName in bbFlux
                    if filterName.startswith(("g", "bp"))
                ]
                interpolateLinearly(spectrum, (590, 630), (720, 760), self.badMask, mode="interpolate")
                interpolateLinearly(spectrum, (835, 855), (855, 875), self.badMask, mode="extrapolate-right")
                return

        if self.log:
            self.log.warning(
                'Flux standard is not used because combination of arms is unexpected: "%s" (%s)',
                arms,
                spectrum.getIdentity(),
            )
        bbFlux[:] = []
        return

    @staticmethod
    def _getLossFunction(*, l1: bool) -> Callable[[float], float]:
        """Get a loss function.

        The loss function takes a relative error and returns a loss.
        The returned loss function is of class C^oo.

        Parameters
        ----------
        l1 : `bool`
            If True, an L1 loss function is returned.
            If False, an L2 loss function is returned.

        Returns
        -------
        lossFunc : `Callable` [[`float`], `float`]
            loss function.
        """
        if l1:

            def l1Loss(relativeErr: float) -> float:
                """L1 loss function.

                This function is not a hard L1 loss function.
                It is smooth around relativeErr=0.

                Parameters
                ----------
                relativeErr : `float`
                    Relative error.

                Returns
                -------
                loss : `float`
                    Loss.
                """
                # This function \simeq l2loss(relativeErr) for relativeErr ~ 0,
                # and \simeq |relativeErr| for relativeErr >> 1.
                return math.hypot(0.5, relativeErr) - 0.5

            return l1Loss
        else:

            def l2Loss(relativeErr: float) -> float:
                """L2 loss function.

                Parameters
                ----------
                relativeErr : `float`
                    Relative error.

                Returns
                -------
                loss : `float`
                    Loss.
                """
                return relativeErr**2

            return l2Loss


def fitFluxCalibToArrays(
    fiberId: np.ndarray,
    wavelengths: np.ndarray,
    values: np.ndarray,
    masks: np.ndarray,
    variances: np.ndarray,
    positions: np.ndarray,
    *,
    robust: bool,
    polyOrder: int,
    polyWavelengthDependent: bool,
    fitPrecisely: bool,
    scales: np.ndarray | None,
    bbChi2: BroadbandFluxChi2,
    tol: float,
    log: logging.Logger | None,
    **kwargs,
) -> "FluxCalib":
    """Fit `FluxCalib` to arrays

    Parameters
    ----------
    fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
        Fiber identifiers.
    wavelengths : `numpy.ndarray` of `float`, shape ``(N, M)``
        Wavelength array. The wavelength array for all the inputs must be
        identical.
    values : `numpy.ndarray` of `float`, shape ``(N, M)``
        Values to fit.
    masks : `numpy.ndarray` of `bool`, shape ``(N, M)``
        Boolean array indicating values to ignore from the fit.
    variances : `numpy.ndarray` of `float`, shape ``(N, M)``
        Variance values to use in fit.
    positions : `numpy.ndarray` of `float`, shape ``(N, 2)``
        Focal-plane positions of fibers.
    robust : `bool`
        Perform robust fit? A robust fit should provide an accurate answer
        in the presense of outliers, even if the answer is less precise
        than desired. A non-robust fit should provide the most precise
        answer while assuming there are no outliers.
    polyOrder : `int`
        Order of the polynomial to be fitted so as to map ``positions``
        to ``scales`` approximately.
    polyWavelengthDependent : `bool`
        Whether the polynomial is wavelength-dependent.
    fitPrecisely : `bool`
        If False, skip time-consuming refinement phase of fitting.
    scales : `numpy.ndarray` of `float`, shape ``(N,)``
        Overall scale for each spectrum. The spectra are divided by
        ``scales`` before being averaged.
    bbChi2 : `BroadbandFluxChi2`
        Function to compute how distant broadband fluxes will be from those
        stored in ``pfsConfig``, when certain flux calib vectors are given.
    tol : `float`
        Tolerance of minimization.
    log : `logging.Logger`, optional
        Logger.
    **kwargs : `dict`
        Fitting parameters.

    Returns
    -------
    fit : `FluxCalib`
        Function fit to input arrays.
    """
    if scales is None:
        scales = np.nanmean(values, axis=(1,))

    constantFocalPlaneFunction = ConstantFocalPlaneFunction.fitArrays(
        fiberId,
        wavelengths,
        values / scales.reshape(len(scales), 1),
        masks,
        variances / np.square(scales).reshape(len(scales), 1),
        positions,
        robust=robust,
    )

    averageCalibVector = constantFocalPlaneFunction.evaluate(wavelengths, fiberId, positions).values

    with bbChi2.temporarilyCalibrateFlux(fiberId, averageCalibVector):
        # Because we have already divided the observed flux by `averageCalibVector`
        # we reset the divider to 1.0
        averageCalibVector = np.ones_like(averageCalibVector)

        # Save to `bbChi2` the result of photometries with the use of this
        # average flux calibration vector.
        bbChi2(fiberId, averageCalibVector, save=True)

        posMin = np.min(positions, axis=(0,))
        posMax = np.max(positions, axis=(0,))
        wlMin = wavelengths[0, 0]
        wlMax = wavelengths[0, -1]
        polyMin = np.array(list(posMin) + [wlMin], dtype=float)
        polyMax = np.array(list(posMax) + [wlMax], dtype=float)

        # First, fit a function independent of \lambda.
        def objective1(params: np.ndarray) -> float:
            """Objective function to minimize.

            Parameters
            ----------
            params : `numpy.ndarray`
                Parameters of `NormalizedPolynomialND`.

            Returns
            -------
            objective : `float`
                Objective.
            """
            poly = NormalizedPolynomialND(params, posMin, posMax)
            scales = np.exp(poly(positions))
            return bbChi2.rescaleFluxCalib(fiberId, scales, l1=robust)

        monitor1 = MinimizationMonitor(objective1, tol=tol, log=log)
        params = NormalizedPolynomialND(polyOrder, posMin, posMax).getParams()

        if log is not None:
            log.debug("Start phase-1 fitting...")

        try:
            result = minimize(objective1, params, callback=monitor1)
            # TBD: Should we test whether minimization has succeeded?
            params = result.x
        except StopIteration:
            # With old scipy, `StopIteration` raised by ``callback``
            # is not caught by ``minimize()``. So we catch it for ourselves.
            params = monitor1.x

        params = NormalizedPolynomialND.getParamsFromLowerVariatePoly(params, [0, 1, None])

        if not polyWavelengthDependent:
            return FluxCalib(params, polyMin, polyMax, constantFocalPlaneFunction)

        # Second, fit a function dependent on \lambda, with approximate chi^2.

        # Low resolution wavelength at which to evaluate the fitted polynomial.
        lowResWL = np.linspace(wlMin, wlMax, num=int(round(wlMax - wlMin)))
        # List of (x, y, lam) at which to evaluate the fitted polynomial.
        polyArgs = np.empty(shape=(len(fiberId), len(lowResWL), 3), dtype=float)
        polyArgs[:, :, :2] = positions.reshape(len(fiberId), 1, 2)
        polyArgs[:, :, 2] = lowResWL.reshape(1, -1)

        def objective2(params: np.ndarray) -> float:
            """Objective function to minimize.

            Parameters
            ----------
            params : `numpy.ndarray`
                Parameters of `NormalizedPolynomialND`.

            Returns
            -------
            objective : `float`
                Objective.
            """
            poly = NormalizedPolynomialND(params, polyMin, polyMax)
            scales = np.exp(poly(polyArgs))
            return bbChi2.rescaleFluxCalibEx(fiberId, scales, polyArgs[:, :, 2], l1=robust)

        monitor2 = MinimizationMonitor(objective2, tol=tol, log=log)
        if log is not None:
            log.debug("Start phase-2 fitting...")

        try:
            result = minimize(objective2, params, callback=monitor2)
            # TBD: Should we test whether minimization has succeeded?
            params = result.x
        except StopIteration:
            # With old scipy, `StopIteration` raised by ``callback``
            # is not caught by ``minimize()``. So we catch it for ourselves.
            params = monitor2.x

        if robust or not fitPrecisely:
            # Phase-3 takes time. We skip it unless this is the last lap of
            # a clipping loop (when robust=True), or unless fitPrecisely=True.
            return FluxCalib(params, polyMin, polyMax, constantFocalPlaneFunction)

        # Third, fit a function dependent on \lambda, with accurate chi^2.

        # List of (x, y, lam) at which to evaluate the fitted polynomial.
        polyArgs = np.empty(shape=values.shape + (3,), dtype=float)
        polyArgs[:, :, :2] = positions.reshape(len(fiberId), 1, 2)
        polyArgs[:, :, 2] = wavelengths

        def objective3(params: np.ndarray) -> float:
            """Objective function to minimize.

            Parameters
            ----------
            params : `numpy.ndarray`
                Parameters of `NormalizedPolynomialND`.

            Returns
            -------
            objective : `float`
                Objective.
            """
            poly = NormalizedPolynomialND(params, polyMin, polyMax)
            fluxCalib = np.exp(poly(polyArgs))
            fluxCalib *= averageCalibVector
            return bbChi2(fiberId, fluxCalib, l1=robust)

        monitor3 = MinimizationMonitor(objective3, tol=tol, log=log)
        if log is not None:
            log.debug("Start phase-3 fitting...")

        try:
            result = minimize(objective3, params, callback=monitor3)
            # TBD: Should we test whether minimization has succeeded?
            params = result.x
        except StopIteration:
            # With old scipy, `StopIteration` raised by ``callback``
            # is not caught by ``minimize()``. So we catch it for ourselves.
            params = monitor3.x

        return FluxCalib(params, polyMin, polyMax, constantFocalPlaneFunction)


def getExistentArms(pfsMerged: PfsMerged) -> dict[int, str]:
    """Get the set of arms, per fiber, that were present when ``pfsMerged``
    was observed.

    This function guesses used arms from the observed spectra instead of
    referring to ``pfsConfig.arms`` or ``pfsMerged.identity.arm`` because
    they are per-visit information.

    Parameters
    ----------
    pfsMerged : `PfsMerged`
        Merged spectra from exposure.

    Returns
    -------
    arms : `dict` [`int`, `str`]
        Mapping from fiberId to one-letter arm names concatenated to be a
        string (e.g. "brn").
    """
    arms: dict[int, str] = {}

    if "NO_DATA" in pfsMerged.flags:
        noData = pfsMerged.flags.get("NO_DATA")
    else:
        noData = 0

    for i, fiberId in enumerate(pfsMerged.fiberId):
        good = 0 == (pfsMerged.mask[i] & noData)
        wavelength = pfsMerged.wavelength[i][good]

        wlRangeList = [
            [450.0, 550.0],
            [750.0, 850.0],
            [910.0, 930.0],
            [1050.0, 1150.0],
        ]

        flags = [np.any((wlMin < wavelength) & (wavelength < wlMax)) for wlMin, wlMax in wlRangeList]

        arms[fiberId] = "".join(
            [
                "b" if flags[0] else "",
                "r" if flags[1] and flags[2] else "",
                "m" if flags[1] and not flags[2] else "",
                "n" if flags[3] else "",
            ]
        )

    return arms


def interpolateLinearly(
    spectrum: PfsSingle,
    wlRange1: tuple[float, float],
    wlRange2: tuple[float, float],
    badMask: list[str],
    mode: Literal["interpolate", "extrapolate-left", "extrapolate-right"],
) -> None:
    """Interpolate or extrapolate spectrum linearly.

    Median fluxes ``flux1``, ``flux2`` are computed in ``wlRange1`` and
    ``wlRange2``. The interpolation line is drawn through ``flux1`` and
    ``flux2``.

    Parameters
    ----------
    spectrum : `PfsSingle`
        Spectrum.
    wlRange1 : `tuple` [`float`, `float`]
        Wavelength range (nm) in which to compute ``flux1``.
    wlRange2 : `tuple` [`float`, `float`]
        Wavelength range (nm) in which to compute ``flux2``.
    badMask : `list` [`str`]
        Mask planes for bad pixels.
    mode : {"interpolate", "extrapolate-left", "extrapolate-right"}
        If mode="interpolate", fluxes between ``wlRange1[1]`` and
        ``wlRange2[0]`` get replaced.
        If mode="extrapolate-left", fluxes on the left of `wlRange1[0]`` get
        replaced.
        If mode="extrapolate-right", fluxes on the right of `wlRange2[1]`` get
        replaced.
    """
    bits = spectrum.flags.get(*(m for m in badMask if m in spectrum.flags))
    isGood = np.isfinite(spectrum.flux) & (0 == (spectrum.mask & bits))

    index1 = isGood & (wlRange1[0] < spectrum.wavelength) & (spectrum.wavelength < wlRange1[1])
    index2 = isGood & (wlRange2[0] < spectrum.wavelength) & (spectrum.wavelength < wlRange2[1])

    wl1 = np.mean(spectrum.wavelength[index1])
    wl2 = np.mean(spectrum.wavelength[index2])
    # We use median instead of mean because we don't want absorption lines
    # to affect the results.
    flux1 = np.median(spectrum.flux[index1])
    flux2 = np.median(spectrum.flux[index2])
    covar1 = np.median(spectrum.covar[:, index1], axis=1, keepdims=True)
    covar2 = np.median(spectrum.covar[:, index2], axis=1, keepdims=True)

    if mode == "interpolate":
        interpoland = (wlRange1[1] < spectrum.wavelength) & (spectrum.wavelength < wlRange2[0])
    elif mode == "extrapolate-left":
        interpoland = spectrum.wavelength < wlRange1[0]
    elif mode == "extrapolate-right":
        interpoland = wlRange2[1] < spectrum.wavelength
    else:
        raise ValueError(f"Invalid mode: '{mode}'")

    spectrum.flux[interpoland] = flux1 + (flux2 - flux1) / (wl2 - wl1) * (
        spectrum.wavelength[interpoland] - wl1
    )

    # Because this spectrum will be photometered, we have to invent some
    # good-looking variance. We interpolate variances just like fluxes,
    # though it is wrong.
    wl1 = wl1.reshape(1, -1)
    wl2 = wl2.reshape(1, -1)
    spectrum.covar[:, interpoland] = covar1 + (covar2 - covar1) / (wl2 - wl1) * (
        spectrum.wavelength[interpoland] - wl1
    )

    # We reset all masks for interpolated points because we don't want other
    # functions to ignore the interpolated points.
    spectrum.mask[interpoland] = 0


class FitFluxCalibFocalPlaneFunctionConfig(FitFocalPlaneConfig):
    """Configuration for fitting a `FluxCalib`

    The ``FluxCalib.fit`` method also needs ``scales`` and ``bbChi2``, ``tol``, ``log``
    input parameters, but those can be determined from the data.
    """

    polyOrder = Field(dtype=int, default=5, doc="Polynomial order")
    polyWavelengthDependent = Field(
        dtype=bool,
        default=True,
        doc="Whether the polynomial is wavelength-dependent.",
    )
    fitPrecisely = Field(
        dtype=bool, default=True, doc="If False, skip time-consuming refinement phase of fitting."
    )


class FitFluxCalibFocalPlaneFunctionTask(FitFocalPlaneTask):
    """Fit a `FluxCalib`"""

    ConfigClass = FitFluxCalibFocalPlaneFunctionConfig
    Function = FluxCalib


class FitFluxCalConfig(PipelineTaskConfig, pipelineConnections=FluxCalibrateConnections):
    """Configuration for FitFluxCalTask"""

    sysErr = Field(
        dtype=float,
        default=1.0e-4,
        doc=(
            "Fraction of value to add to variance before fitting. This attempts to offset the "
            "loss of variance as covariance when we resample, the result of which is "
            "underestimated errors and excess rejection."
        ),
    )
    fitFocalPlane = ConfigurableField(
        target=FitFluxCalibFocalPlaneFunctionTask, doc="Fit flux calibration model"
    )
    fluxTable = ConfigurableField(target=FluxTableTask, doc="Flux table")
    doWrite = Field(dtype=bool, default=True, doc="Write outputs?")
    adjustCalibVectorsRangeStart = Field(
        dtype=float, default=600, doc="Start of wavelength range to use for height adjustment [nm]."
    )
    adjustCalibVectorsRangeStop = Field(
        dtype=float, default=700, doc="Stop of wavelength range to use for height adjustment [nm]."
    )
    badMask = ListField(
        dtype=str, default=["BAD", "SAT", "CR", "NO_DATA", "SUSPECT"], doc="Mask planes for bad pixels"
    )
    broadbandFluxType = ChoiceField(
        doc="Type of broadband fluxes to use.",
        dtype=str,
        allowed={
            "fiber": "Use `psfConfig.fiberFlux`.",
            "psf": "Use `psfConfig.psfFlux`.",
            "total": "Use `psfConfig.totalFlux`.",
        },
        default="psf",
        optional=False,
    )
    softenBroadbandFluxErr = Field(
        doc="Soften broadband flux errors: err**2 -> err**2 + (soften*flux)**2",
        dtype=float,
        default=0.01,
        optional=False,
    )
    fabricatedBroadbandFluxErrSNR = Field(
        dtype=float,
        default=0,
        doc="If positive, fabricate flux errors in pfsConfig if all of them are NaN"
        " (for old engineering data). The fabricated flux errors are such that S/N is this much.",
    )
    smoothFilterWidth = Field(
        dtype=float,
        default=1.0,
        doc="Width of smoothing filter (median filter) applied to spectra"
        " before they are used to compute broadband photometry [nm].",
    )
    minIntegrandWavelength = Field(
        dtype=float,
        default=380,
        doc="Mininum wavelength in the spectra to integrate for broadband photometry [nm].",
    )
    maxIntegrandWavelength = Field(
        dtype=float,
        default=math.inf,
        doc="Maximum wavelength in the spectra to integrate for broadband photometry [nm].",
    )
    minimizationTolerance = Field(
        dtype=float,
        default=1e-3,
        doc="Minimizer stops when `stddev(f) < minimizationTolerance * f`",
    )


class FitFluxCalTask(PipelineTask):
    """Measure and apply the flux calibration"""

    ConfigClass = FitFluxCalConfig
    _DefaultName = "fitFluxCal"

    fitFocalPlane: FitFluxCalibFocalPlaneFunctionTask
    fluxTable: FluxTableTask

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.makeSubtask("fitFocalPlane")
        self.makeSubtask("fluxTable")

        self.debugInfo = lsstDebug.Info(__name__)

    def run(
        self,
        pfsMerged: PfsMerged,
        pfsMergedLsf: LsfDict,
        references: PfsFluxReference,
        pfsConfig: PfsConfig,
        pfsArmList: list[PfsArm],
        sky1dList: Iterable[FluxCalib],
    ) -> Struct:
        """Measure and apply the flux calibration

        Parameters
        ----------
        pfsMerged : `PfsMerged`
            Merged spectra, containing observations of ``FLUXSTD`` sources.
        pfsMergedLsf : `LsfDict`
            Line-spread functions for merged spectra.
        references : `PfsFluxReference`
            Reference spectra.
        pfsConfig : `PfsConfig`
            PFS fiber configuration.
        pfsArmList : iterable of `PfsArm`
            List of extracted spectra, for constructing the flux table.
        sky1dList : iterable of `FluxCalib`
            Corresponding list of 1d sky subtraction models.

        Returns
        -------
        fluxCal : `FluxCalib`
            Flux calibration solution.
        pfsCalibrated : `PfsCalibratedSpectra`
            Calibrated spectra.
        pfsCalibratedLsf : `LsfDict`
            Line-spread functions for calibrated spectra.
        """
        removeBadFluxes(pfsConfig, self.config.broadbandFluxType, self.config.fabricatedBroadbandFluxErrSNR)
        fluxCal = self.calculateCalibrations(pfsConfig, pfsMerged, pfsMergedLsf, references)
        fluxCalibrate(pfsMerged, pfsConfig, fluxCal)

        calibrated = []
        fiberToArm = defaultdict(list)
        for ii, (pfsArm, sky1d) in enumerate(zip(pfsArmList, sky1dList)):
            subtractSky1d(pfsArm, pfsConfig, sky1d)
            fluxCalibrate(pfsArm, pfsConfig, fluxCal)
            for ff in pfsArm.fiberId:
                fiberToArm[ff].append(ii)
            calibrated.append(pfsArm)

        selection = pfsConfig.getSelection(fiberStatus=FiberStatus.GOOD)
        selection &= ~pfsConfig.getSelection(targetType=TargetType.ENGINEERING)
        fiberId = pfsMerged.fiberId[np.isin(pfsMerged.fiberId, pfsConfig.fiberId[selection])]

        pfsCalibrated: dict[Target, PfsSingle] = {}
        pfsCalibratedLsf: dict[Target, Lsf] = {}
        for ff in fiberId:
            extracted = pfsMerged.extractFiber(PfsSingle, pfsConfig, ff)
            extracted.fluxTable = self.fluxTable.run(
                [calibrated[ii].identity.getDict() for ii in fiberToArm[ff]],
                [pfsArmList[ii].extractFiber(PfsSingle, pfsConfig, ff) for ii in fiberToArm[ff]],
            )
            extracted.metadata = getPfsVersions()

            target = extracted.target
            pfsCalibrated[target] = extracted
            pfsCalibratedLsf[target] = pfsMergedLsf[ff]

        return Struct(
            fluxCal=fluxCal,
            pfsCalibrated=PfsCalibratedSpectra(pfsCalibrated.values()),
            pfsCalibratedLsf=LsfDict(pfsCalibratedLsf),
        )

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `QuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        armInputs = readDatasetRefs(butler, inputRefs, "pfsArm", "sky1d")
        inputs = butler.get(inputRefs)

        outputs = self.run(**inputs, pfsArmList=armInputs.pfsArm, sky1dList=armInputs.sky1d)
        butler.put(outputs, outputRefs)

    @classmethod
    def _makeArgumentParser(cls) -> ArgumentParser:
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(
            name="--id", datasetType="pfsMerged", level="Visit", help="data IDs, e.g. --id exp=12345"
        )
        return parser

    def runDataRef(self, dataRef: lsst.daf.persistence.ButlerDataRef) -> Struct:
        """Measure and apply the flux calibration

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for merged spectrum.

        Returns
        -------
        fluxCal : `FluxCalib`
            Flux calibration.
        pfsCalibrated : `PfsCalibratedSpectra`
            Calibrated spectra.
        pfsCalibratedLsf : `LsfDict`
            Line-spread functions for calibrated spectra.
        """
        pfsMerged = dataRef.get("pfsMerged")
        pfsMergedLsf = dataRef.get("pfsMergedLsf")
        pfsConfig = dataRef.get("pfsConfig")
        references = dataRef.get("pfsFluxReference")

        butler = dataRef.getButler()
        armRefList = list(butler.subset("raw", dataId=dataRef.dataId))
        pfsArmList = [armRef.get("pfsArm") for armRef in armRefList]
        sky1dList = [armRef.get("sky1d") for armRef in armRefList]

        outputs = self.run(pfsMerged, pfsMergedLsf, references, pfsConfig, pfsArmList, sky1dList)

        if self.config.doWrite:
            dataRef.put(outputs.fluxCal, "fluxCal")

            # Gen2 writes the pfsCalibrated spectra individually
            for target in outputs.pfsCalibrated:
                pfsSingle = outputs.pfsCalibrated[target]
                dataId = pfsSingle.getIdentity().copy()
                dataId.update(dataRef.dataId)
                self.forceSpectrumToBePersistable(pfsSingle)
                butler.put(pfsSingle, "pfsSingle", dataId)
                butler.put(outputs.pfsCalibratedLsf[target], "pfsSingleLsf", dataId)

        return outputs

    def calculateCalibrations(
        self,
        pfsConfig: PfsConfig,
        pfsMerged: PfsMerged,
        pfsMergedLsf: LsfDict,
        pfsFluxReference: PfsFluxReference,
    ) -> FluxCalib:
        """Model flux calibration over the focal plane

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            PFS fiber configuration.
        pfsMerged : `PfsMerged`
            Merged spectra, containing observations of ``FLUXSTD`` sources.
        pfsMergedLsf : `LsfDict`
            Line-spread functions for merged spectra.
        pfsFluxReference: `pfs.datamodel.pfsFluxReference.PfsFluxReference`
            Model reference template set for flux calibration.

        Returns
        -------
        fluxCal: `FluxCalib`
            Flux calibration.
        """
        c = const.c.to("km/s").value

        # We don't need any flux references with any failure flags
        pfsFluxReference = pfsFluxReference[pfsFluxReference.fitFlag == 0]
        if len(pfsFluxReference) == 0:
            raise RuntimeError("No available flux reference (maybe every fitting procecss has failed)")

        # This is going to be (observed spectra) / (reference spectra)
        calibVectors = pfsMerged[np.isin(pfsMerged.fiberId, pfsFluxReference.fiberId)]

        ref = np.empty_like(calibVectors.flux)
        for i, fiberId in enumerate(calibVectors.fiberId):
            refSpec = pfsFluxReference.extractFiber(PfsSimpleSpectrum, pfsConfig, fiberId)

            # We convolve `refSpec` with LSF before resampling
            # because the resampling interval is not short enough
            # compared to `refSpec`'s inherent LSF.
            refLsf = warpLsf(pfsMergedLsf[fiberId], calibVectors.wavelength[i, :], refSpec.wavelength)
            refSpec.flux = refLsf.computeKernel((len(refSpec) - 1) / 2.0).convolve(refSpec.flux)

            # Then we stretch `refSpec` according to its radial velocity.
            # (Resampling takes place in so doing.)
            # The LSF gets slightly wider or narrower by this operation,
            # but we hope it negligible.
            beta = pfsFluxReference.fitParams["radial_velocity"][i].astype(float) / c
            # `refSpec.wavelength[...]` is not mutable. We replace this member.
            refSpec.wavelength = refSpec.wavelength * np.sqrt((1.0 + beta) / (1.0 - beta))
            refSpec = refSpec.resample(calibVectors.wavelength[i, :])

            ref[i, :] = refSpec.flux

        calibVectors.covar[:, 0] += self.config.sysErr * calibVectors.flux  # add systematic error
        calibVectors /= calibVectors.norm
        calibVectors /= ref
        calibVectors.norm[...] = 1.0  # We're deliberately changing the normalisation

        scales = self.getHeightsOfCalibVectors(calibVectors)

        # TODO: Smooth the flux calibration vectors.

        if self.debugInfo.doWriteCalibVector:
            debugging.writeExtraData(
                f"fitFluxCal-output/calibVector-{pfsMerged.filename}.pickle",
                fiberId=calibVectors.fiberId,
                calibVector=calibVectors.flux,
            )

        # Before the call to `fitFocalPlane`, we have to ensure
        # that all the bad flags in `config.mask` are contained in `flags`.
        # This operation modifies `pfsMerged`, but we hope it won't be harmful.
        for name in self.fitFocalPlane.config.mask:
            calibVectors.flags.add(name)

        fluxStdConfig = pfsConfig[np.isin(pfsConfig.fiberId, pfsFluxReference.fiberId)]
        fluxStdMerged = pfsMerged[np.isin(pfsMerged.fiberId, pfsFluxReference.fiberId)]
        bbChi2 = BroadbandFluxChi2(
            pfsConfig=fluxStdConfig,
            pfsMerged=fluxStdMerged,
            broadbandFluxType=self.config.broadbandFluxType,
            badMask=self.config.badMask,
            softenBroadbandFluxErr=self.config.softenBroadbandFluxErr,
            smoothFilterWidth=self.config.smoothFilterWidth,
            minIntegrandWavelength=self.config.minIntegrandWavelength,
            maxIntegrandWavelength=self.config.maxIntegrandWavelength,
            log=self.log,
        )
        return self.fitFocalPlane.run(
            calibVectors,
            fluxStdConfig,
            fitter=fitFluxCalibToArrays,
            scales=scales,
            bbChi2=bbChi2,
            tol=self.config.minimizationTolerance,
            log=self.log,
        )

    def getHeightsOfCalibVectors(self, calibVectors: PfsMerged) -> np.ndarray:
        """Get relative heights of calib vectors (observed spectra) / (reference spectra).

        The "relative height" of a calib vector is the ratio of
        (typical height of this calib vector) / (typical height of the highest calib vector).

        Parameters
        ----------
        calibVectors : `PfsMerged`
            Calib vectors. ``calibVectors.norm`` must be 1 (constant).

        Returns
        -------
        heights : `numpy.ndarray`
            Relative heights.
        """
        badMask = calibVectors.flags.get(*(m for m in self.config.badMask if m in calibVectors.flags))
        n = len(calibVectors)
        heights = np.empty(shape=(n,), dtype=float)
        for i in range(n):
            selection = (
                (self.config.adjustCalibVectorsRangeStart <= calibVectors.wavelength[i, :])
                & (calibVectors.wavelength[i, :] <= self.config.adjustCalibVectorsRangeStop)
                & (0 == (calibVectors.mask[i, :] & badMask))
            )
            heights[i] = np.nanmedian(calibVectors.flux[i, selection])

        # We use the highest calib vector as the reference to which the other
        # calib vectors are adjusted, for we assume that the fiber of the
        # highest calib vector was positioned better than any other fiber.
        reference = np.nanmax(heights)
        return heights / reference

    def forceSpectrumToBePersistable(self, spectrum: PfsFiberArray) -> None:
        """Force ``spectrum`` to be able to be written to file.

        Parameters
        ----------
        spectrum : `PfsFiberArray`
            An observed spectrum.
        """
        if not (math.isfinite(spectrum.target.ra) and math.isfinite(spectrum.target.dec)):
            # Because target's (ra, dec) is written in the FITS header,
            # these values must be finite.
            self.log.warning(
                "Target's (ra, dec) is not finite. Replaced by 0 in the FITS header (%s)",
                spectrum.getIdentity(),
            )
            # Even if ra or dec is finite, we replace both with zero, for
            # (0, 0) looks more alarming than, say, (9.87654321, 0) to users.
            spectrum.target.ra = 0
            spectrum.target.dec = 0

    def _getMetadataName(self) -> None:
        return None
