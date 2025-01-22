from lsst.pex.config import Config, Field, ChoiceField, ListField
from lsst.pipe.base import Struct, Task
from pfs.datamodel import PfsFiberArray, PfsSimpleSpectrum
from pfs.drp.stella.interpolate import interpolateFlux

import numpy as np
from astropy import constants as const
import scipy.optimize

import math

from typing import Sequence
from typing import Any

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Sequence


__all__ = ["EstimateRadialVelocityConfig", "EstimateRadialVelocityTask"]


class CrossCorrelationFunction:
    r"""Cross correlation function:
    ccf[k] = \sum_{i} flux[i] model(\lambda[i] * scale[k]).

    Parameters
    ----------
    flux : `np.ndarray` of `float`
        Shape ``(numWavelength,)``. The array ``flux`` in the equation above.
    variance : `np.ndarray` of `float`
        Shape ``(numWavelength,)``. Variance of ``flux``.
    scaledModel : `np.ndarray` of `float`
        Shape ``(numVelocities, numWavelength)``.
        The matrix ``model(\lambda[i] * scale[k])`` in the equation above.
        The first index is ``k``, and the second index is ``i``.
    """

    ccf: NDArray[np.float64]
    """Shape ``(numVelocities,)``. Cross correlation function.
    """

    def __init__(
        self, flux: NDArray[np.float64], variance: NDArray[np.float64], scaledModel: NDArray[np.float64]
    ) -> None:
        self.__variance = variance
        self.__scaledModel = scaledModel
        self.ccf = scaledModel @ flux

    def getCovar(self, slice: Any, amplifier: float = 1) -> NDArray[np.float64]:
        """Get the covariance matrix of ``self.ccf``.

        Parameters
        ----------
        slice : `Any`
            Slice to thin out ``self.ccf``.
            It is a boolean array of length ``numVelocities``, for example.
            Pass ``Ellipsis`` to get the entire covariance matrix.
        amplifier : `float`
            A constant by which to multiply all elements of the covariance.
            The multiplication is executed most efficiently.

        Returns
        -------
        covar : `np.ndarray` of `float`
            Covariance matrix of ``ccf[slice]``
        """
        variance = self.__variance
        scaledModel = self.__scaledModel[slice, :]
        if amplifier == 1:
            return (scaledModel * variance.reshape(1, -1)) @ np.transpose(scaledModel)
        elif len(variance) < len(scaledModel) ** 2:
            return (scaledModel * (amplifier * variance.reshape(1, -1))) @ np.transpose(scaledModel)
        else:
            return amplifier * ((scaledModel * variance.reshape(1, -1)) @ np.transpose(scaledModel))


class EstimateRadialVelocityConfig(Config):
    """Configuration for EstimateRadialVelocityTask"""

    findMethod = ChoiceField(
        doc="Peak-finding method.",
        dtype=str,
        allowed={
            "peak": "The sampled point at which the cross-correlation is maximum.",
            "gauss": "Peak of a Gaussian fit to the cross-correlation.",
            "quadratic": "Peak of the quadratic interpolation around the argmax of the cross-correlation.",
        },
        default="quadratic",
        optional=False,
    )

    searchMin = Field(doc="Minimum of searched range of radial velocity, in km/s.", dtype=float, default=-500)

    searchMax = Field(doc="Maximum of searched range of radial velocity, in km/s.", dtype=float, default=500)

    searchStep = Field(
        doc="Step of searched range of radial velocity, in km/s."
        " The actual step may be slightly smaller than this value.",
        dtype=float,
        default=10,
    )

    peakRange = Field(
        doc='Velocity range, in km/s, used in fitting gaussian (valid when `findMethod` = "gauss")',
        dtype=float,
        default=100,
    )

    useCovar = Field(
        doc='Whether to use covariance.  (valid when `findMethod` = "gauss")'
        " If False, use variance only."
        " If covariance is used, the returned error bar will be more correct,"
        " but this task will be far less robust.",
        dtype=bool,
        default=True,
    )

    mask = ListField(
        doc="Mask planes for bad pixels",
        dtype=str,
        default=["BAD", "SAT", "CR", "NO_DATA"],
    )


class EstimateRadialVelocityTask(Task):
    """Estimate the radial velocity."""

    ConfigClass = EstimateRadialVelocityConfig
    _DefaultName = "estimateRadialVelocity"

    def run(self, spectrum: PfsFiberArray, modelSpectrum: PfsSimpleSpectrum) -> Struct:
        """Get the radial velocity of ``spectrum``
        in comparison with ``modelSpectrum``.

        Parameters
        ----------
        spectrum : `pfs.datamodel.pfsFiberArray.PfsFiberArray`
            Observed spectrum.
            It must be whitened (Continuum is 1.0 everywhere.)
        modelSpectrum : `pfs.datamodel.pfsSimpleSpectrum.PfsSimpleSpectrum`
            Model spectrum as ``spectrum`` would be
            were it not for the radial velocity.
            It must be whitened (Continuum is 1.0 everywhere.)

        Returns
        -------
        velocity : `float`
            Radial velocity in km/s.
        error : `float`
            Standard deviation of ``velocity``.
        fail : `bool`
            True if measuring ``velocity`` failed.
        crossCorr : `numpy.array`
            This is a structured array of
            `dtype=[("velocity", float), ("crosscorr", float)]`.
            ``"velocity"`` is radial velocity in km/s.
            ``"crosscorr"`` is cross correlation.
        """
        # TODO: This method should be wholly rewritten so that it will use
        # a log-scaled wavelength for the sake of FFT convolution.
        searchMin = self.config.searchMin
        searchMax = self.config.searchMax
        searchStep = self.config.searchStep
        searchNum = 1 + int(math.ceil((searchMax - searchMin) / searchStep))
        searchVelocity = np.linspace(searchMin, searchMax, num=searchNum, endpoint=True)
        beta = searchVelocity / const.c.to("km/s").value
        doppler = np.sqrt((1.0 + beta) / (1.0 - beta))

        goodIndex = 0 == (
            spectrum.mask & spectrum.flags.get(*(m for m in self.config.mask if m in spectrum.flags))
        )
        wavelength = spectrum.wavelength[goodIndex]
        flux = spectrum.flux[goodIndex] - 1.0
        variance = spectrum.variance[goodIndex]

        goodIndex = 0 == (
            modelSpectrum.mask
            & modelSpectrum.flags.get(*(m for m in self.config.mask if m in modelSpectrum.flags))
        )
        modelWavelength = modelSpectrum.wavelength[goodIndex]
        modelFlux = modelSpectrum.flux[goodIndex] - 1.0

        # Make scaledModel[i,:] = modelSpectrum moving at searchVelocity[i]
        scaledWavelength = wavelength.reshape(1, -1) / doppler.reshape(-1, 1)
        scaledModel = interpolateFlux(
            modelWavelength, modelFlux, scaledWavelength.reshape(-1)
        ).reshape(len(searchVelocity), -1)
        # We divide the model flux by `dopper`
        # assuming that line spectra contribute to the correlation
        # much more than the continuum, not subtracted perfectly, does.
        # If this assumption is wrong, we must not divide it by `dopp`.
        scaledModel /= doppler.reshape(-1, 1)

        # This is cross correlation function
        ccf = CrossCorrelationFunction(flux, variance, scaledModel)

        # c.c.f. is returned to the caller in this format for debugging.
        crossCorr = np.empty(len(ccf.ccf), dtype=[("velocity", float), ("crosscorr", float)])
        crossCorr["velocity"] = searchVelocity
        crossCorr["crosscorr"] = ccf.ccf

        # Find the peak of CCF
        retvalue = self.findPeak(searchVelocity, ccf)
        retvalue.crossCorr = crossCorr
        return retvalue

    def findPeak(self, searchVelocity: NDArray[np.float64], ccf: CrossCorrelationFunction) -> Struct:
        """Find the peak of ``ccf``.

        Parameters
        ----------
        searchVelocity : `np.ndarray` of `float`
            Shape ``(numVelocities)``. Velocity in km/s.
            This is x-axis of ``ccf``.
        ccf : `CrossCorrelationFunction`
            Cross correlation function.

        Returns
        -------
        velocity : `float`
            Radial velocity in km/s.
        error : `float`
            Standard deviation of ``velocity``.
        fail : `bool`
            True if measuring ``velocity`` failed.
        """
        if self.config.findMethod == "peak":
            return findPeakNaive(searchVelocity, ccf, self.config)
        elif self.config.findMethod == "quadratic":
            return findPeakQuadratic(searchVelocity, ccf, self.config)
        elif self.config.findMethod == "gauss":
            return findPeakGauss(searchVelocity, ccf, self.config)
        else:
            raise RuntimeError(f"config.findMethod has a wrong value: '{self.config.findMethod}'.")


def findPeakNaive(
    searchVelocity: NDArray[np.float64], ccf: CrossCorrelationFunction, config: Config
) -> Struct:
    """Find the peak of ``ccf`` naively.

    Parameters
    ----------
    searchVelocity : `np.ndarray` of `float`
        Shape ``(numVelocities)``. Velocity in km/s.
        This is x-axis of ``ccf``.
    ccf : `CrossCorrelationFunction`
        Cross correlation function.
    config : `Config`
        Not used.

    Returns
    -------
    velocity : `float`
        Radial velocity in km/s.
    error : `float`
        Standard deviation of ``velocity``.
        This is always ``nan``.
    fail : `bool`
        True if measuring ``velocity`` failed.
    """
    iMax = np.argmax(ccf.ccf)
    velocity = searchVelocity[iMax]
    fail = iMax == 0 or iMax + 1 == len(ccf.ccf)
    return Struct(velocity=velocity, error=np.nan, fail=fail)


def findPeakQuadratic(
    searchVelocity: NDArray[np.float64], ccf: CrossCorrelationFunction, config: Config
) -> Struct:
    """Find the peak of ``ccf`` with quadratic interpolation.

    Parameters
    ----------
    searchVelocity : `np.ndarray` of `float`
        Shape ``(numVelocities)``. Velocity in km/s.
        This is x-axis of ``ccf``.
    ccf : `CrossCorrelationFunction`
        Cross correlation function.
    config : `Config`
        Not used.

    Returns
    -------
    velocity : `float`
        Radial velocity in km/s.
    error : `float`
        Standard deviation of ``velocity``.
    fail : `bool`
        True if measuring ``velocity`` failed.
    """
    iMax = np.argmax(ccf.ccf)
    velocity = searchVelocity[iMax]
    fail = iMax == 0 or iMax + 1 == len(ccf.ccf)
    if fail:
        return Struct(velocity=velocity, error=np.nan, fail=fail)

    velocityL = searchVelocity[iMax - 1]
    velocityR = searchVelocity[iMax + 1]
    ccfL = ccf.ccf[iMax - 1]
    ccf0 = ccf.ccf[iMax]
    ccfR = ccf.ccf[iMax + 1]

    # Find f(v) = a*v**2 + b*v + c
    # such that ccfL = f(velocityL), ccf0 = f(velocity),
    # and ccfR = f(velocityR).
    m00 = velocityL**2 - velocity**2
    m01 = velocityL - velocity
    m10 = velocityR**2 - velocity**2
    m11 = velocityR - velocity
    x0 = ccfL - ccf0
    x1 = ccfR - ccf0
    det = m00 * m11 - m01 * m10
    a = (m11 * x0 - m01 * x1) / det
    b = (m00 * x1 - m10 * x0) / det

    # Peak position of f(v) = a*v**2 + b*v + c
    rv = -b / (2 * a)

    # \frac{\partial rv}{\partial ccf_i}
    # where ccf_0 = ccfL, ccf_1 = ccf0, ccf_2 = ccfR.
    drv_over_dccf = np.empty(shape=(3, 1), dtype=float)
    drv_over_dccf[0, 0] = (m10 * a + m11 * b) / (2 * det * a**2)
    drv_over_dccf[2, 0] = -(m00 * a + m01 * b) / (2 * det * a**2)
    drv_over_dccf[1, 0] = -drv_over_dccf[0, 0] - drv_over_dccf[2, 0]

    peakIndex = np.arange(iMax - 1, iMax + 2)
    covarCcf = ccf.getCovar(peakIndex)
    varRv = np.transpose(drv_over_dccf) @ covarCcf @ drv_over_dccf

    return Struct(velocity=rv, error=math.sqrt(varRv[0, 0]), fail=fail)


def findPeakGauss(
    searchVelocity: NDArray[np.float64], ccf: CrossCorrelationFunction, config: Config
) -> Struct:
    """Find the peak of ``ccf`` with Gaussian fit.

    Parameters
    ----------
    searchVelocity : `np.ndarray` of `float`
        Shape ``(numVelocities)``. Velocity in km/s.
        This is x-axis of ``ccf``.
    ccf : `CrossCorrelationFunction`
        Cross correlation function.
    config : `Config`
        This must contain the following members:

        peakRange : `float`
            Velocity range, in km/s, used in fitting Gaussian.
        useCovar : `bool`
            Whether to use covariance.

    Returns
    -------
    velocity : `float`
        Radial velocity in km/s.
    error : `float`
        Standard deviation of ``velocity``.
        This is correct only if ``config.useCovar=True``.
    fail : `bool`
        True if measuring ``velocity`` failed.
    """

    def gauss(v, a, v_est, sigma):
        return a * np.exp((v - v_est) ** 2 / (-2 * sigma**2))

    iMax = np.argmax(ccf.ccf)
    velocity = searchVelocity[iMax]
    fail = iMax == 0 or iMax + 1 == len(ccf.ccf)
    if fail:
        return Struct(velocity=velocity, error=np.nan, fail=fail)

    coeff = 1.0 / ccf.ccf[iMax]
    fitIndex = (searchVelocity > (velocity - config.peakRange / 2)) & (
        searchVelocity < (velocity + config.peakRange / 2)
    )
    fitVelocity = searchVelocity[fitIndex]
    fitCcf = coeff * ccf.ccf[fitIndex]
    fitCovar = ccf.getCovar(fitIndex, amplifier=coeff * coeff)

    if not config.useCovar:
        # `scipy.optimize.curve_fit()` takes standard deviation
        # rather than variance if `sigma` argument is not a matrix.
        fitCovar = np.sqrt(np.diag(fitCovar))

    iniParam = [1.0, velocity, config.peakRange]
    pfit, pcov = scipy.optimize.curve_fit(
        gauss, fitVelocity, fitCcf, sigma=fitCovar, p0=iniParam, absolute_sigma=True
    )

    return Struct(velocity=pfit[1], error=np.sqrt(pcov[1][1]), fail=fail)
