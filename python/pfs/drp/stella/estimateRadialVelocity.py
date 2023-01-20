from lsst.pex.config import Config, Field, ChoiceField, ListField
from lsst.pipe.base import Struct, Task
from pfs.datamodel import PfsFiberArray, PfsSimpleSpectrum
from pfs.drp.stella.interpolate import interpolateFlux

import numpy as np
from astropy import constants as const
import scipy.optimize

import math


class EstimateRadialVelocityConfig(Config):
    """Configuration for EstimateRadialVelocityTask"""

    findMethod = ChoiceField(
        doc="Peak-finding method.",
        dtype=str,
        allowed={
            "peak": "The sampled point at which the cross-correlation is maximum.",
            "gauss": "Peak of a Gaussian fit to the cross-correlation.",
        },
        default="gauss",
        optional=False,
    )

    searchMin = Field(doc="Minimum of searched range of radial velocity, in km/s.", dtype=float, default=-500)

    searchMax = Field(doc="Maximum of searched range of radial velocity, in km/s.", dtype=float, default=500)

    searchStep = Field(
        doc="Step of searched range of radial velocity, in km/s."
        " The actual step may be slightly smaller than this value.",
        dtype=float,
        default=5.0,
    )

    peakRange = Field(
        doc='Velocity range, in km/s, used in fitting gaussian (valid when `findMethod` = "gauss")',
        dtype=float,
        default=100,
    )

    useCovar = Field(
        doc="Whether to use covariance. If False, use variance only."
        " Covariance used, the returned error bar will be more correct,"
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
            This is reliable only if ``config.findMethod="gauss"``
            and ``config.useCovar=True``.
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
        variance = spectrum.covar[0][goodIndex]

        goodIndex = 0 == (
            modelSpectrum.mask
            & modelSpectrum.flags.get(*(m for m in self.config.mask if m in modelSpectrum.flags))
        )
        modelWavelength = modelSpectrum.wavelength[goodIndex]
        modelFlux = modelSpectrum.flux[goodIndex] - 1.0

        # Make scaledModel[i,:] = modelSpectrum moving at searchVelocity[i]
        scaledWavelength = wavelength.reshape(1, -1) / doppler.reshape(-1, 1)
        scaledModel = interpolateFlux(
            modelWavelength, modelFlux, scaledWavelength.reshape(-1), jacobian=False
        ).reshape(len(searchVelocity), -1)
        # We divide the model flux by `dopper`
        # assuming that line spectra contribute to the correlation
        # much more than the continuum, not subtracted perfectly, does.
        # If this assumption is wrong, we must not divide it by `dopp`.
        scaledModel /= doppler.reshape(-1, 1)

        # This is cross correlation function
        ccf = scaledModel @ flux

        # c.c.f. is returned to the caller in this format for debugging.
        crossCorr = np.empty(len(ccf), dtype=[("velocity", float), ("crosscorr", float)])
        crossCorr["velocity"] = searchVelocity
        crossCorr["crosscorr"] = ccf

        # Find the peak of CCF
        if self.config.findMethod == "peak":
            iMax = np.argmax(ccf)
            velocity = searchVelocity[iMax]
            fail = iMax == 0 or iMax + 1 == len(ccf)
            # We have to think of the error...
            return Struct(velocity=velocity, error=np.nan, fail=fail, crossCorr=crossCorr)

        # Gaussian fit
        if self.config.findMethod == "gauss":

            def gauss(v, a, v_est, sigma):
                return a * np.exp((v - v_est) ** 2 / (-2 * sigma**2))

            iMax = np.argmax(ccf)
            velocity = searchVelocity[iMax]
            fail = iMax == 0 or iMax + 1 == len(ccf)
            if fail:
                return Struct(velocity=velocity, error=np.nan, fail=fail, crossCorr=crossCorr)

            coeff = 1.0 / ccf[iMax]
            fitIndex = (searchVelocity > (velocity - self.config.peakRange / 2)) & (
                searchVelocity < (velocity + self.config.peakRange / 2)
            )
            fitVelocity = searchVelocity[fitIndex]
            fitCcf = coeff * ccf[fitIndex]
            scaledModel = scaledModel[fitIndex, :]
            fitCovar = (scaledModel * (coeff * coeff * variance.reshape(1, -1))) @ np.transpose(scaledModel)

            if not self.config.useCovar:
                # `scipy.optimize.curve_fit()` takes standard deviation
                # rather than variance if `sigma` argument is not a matrix.
                fitCovar = np.sqrt(np.diag(fitCovar))

            iniParam = [1.0, velocity, self.config.peakRange]
            pfit, pcov = scipy.optimize.curve_fit(
                gauss, fitVelocity, fitCcf, sigma=fitCovar, p0=iniParam, absolute_sigma=True
            )

            return Struct(velocity=pfit[1], error=np.sqrt(pcov[1][1]), fail=fail, crossCorr=crossCorr)

        raise RuntimeError("config.findMethod has a wrong value.")
