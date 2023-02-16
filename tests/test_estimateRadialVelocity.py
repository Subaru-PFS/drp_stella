import lsst.utils.tests

from pfs.drp.stella.estimateRadialVelocity import EstimateRadialVelocityTask, EstimateRadialVelocityConfig
from pfs.drp.stella.tests import methodParameters, runTests
from pfs.datamodel import MaskHelper
from pfs.datamodel.observations import Observations
from pfs.datamodel.drp import PfsSingle
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.datamodel.target import Target

import astropy.constants
import numpy as np
import scipy.interpolate
import scipy.signal

import contextlib


class EstimateRadialVelocityTestCase(lsst.utils.tests.TestCase):
    obsLsfSigma = 0.23
    """Width of the LSF of a fake observed spectrum, in nm.
    """

    obsSpecRange = [400, 1200, 12000]
    """Wavelength range of a fake observed spectrum, in nm:
    `[min, max, numSamples]`.
    """

    refLsfSigma = 0.087
    """Width of the LSF of the fake reference spectrum, in nm.
    """

    refSpecRange = [300, 1300, 100000]
    """Wavelength range of the fake reference spectrum, in nm:
    `[min, max, numSamples]`.
    """

    permissibleError = 1
    """Permissible error of radial velocity estimate, in km/s,
    that cannot be explained by the statistical error.
    """

    varianceTolerance = 1.2
    """Error estimate (`retvalue.error`) must satisfy `1 / varianceTolerance < ratio < varianceTolerance`,
    where `ratio = (variance of retvalue.velocity) / (average of retvalue.error**2)`.
    """

    numTrials = 100
    """Number of trials for acquisition of statistics
    """

    def setUp(self):
        self.estimateRadialVelocity = EstimateRadialVelocityTask(config=EstimateRadialVelocityConfig())
        self.np_random = np.random.RandomState(seed=0x0123456)
        self.refSpectrum = self.createRefSpectrum(self.refLsfSigma)

    @methodParameters(snr=[10, 20, 30, 40, 50], trueVelocity=[-200, -100, 0, 100, 200])
    def test(self, snr, trueVelocity):
        """Test run()

        Parameters
        ----------
        snr : `float`
            Signal-to-noise ratio of the fake observed spectrum.
        trueVelocity : `float`
            Radial velocity of the fake observed spectrum, in km/s.
        """
        sum_1 = 0
        sum_x = 0
        sum_xx = 0
        sum_v = 0
        for i in range(self.numTrials):
            obsSpectrum = self.createObsSpectrum(self.refSpectrum, trueVelocity, self.obsLsfSigma, snr)
            with temporarilyRuinMaskedRegion(obsSpectrum):
                with temporarilyRuinMaskedRegion(self.refSpectrum):
                    result = self.estimateRadialVelocity.run(obsSpectrum, self.refSpectrum)
                    sum_1 += 1
                    sum_x += result.velocity
                    sum_xx += result.velocity**2
                    sum_v += result.error**2

        average = sum_x / sum_1
        variance = (sum_xx / sum_1 - average**2) * (sum_1 / (sum_1 - 1))
        estimatedVar = sum_v / sum_1
        varianceRatio = (estimatedVar + self.permissibleError**2) / variance
        error = (estimatedVar + self.permissibleError**2) / np.sqrt(sum_1)

        self.assertLess(abs(average - trueVelocity), 2*error)
        self.assertGreater(varianceRatio, 1 / self.varianceTolerance)
        self.assertLess(varianceRatio, self.varianceTolerance)

    def createMask(self, shape, density):
        """Create a fake mask array

        Parameters
        ----------
        shape : `int` or `Tuple[int]`
            The shape of returned array.
        density : `float`
            Density of masked region, from 0 to 1.

        Returns
        -------
        mask : `numpy.array` of `int`
            Mask array.
        flags : `pfs.datamodel.masks.MaskHelper`
            Flag names.
        """
        numFlags = len(self.estimateRadialVelocity.config.mask)
        mask = np.zeros(shape=shape, dtype=int)
        flags = MaskHelper()
        for flag in self.estimateRadialVelocity.config.mask:
            value = flags.add(flag)
            mask |= self.np_random.choice(
                np.array([0, value], dtype=int),
                size=shape,
                replace=True,
                p=[1 - density / numFlags, density / numFlags]
            )
        return mask, flags

    def createRefSpectrum(self, lsfSigma):
        """Create a fake reference spectrum.

        Parameters
        ----------
        lsfSigma : `float`
            Standard deviation of the LSF, in nm

        Returns
        -------
        spectrum : `pfs.datamodel.pfsSimpleSpectrum.PfsSimpleSpectrum`
            A fake reference spectrum.
        """
        wavelength = np.linspace(*self.refSpecRange)

        peak = self.np_random.uniform(wavelength[0], wavelength[-1], size=200).reshape(-1, 1)
        depth = self.np_random.beta(1, 5, size=peak.shape)
        flux = np.prod(1.0 - depth * createRefLsf(wavelength.reshape(1, -1), peak, lsfSigma), axis=0)

        target = Target(0, 0, "0,0", 0)
        mask, flags = self.createMask(wavelength.shape, density=0.001)

        return PfsSimpleSpectrum(target, wavelength, flux, mask, flags)

    def createObsSpectrum(self, refSpectrum, trueVelocity, lsfSigma, snr):
        """Create a fake observed spectrum.

        Parameters
        ----------
        refSpectrum : `pfs.datamodel.pfsSimpleSpectrum.PfsSimpleSpectrum`
            Reference spectrum.
        trueVelocity : `float`
            Radial velocity of the observed object, in km/s
        lsfSigma : `float`
            Standard deviation of the LSF, in nm
        snr : `float`
            Signal-to-noise ratio.

        Returns
        -------
        spectrum : `pfs.datamodel.PfsSingle`
            A fake observed spectrum.
        """
        refWavelength = refSpectrum.wavelength
        refFlux = refSpectrum.flux

        step = refWavelength[len(refWavelength)//2 + 1] - refWavelength[len(refWavelength)//2]
        # 7-sigma on both sides of the LSF peak.
        # `numLsfSamples` must be made an odd number.
        numLsfSamples = int(2 * 7 * lsfSigma / step) // 2 * 2 + 1
        lsfWavelength = np.linspace(0.0, step * (numLsfSamples - 1), numLsfSamples)

        lsfFlux = createLsf(lsfWavelength, lsfWavelength[len(lsfWavelength)//2], lsfSigma)
        lsfFlux /= np.sum(lsfFlux)

        c = astropy.constants.c.to('km/s').value
        doppler = np.sqrt((1 + trueVelocity / c) / (1 - trueVelocity / c))

        wavelength = np.linspace(*self.obsSpecRange)
        flux = scipy.signal.fftconvolve(refFlux, lsfFlux, mode="same")
        flux = scipy.interpolate.interp1d(refWavelength*doppler, flux, kind="cubic")(wavelength)

        variance = flux / (snr*snr)
        flux = self.np_random.poisson(flux * (snr*snr)) / (snr*snr)

        target = Target(0, 0, "0,0", 0)
        observations = Observations(
            np.zeros(0, dtype=int),
            np.zeros(0, dtype="U0"),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(shape=(0, 2), dtype=float),
            np.zeros(shape=(0, 2), dtype=float)
        )
        mask, flags = self.createMask(wavelength.shape, density=0.01)
        sky = np.zeros(len(wavelength), dtype=float)
        covar = np.zeros(shape=(3, len(wavelength)), dtype=float)
        covar[0, :] = variance
        covar2 = np.zeros(shape=(0, 0), dtype=float)

        return PfsSingle(target, observations, wavelength, flux, mask, sky, covar, covar2, flags)


def createLsf(wavelength, peak, sigma):
    """Fake line spread function for observed spectra.
    It is normalized so that its height is 1.

    Parameters
    ----------
    wavelength : `numpy.array`
        Wavelength in nm.
    peak : `float` or `numpy.array`
        Peak position in nm.
        If this is `numpy.array`, it must be broadcastable to ``wavelength``
        (For example, ``wavelength.shape == (1, N)``
        and ``peak.shape == (M, 1)``,
        in which case the return value has shape ``(M, N)``.)
    sigma : `float`
        Standard deviation of the peak, in nm.

    Returns
    -------
    flux : `numpy.array`
        An LSF whose peak is at `peak`.
    """
    # This constant makes `sigma` the standard deviation
    a = 1.0399832613
    return np.exp(-np.abs((wavelength - peak) / (a * sigma))**(4/3.0))


def createRefLsf(wavelength, peak, sigma):
    """Fake line spread function for reference spectra.
    It is normalized so that its height is 1.

    Parameters
    ----------
    wavelength : `numpy.array`
        Wavelength in nm.
    peak : `float` or `numpy.array`
        Peak position in nm.
        If this is `numpy.array`, it must be broadcastable to ``wavelength``
        (For example, ``wavelength.shape == (1, N)``
        and ``peak.shape == (M, 1)``,
        in which case the return value has shape ``(M, N)``.)
    sigma : `float`
        Standard deviation of the peak, in nm.

    Returns
    -------
    flux : `numpy.array`
        An LSF whose peak is at `peak`.
    """
    return np.exp(np.square(wavelength - peak) / (-2*sigma*sigma))


@contextlib.contextmanager
def temporarilyRuinMaskedRegion(spectrum):
    """Set very bad values at places where ``spectrum.mask != 0``.
    The ruined region will be restored on exiting the context.

    Parameters
    ----------
    spectrum : `pfs.datamodel.pfsSimpleSpectrum.PfsSimpleSpectrum`
        Spectrum to ruin.
    """
    flux = np.copy(spectrum.flux)
    covar = getattr(spectrum, "covar", None)
    if covar is not None:
        covar = np.copy(covar)

    try:
        spectrum.flux[spectrum.mask != 0] = np.nan
        if covar is not None:
            spectrum.covar[:, spectrum.mask != 0] = np.nan
        yield
    finally:
        spectrum.flux[...] = flux
        if covar is not None:
            spectrum.covar[...] = covar


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
