import numpy as np

import lsst.utils.tests
from lsst.pipe.base import Struct
from lsst.afw.math import offsetImage
from lsst.afw.image import ImageD

from pfs.drp.stella.interpolate import (
    calculateDispersion,
    interpolateFlux,
    interpolateVariance,
    interpolate,
    interpolateCovariance,
)
from pfs.drp.stella.tests import runTests, classParameters, methodParameters, methodParametersProduct

display = None


@classParameters(order=(1, 3))
class InterpolateTestCase(lsst.utils.tests.TestCase):
    order: int  # Interpolation order; set by classParameters

    def setUp(self):
        self.min = 650
        self.max = 950

    def makeSpectrum(self, dispersion, flux=1.0):
        """Create a spectrum

        Parameters
        ----------
        dispersion : `float`
            Dispersion, in nm/pixel.
        flux : `float`
            Flux value

        Returns
        -------
        wavelength : `numpy.ndarray` of `float`
            Array of wavelengths.
        flux : `numpy.ndarray` of `float`
            Array of flux.
        """
        num = int((self.max - self.min)/dispersion)
        wavelength = np.linspace(self.min, self.max, num + 1, dtype=float)
        fluxArray = np.full_like(wavelength, flux)
        return Struct(wavelength=wavelength, flux=fluxArray)

    @methodParameters(dispersion=(0.1, 0.3, 0.5))
    def testCalculateDispersion(self, dispersion):
        """Test calculateDispersion function

        We create a spectrum with a particular dispersion. The function should
        measure the correct dispersion.

        Parameters
        ----------
        dispersion : `float`
            Dispersion, in nm/pixel.
        """
        spectrum = self.makeSpectrum(dispersion)
        self.assertFloatsAlmostEqual(calculateDispersion(spectrum.wavelength), dispersion, atol=1.0e-13)

    @methodParametersProduct(inDispersion=(0.1, 0.3, 0.5),
                             outDispersion=(0.01, 0.3, 1.0))
    def testFlux(self, inDispersion, outDispersion):
        """Test interpolateFlux function

        Since all we're doing is interpolating, the flux should stay the same.

        Parameters
        ----------
        inDispersion : `float`
            Dispersion of input spectrum.
        outDispersion : `float`
            Dispersion of output spectrum.
        """
        flux = 123.45
        inSpectrum = self.makeSpectrum(inDispersion, flux)
        expected = self.makeSpectrum(outDispersion, flux)
        outSpectrum = interpolateFlux(
            inSpectrum.wavelength, inSpectrum.flux, expected.wavelength, order=self.order
        )
        self.assertFloatsAlmostEqual(outSpectrum, flux, atol=1.0e-12)

    @methodParameters(inDispersion=(3.0, 5.0, 10.0),
                      outDispersion=(1.23, 4.321, 7.654)
                      )
    def testVariance(self, inDispersion, outDispersion):
        """Test interpolateVariance"""
        flux = 10000.0
        noise = 30.0
        inSpectrum = self.makeSpectrum(inDispersion, flux)
        expected = self.makeSpectrum(outDispersion, flux)
        rng = np.random.RandomState(12345)
        inSpectrum.flux += rng.normal(scale=noise, size=inSpectrum.flux.shape)
        inVariance = np.full_like(inSpectrum.flux, noise**2)

        outFlux = interpolateFlux(
            inSpectrum.wavelength, inSpectrum.flux, expected.wavelength, order=self.order
        )
        outVariance = interpolateVariance(
            inSpectrum.wavelength, inVariance, expected.wavelength, order=self.order
        )

        mean = np.average(outFlux)
        stdChi = np.std((outFlux - flux)/np.sqrt(outVariance))
        self.assertFloatsAlmostEqual(mean, flux, rtol=1.5e-3)
        self.assertFloatsAlmostEqual(stdChi, 1.0, rtol=0.15)

    @methodParameters(
        inDispersion=(3.0, 5.0, 10.0),
        outDispersion=(1.23, 4.321, 7.654)
    )
    def testInterpolate(self, inDispersion: float, outDispersion: float):
        """Test general-purpose interpolate function"""
        flux = 10000.0
        noise = 30.0
        inSpectrum = self.makeSpectrum(inDispersion, flux)
        expected = self.makeSpectrum(outDispersion, flux)
        rng = np.random.RandomState(12345)
        inSpectrum.flux += rng.normal(scale=noise, size=inSpectrum.flux.shape)
        inVariance = np.full_like(inSpectrum.flux, noise**2)

        result = interpolate(
            inSpectrum.wavelength,
            inSpectrum.flux,
            np.zeros_like(inSpectrum.flux, dtype=int),
            inVariance,
            expected.wavelength,
            order=self.order,
        )
        outFlux = result.flux
        outVariance = result.variance

        mean = np.average(outFlux)
        stdChi = np.std((outFlux - flux)/np.sqrt(outVariance))
        self.assertFloatsAlmostEqual(mean, flux, rtol=1.5e-3)
        self.assertFloatsAlmostEqual(stdChi, 1.0, rtol=0.15)

    def testLsst(self):
        """Test that we get the same result as lsst.afw.math.offsetImage"""
        algorithm = "bilinear" if self.order <= 1 else f"lanczos{self.order}"

        sigma = 3.21
        size = 101
        center = (size // 2) + 0.321
        shift = 1.25

        xx = np.arange(size, dtype=float)
        yy = np.exp(-0.5*((xx - center)/sigma)**2)

        # Make an image containing the spectrum, and use LSST's offsetImage to shift it
        array = np.zeros((2*self.order + 1, size), dtype=float)
        array[self.order] = yy
        image = ImageD(array)
        offset = offsetImage(image, shift, 0, algorithm)
        x0 = offset.getX0()
        lsst = np.zeros_like(yy)
        lsst[x0:] = offset.array[self.order][:-x0]

        pfs = interpolateFlux(xx, yy, xx - shift, order=self.order)

        select = slice(self.order, -self.order)
        self.assertFloatsAlmostEqual(pfs[select], lsst[select], atol=1.0e-6)

    def testGaussian(self):
        """Test interpolateFlux with a Gaussian"""
        sigma = 3.21
        size = 101
        center = (size // 2) + 0.321
        shift = 1.25

        xx = np.arange(size, dtype=float)
        yy = np.exp(-0.5*((xx - center)/sigma)**2)

        result = interpolateFlux(xx, yy, xx - shift, order=self.order)

        result[0:self.order] = 0.0  # Ignore edge effects
        result[-self.order:] = 0.0
        expected = np.exp(-0.5*((xx - center - shift)/sigma)**2)

        if False:
            import matplotlib.pyplot as plt
            plt.plot(xx, yy, "r-", label="input")
            plt.plot(xx, expected, "b-", label="expected")
            plt.plot(xx, result, "k-", label="result")
            plt.axvline(center, color="k", ls=":")
            plt.legend()
            plt.ylim(0, 1)
            plt.show()

        moment1 = np.sum(result*xx)/np.sum(result) - center
        moment2 = np.sqrt(np.sum(result*(xx - center - moment1)**2)/np.sum(result))

        if self.order <= 1:
            # Bilinear gets the centroid right, but grows the width
            atol1 = 1.0e-6
            atol2 = 3.0e-2
        else:
            # Lanczos isn't as good on centroid, but the width is much better
            atol1 = 2.0e-2
            atol2 = 1.0e-4

        self.assertFloatsAlmostEqual(result, expected, atol=1.0e-2)
        self.assertFloatsAlmostEqual(moment1, shift, atol=atol1)
        self.assertFloatsAlmostEqual(moment2, sigma, atol=atol2)

    @methodParameters(
        inDispersion=(3.0, 5.0, 10.0),
        outDispersion=(1.23, 4.321, 7.654)
    )
    def testCovariance(self, inDispersion: float, outDispersion: float):
        """Test interpolateCovariance function"""
        fluxValue = 100.0
        varianceValue = 1.0
        numSamples = 10000
        numCovar = self.order + 2  # +1 to test one beyond order, +1 for zeroth order

        inSpectrum = self.makeSpectrum(inDispersion, fluxValue)
        variance = np.full_like(inSpectrum.flux, varianceValue)
        outSpectrum = self.makeSpectrum(outDispersion, fluxValue)

        covar = interpolateCovariance(
            inSpectrum.wavelength, variance, outSpectrum.wavelength, order=self.order, numCovar=numCovar
        )
        self.assertEqual(covar.shape, (numCovar, outSpectrum.wavelength.size))

        rng = np.random.RandomState(12345)
        samples = np.zeros((numSamples, outSpectrum.wavelength.size), dtype=float)
        for ii in range(numSamples):
            flux = rng.normal(fluxValue, np.sqrt(varianceValue), size=inSpectrum.wavelength.size)
            samples[ii] = interpolateFlux(
                inSpectrum.wavelength, flux, outSpectrum.wavelength, order=self.order
            )
        residuals = samples - fluxValue

        outVar = interpolateVariance(
            inSpectrum.wavelength, variance, outSpectrum.wavelength, order=self.order
        )
        measVar = np.nanvar(residuals, axis=0)
        self.assertFloatsAlmostEqual(covar[0], outVar, atol=1.0e-6)
        self.assertFloatsAlmostEqual(measVar, outVar, atol=4.0e-2)

        for offset in range(numCovar):
            calculated = covar[offset]
            measured = np.mean(residuals[:, :-offset or None]*residuals[:, offset or None:], axis=0)
            self.assertFloatsAlmostEqual(measured, calculated[:-offset or None], rtol=0.1, atol=0.05)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    runTests(globals())
