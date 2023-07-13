import numpy as np

import lsst.utils.tests
from lsst.pipe.base import Struct

from pfs.drp.stella.interpolate import calculateDispersion, interpolateFlux, interpolateVariance
from pfs.drp.stella.tests import runTests, methodParameters, methodParametersProduct

display = None


class InterpolateTestCase(lsst.utils.tests.TestCase):
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
        outSpectrum = interpolateFlux(inSpectrum.wavelength, inSpectrum.flux, expected.wavelength)
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

        outFlux = interpolateFlux(inSpectrum.wavelength, inSpectrum.flux, expected.wavelength)
        outVariance = interpolateVariance(inSpectrum.wavelength, inVariance, expected.wavelength)

        mean = np.average(outFlux)
        stdev = np.std(outFlux)
        stdChi = np.std((outFlux - flux)/np.sqrt(outVariance))
        self.assertFloatsAlmostEqual(mean, flux, rtol=1.0e-3)
        self.assertFloatsAlmostEqual(stdChi, 1.0, rtol=0.15)
        self.assertFloatsAlmostEqual(stdev, np.sqrt(np.median(outVariance)), rtol=0.15)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    runTests(globals())
