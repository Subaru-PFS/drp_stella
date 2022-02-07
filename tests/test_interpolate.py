import numpy as np

import lsst.utils.tests
from lsst.pipe.base import Struct

from pfs.drp.stella.interpolate import calculateDispersion, interpolateFlux
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
            Flux density, in units/nm.

        Returns
        -------
        wavelength : `numpy.ndarray` of `float`
            Array of wavelengths.
        flux : `numpy.ndarray` of `float`
            Array of flux (densities).
        """
        num = (self.max - self.min)/dispersion
        wavelength = np.linspace(self.min, self.max, num + 1, dtype=float)
        fluxArray = np.full_like(wavelength, flux*dispersion)
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
    def testInterpolationWithJacobian(self, inDispersion, outDispersion):
        """Test interpolateFlux function with Jacobian

        We test that the flux density is unchanged when we call
        ``interpolateFlux`` with ``jacobian=True``.

        Parameters
        ----------
        inDispersion : `float`
            Dispersion of input spectrum.
        outDispersion : `float`
            Dispersion of output spectrum.
        """
        flux = 543.21
        inSpectrum = self.makeSpectrum(inDispersion, flux)
        expected = self.makeSpectrum(outDispersion, flux)
        outSpectrum = interpolateFlux(inSpectrum.wavelength, inSpectrum.flux, expected.wavelength,
                                      jacobian=True)
        self.assertFloatsAlmostEqual(outSpectrum, expected.flux, atol=1.0e-9)

    @methodParametersProduct(inDispersion=(0.1, 0.3, 0.5),
                             outDispersion=(0.01, 0.3, 1.0))
    def testInterpolationWithoutJacobian(self, inDispersion, outDispersion):
        """Test interpolateFlux function without Jacobian

        Since we're not correcting for the change in the dispersion, the output
        flux density should differ by the ratio of the two dispersions.

        Parameters
        ----------
        inDispersion : `float`
            Dispersion of input spectrum.
        outDispersion : `float`
            Dispersion of output spectrum.
        """
        flux = 123.45
        inSpectrum = self.makeSpectrum(inDispersion, flux)
        expected = self.makeSpectrum(outDispersion, flux*inDispersion/outDispersion)
        outSpectrum = interpolateFlux(inSpectrum.wavelength, inSpectrum.flux, expected.wavelength,
                                      jacobian=False)
        self.assertFloatsAlmostEqual(outSpectrum, expected.flux, atol=1.0e-12)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    runTests(globals())
