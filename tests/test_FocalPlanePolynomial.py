import numpy as np

import lsst.utils.tests

from pfs.drp.stella.focalPlaneFunction import FocalPlanePolynomial
from pfs.drp.stella.math import NormalizedPolynomial2D

from pfs.drp.stella.tests import runTests, methodParameters

display = None


class FocalPlanePolynomialTestCase(lsst.utils.tests.TestCase):
    """Test FocalPlanePolynomial"""
    def setUp(self):
        self.halfWidth = 100.0
        self.rng = np.random.RandomState(12345)

    def makeFocalPlanePolynomial(self, order: int = 1) -> FocalPlanePolynomial:
        """Make a random FocalPlanePolynomial with the given order."""
        numCoeffs = NormalizedPolynomial2D.nParametersFromOrder(order)
        coeffs = self.rng.uniform(size=numCoeffs)
        rms = self.rng.uniform()
        return FocalPlanePolynomial(coeffs=coeffs, halfWidth=self.halfWidth, rms=rms)

    def assertFocalPlanePolynomialEqual(self, left, right):
        """Assert that two FocalPlanePolynomials are equal."""
        self.assertFloatsEqual(left.coeffs, right.coeffs)
        self.assertEqual(left.halfWidth, right.halfWidth)
        self.assertEqual(left.rms, right.rms)

    @methodParameters(order=(1, 3, 7))
    def testBasic(self, order: int):
        """Test basic functionality of FocalPlanePolynomial."""
        numFibers = 10
        numPixels = 27
        poly = self.makeFocalPlanePolynomial(order)
        positions = self.rng.uniform(size=(numFibers, 2))
        wavelength = np.linspace(300, 1000, numPixels)
        fiberId = np.arange(0, numFibers, dtype=int)
        result = poly.evaluate(np.tile(wavelength, (numFibers, 1)), fiberId, positions)
        for ii, point in enumerate(positions):
            single = poly.evaluate(wavelength[None, :], fiberId[ii], point[None, :])
            self.assertEqual(single.values.shape, (1, numPixels))
            self.assertEqual(single.variances.shape, (1, numPixels))
            self.assertEqual(single.masks.shape, (1, numPixels))
            self.assertFloatsEqual(result.values[ii], single.values[0])
            self.assertFloatsEqual(result.variances[ii], single.variances[0])
            self.assertTrue(np.array_equal(result.masks[ii], single.masks[0]))

    def testIO(self):
        """Test reading and writing FocalPlanePolynomial to/from FITS files."""
        order = 5
        poly = self.makeFocalPlanePolynomial(order)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            poly.writeFits(filename)
            new = FocalPlanePolynomial.readFits(filename)
            self.assertFocalPlanePolynomialEqual(poly, new)

    def testFit(self):
        """Test fitting a FocalPlanePolynomial to random data."""
        numFibers = 45
        numPixels = 1234
        order = 2
        halfWidth = 100.0

        wavelength = np.tile(np.linspace(300, 1000, numPixels), (numFibers, 1))
        fiberId = np.arange(0, numFibers, dtype=int)
        positions = self.rng.uniform(low=-halfWidth, high=halfWidth, size=(numFibers, 2))
        flux = self.rng.uniform(low=-1, high=1, size=(numFibers, numPixels))
        mask = np.zeros((numFibers, numPixels), dtype=bool)
        variance = np.ones((numFibers, numPixels), dtype=float)

        poly = FocalPlanePolynomial.fitArrays(
            fiberId, wavelength, flux, mask, variance, positions, order=order, halfWidth=halfWidth
        )
        self.assertFloatsAlmostEqual(poly.coeffs, 0.0, atol=1.0e-2)
        self.assertEqual(poly.halfWidth, halfWidth)
        self.assertFloatsAlmostEqual(poly.rms, 0.0, atol=3.0e-2)
        self.assertGreater(poly.rms, 0.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
