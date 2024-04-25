import numpy as np

import lsst.utils.tests

from pfs.drp.stella.math import NormalizedPolynomial1D
from pfs.drp.stella.tests.utils import runTests

display = None


class NormalizedPolynomialTestCase(lsst.utils.tests.TestCase):
    """Tests for pfs.drp.stella.math.NormalizedPolynomial1D"""
    def testInit(self):
        """Test NormalizedPolynomial1D.__init__

        An array with a single element as the first argument in the constructor
        was getting the wrong overload: the array was being interpreted as a
        scalar and being cast to an integer. This has been fixed, and this test
        ensures it remains fixed.
        """
        coeff = np.array([1.2345])  # A single value
        minMax = (-123.4, 456.7)
        poly = NormalizedPolynomial1D(coeff, *minMax)
        self.assertEqual(poly.getOrder(), 0)
        self.assertEqual(poly.getMin(), minMax[0])
        self.assertEqual(poly.getMax(), minMax[1])
        self.assertFloatsEqual(poly.getParameters(), coeff)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
