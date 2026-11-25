import numpy as np
from scipy.stats import binned_statistic

import lsst.utils.tests

from pfs.drp.stella.math import NormalizedPolynomial1D
from pfs.drp.stella.utils.math import robustRms, binnedRobustRms
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


class StatisticsTestCase(lsst.utils.tests.TestCase):
    """Tests for the vectorised statistics in pfs.drp.stella.utils.math"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)

    def testBinnedRobustRms(self):
        """binnedRobustRms agrees with binned_statistic(..., robustRms, ...)"""
        xx = np.sort(self.rng.uniform(0, 100, size=5000))
        values = self.rng.normal(size=xx.size)
        values[123] = np.nan                                   # a bin with a non-finite value
        bins = np.concatenate(([xx[0]], np.linspace(1, 99, 300), [50.05, 50.06], [xx[-1]]))
        bins = np.unique(bins)                                 # includes some empty bins
        xx[-1] = bins[-1]                                      # a value on the last edge
        expected = binned_statistic(xx, values, statistic=robustRms, bins=bins)[0]
        got = binnedRobustRms(xx, values, bins)
        self.assertEqual(got.shape, expected.shape)
        np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
        self.assertTrue(np.any(np.isnan(expected)))
        np.testing.assert_array_equal(got, expected)   # identical, including the NaNs

        for num in (1, 2, 3, 4, 5):                            # small bins exercise the interpolation
            xs = np.arange(num, dtype=float)
            vs = self.rng.normal(size=num)
            self.assertEqual(binnedRobustRms(xs, vs, [-0.5, num - 0.5])[0], robustRms(vs))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
