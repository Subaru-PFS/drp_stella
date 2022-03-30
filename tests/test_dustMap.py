import lsst.utils.tests

import lsst.utils
from pfs.drp.stella.dustMap import DustMap
from pfs.drp.stella.tests import runTests

import astropy.units
import numpy as np

import unittest

try:
    dataDir = lsst.utils.getPackageDir("dustmaps_cachedata")
except LookupError:
    dataDir = None


@unittest.skipIf(dataDir is None, "dustmaps_cachedata not setup")
class DustMapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.dustMap = DustMap()

    def test(self):
        """Test __call__()
        """
        ralist = np.linspace(0, 360, 5, dtype=float)
        declist = np.linspace(-90, 90, 5, dtype=float)

        # calls with scalar arguments
        result1 = [self.dustMap(float(ra), float(dec)) for ra, dec in zip(ralist, declist)]
        result1 = np.asarray(result1)

        # call with numpy arrays
        result2 = self.dustMap(ralist, declist)

        # call with astropy.units.quantity.Quantity
        result3 = self.dustMap(ralist * astropy.units.degree, declist * astropy.units.degree)

        # I don't know why, but these two results are not quite equal.
        self.assertFloatsAlmostEqual(result1, result2, rtol=3e-14)
        self.assertFloatsAlmostEqual(result2, result3)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
