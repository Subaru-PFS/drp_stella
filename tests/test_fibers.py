import numpy as np

import lsst.utils.tests
from pfs.utils.fibers import FIBERS_PER_SPECTROGRAPH
from pfs.utils.fibers import calculateFiberId, spectrographFromFiberId, fiberHoleFromFiberId
from pfs.drp.stella.tests.utils import runTests


class FibersTestCase(lsst.utils.tests.TestCase):
    """Test that the fiberId calculations from pfs_utils work

    This is here, because there are no tests in pfs_utils.
    """
    def testRoundTrip(self):
        """Check that values round-trip"""
        fiberHole = np.arange(FIBERS_PER_SPECTROGRAPH, dtype=int) + 1  # Unit-indexed!
        for spectrograph in (1, 2, 3, 4):
            fiberId = calculateFiberId(spectrograph, fiberHole)
            self.assertTrue(np.all(fiberId >= 1))
            self.assertTrue(np.all(fiberId <= 2604))
            self.assertFloatsEqual(spectrographFromFiberId(fiberId), spectrograph)
            self.assertFloatsEqual(fiberHoleFromFiberId(fiberId), fiberHole)

    def testCalculations(self):
        """Check that the calculations are correct"""
        spectrograph = np.array([1, 1, 2, 2, 3, 3, 4, 4])
        fiberHole = np.array([1, FIBERS_PER_SPECTROGRAPH]*4)
        fiberId = np.array([1, 651, 652, 1302, 1303, 1953, 1954, 2604])

        self.assertFloatsEqual(calculateFiberId(spectrograph, fiberHole), fiberId)
        self.assertFloatsEqual(spectrographFromFiberId(fiberId), spectrograph)
        self.assertFloatsEqual(fiberHoleFromFiberId(fiberId), fiberHole)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
