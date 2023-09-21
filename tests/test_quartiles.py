import numpy as np

import lsst.utils.tests
from pfs.drp.stella.math import calculateMedian, calculateQuartiles
from pfs.drp.stella.tests.utils import runTests, methodParameters


class QuartilesTestCase(lsst.utils.tests.TestCase):
    """Test that the median and quartiles calculations work"""
    def setUp(self):
        self.rng = np.random.RandomState(12345)

    @methodParameters(
        num=(1, 2, 10, 1000),
        numMasked=(0, 1, 4, 100),
    )
    def testMedian(self, num, numMasked):
        """Test that the median is correct"""
        values = self.rng.uniform(size=num)
        if numMasked > 0:
            mask = np.zeros_like(values, dtype=bool)
            mask[:numMasked] = True
            self.rng.shuffle(mask)
            median = calculateMedian(values, mask)
            expected = np.median(values[~mask])
        else:
            median = calculateMedian(values)
            expected = np.median(values)
        self.assertFloatsEqual(median, expected)

    @methodParameters(
        num=(1, 2, 10, 1000),
        numMasked=(0, 1, 4, 100),
    )
    def testQuartiles(self, num, numMasked):
        """Test that the quarties are correct"""
        values = self.rng.uniform(size=num)
        percentiles = (25.0, 50.0, 75.0)
        if numMasked > 0:
            mask = np.zeros_like(values, dtype=bool)
            mask[:numMasked] = True
            self.rng.shuffle(mask)
            quartiles = calculateQuartiles(values, mask)
            expected = np.percentile(values[~mask], percentiles)
        else:
            quartiles = calculateQuartiles(values)
            expected = np.percentile(values, percentiles)
        self.assertFloatsEqual(quartiles[0], expected[0])
        self.assertFloatsEqual(quartiles[1], expected[1])
        self.assertFloatsEqual(quartiles[2], expected[2])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
