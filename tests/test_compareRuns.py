"""Claude: the array comparisons used by pfsCompareRuns.py"""
import unittest

import numpy as np

import lsst.utils.tests
from pfs.drp.stella.compareRuns import compareArrays, noiseChange


class CompareRunsTestCase(lsst.utils.tests.TestCase):
    def testCompareArrays(self):
        rng = np.random.default_rng(12345)
        x = rng.normal(size=(3, 100))
        x[0, 5] = np.nan
        self.assertEqual(compareArrays(x, x.copy()), (True, 0.0))  # NaNs compare equal
        y = x.copy()
        y[1, 7] += 0.5
        same, rel = compareArrays(x, y)
        self.assertFalse(same)
        finite = np.isfinite(x)
        self.assertFloatsAlmostEqual(rel, 0.5/np.sqrt(np.mean(x[finite]**2)), rtol=1e-12)
        self.assertEqual(compareArrays(x, x[:2]), (False, np.inf))

    def testNoiseChange(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0])
        new = np.array([1.0, 2.5, 3.0, 5.0])
        variance = np.array([1.0, 0.25, 1.0, 4.0])
        refMask = np.array([0, 0, 1, 0])
        newMask = np.array([0, 0, 0, 0])
        change, use = noiseChange(ref, new, variance, refMask, newMask)
        np.testing.assert_array_equal(use, [True, True, False, True])
        np.testing.assert_allclose(change, [0.0, 1.0, 0.0, 0.5])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
