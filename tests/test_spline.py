import numpy as np
import scipy.interpolate

import lsst.utils.tests
from lsst.pex.exceptions import LengthError

from pfs.drp.stella.tests.utils import runTests
from pfs.drp.stella import SplineD

display = None


def makeSpline(xx, yy):
    """Construct a spline with an alternate implementation

    Spline interpolates from ``xx`` to ``yy``.

    Parameters
    ----------
    xx, yy : `numpy.ndarray`
        Arrays for interpolation.

    Returns
    -------
    spline : `scipy.interpolate.CubicSpline`
        Spline object.
    """
    return scipy.interpolate.CubicSpline(xx, yy, bc_type='not-a-knot')


class SplineTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Create a spline to play with"""
        self.num = 100
        self.rng = np.random.RandomState(12345)
        self.xx = np.arange(0.0, self.num)
        self.yy = self.rng.uniform(size=self.num)
        self.spline = SplineD(self.xx, self.yy)

    def testBasic(self):
        """Test basic properties"""
        self.assertFloatsEqual(self.spline.getX(), self.xx)
        self.assertFloatsEqual(self.spline.getY(), self.yy)
        values = np.array([self.spline(x) for x in self.xx])
        self.assertFloatsAlmostEqual(values, self.yy, atol=1.0e-7)

    def testCompare(self):
        """Compare with alternate implementation"""
        alternate = makeSpline(self.xx, self.yy)
        rand = self.rng.uniform(size=self.num)*(self.num - 1)
        ours = np.array([self.spline(x) for x in rand])
        theirs = alternate(rand)
        self.assertFloatsAlmostEqual(ours, theirs, atol=3.0e-6)

    def testSizeMismatch(self):
        """Test a mismatch of sizes in the ctor"""
        with self.assertRaises(LengthError):
            SplineD(np.arange(5, dtype=float), np.arange(7, dtype=float))

    def testExtrapolation(self):
        """Test extrapolation controls"""
        delta = np.average(self.xx[1:] - self.xx[:-1])
        low = self.xx[0] - 0.5*delta
        high = self.xx[-1] + 0.5*delta
        middle = np.median(self.xx)

        spline = self.spline
        extrapolation = SplineD.ExtrapolationTypes.ALL
        self.assertEqual(spline.getExtrapolationType(), extrapolation)
        self.assertEqual(spline.extrapolationType, extrapolation)
        self.assertFalse(np.isnan(spline(low)))
        self.assertFalse(np.isnan(spline(high)))
        self.assertFalse(np.isnan(spline(middle)))

        extrapolation = SplineD.ExtrapolationTypes.NONE
        spline = SplineD(self.xx, self.yy, self.spline.getInterpolationType(), extrapolation)
        self.assertEqual(spline.getExtrapolationType(), extrapolation)
        self.assertEqual(spline.extrapolationType, extrapolation)
        self.assertTrue(np.isnan(spline(low)))
        self.assertTrue(np.isnan(spline(high)))
        self.assertFalse(np.isnan(spline(middle)))

        extrapolation = SplineD.ExtrapolationTypes.SINGLE
        spline = SplineD(self.xx, self.yy, self.spline.getInterpolationType(), extrapolation)
        self.assertEqual(spline.getExtrapolationType(), extrapolation)
        self.assertEqual(spline.extrapolationType, extrapolation)
        self.assertFalse(np.isnan(spline(low)))
        self.assertFalse(np.isnan(spline(high)))
        self.assertFalse(np.isnan(spline(middle)))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
