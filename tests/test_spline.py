import numpy as np
import scipy.interpolate

import lsst.utils.tests

from pfs.drp.stella.tests.utils import runTests
from pfs.drp.stella import SplineF

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
        self.xx = np.arange(0.0, self.num, dtype=np.float32)
        self.yy = self.rng.uniform(size=self.num).astype(np.float32)
        self.spline = SplineF(self.xx, self.yy)

    def testBasic(self):
        """Test basic properties"""
        self.assertFloatsEqual(self.spline.getX(), self.xx)
        self.assertFloatsEqual(self.spline.getY(), self.yy)
        values = np.array([self.spline(x) for x in self.xx])
        self.assertFloatsEqual(values, self.yy)

    def testCompare(self):
        """Compare with alternate implementation"""
        alternate = makeSpline(self.xx, self.yy)
        rand = self.rng.uniform(size=self.num)*(self.num - 1)
        ours = np.array([self.spline(x) for x in rand])
        theirs = alternate(rand)
        self.assertFloatsAlmostEqual(ours, theirs, atol=3.0e-6)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
