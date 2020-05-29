import matplotlib
matplotlib.use("Agg")  # noqa E402: disable showing plots
import matplotlib.pyplot as plt

from types import SimpleNamespace
import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils

from pfs.drp.stella.fitPolynomial import FitPolynomialTask
from pfs.drp.stella.tests.utils import runTests, methodParameters

display = None


class FitPolynomialTaskTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.config = FitPolynomialTask.ConfigClass()
        self.task = FitPolynomialTask(config=self.config)

    def makeData(self, coeffs, noise, xMin=0, xMax=4000):
        """Make some data we can fit

        Parameters
        ----------
        coeffs : array_like
            Polynomial coefficients.
        noise : `float`
            Gaussian sigma for the noise to be added.
        xMin, xMax : `int`
            Minimum and maximum x values.

        Returns
        -------
        xx : `numpy.ndarray`
            Ordinates.
        yy : `numpy.ndarray`
            Values of the function at the ordinates, plus Gaussian noise.
        func : callable
            Polynomial function used to construct the ``yy``.
        order : `int`
            Polynomial order; equal to the length of the input ``coeffs``.
        noise : `float`
            Gaussian sigma of the noise added.
        """
        func = np.polynomial.Chebyshev(coeffs, domain=(xMin, xMax))
        xx = np.arange(xMin, xMax, dtype=int)
        yy = func(xx) + self.rng.normal(0.0, noise, xx.shape)
        return SimpleNamespace(xx=xx, yy=yy, func=func, order=len(coeffs), noise=noise)

    def assertResult(self, data, result, bad=None):
        """Assert that the fit results are as expected

        Parameters
        ----------
        data : `types.SimpleNamespace`
            Output from ``makeData``.
        result : `lsst.pipe.base.Struct`
            Output from ``FitPolynomialTask.run``.
        bad : array_like of `bool`
            Boolean array indicating whether a measurement is known to be bad.
        """
        if bad is None:
            bad = np.zeros_like(data.xx, dtype=bool)
        self.assertFloatsEqual(result.fit, result.func(data.xx))
        self.assertFloatsEqual(result.residuals, data.yy - result.func(data.xx))
        self.assertGreater(result.good.sum(), data.order)
        self.assertLessEqual(result.rms, 1.1*result.residuals[result.good].std(),
                             "robust rms should be better than standard rms")
        self.assertEqual(result.good.dtype, bool)

        if np.any(bad):
            self.assertFalse(np.any(result.good[bad]), "correctly identified all bad points")

        residuals = result.func(data.xx) - data.func(data.xx)
        self.assertFloatsAlmostEqual(residuals[~bad].std(), 0.0, atol=1.2*data.noise)

    @methodParameters(
        coeffs=((3.21, 1.23),
                (1234.567, 1.2345, 0.12345),
                (9.87, 6.543, 2.1, 0.987, 0.0654),
                ),
        noise=(0.321, 0.321, 0.54321)
    )
    def testBasic(self, coeffs, noise):
        """Test basic function of FitPolynomialTask"""
        data = self.makeData(coeffs, noise)
        self.config.order = len(coeffs)
        result = self.task.run(data.xx, data.yy)
        self.assertResult(data, result)
        self.task.plot(data.xx, data.yy, result.good, result)
        plt.close("all")

    @methodParameters(badFrac=(0.01, 0.03, 0.05, 0.1, 0.25))
    def testRobust(self, badFrac=0.25):
        """Test robustness of FitPolynomialTask in the presence of junk

        We add a bunch of junk and see how FitPolynomialTask behaves, both when
        it knows where the junk is, and when it has to find the junk itself.
        """
        coeffs = (1234.567, 1.2345, -0.12345)
        noise = 0.321
        self.config.order = 3
        data = self.makeData(coeffs, noise)
        num = len(data.xx)
        bad = np.zeros_like(data.xx, dtype=bool)
        bad[self.rng.randint(num, size=int(badFrac*num))] = True

        # No junk
        result = self.task.run(data.xx, data.yy)
        self.assertResult(data, result)

        data.yy[bad] += np.max(data.yy)

        # Fit when I tell you where the junk is
        result = self.task.run(data.xx, data.yy, good=~bad)
        self.assertResult(data, result, bad)

        # Fit when you have to find the junk yourself
        result = self.task.run(data.xx, data.yy)
        self.assertResult(data, result, bad)

    def testEaseOff(self):
        """Test easing off the rejection

        This feature guards against rejecting all the data.

        We make a bimodal distribution of points, and set the rejection
        threshold very small, so ordinarily all the points would be rejected.
        """
        xx = np.arange(1000, dtype=int)
        yy = 0.5 + 1.0*(xx % 2)
        self.config.order = 0
        self.config.rejThresh = 0.1

        self.config.easeOff = 0
        with self.assertRaises(RuntimeError):
            result = self.task.run(xx, yy)

        self.config.easeOff = 3
        result = self.task.run(xx, yy)
        self.assertFloatsAlmostEqual(result.fit, 1.0, rtol=1.0e-14)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
