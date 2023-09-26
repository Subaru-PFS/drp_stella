from typing import Optional

import numpy as np

import lsst.utils.tests
from lsst.pipe.base import Struct

from pfs.drp.stella.math import solveLeastSquaresDesign
from pfs.drp.stella.tests.utils import runTests, methodParametersProduct

display = None


class SolveLeastSquaresTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)

    def doFit(
        self, num: int, noise: float, *, coeffs: Optional[np.ndarray] = None, debug: bool = False, **kwargs
    ) -> Struct:
        """Perform a least-squares fit

        We fit a quadratic function by calculating the design matrix and
        calling solveLeastSquaresDesign.

        Parameters
        ----------
        num : `int`
            Number of points.
        noise : `float`
            Noise level.
        coeffs : `numpy.ndarray`, optional
            Coefficients of the quadratic function.
        debug : `bool`, optional
            Display debugging information?
        **kwargs
            Additional arguments for `solveLeastSquaresDesign`.

        Returns
        -------
        x : `numpy.ndarray`
            Independent variable.
        y : `numpy.ndarray`
            Dependent variable.
        params : `numpy.ndarray`
            Fitted parameters.
        fit : `numpy.ndarray`
            Fitted values.
        resid : `numpy.ndarray`
            Residuals.
        """
        if coeffs is None:
            coeffs = np.array([1.0, 2.0, 3.0])

        xx = np.linspace(0, 10.0, num, dtype=float)
        design = np.vstack([xx**2, xx, np.ones_like(xx)]).T.copy()
        yy = np.dot(design, coeffs)

        meas = yy + self.rng.normal(0, noise, size=xx.shape)
        err = np.full_like(meas, noise)

        params = solveLeastSquaresDesign(design, meas, err, **kwargs)
        fit = np.dot(design, params)
        resid = meas - fit

        if debug:
            import matplotlib.pyplot as plt

            plt.plot(xx, meas, "ko")
            plt.plot(xx, yy, "k-")
            plt.plot(xx, fit, "b-")
            plt.show()

        return Struct(x=xx, y=yy, coeffs=coeffs, params=params, fit=fit, resid=resid)

    @methodParametersProduct(num=(10, 100, 1000), noise=(1.0e-1, 1.0e-2, 1.0e-3))
    def testBasic(self, num: int, noise: float):
        """Test basic operation

        We check that the fitting results are as expected.

        Parameters
        ----------
        num : `int`
            Number of points.
        noise : `float`
            Noise level.
        """
        results = self.doFit(num, noise)
        self.assertFloatsAlmostEqual(results.fit, results.y, atol=3 * noise)
        self.assertFloatsAlmostEqual(np.average(results.resid), 0.0, atol=1.0e-1)
        self.assertLess(np.std(results.resid), 1.2 * noise)

    def testForced(self, num=100, noise=1.0e-1):
        """Test forcing a parameter

        We check that the forced parameter is as expected.

        Parameters
        ----------
        num : `int`
            Number of points.
        noise : `float`
            Noise level.
        """
        # Check that we get the crazy parameter out if we force it
        value = 3.21
        results = self.doFit(
            num, noise, forced=np.array([True, False, False]), params=np.array([value, np.nan, np.nan])
        )
        self.assertEqual(results.params[0], value)

        # Check that we get good fit results if we give it a good parameter
        coeffs = np.array([value, 2.0, 3.0])
        results = self.doFit(
            num,
            noise,
            coeffs=coeffs,
            forced=np.array([True, False, False]),
            params=np.array([value, np.nan, np.nan]),
        )
        self.assertEqual(results.params[0], value)
        self.assertFloatsAlmostEqual(results.params[1:], coeffs[1:], atol=2.0e-2)
        self.assertFloatsAlmostEqual(results.fit, results.y, atol=3 * noise)
        self.assertFloatsAlmostEqual(np.average(results.resid), 0.0, atol=1.0e-1)
        self.assertLess(np.std(results.resid), 1.2 * noise)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
