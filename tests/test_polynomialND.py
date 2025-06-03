import numpy as np

import lsst.utils.tests
from lsst.pex.exceptions import InvalidParameterError
from pfs.drp.stella.tests.utils import runTests, methodParameters
from pfs.drp.stella.math import NormalizedPolynomialND

import math

eps = np.nextafter(np.float64(1), np.inf) - np.float64(1)


def getExponents(order, nVars):
    """Get the exponents for a normalized polynomial of given order and number of variables."""
    return np.array(NormalizedPolynomialND.getExponents(nVars, order))


getPolynomialOrder = NormalizedPolynomialND.orderFromNParameters


class NormalizedPolynomialNDTestCase(lsst.utils.tests.TestCase):
    """NormalizedPolynomialND test case"""

    def testGetPolynomialOrder(self):
        """Test getPolynomialOrder(nVars, nParams)"""
        with self.assertRaises(InvalidParameterError):
            getPolynomialOrder(0, 1)

        with self.assertRaises(InvalidParameterError):
            getPolynomialOrder(1, 0)

        with self.assertRaises(InvalidParameterError):
            getPolynomialOrder(4, 2)

        self.assertEqual(getPolynomialOrder(4, math.comb(4 + 9, 9)), 9)
        self.assertEqual(getPolynomialOrder(5, math.comb(5 + 10, 10)), 10)

    @methodParameters(
        order=(1, 1, 7, 7),
        nVars=(1, 5, 1, 5),
    )
    def testGetExponents(self, order, nVars):
        """Test getExponents(order, nVars)"""
        exponents = getExponents(order, nVars)
        self.assertIs(exponents.dtype, np.array([1]).dtype)
        self.assertEqual(exponents.shape, (math.comb(order + nVars, nVars), nVars))

        self.assertTrue(np.all(exponents >= 0))
        self.assertTrue(np.all(np.sum(exponents, axis=(1,)) <= order))

        self.assertTrue(
            all(
                (sum(p),) + tuple(p) > (sum(q),) + tuple(q)
                for p, q in zip(exponents[:-1, :], exponents[1:, :])
            ),
        )

    def testUnivariate(self):
        """Test `NormalizedPolynomialND` when N = 1"""
        np.random.seed(54321)

        order = 5
        nVars = 1
        nParams = math.comb(order + nVars, nVars)
        nPoints = 4
        left = -3.0
        right = 2.0

        params = np.random.uniform(-2, 2, size=(nParams,))
        x = np.random.uniform(-2, 2, size=(nPoints, 1))

        poly = NormalizedPolynomialND(params, np.array([left]), np.array([right]), False)  # using old norm

        self.assertIsInstance(poly(x[0]), float)
        self.assertEqual(poly(x).shape, (nPoints,))

        self.assertFloatsAlmostEqual(poly(x)[0], poly(x[0]), rtol=16 * eps)

        normalizedX1 = (x[1] - left) / (right - left)  # Old normalization scheme
        exponents = np.asarray(getExponents(order, nVars))[:, 0]

        self.assertFloatsAlmostEqual(normalizedX1**exponents @ params, poly(x[1]), rtol=16 * eps)

    def testMultivariate(self):
        """Test `NormalizedPolynomialND` when N > 1"""
        np.random.seed(12345)

        order = 5
        nVars = 3
        nParams = math.comb(order + nVars, nVars)
        nPoints = 4
        minVertex = np.array([-3, -4, -5], dtype=float)
        maxVertex = np.array([2, 1, 6], dtype=float)

        params = np.random.uniform(-2, 2, size=(nParams,))
        x = np.random.uniform(-2, 2, size=(nPoints, nVars))

        poly = NormalizedPolynomialND(params, minVertex, maxVertex, False)  # using old normalization

        self.assertFloatsAlmostEqual(poly(x)[0], poly(x[0, :]), rtol=16 * eps)

        normalizedX1 = (x[1, :] - minVertex) / (maxVertex - minVertex)  # Old normalization scheme

        self.assertFloatsAlmostEqual(poly.normalize(x[1, :]), normalizedX1)

        self.assertFloatsAlmostEqual(
            np.prod(normalizedX1.reshape(1, nVars) ** getExponents(order, nVars), axis=(1,)) @ params,
            poly(x[1, :]),
            rtol=16 * eps,
        )


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
