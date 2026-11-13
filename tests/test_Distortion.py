"""Tests of Distortion subclasses

Distortions can be persisted on their own (FITS file, bytes, pickle), in
addition to being persisted as part of a detectorMap. These tests check that
the standalone persistence round-trips for each registered subclass, and that
the analytic distortions can be fit.
"""

import os
import pickle

import numpy as np

import lsst.utils.tests
from lsst.geom import Box2D, Box2I, Extent2I, Point2D, Point2I

from pfs.drp.stella import Distortion, MosaicPolynomialDistortion, PolynomialDistortion
from pfs.drp.stella.tests.utils import runTests


class DistortionTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.order = 3
        self.bbox = Box2I(Point2I(0, 0), Extent2I(4096, 4176))
        self.range = Box2D(self.bbox)
        self.rng = np.random.default_rng(12345)
        numPoints = 100
        self.xx = self.rng.uniform(self.range.getMinX(), self.range.getMaxX(), numPoints)
        self.yy = self.rng.uniform(self.range.getMinY(), self.range.getMaxY(), numPoints)

    def makePolynomialDistortion(
        self, xOrder: int | None = None, yOrder: int | None = None
    ) -> PolynomialDistortion:
        """Construct a PolynomialDistortion with random coefficients

        The orders default to ``self.order``; if both are defaulted, the
        single-order constructor is used.
        """
        if xOrder is None and yOrder is None:
            numCoeff = PolynomialDistortion.getNumDistortionForOrder(self.order)
            xCoeff = self.rng.normal(size=numCoeff)
            yCoeff = self.rng.normal(size=numCoeff)
            return PolynomialDistortion(self.order, self.range, xCoeff, yCoeff)
        xOrder = self.order if xOrder is None else xOrder
        yOrder = self.order if yOrder is None else yOrder
        xCoeff = self.rng.normal(size=PolynomialDistortion.getNumDistortionForOrder(xOrder))
        yCoeff = self.rng.normal(size=PolynomialDistortion.getNumDistortionForOrder(yOrder))
        return PolynomialDistortion(xOrder, yOrder, self.range, xCoeff, yCoeff)

    def makeMosaicPolynomialDistortion(
        self, xOrder: int | None = None, yOrder: int | None = None
    ) -> MosaicPolynomialDistortion:
        """Construct a MosaicPolynomialDistortion with random coefficients

        The orders default to ``self.order``; if both are defaulted, the
        single-order constructor is used.
        """
        if xOrder is None and yOrder is None:
            numParams = MosaicPolynomialDistortion.getNumParametersForOrder(self.order)
            return MosaicPolynomialDistortion(self.order, self.range, self.rng.normal(size=numParams))
        xOrder = self.order if xOrder is None else xOrder
        yOrder = self.order if yOrder is None else yOrder
        numParams = MosaicPolynomialDistortion.getNumParametersForOrder(xOrder, yOrder)
        return MosaicPolynomialDistortion(xOrder, yOrder, self.range, self.rng.normal(size=numParams))

    def assertDistortionsEqual(self, left: Distortion, right: Distortion):
        """Assert that two distortions are identical"""
        self.assertIs(type(left), type(right))
        self.assertEqual(left.getXOrder(), right.getXOrder())
        self.assertEqual(left.getYOrder(), right.getYOrder())
        self.assertEqual(left.getRange(), right.getRange())
        np.testing.assert_array_equal(left.getCoefficients(), right.getCoefficients())
        np.testing.assert_array_equal(left(self.xx, self.yy), right(self.xx, self.yy))

    def checkPersistence(self, distortion: Distortion):
        """Check that persistence round-trips

        Parameters
        ----------
        distortion : `Distortion`
            Distortion to check.
        """
        self.assertDistortionsEqual(pickle.loads(pickle.dumps(distortion)), distortion)

        data = distortion.toBytes()
        self.assertDistortionsEqual(Distortion.fromBytes(data), distortion)
        self.assertDistortionsEqual(type(distortion).fromBytes(data), distortion)

        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            distortion.writeFits(filename)
            self.assertTrue(os.path.exists(filename))
            self.assertDistortionsEqual(Distortion.readFits(filename), distortion)
            self.assertDistortionsEqual(type(distortion).readFits(filename), distortion)

    def setNonPixelRange(self):
        """Use a range that isn't aligned with pixel boundaries"""
        self.range = Box2D(Point2D(-1.3, -200.7), Point2D(1.3, 200.7))
        self.xx = self.rng.uniform(self.range.getMinX(), self.range.getMaxX(), len(self.xx))
        self.yy = self.rng.uniform(self.range.getMinY(), self.range.getMaxY(), len(self.yy))

    def testPolynomialDistortion(self):
        """Test persistence of PolynomialDistortion"""
        self.checkPersistence(self.makePolynomialDistortion())

    def testMosaicPolynomialDistortion(self):
        """Test persistence of MosaicPolynomialDistortion"""
        self.checkPersistence(self.makeMosaicPolynomialDistortion())

    def testNonPixelRange(self):
        """Test persistence with a range not aligned with pixel boundaries"""
        self.setNonPixelRange()
        self.checkPersistence(self.makePolynomialDistortion())
        self.checkPersistence(self.makeMosaicPolynomialDistortion())

    def fitDistortion(
        self,
        truth: Distortion,
        numLines: int = 1000,
        numTraces: int = 0,
        traceErr: float = 0.01,
        forced: np.ndarray | None = None,
        params: np.ndarray | None = None,
    ) -> Distortion:
        """Fit a distortion to noiseless measurements generated from truth

        Parameters
        ----------
        truth : `Distortion`
            Distortion used to generate the measurements.
        numLines : `int`
            Number of line measurements (both x and y).
        numTraces : `int`
            Number of trace measurements (x only).
        traceErr : `float`
            Error in x for trace measurements.
        forced : `numpy.ndarray` of `bool`, optional
            Which parameters are forced.
        params : `numpy.ndarray` of `float`, optional
            Values of forced parameters.

        Returns
        -------
        distortion : `Distortion`
            Fit distortion.
        """
        num = numLines + numTraces
        xx = self.rng.uniform(self.range.getMinX(), self.range.getMaxX(), num)
        yy = self.rng.uniform(self.range.getMinY(), self.range.getMaxY(), num)
        isLine = np.zeros(num, dtype=bool)
        isLine[:numLines] = True
        meas = truth(xx, yy)
        xMeas = meas[:, 0].copy()
        yMeas = meas[:, 1].copy()
        yMeas[~isLine] = 1.0e6  # Traces have no y measurement: this should be ignored
        xErr = np.where(isLine, 0.01, traceErr)
        yErr = np.full(num, 0.01, dtype=float)
        if truth.getXOrder() == truth.getYOrder():
            # Exercise the single-order interface
            return type(truth).fit(
                truth.getXOrder(), self.range, xx, yy, xMeas, yMeas, xErr, yErr, isLine,
                forced=forced, params=params,
            )
        return type(truth).fit(
            truth.getXOrder(), truth.getYOrder(), self.range, xx, yy, xMeas, yMeas, xErr, yErr, isLine,
            forced=forced, params=params,
        )

    def assertFitMatches(self, fit: Distortion, truth: Distortion):
        """Assert that a fit distortion matches the truth"""
        self.assertIs(type(fit), type(truth))
        self.assertFloatsAlmostEqual(fit(self.xx, self.yy), truth(self.xx, self.yy), atol=1.0e-7)

    def testFitPolynomialDistortion(self):
        """Test fitting PolynomialDistortion"""
        truth = self.makePolynomialDistortion()
        self.assertFitMatches(self.fitDistortion(truth), truth)

    def testFitMosaicPolynomialDistortion(self):
        """Test fitting MosaicPolynomialDistortion"""
        truth = self.makeMosaicPolynomialDistortion()
        self.assertFitMatches(self.fitDistortion(truth), truth)

    def testDifferentOrders(self):
        """Test distortions with different orders in x and y"""
        xOrder, yOrder = 4, 2
        numX = PolynomialDistortion.getNumDistortionForOrder(xOrder)
        numY = PolynomialDistortion.getNumDistortionForOrder(yOrder)
        for truth, numParams in (
            (self.makePolynomialDistortion(xOrder, yOrder), numX + numY),
            (self.makeMosaicPolynomialDistortion(xOrder, yOrder), 6 + numX + numY),
        ):
            Class = type(truth)
            with self.subTest(distortion=Class.__name__):
                self.assertEqual(truth.getXOrder(), xOrder)
                self.assertEqual(truth.getYOrder(), yOrder)
                with self.assertRaises(RuntimeError):
                    truth.getOrder()
                self.assertEqual(Class.getNumParametersForOrder(xOrder, yOrder), numParams)
                self.assertEqual(truth.getNumParameters(), numParams)
                self.assertEqual(len(truth.getCoefficients()), numParams)
                self.assertEqual(truth.getXNumDistortion(), numX)
                self.assertEqual(truth.getYNumDistortion(), numY)
                self.assertEqual(len(truth.getXCoefficients()), numX)
                self.assertEqual(len(truth.getYCoefficients()), numY)
                self.checkPersistence(truth)
                self.assertFitMatches(self.fitDistortion(truth, numTraces=1000), truth)

    def testSameOrders(self):
        """Test that the single-order interfaces match the two-order ones"""
        for Class in (PolynomialDistortion, MosaicPolynomialDistortion):
            with self.subTest(distortion=Class.__name__):
                self.assertEqual(
                    Class.getNumParametersForOrder(self.order),
                    Class.getNumParametersForOrder(self.order, self.order),
                )
        distortion = self.makePolynomialDistortion()
        self.assertEqual(distortion.getOrder(), self.order)
        self.assertEqual(distortion.getXOrder(), self.order)
        self.assertEqual(distortion.getYOrder(), self.order)

    def testFitWithDominantTraces(self):
        """Test fitting when traces dominate the x measurements

        Traces provide only x measurements. When they carry far more weight
        than the lines, the y fit should be unaffected. This failed when x and
        y were fit together, because the eigenvalue threshold for the matrix
        solution was set by the x measurements.
        """
        for truth in (self.makePolynomialDistortion(), self.makeMosaicPolynomialDistortion()):
            with self.subTest(distortion=type(truth).__name__):
                fit = self.fitDistortion(truth, numTraces=10000, traceErr=1.0e-6)
                self.assertFitMatches(fit, truth)

    def testFitForcedPolynomialDistortion(self):
        """Test fitting PolynomialDistortion with forced parameters

        Parameters are ordered as x coefficients then y coefficients.
        """
        truth = self.makePolynomialDistortion()
        numCoeff = PolynomialDistortion.getNumDistortionForOrder(self.order)
        forced = np.zeros(2*numCoeff, dtype=bool)
        forced[[1, numCoeff + 2]] = True
        params = np.concatenate((truth.getXCoefficients(), truth.getYCoefficients()))

        # Forcing to the true values should recover the truth
        fit = self.fitDistortion(truth, forced=forced, params=params)
        self.assertFitMatches(fit, truth)

        # Forcing to different values should be respected
        params[forced] += 1.0
        fit = self.fitDistortion(truth, forced=forced, params=params)
        self.assertEqual(fit.getXCoefficients()[1], params[1])
        self.assertEqual(fit.getYCoefficients()[2], params[numCoeff + 2])

    def testFitForcedMosaicPolynomialDistortion(self):
        """Test fitting MosaicPolynomialDistortion with forced parameters

        Parameters are ordered as x coefficients, y coefficients, then the
        affine terms X, XX, XY, Y, YX, YY.
        """
        truth = self.makeMosaicPolynomialDistortion()
        numCoeff = MosaicPolynomialDistortion.getNumDistortionForOrder(self.order)
        affine = truth.getAffine()
        translation = affine.getTranslation()
        matrix = affine.getLinear().getMatrix()
        affineParams = [
            translation.getX(), matrix[0, 0], matrix[0, 1], translation.getY(), matrix[1, 0], matrix[1, 1]
        ]
        params = np.concatenate((truth.getXCoefficients(), truth.getYCoefficients(), affineParams))
        forced = np.zeros_like(params, dtype=bool)
        forcedIndices = [0, numCoeff + 1, 2*numCoeff + 2, 2*numCoeff + 3]  # x, y, affine XY and Y
        forced[forcedIndices] = True

        # Forcing to the true values should recover the truth
        fit = self.fitDistortion(truth, forced=forced, params=params)
        self.assertFitMatches(fit, truth)

        # Forcing to different values should be respected
        params[forced] += 1.0
        fit = self.fitDistortion(truth, forced=forced, params=params)
        self.assertEqual(fit.getXCoefficients()[0], params[0])
        self.assertEqual(fit.getYCoefficients()[1], params[numCoeff + 1])
        fitAffine = fit.getAffine()
        self.assertEqual(fitAffine.getLinear().getMatrix()[0, 1], params[2*numCoeff + 2])
        self.assertEqual(fitAffine.getTranslation().getY(), params[2*numCoeff + 3])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
