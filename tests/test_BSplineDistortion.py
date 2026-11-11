import pickle

import numpy as np
from scipy.interpolate import NdBSpline

import lsst.utils.tests
from lsst.geom import Box2I, Point2D, Point2I, Extent2I

from pfs.drp.stella import BSpline2D, BSplineDistortion, Distortion
from pfs.drp.stella.OpticalModel import DetectorModel
from pfs.drp.stella.math import makeAffineTransform
from pfs.drp.stella.tests.utils import runTests

display = None


def makeKnots(low, high, spacing, order=3):
    """Make a full knot vector with uniform interior spacing"""
    num = int(np.round((high - low) / spacing))
    interior = np.linspace(low, high, num + 1)[1:-1]
    return np.concatenate([[low] * (order + 1), interior, [high] * (order + 1)])


def makeSpline(rng, xLow, xHigh, yLow, yHigh, spacing, amplitude=0.05, order=3):
    """Make a spline with random coefficients"""
    xKnots = makeKnots(xLow, xHigh, spacing, order)
    yKnots = makeKnots(yLow, yHigh, spacing, order)
    coeff = rng.normal(0.0, amplitude, size=(len(xKnots) - order - 1, len(yKnots) - order - 1))
    return BSpline2D(order, xKnots, yKnots, coeff)


class BSpline2DTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)

    def testEvaluate(self):
        """Evaluation agrees with scipy, including clamping out of range"""
        for order in (1, 2, 3):
            with self.subTest(order=order):
                spline = makeSpline(self.rng, -0.5, 1000.5, 10.0, 2000.0, 150.0, order=order)
                reference = NdBSpline(
                    (spline.getXKnots(), spline.getYKnots()), spline.getCoefficients(), order
                )
                xx = self.rng.uniform(-100, 1100, size=1000)
                yy = self.rng.uniform(-100, 2100, size=1000)
                values = spline(xx, yy)
                expected = reference(np.vstack([np.clip(xx, -0.5, 1000.5), np.clip(yy, 10.0, 2000.0)]).T)
                self.assertFloatsAlmostEqual(values, expected, atol=1.0e-12)
                self.assertFloatsAlmostEqual(spline(xx[0], yy[0]), expected[0], atol=1.0e-12)
                self.assertFloatsAlmostEqual(spline(Point2D(xx[0], yy[0])), expected[0], atol=1.0e-12)

    def testEmpty(self):
        """An empty spline evaluates to zero"""
        spline = BSpline2D()
        self.assertTrue(spline.isEmpty())
        self.assertEqual(spline.getNumParameters(), 0)
        self.assertEqual(spline(123.0, 456.0), 0.0)
        self.assertFloatsEqual(spline(np.array([1.0, 2.0]), np.array([3.0, 4.0])), 0.0)

    def testValidation(self):
        """Bad inputs are caught"""
        xKnots = makeKnots(0, 100, 20)
        yKnots = makeKnots(0, 200, 20)
        numX = len(xKnots) - 4
        numY = len(yKnots) - 4
        with self.assertRaises(Exception):
            BSpline2D(3, xKnots, yKnots, np.zeros((numX + 1, numY)))
        with self.assertRaises(Exception):
            BSpline2D(3, xKnots[::-1].copy(), yKnots, np.zeros((numX, numY)))
        with self.assertRaises(Exception):
            BSpline2D(10, xKnots, yKnots, np.zeros((numX, numY)))

    def testPickle(self):
        """Round-trip pickle"""
        for spline in (makeSpline(self.rng, 0, 100, 0, 200, 20), BSpline2D()):
            copy = pickle.loads(pickle.dumps(spline))
            self.assertEqual(copy.isEmpty(), spline.isEmpty())
            self.assertFloatsEqual(copy.getXKnots(), spline.getXKnots())
            self.assertFloatsEqual(copy.getYKnots(), spline.getYKnots())
            self.assertFloatsEqual(copy.getCoefficients(), spline.getCoefficients())


class BSplineDistortionTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)
        self.bbox = Box2I(Point2I(0, 0), Extent2I(4096, 4176))
        self.boundary = 2047.5
        self.xSplines = [
            makeSpline(self.rng, -0.5, self.boundary, -0.5, 4175.5, 300.0),
            makeSpline(self.rng, self.boundary, 4095.5, -0.5, 4175.5, 200.0),
        ]
        self.ySplines = [BSpline2D(), makeSpline(self.rng, self.boundary, 4095.5, -0.5, 4175.5, 500.0)]
        self.distortion = BSplineDistortion(np.array([self.boundary]), self.xSplines, self.ySplines)

    def assertDistortionsEqual(self, lhs, rhs):
        """Assert that the distortions are the same"""
        self.assertIsInstance(lhs, BSplineDistortion)
        self.assertIsInstance(rhs, BSplineDistortion)
        self.assertFloatsEqual(lhs.getXBoundaries(), rhs.getXBoundaries())
        for ll, rr in zip(lhs.getXSplines() + lhs.getYSplines(), rhs.getXSplines() + rhs.getYSplines()):
            self.assertEqual(ll.isEmpty(), rr.isEmpty())
            self.assertEqual(ll.getOrder(), rr.getOrder())
            self.assertFloatsEqual(ll.getXKnots(), rr.getXKnots())
            self.assertFloatsEqual(ll.getYKnots(), rr.getYKnots())
            self.assertFloatsEqual(ll.getCoefficients(), rr.getCoefficients())

    def testBasic(self):
        """Regions, number of parameters and evaluation"""
        distortion = self.distortion
        self.assertEqual(distortion.getNumRegions(), 2)
        self.assertEqual(distortion.getRegion(100.0), 0)
        self.assertEqual(distortion.getRegion(self.boundary), 1)
        self.assertEqual(distortion.getRegion(4000.0), 1)
        numParams = sum(ss.getNumParameters() for ss in self.xSplines + self.ySplines)
        self.assertEqual(distortion.getNumParameters(), numParams)

        xx = self.rng.uniform(0, 4095, size=1000)
        yy = self.rng.uniform(0, 4175, size=1000)
        result = distortion(xx, yy)
        left = xx < self.boundary
        self.assertFloatsEqual(result[left, 0], self.xSplines[0](xx[left], yy[left]))
        self.assertFloatsEqual(result[~left, 0], self.xSplines[1](xx[~left], yy[~left]))
        self.assertFloatsEqual(result[left, 1], 0.0)
        self.assertFloatsEqual(result[~left, 1], self.ySplines[1](xx[~left], yy[~left]))
        point = distortion(Point2D(xx[0], yy[0]))
        self.assertFloatsEqual(point.getX(), result[0, 0])
        self.assertFloatsEqual(point.getY(), result[0, 1])

    def testClone(self):
        """Clone is a deep copy"""
        self.assertDistortionsEqual(self.distortion.clone(), self.distortion)

    def testValidation(self):
        """Wrong number of splines is caught"""
        with self.assertRaises(Exception):
            BSplineDistortion(np.array([self.boundary]), self.xSplines[:1], self.ySplines)
        with self.assertRaises(Exception):
            BSplineDistortion(np.array([]), self.xSplines, self.ySplines)

    def testFits(self):
        """Round-trip through FITS"""
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            self.distortion.writeFits(filename)
            copy = Distortion.readFits(filename)
        self.assertDistortionsEqual(copy, self.distortion)

    def testPickle(self):
        """Round-trip pickle"""
        copy = pickle.loads(pickle.dumps(self.distortion))
        self.assertDistortionsEqual(copy, self.distortion)

    def testDetectorModel(self):
        """DetectorModel applies the distortion in the pixel frame, and inverts it"""
        rightCcd = makeAffineTransform(np.array([0.1, -0.2, 1.0e-4, -2.0e-4, 3.0e-4, 1.0e-4]))
        model = DetectorModel(self.bbox, rightCcd, [self.distortion])
        reference = DetectorModel(self.bbox, rightCcd)
        self.assertEqual(len(model.getDistortions()), 1)

        xx = np.concatenate([self.rng.uniform(5, 2040, size=500), self.rng.uniform(2060, 4090, size=500)])
        yy = self.rng.uniform(5, 4170, size=1000)
        pixels = model.detectorToPixels(xx, yy)
        undistorted = reference.detectorToPixels(xx, yy)
        expected = undistorted + self.distortion(undistorted[:, 0].copy(), undistorted[:, 1].copy())
        self.assertFloatsAlmostEqual(pixels, expected, atol=1.0e-12)

        # Round trip
        detector = model.pixelsToDetector(pixels[:, 0].copy(), pixels[:, 1].copy())
        self.assertFloatsAlmostEqual(detector[:, 0], xx, atol=2.0e-3)
        self.assertFloatsAlmostEqual(detector[:, 1], yy, atol=2.0e-3)

        # Copy carries the distortions
        copy = model.copy()
        self.assertFloatsEqual(copy.detectorToPixels(xx, yy), pixels)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
