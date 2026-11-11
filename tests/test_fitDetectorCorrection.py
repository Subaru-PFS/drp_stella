import unittest

import numpy as np

import lsst.utils.tests
from lsst.geom import Box2I, Extent2I, Point2I

from pfs.drp.stella.fitDetectorCorrection import FitDetectorCorrectionTask, makePenaltyMatrix
from pfs.drp.stella.fitDetectorCorrection import makePolynomialMatrix


class FitDetectorCorrectionTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.bbox = Box2I(Point2I(0, 0), Extent2I(2048, 2048))
        self.amplitude = 0.05
        self.center = (1000.0, 1000.0)
        self.sigma = 60.0
        self.noise = 0.02

        # Traces cover only part of the detector, ending part way through knot intervals
        self.xMin = 300.0
        self.xMax = 1700.0
        self.yMin = 50.0
        self.yMax = 1990.0
        fiberX = np.arange(self.xMin, self.xMax, 6.2)
        rows = np.arange(self.yMin, self.yMax, 1.0)
        self.fiberId = np.repeat(np.arange(len(fiberX)), len(rows))
        self.xx = np.repeat(fiberX, len(rows))
        self.yy = np.tile(rows, len(fiberX))
        self.dx = self.bump(self.xx, self.yy) + self.rng.normal(0.0, self.noise, self.xx.size)

    def bump(self, xx, yy):
        """Model of the detector effect"""
        radius2 = (xx - self.center[0]) ** 2 + (yy - self.center[1]) ** 2
        return self.amplitude * np.exp(-0.5 * radius2 / self.sigma**2)

    def fit(self, dx=None, isDivided=False, **kwargs):
        """Fit the correction, with config overrides"""
        config = FitDetectorCorrectionTask.ConfigClass()
        for key, value in kwargs.items():
            setattr(config, key, value)
        task = FitDetectorCorrectionTask(config=config)
        if dx is None:
            dx = self.dx
        return task.run(self.bbox, isDivided, self.fiberId, self.xx, self.yy, dx)

    def evaluateOutside(self, correction):
        """Return the maximum absolute correction outside the data"""
        xGrid, yGrid = np.meshgrid(np.arange(0, 2048, 4.0), np.arange(0, 2048, 4.0))
        outside = (xGrid < self.xMin) | (xGrid > self.xMax) | (yGrid < self.yMin) | (yGrid > self.yMax)
        values = correction(xGrid[outside], yGrid[outside])[:, 0]
        return np.max(np.abs(values))

    def testRecovery(self):
        """The regularized fit recovers the bump, and stays small outside"""
        correction = self.fit()
        self.assertTrue(all(spline.isEmpty() for spline in correction.getYSplines()))

        # Recovery where there are data. The per-fiber medians are subtracted, so compare the shape.
        xGrid, yGrid = np.meshgrid(np.arange(800, 1200, 4.0), np.arange(800, 1200, 4.0))
        values = correction(xGrid.flatten(), yGrid.flatten())[:, 0]
        expected = self.bump(xGrid.flatten(), yGrid.flatten())
        self.assertLess(np.max(np.abs(values - expected)), 0.15 * self.amplitude)

        # No wild extrapolation where there are no data
        self.assertLess(self.evaluateOutside(correction), 0.3 * self.amplitude)

    def testUnregularized(self):
        """Without regularization, the fit is wild outside the data

        This demonstrates that the regularization is necessary (and that the
        test of the regularization is meaningful).
        """
        regularized = self.evaluateOutside(self.fit())
        unregularized = self.evaluateOutside(self.fit(smoothing=0.0, damp=0.0))
        self.assertGreater(unregularized, 2 * regularized)

    def testGradient(self):
        """Large-scale gradients are excluded from the correction

        Each region (each CCD of a divided detector) gets its own gradient.
        """
        middle = self.bbox.getCenterX()
        slope = np.where(self.xx < middle, 0.03, -0.02)  # Different for each half
        gradient = slope * (self.yy - 1000.0) / 1000.0 + 0.01 * (self.xx - 1000.0) / 1000.0
        dx = self.dx + gradient

        xGrid, yGrid = np.meshgrid(np.arange(400, 1600, 8.0), np.arange(100, 1900, 8.0))
        # Avoid the boundary between regions, where a divided detector has a discontinuity
        awayFromBoundary = np.abs(xGrid.flatten() - middle) > 50
        xGrid = xGrid.flatten()[awayFromBoundary]
        yGrid = yGrid.flatten()[awayFromBoundary]
        bump = self.bump(xGrid, yGrid)
        gridGradient = np.where(xGrid < middle, 0.03, -0.02) * (yGrid - 1000.0) / 1000.0
        for isDivided in (False, True):
            with self.subTest(isDivided=isDivided):
                values = self.fit(dx, isDivided)(xGrid, yGrid)[:, 0]
                if isDivided:
                    # The gradient is removed, leaving the bump
                    self.assertLess(np.max(np.abs(values - bump)), 0.15 * self.amplitude)
                else:
                    # A single plane can't remove gradients that differ between the halves, but what's left
                    # is the difference from the mean gradient.
                    self.assertGreater(np.max(np.abs(values - bump)), 0.2 * self.amplitude)

        # Without removing the polynomial, the gradient remains in the correction
        values = self.fit(dx, True, polynomialOrder=-1)(xGrid, yGrid)[:, 0]
        self.assertGreater(np.max(np.abs(values - bump)), 0.5 * np.max(np.abs(gridGradient)))
        self.assertLess(np.max(np.abs(values - bump - gridGradient)), 0.15 * self.amplitude)

    def testPolynomialMatrix(self):
        """The polynomial design matrix has the expected terms"""
        xx = np.array([0.5, -0.25, 1.0])
        yy = np.array([2.0, 0.5, -1.0])
        self.assertEqual(makePolynomialMatrix(-1, xx, yy).shape, (3, 0))
        self.assertFloatsEqual(makePolynomialMatrix(0, xx, yy).toarray(), np.ones((3, 1)))
        expected = np.vstack([np.ones(3), xx, yy, xx**2, xx * yy, yy**2]).T
        self.assertFloatsAlmostEqual(makePolynomialMatrix(2, xx, yy).toarray(), expected, atol=1.0e-15)

    def testPenaltyMatrix(self):
        """The penalty is zero for linear coefficients, and has the right shape"""
        numX, numY = 5, 7
        penalty = makePenaltyMatrix(numX, numY, 1.0, 0.0)
        self.assertEqual(penalty.shape, ((numX - 2) * numY + numX * (numY - 2), numX * numY))
        xIndex, yIndex = np.meshgrid(np.arange(numX), np.arange(numY), indexing="ij")
        linear = (1.0 + 2.0 * xIndex - 3.0 * yIndex).flatten()  # y varying fastest
        self.assertFloatsAlmostEqual(penalty @ linear, 0.0, atol=1.0e-12)
        damped = makePenaltyMatrix(numX, numY, 0.0, 4.0)
        self.assertFloatsAlmostEqual(damped @ linear, 2.0 * linear, atol=1.0e-12)
        self.assertEqual(makePenaltyMatrix(numX, numY, 0.0, 0.0).shape, (0, numX * numY))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
