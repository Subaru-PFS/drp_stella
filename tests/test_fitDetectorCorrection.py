import unittest

import numpy as np

import lsst.utils.tests
from lsst.geom import Box2I, Extent2I, Point2I

from pfs.drp.stella.fitDetectorCorrection import FitDetectorCorrectionTask, makePenaltyMatrix


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

    def fit(self, **kwargs):
        """Fit the correction, with config overrides"""
        config = FitDetectorCorrectionTask.ConfigClass()
        for key, value in kwargs.items():
            setattr(config, key, value)
        task = FitDetectorCorrectionTask(config=config)
        return task.run(self.bbox, False, self.fiberId, self.xx, self.yy, self.dx)

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
