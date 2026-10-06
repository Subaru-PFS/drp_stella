import numpy as np

import lsst.utils.tests

from pfs.drp.stella.synthetic import SyntheticPsfConfig, makeSyntheticPsfArc
from pfs.drp.stella.tests import runTests


class SyntheticPsfConfigTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticPsfConfig()

    def testTraceCentersPitch(self):
        """Fiber centers within a block are separated by the fiber pitch;
        at a block boundary, they're separated by the pitch plus the gap"""
        centers = self.config.traceCenters
        spacing = np.diff(centers)
        withinBlock = np.delete(
            spacing, np.arange(self.config.blockSize - 1, len(spacing), self.config.blockSize)
        )
        self.assertFloatsAlmostEqual(withinBlock, self.config.separation, atol=1.0e-9)

        atGap = spacing[self.config.blockSize - 1 :: self.config.blockSize]
        expectedGap = self.config.separation * (1 + self.config.gapWidth)
        self.assertFloatsAlmostEqual(atGap, expectedGap, atol=1.0e-9)

    def testBlockEdgeIndices(self):
        """Block edges bracket every gap, plus the two outer edges"""
        indices = self.config.blockEdgeIndices
        self.assertIn(0, indices)
        self.assertIn(self.config.numFibers - 1, indices)
        self.assertIn(self.config.blockSize - 1, indices)
        self.assertIn(self.config.blockSize, indices)

    def testEvaluatePsfUnitFlux(self):
        """The ground-truth PSF (parametric backbone + injected bump) integrates to 1"""
        xx = np.linspace(-40, 40, 1601)
        dx, dy = np.meshgrid(xx, xx, indexing="xy")
        area = (xx[1] - xx[0]) ** 2
        values = self.config.evaluatePsf(dx, dy)
        self.assertFloatsAlmostEqual(values.sum() * area, 1.0, atol=5.0e-3)


class MakeSyntheticPsfArcTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticPsfConfig()

    def testLineListPopulated(self):
        """Most fibers get every nominal line rendered (a few are dropped at
        the top/bottom of the detector because of the trace slope/curvature)"""
        numLines = 15
        rng = np.random.RandomState(12345)
        result = makeSyntheticPsfArc(self.config, numLines=numLines, addNoise=False, rng=rng)
        maxLines = numLines * self.config.numFibers
        self.assertGreater(len(result.row), 0.9 * maxLines)
        self.assertLessEqual(len(result.row), maxLines)
        for array in (result.fiberIndex, result.fiberId, result.row, result.xCenter, result.amplitude):
            self.assertEqual(len(array), len(result.row))

    def testFluxConservation(self):
        """Without noise, the total image flux matches the sum of line
        amplitudes, up to the flux truncated by the finite stamp half-size"""
        rng = np.random.RandomState(54321)
        result = makeSyntheticPsfArc(self.config, numLines=10, addNoise=False, rng=rng)
        self.assertFloatsAlmostEqual(result.image.array.sum(), result.amplitude.sum(), rtol=1.0e-2)

    def testReproducible(self):
        """The same seed gives the same result"""
        result1 = makeSyntheticPsfArc(self.config, numLines=10, addNoise=True, rng=np.random.RandomState(999))
        result2 = makeSyntheticPsfArc(self.config, numLines=10, addNoise=True, rng=np.random.RandomState(999))
        self.assertFloatsEqual(result1.image.array, result2.image.array)
        self.assertFloatsEqual(result1.row, result2.row)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
