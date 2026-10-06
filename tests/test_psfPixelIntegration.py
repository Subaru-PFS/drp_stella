import numpy as np

import lsst.utils.tests

from pfs.drp.stella.psfPixelIntegration import (
    analyticGaussianPixelIntegral,
    binningMatrix1D,
    binningMatrix2D,
    integrateOverPixels,
    makeOversampledGrid,
)
from pfs.drp.stella.psfProfiles import gaussian1D
from pfs.drp.stella.tests import methodParametersProduct, runTests


class BinningMatrixTestCase(lsst.utils.tests.TestCase):
    @methodParametersProduct(numPixels=(1, 5, 8), oversampling=(1, 3, 4))
    def testBinningMatrix1DOfOnesIsOnes(self, numPixels, oversampling):
        """Binning a field of all ones gives all ones (it's a weighted mean)"""
        operator = binningMatrix1D(numPixels, oversampling)
        self.assertEqual(operator.shape, (numPixels, numPixels * oversampling))
        ones = np.ones(numPixels * oversampling)
        self.assertFloatsAlmostEqual(operator @ ones, np.ones(numPixels), atol=1.0e-12)

    def testBinningMatrix1DInvalidOversampling(self):
        """oversampling < 1 should raise"""
        with self.assertRaises(ValueError):
            binningMatrix1D(5, 0)

    @methodParametersProduct(numX=(3, 7), numY=(2, 5), oversampling=(1, 4))
    def testBinningMatrix2DOfOnesIsOnes(self, numX, numY, oversampling):
        """Binning a 2D field of all ones gives all ones"""
        operator = binningMatrix2D(numX, numY, oversampling)
        self.assertEqual(operator.shape, (numX * numY, numX * numY * oversampling**2))
        ones = np.ones(numX * numY * oversampling**2)
        self.assertFloatsAlmostEqual(operator @ ones, np.ones(numX * numY), atol=1.0e-12)


class IntegrateOverPixelsTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.xIndices = np.arange(-10, 11)
        self.yIndices = np.arange(-10, 11)

    def testOversampledGridBounds(self):
        """The oversampled grid spans each pixel's [-0.5, 0.5] extent"""
        xGrid, yGrid = makeOversampledGrid(np.array([0.0]), np.array([0.0]), 2)
        self.assertFloatsAlmostEqual(np.sort(np.unique(xGrid)), np.array([-0.25, 0.25]), atol=1.0e-12)
        self.assertFloatsAlmostEqual(np.sort(np.unique(yGrid)), np.array([-0.25, 0.25]), atol=1.0e-12)

    @methodParametersProduct(sigmaX=(0.8, 1.5), sigmaY=(0.7, 1.2))
    def testMatchesAnalyticGaussian(self, sigmaX, sigmaY):
        """Oversample-and-bin integration matches the analytic erf-based
        pixel integral, for a purely Gaussian profile"""

        def evaluate(dx, dy):
            return gaussian1D(dx, sigmaX) * gaussian1D(dy, sigmaY)

        numeric = integrateOverPixels(evaluate, self.xIndices, self.yIndices, oversampling=8)
        analytic = analyticGaussianPixelIntegral(self.xIndices, self.yIndices, sigmaX, sigmaY)
        self.assertFloatsAlmostEqual(numeric, analytic, atol=5.0e-4)
        self.assertFloatsAlmostEqual(numeric.sum(), 1.0, atol=1.0e-6)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
