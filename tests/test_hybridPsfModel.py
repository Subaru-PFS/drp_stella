import numpy as np

import lsst.utils.tests

from pfs.drp.stella.hybridPsfModel import HybridPsfModel
from pfs.drp.stella.psfProfiles import ParametricPsfModel
from pfs.drp.stella.psfSpline import PsfSplineBasis, SplineAxisConfig
from pfs.drp.stella.tests import runTests


def makeParametricModel() -> ParametricPsfModel:
    model = ParametricPsfModel(0, 0, (0, 10), (0, 100))
    model.setInitialGuess(sigmaX=1.0, sigmaY=1.0, tophatWidth=1.0)
    return model


def makeSplineBasis() -> PsfSplineBasis:
    return PsfSplineBasis(SplineAxisConfig(extent=15.0), SplineAxisConfig(extent=12.0))


class HybridPsfModelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.parametric = makeParametricModel()
        self.splineBasis = makeSplineBasis()
        self.model = HybridPsfModel(self.parametric, self.splineBasis)
        dx = np.linspace(-5.0, 5.0, 11)
        dy = np.linspace(-5.0, 5.0, 11)
        self.xGrid, self.yGrid = np.meshgrid(dx, dy)

    def testZeroSplineCoefficientsMatchParametricOnly(self):
        """With no ``dP`` set, the hybrid model reduces to ``P_param``"""
        values = self.model.evaluate(self.xGrid, self.yGrid, 0, 0)
        expected = self.parametric.evaluate(self.xGrid, self.yGrid, 0, 0)
        self.assertFloatsAlmostEqual(values, expected, atol=1.0e-12)

    def testNonzeroSplineCoefficientsPerturbResult(self):
        """A non-zero (constraint-satisfying) ``dP`` changes the evaluated PSF"""
        rng = np.random.RandomState(4)
        nullSpace = self.splineBasis.nullSpaceBasis()
        zz = rng.normal(size=nullSpace.shape[1]) * 1.0e-3
        coeffs = nullSpace @ zz
        self.model.setSplineCoefficients(coeffs)
        values = self.model.evaluate(self.xGrid, self.yGrid, 0, 0)
        paramOnly = self.parametric.evaluate(self.xGrid, self.yGrid, 0, 0)
        self.assertTrue(np.any(np.abs(values - paramOnly) > 1.0e-8))

    def testSetSplineCoefficientsRejectsWrongLength(self):
        with self.assertRaises(ValueError):
            self.model.setSplineCoefficients(np.zeros(self.splineBasis.numCoefficients + 1))

    def testCheckNonNegativityAllPositiveForSmoothCore(self):
        """With no ``dP`` correction, a smooth unit-flux core+tophat stays non-negative"""
        result = self.model.checkNonNegativity(np.arange(-10, 11), np.arange(-10, 11), 0, 0, oversampling=2)
        self.assertGreaterEqual(result.minValue, 0.0)
        self.assertEqual(result.fractionNegative, 0.0)

    def testCheckNonNegativityDetectsNegativeDp(self):
        """A large negative ``dP`` bump can drive the total model negative,
        and `checkNonNegativity` should report it rather than hide it"""
        coeffs = np.zeros(self.splineBasis.numCoefficients)
        centerIndex = (self.splineBasis.numBasisY // 2) * self.splineBasis.numBasisX + (
            self.splineBasis.numBasisX // 2
        )
        coeffs[centerIndex] = -10.0
        self.model.setSplineCoefficients(coeffs)
        result = self.model.checkNonNegativity(np.arange(-10, 11), np.arange(-10, 11), 0, 0, oversampling=2)
        self.assertLess(result.minValue, 0.0)
        self.assertGreater(result.fractionNegative, 0.0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
