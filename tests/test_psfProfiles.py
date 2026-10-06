import numpy as np

import lsst.utils.tests

from pfs.drp.stella.psfProfiles import (
    ChebyshevSurface2D,
    ParametricPsfModel,
    PsfParams,
    WingParams,
    evaluateParametricPsf,
    gaussian1D,
    moffat2D,
    tophatConvGaussian1D,
)
from pfs.drp.stella.tests import methodParametersProduct, runTests


class Gaussian1DTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.xx = np.linspace(-30, 30, 6001)
        self.step = self.xx[1] - self.xx[0]

    @methodParametersProduct(sigma=(0.5, 1.0, 3.21))
    def testUnitIntegral(self, sigma):
        """gaussian1D integrates to 1"""
        values = gaussian1D(self.xx, sigma)
        self.assertFloatsAlmostEqual(values.sum() * self.step, 1.0, atol=1.0e-9)


class TophatConvGaussian1DTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.xx = np.linspace(-30, 30, 6001)
        self.step = self.xx[1] - self.xx[0]

    @methodParametersProduct(width=(0.0, 1.0, 2.5, 5.0), sigma=(0.5, 1.3))
    def testUnitIntegral(self, width, sigma):
        """tophatConvGaussian1D integrates to 1, for any width"""
        values = tophatConvGaussian1D(self.xx, width, sigma)
        self.assertFloatsAlmostEqual(values.sum() * self.step, 1.0, atol=1.0e-9)

    def testReducesToGaussian(self):
        """Zero width reduces exactly to a pure Gaussian"""
        sigma = 1.23
        self.assertFloatsAlmostEqual(
            tophatConvGaussian1D(self.xx, 0.0, sigma), gaussian1D(self.xx, sigma), atol=1.0e-12
        )


class Moffat2DTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        xx = np.linspace(-40, 40, 1601)
        self.dx, self.dy = np.meshgrid(xx, xx, indexing="xy")
        self.area = (xx[1] - xx[0]) ** 2

    @methodParametersProduct(scaleX=(2.0, 3.0), scaleY=(2.0, 2.5), beta=(2.0, 2.5, 4.0))
    def testUnitIntegral(self, scaleX, scaleY, beta):
        """moffat2D integrates to 1 (to the truncation/discretization tolerance)"""
        values = moffat2D(self.dx, self.dy, scaleX, scaleY, beta)
        self.assertFloatsAlmostEqual(values.sum() * self.area, 1.0, atol=5.0e-3)

    def testInvalidBeta(self):
        """beta <= 1 has no finite integral, and should raise"""
        with self.assertRaises(ValueError):
            moffat2D(self.dx, self.dy, 1.0, 1.0, 1.0)


class EvaluateParametricPsfTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        xx = np.linspace(-40, 40, 1601)
        self.dx, self.dy = np.meshgrid(xx, xx, indexing="xy")
        self.area = (xx[1] - xx[0]) ** 2

    def testUnitFluxAndZeroCentroid(self):
        """The combined core+wings profile is unit-flux and centered"""
        params = PsfParams(
            sigmaX=0.9,
            sigmaY=0.8,
            tophatWidth=2.0,
            wings=[
                WingParams(scaleX=3.0, scaleY=2.5, beta=2.5, fraction=0.05),
                WingParams(scaleX=8.0, scaleY=6.0, beta=1.8, fraction=0.02),
            ],
        )
        values = evaluateParametricPsf(self.dx, self.dy, params)
        self.assertFloatsAlmostEqual(values.sum() * self.area, 1.0, atol=5.0e-3)
        self.assertFloatsAlmostEqual((values * self.dx).sum() * self.area, 0.0, atol=1.0e-9)
        self.assertFloatsAlmostEqual((values * self.dy).sum() * self.area, 0.0, atol=1.0e-9)


class ChebyshevSurface2DTestCase(lsst.utils.tests.TestCase):
    def testCoefficientRoundTrip(self):
        """Coefficients set via setCoefficients are returned by getCoefficients,
        and the surface evaluates as expected"""
        surface = ChebyshevSurface2D(2, (0, 300), (0, 4000))
        rng = np.random.RandomState(12345)
        coeffs = rng.uniform(-1, 1, surface.numCoefficients)
        surface.setCoefficients(coeffs)
        self.assertFloatsAlmostEqual(surface.getCoefficients(), coeffs, atol=1.0e-12)

        # A pure constant surface should evaluate to that constant everywhere
        constantSurface = ChebyshevSurface2D(2, (0, 300), (0, 4000), constant=3.21)
        self.assertFloatsAlmostEqual(constantSurface(150, 2000), 3.21, atol=1.0e-12)
        self.assertFloatsAlmostEqual(constantSurface(0, 0), 3.21, atol=1.0e-12)
        self.assertFloatsAlmostEqual(constantSurface(300, 4000), 3.21, atol=1.0e-12)


class ParametricPsfModelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.wings = [WingParams(scaleX=3.0, scaleY=2.5, beta=2.5, fraction=0.05)]
        self.model = ParametricPsfModel(1, len(self.wings), (0, 300), (0, 4000))
        self.model.setInitialGuess(0.9, 0.8, 2.0, self.wings)

    def testWrongNumberOfWings(self):
        """Setting the wrong number of wings should raise"""
        with self.assertRaises(ValueError):
            self.model.setInitialGuess(0.9, 0.8, 2.0, [])

    def testConstantModelEvaluatesEverywhere(self):
        """A model set with setInitialGuess is constant across the detector"""
        expected = PsfParams(sigmaX=0.9, sigmaY=0.8, tophatWidth=2.0, wings=self.wings)
        for fiberIndex, row in ((0, 0), (150, 2000), (299, 3999)):
            self.assertEqual(self.model.getParamsAt(fiberIndex, row), expected)

    def testParameterVectorRoundTrip(self):
        """Packing and unpacking the parameter vector recovers the same model"""
        expected = self.model.getParamsAt(150, 2000)
        vector = self.model.getParameterVector()

        other = ParametricPsfModel(1, len(self.wings), (0, 300), (0, 4000))
        other.setParameterVector(vector)
        self.assertEqual(other.getParamsAt(150, 2000), expected)

    def testParameterVectorWrongLength(self):
        """Setting a vector of the wrong length should raise"""
        vector = self.model.getParameterVector()
        with self.assertRaises(ValueError):
            self.model.setParameterVector(vector[:-1])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
