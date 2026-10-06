import numpy as np

import lsst.utils.tests

from pfs.drp.stella.psfPixelIntegration import makeOversampledGrid
from pfs.drp.stella.psfSpline import (
    BSplineAxis,
    PsfSplineBasis,
    RegularizationConfig,
    SplineAxisConfig,
    buildNonuniformKnots,
)
from pfs.drp.stella.tests import runTests


class BuildNonuniformKnotsTestCase(lsst.utils.tests.TestCase):
    def testSymmetricAndReachesExtent(self):
        config = SplineAxisConfig(extent=15.0)
        breakpoints = buildNonuniformKnots(config)
        self.assertFloatsAlmostEqual(breakpoints[0], -15.0, atol=1.0e-12)
        self.assertFloatsAlmostEqual(breakpoints[-1], 15.0, atol=1.0e-12)
        self.assertFloatsAlmostEqual(breakpoints, -breakpoints[::-1], atol=1.0e-12)
        self.assertTrue(np.all(np.diff(breakpoints) > 0))

    def testSpacingCoarsensWithRadius(self):
        config = SplineAxisConfig(extent=15.0)
        breakpoints = buildNonuniformKnots(config)
        positive = breakpoints[breakpoints >= 0]
        spacing = np.diff(positive)
        # Spacing at the end of the array (large radius) should be
        # coarser than spacing at the start (small radius).
        self.assertGreater(spacing[-1], spacing[0])

    def testRejectsBadOrdering(self):
        with self.assertRaises(ValueError):
            buildNonuniformKnots(SplineAxisConfig(extent=15.0, fineRadius=5.0, mediumRadius=4.0))
        with self.assertRaises(ValueError):
            buildNonuniformKnots(SplineAxisConfig(extent=3.0, mediumRadius=4.0))


class BSplineAxisTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.axis = BSplineAxis(SplineAxisConfig(extent=15.0))

    def testPartitionOfUnity(self):
        """Basis functions sum to 1 everywhere within the domain"""
        xx = np.linspace(-14.99, 14.99, 500)
        indices, data, inside = self.axis.designMatrix(xx)
        total = np.zeros(len(xx))
        np.add.at(total, np.repeat(np.arange(len(xx)), indices.shape[1]), data.ravel())
        self.assertTrue(np.all(inside))
        self.assertFloatsAlmostEqual(total, 1.0, atol=1.0e-10)

    def testZeroOutsideExtent(self):
        indices, data, inside = self.axis.designMatrix(np.array([-20.0, 0.0, 20.0]))
        np.testing.assert_array_equal(inside, np.array([False, True, False]))

    def testIntegralAgainstDirectQuadrature(self):
        """Analytic per-basis integral matches direct numerical integration"""
        from scipy.interpolate import BSpline

        mid = self.axis.numBasis // 2
        coeffs = np.zeros(self.axis.numBasis)
        coeffs[mid] = 1.0
        spline = BSpline(self.axis.knots, coeffs, self.axis.degree, extrapolate=False)
        xx = np.linspace(self.axis.breakpoints[0], self.axis.breakpoints[-1], 200001)
        yy = np.nan_to_num(spline(xx))
        numericalIntegral = np.trapezoid(yy, xx)
        numericalMoment = np.trapezoid(yy * xx, xx)
        self.assertFloatsAlmostEqual(self.axis.integral()[mid], numericalIntegral, atol=1.0e-6)
        self.assertFloatsAlmostEqual(self.axis.firstMoment()[mid], numericalMoment, atol=1.0e-6)


class PsfSplineBasisTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.basis = PsfSplineBasis(SplineAxisConfig(extent=15.0), SplineAxisConfig(extent=12.0))

    def testConstraintsSatisfiedByNullSpace(self):
        """Any coefficient vector built from the null-space basis satisfies
        the hard integral/moment constraints to numerical precision"""
        constraints = self.basis.constraintMatrix()
        self.assertEqual(constraints.shape, (3, self.basis.numCoefficients))
        nullSpace = self.basis.nullSpaceBasis()
        self.assertEqual(nullSpace.shape[0], self.basis.numCoefficients)
        self.assertFloatsAlmostEqual(constraints @ nullSpace, 0.0, atol=1.0e-9)

        rng = np.random.RandomState(1)
        zz = rng.normal(size=nullSpace.shape[1])
        coeffs = nullSpace @ zz
        self.assertFloatsAlmostEqual(constraints @ coeffs, 0.0, atol=1.0e-9)

    def testEvaluateGridMatchesDirectProduct(self):
        """`evaluateGrid` at a single point matches the direct outer product
        of the two 1D basis evaluations"""
        rng = np.random.RandomState(2)
        coeffs = rng.normal(size=self.basis.numCoefficients)
        point = (2.3, -1.7)
        indicesX, dataX, _ = self.basis.xAxis.designMatrix(np.array([point[0]]))
        indicesY, dataY, _ = self.basis.yAxis.designMatrix(np.array([point[1]]))
        xRow = np.zeros(self.basis.numBasisX)
        yRow = np.zeros(self.basis.numBasisY)
        xRow[indicesX[0]] = dataX[0]
        yRow[indicesY[0]] = dataY[0]
        direct = np.outer(yRow, xRow).ravel() @ coeffs

        design = self.basis.evaluateGrid(np.array([[point[0]]]), np.array([[point[1]]]))
        viaDesign = (design @ coeffs)[0]
        self.assertFloatsAlmostEqual(direct, viaDesign, atol=1.0e-10)

    def testEvaluateGridZeroOutsideExtent(self):
        xGrid, yGrid = makeOversampledGrid(np.array([100.0]), np.array([0.0]), oversampling=1)
        design = self.basis.evaluateGrid(xGrid, yGrid)
        self.assertEqual(design.nnz, 0)

    def testRegularizationMatrixIsSymmetricPositiveSemiDefinite(self):
        regularization = self.basis.regularizationMatrix(RegularizationConfig())
        dense = regularization.toarray()
        self.assertFloatsAlmostEqual(dense, dense.T, atol=1.0e-10)
        nullSpace = self.basis.nullSpaceBasis()
        projected = nullSpace.T @ dense @ nullSpace
        eigenvalues = np.linalg.eigvalsh(projected)
        self.assertGreater(eigenvalues.min(), -1.0e-8)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
