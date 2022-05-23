import sys
import unittest

import numpy as np

import lsst.utils.tests

from pfs.drp.stella.symmetricTridiagonal import solveSymmetricTridiagonal, invertSymmetricTridiagonal

display = None


def getMatrix(diagonal, offDiag):
    """Convert symmetric tridiagonal formulation to a full matrix

    Parameters
    ----------
    diagonal : `numpy.ndarray` of shape ``(N)``
        Diagonal elements of matrix.
    offDiag : `numpy.ndarray` of shape ``(N-1)``
        Off-diagonal elements of matrix.
    """
    size = len(diagonal)
    assert len(offDiag) == size - 1

    matrix = np.zeros((size, size))
    for ii in range(size - 1):
        matrix[ii, ii] = diagonal[ii]
        matrix[ii, ii + 1] = offDiag[ii]
        matrix[ii + 1, ii] = offDiag[ii]
    matrix[size - 1, size - 1] = diagonal[size - 1]
    return matrix


class SymmetricTridiagonalTestCase(lsst.utils.tests.TestCase):
    """Tests for the symmetric tridiagonal matrix functions"""
    def checkInverse(self, size, rng):
        """Check that we can calculate the correct inverse values

        We generate a random matrix, and check our inverse against the numpy
        inverse.

        Parameters
        ----------
        size : `int`
            Size of the matrix.
        rng : `numpy.random.RandomState`
            Random number generator.
        """
        diagonal = rng.uniform(size=size)*10  # Make it diagonally dominant, like a least-squares matrix
        offDiag = rng.uniform(size=size - 1)
        inverseDiagonal, inverseOffDiag = invertSymmetricTridiagonal(diagonal, offDiag)

        inverse = np.linalg.inv(getMatrix(diagonal, offDiag))
        expectDiagonal = np.array([inverse[ii, ii] for ii in range(size)])
        expectOffDiag = np.array([inverse[ii, ii + 1] for ii in range(size - 1)])

        self.assertFloatsAlmostEqual(inverseDiagonal, expectDiagonal, atol=1.0e-13)
        self.assertFloatsAlmostEqual(inverseOffDiag, expectOffDiag, atol=1.0e-13)

    def testInverse(self):
        """Test that we can successfully invert matrices"""
        rng = np.random.RandomState(12345)
        for size in (10, 100, 1000):
            self.checkInverse(size, rng)

    def checkSolve(self, size, rng):
        """Check that we can calculate the correct solution values

        We generate a random matrix, and check our solution against the numpy
        solution.

        Parameters
        ----------
        size : `int`
            Size of the matrix.
        rng : `numpy.random.RandomState`
            Random number generator.
        """
        diagonal = rng.uniform(size=size)*10
        offDiag = rng.uniform(size=size - 1)
        answer = rng.uniform(size=size)

        solution = solveSymmetricTridiagonal(diagonal, offDiag, answer)
        expected = np.linalg.solve(getMatrix(diagonal, offDiag), answer)
        self.assertFloatsAlmostEqual(solution, expected, atol=1.0e-13)

    def testSolve(self):
        """Test that we can successfully solve matrix equations"""
        rng = np.random.RandomState(12345)
        for size in (10, 100, 1000):
            self.checkSolve(size, rng)

    def testSingular(self):
        """Test that we can solve singular matrices

        These values are taken from fiber indices 510 through 519 of
        visit=76003 at y=1025, where a cosmic ray has wiped out data and left
        only a single pixel between fiber indices 515 and 516, the full value of
        which is claimed by both.
        """
        diagonal = np.array(
            [
                2.04603214245271892e-01,
                1.77478682035331681e-01,
                1.86878709397182152e-01,
                2.30714251670864207e-01,
                1.93489959968508124e-01,
                2.72308226631381165e-04,
                2.74896376556077015e-04,
                2.33667285475434305e-01,
                2.31162677238954783e-01,
                2.31444225252564806e-01,
            ]
        )
        offDiag = np.array(
            [
                8.29542189990942528e-04,
                1.50040111291657426e-03,
                7.35440581909623209e-04,
                5.02985261367372362e-04,
                0.00000000000000000e00,
                2.73599241240500741e-04,
                0.00000000000000000e00,
                3.89897626215648507e-04,
                3.69925928467314269e-04,
            ]
        )
        answer = np.array(
            [
                2.28096719322151216e02,
                1.88301068255664319e02,
                1.72575595095778453e02,
                2.59294852978659037e02,
                1.90605045408182889e02,
                2.92346131950238675e-01,
                2.93732146364626023e-01,
                2.55098123020953864e02,
                2.74973877207476960e02,
                2.50663980370680378e02,
            ]
        )
        self.assertEqual(diagonal[5]*diagonal[6] - offDiag[5]**2, 0.0)  # Singular
        solution = solveSymmetricTridiagonal(diagonal, offDiag, answer)
        good = np.ones_like(solution, dtype=bool)
        good[6] = False
        self.assertTrue(np.all(np.isfinite(solution[good])))
        self.assertTrue(np.all(np.isnan(solution[~good])))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    from argparse import ArgumentParser

    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display backend")
    args, argv = parser.parse_known_args()
    display = args.display
    unittest.main(failfast=True, argv=[__file__] + argv)
