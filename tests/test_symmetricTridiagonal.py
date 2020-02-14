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
