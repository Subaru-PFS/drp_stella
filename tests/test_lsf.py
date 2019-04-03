import sys
import unittest
import pickle

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.geom
import lsst.afw.image
import lsst.afw.image.testUtils
import lsst.afw.display

from pfs.drp.stella import Kernel, GaussianKernel, GaussianLsf, FixedEmpiricalLsf

display = None


def calculateMoments(array):
    """Calculate first and second moments of an array

    Parameters
    ----------
    array : array-like
        Array for which to calculate moments.

    Returns
    -------
    centroid : `float`
        First moment.
    stdev : `float`
        Square root of the second moment.
    """
    indices = np.arange(len(array))
    centroid = np.sum((indices*array).astype(np.float64))
    stdev = np.sqrt(np.sum((array*(indices - centroid)**2).astype(np.float64)))
    return centroid, stdev


class KernelTestCase(lsst.utils.tests.TestCase):
    """Tests for pfs.drp.stella.Kernel"""
    def setUp(self):
        self.min = -3
        self.max = 5
        self.center = -self.min
        self.values = np.arange(self.min, self.max + 1, dtype=float)
        self.kernel = Kernel(self.values.copy(), self.center, normalize=False)

    def testGetItem(self):
        """Test Kernel.__getitem__

        It should work with both scalar and array indexing.
        """
        num = 0
        for (index, value), expect in zip(self.kernel, self.values):
            self.assertEqual(index, int(expect))
            self.assertEqual(self.kernel[index], expect)
            num += 1
        self.assertEqual(num, len(self.values))
        self.assertFloatsEqual(self.kernel[self.kernel.indices], self.values)
        self.assertFloatsEqual(self.kernel.values, self.values)

    def testSetItem(self):
        """Test Kernel.__setitem__

        It should work with both scalar and array indexing.
        """
        value = 1.2345
        for index in self.kernel.indices:
            self.kernel[index] = value
        self.assertFloatsEqual(self.kernel.values, value)

        value = 5.4321
        self.kernel[self.kernel.indices] = value
        self.assertFloatsEqual(self.kernel.values, value)

        value = 9.8765
        self.kernel[self.kernel.indices] = value*np.ones_like(self.kernel.values)
        self.assertFloatsEqual(self.kernel.values, value)

    def testMath(self):
        """Test in-place multiplication and division"""
        value = 1.2345
        self.kernel *= value
        self.assertFloatsEqual(self.kernel.values, self.values*value)

        self.kernel /= value
        self.assertFloatsEqual(self.kernel.values, self.values)

    def testNormalization(self):
        """Test Kernel.normalization"""
        self.assertEqual(self.kernel.normalization(), np.sum(self.values))

    def testToArray(self):
        """Test Kernel.toArray"""
        length = 123

        # No sub-pixel shift: should get the kernel exactly
        center = 54.0
        expect = np.zeros(length, dtype=float)
        expect[int(center) + self.min:int(center) + self.max + 1] = self.values
        array = self.kernel.toArray(length, center)
        self.assertFloatsEqual(array, expect)

        # Sub-pixel shift on a kernel that is a delta function.
        # Should get the single pixel from the kernel split between two pixels
        # with linear interpolation.
        kernelCenter = 2
        values = np.zeros_like(self.values)
        values[kernelCenter] = 1.0
        kernel = Kernel(values, kernelCenter)
        center = 54.321
        array = kernel.toArray(length, center)
        expect = np.zeros(length, dtype=float)
        fracCenter = center - int(center)
        expect[int(center)] = 1.0 - fracCenter
        expect[int(center) + 1] = fracCenter
        self.assertFloatsEqual(array, expect)

    def testConvolve(self):
        """Test Kernel.convolve"""
        halfSize = 3
        value = 1.0
        kernel = Kernel.makeEmpty(halfSize)
        kernel[0] = value

        # Convolve with a delta-function kernel
        length = 123
        center = 42
        array = np.zeros(length, dtype=float)
        array[center] = value
        convolved = kernel.convolve(array)
        self.assertFloatsEqual(convolved, array)

        # Convolve with a 2*delta-function kernel
        multiplier = 2.0
        kernel *= multiplier
        expect = array*multiplier
        convolved = kernel.convolve(array)
        self.assertFloatsEqual(convolved, expect)

        # Convolve with an offset-delta-function kernel
        offset = 1
        kernel[0] = 0.0
        kernel[offset] = value
        expect = np.zeros_like(array)
        expect[center + offset] = value
        convolved = kernel.convolve(array)
        self.assertFloatsEqual(convolved, expect)

    def testPickle(self):
        """Test pickling"""
        copy = pickle.loads(pickle.dumps(self.kernel))
        self.assertEqual(self.kernel, copy)


class GaussianLsfTestCase(lsst.utils.tests.TestCase):
    """Tests of pfs.drp.stella.GaussianLsf"""
    def setUp(self):
        self.length = 123
        self.sigma = 2.345
        self.lsf = GaussianLsf(self.length, self.sigma)

    def testComputeArray(self):
        """Test GaussianLsf.computeArray

        Should be the same no matter what position we give it.
        """
        for center in (self.length/3, self.length/2, self.length*3/4):
            array = self.lsf.computeArray(center)
            centroid, stdev = calculateMoments(array)
            self.assertFloatsAlmostEqual(centroid, center, atol=1.0e-2)
            self.assertFloatsAlmostEqual(stdev, self.sigma, atol=5.0e-2)

    def testComputeKernel(self):
        """Test GaussianLsf.computeKernel

        Should be the same no matter what position we give it.
        """
        for center in (self.length/3, self.length/2, self.length*3/4):
            kernel = self.lsf.computeKernel(center)
            centroid, stdev = calculateMoments(kernel.values)
            halfSize = (len(kernel) - 1)//2
            self.assertFloatsAlmostEqual(centroid, halfSize)
            self.assertFloatsAlmostEqual(stdev, self.sigma, atol=1.0e-3)

    def testComputeShape(self, precise=True):
        """Test GaussianLsf.computeShape

        Should be the same no matter what position we give it.

        Allow imprecise shapes. For GaussianLsf, we know the shape exactly,
        but for other types of Lsf it may have to be calculated, so the
        answer would not be exact.
        """
        for center in (self.length/3, self.length/2, self.length*3/4):
            stdev = self.lsf.computeStdev(center)
            if precise:
                self.assertFloatsEqual(stdev, self.sigma)
            else:
                self.assertFloatsAlmostEqual(stdev, self.sigma, atol=1.0e-3)


class FixedEmpiricalLsfTestCase(GaussianLsfTestCase):
    """Tests for pfs.drp.stella.FixedEmpiricalLsf

    We grab the values from a GaussianLsf's kernel and use that as our kernel;
    then we can just recycle the tests for GaussianLsf.
    """
    def setUp(self):
        super().setUp()
        kernel = Kernel(self.lsf.kernel.values, self.lsf.kernel.center)
        self.lsf = FixedEmpiricalLsf(kernel, self.length)

    def testComputeShape(self):
        """Test FixedEmpiricalLsf.computeShape

        FixedEmpiricalLsf has to calculate the shape rather than knowing the
        exact value.
        """
        super().testComputeShape(precise=False)


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
