import sys
import unittest
import pickle

import numpy as np
import scipy.integrate

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.geom
import lsst.afw.image
import lsst.afw.image.testUtils
import lsst.afw.display

from pfs.drp.stella import Kernel1D, GaussianLsf, FixedEmpiricalLsf

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
    rms : `float`
        Square root of the second moment.
    """
    indices = np.arange(len(array))
    centroid = np.sum((indices*array).astype(np.float64))
    rms = np.sqrt(np.sum((array*(indices - centroid)**2).astype(np.float64)))
    return centroid, rms


class KernelTestCase(lsst.utils.tests.TestCase):
    """Tests for pfs.drp.stella.Kernel1D"""
    def setUp(self):
        self.min = -3
        self.max = 5
        self.center = -self.min
        self.values = np.arange(self.min, self.max + 1, dtype=float)
        self.kernel = Kernel1D(self.values.copy(), self.center, normalize=False)

    def testGetItem(self):
        """Test Kernel1D.__getitem__

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
        """Test Kernel1D.__setitem__

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
        """Test Kernel1D.normalization"""
        self.assertEqual(self.kernel.normalization(), np.sum(self.values))

    def testToArray(self):
        """Test Kernel1D.toArray"""
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
        kernel = Kernel1D(values, kernelCenter)
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
        kernel = Kernel1D.makeEmpty(halfSize)
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
        self.width = 2.345
        self.lsf = GaussianLsf(self.length, self.width)

    def testComputeArray(self):
        """Test GaussianLsf.computeArray

        Should be the same no matter what position we give it.
        """
        for center in (self.length/3, self.length/2, self.length*3/4):
            array = self.lsf.computeArray(center)
            centroid, rms = calculateMoments(array)
            self.assertFloatsAlmostEqual(centroid, center, atol=1.0e-2)
            self.assertFloatsAlmostEqual(rms, self.width, atol=5.0e-2)

            point = lsst.geom.Point2D(center, 12345)
            image = self.lsf.computeImage(point)
            self.assertEqual(image.getWidth(), self.length)
            self.assertEqual(image.getHeight(), 1)
            self.assertFloatsEqual(image.array[0], array.astype(np.float32))

    def testComputeKernel(self):
        """Test GaussianLsf.computeKernel

        Should be the same no matter what position we give it.
        """
        for center in (self.length/3, self.length/2, self.length*3/4):
            kernel = self.lsf.computeKernel(center)
            centroid, rms = calculateMoments(kernel.values)
            halfSize = (len(kernel) - 1)//2
            self.assertFloatsAlmostEqual(centroid, halfSize)
            self.assertFloatsAlmostEqual(rms, self.width, atol=1.0e-3)

            point = lsst.geom.Point2D(center, 12345)
            image = self.lsf.computeKernelImage(point)
            self.assertEqual(image.getWidth(), len(kernel))
            self.assertEqual(image.getHeight(), 1)
            self.assertFloatsEqual(image.array[0], kernel.values.astype(np.float32))

            peak = self.lsf.computePeak(point)
            self.assertEqual(peak, kernel[0])

            local = self.lsf.getLocalKernel(point)
            self.assertEqual(local.getWidth(), len(kernel))
            self.assertEqual(local.getHeight(), 1)
            image = lsst.afw.image.ImageD(local.getDimensions())
            local.computeImage(image, False)
            self.assertFloatsEqual(image.array[0], kernel.values)

            atol = 1.0e-2  # Tolerance for flux comparison (trapz in implementation vs quad here)
            for radius in 3, 4.567, 9:
                flux = self.lsf.computeApertureFlux(radius, point)

                def integrand(xx):
                    """Function to integrate"""
                    return np.interp(xx, kernel.indices, kernel.values, left=0, right=0)

                expected = scipy.integrate.quad(integrand, -radius, radius,
                                                epsabs=atol, epsrel=0.0)[0]
                self.assertFloatsAlmostEqual(flux, expected, atol=atol)

    def testComputeShape(self, atol=0.0):
        """Test GaussianLsf.computeShape

        Should be the same no matter what position we give it.
        """
        for center in (self.length/3, self.length/2, self.length*3/4):
            shape = self.lsf.computeShape1D(center)
            self.assertFloatsAlmostEqual(shape, self.width, atol=atol)

            point = lsst.geom.Point2D(center, 12345)
            shape = self.lsf.computeShape(point)
            self.assertFloatsAlmostEqual(shape.getIxx(), self.width**2, atol=atol)
            self.assertFloatsEqual(shape.getIxy(), 0.0)
            self.assertFloatsEqual(shape.getIyy(), 0.0)


class FixedEmpiricalLsfTestCase(GaussianLsfTestCase):
    """Tests for pfs.drp.stella.FixedEmpiricalLsf

    We grab the values from a GaussianLsf's kernel and use that as our kernel;
    then we can just recycle the tests for GaussianLsf.
    """
    def setUp(self):
        super().setUp()
        kernel = Kernel1D(self.lsf.kernel.values, self.lsf.kernel.center)
        self.lsf = FixedEmpiricalLsf(kernel, self.length)

    def testComputeShape(self):
        """Test FixedEmpiricalLsf.computeShape

        FixedEmpiricalLsf has to calculate the shape rather than knowing the
        exact value.
        """
        super().testComputeShape(atol=5.0e-3)


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
