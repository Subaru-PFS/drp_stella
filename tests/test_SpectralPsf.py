import sys
import unittest
import pickle

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.detection
import lsst.afw.image.testUtils

import pfs.drp.stella
import pfs.drp.stella.synthetic
from pfs.drp.stella.images import getIndices, calculateCentroid, calculateSecondMoments
from pfs.drp.stella.tests.oversampledPsf import GaussianOversampledPsf
from pfs.drp.stella.tests.utils import classParameters, methodParameters

display = None


class ImagingPsfTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.sigma = 2.345
        self.size = 2*int(4*self.sigma) + 1
        self.imagePsf = lsst.afw.detection.GaussianPsf(self.size, self.size, self.sigma)

        self.synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        self.detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(self.synthConfig)

        self.psf = pfs.drp.stella.ImagingSpectralPsf(self.imagePsf, self.detMap)

    def assertImagePsf(self, psf):
        """Assert that the point-spread function object is as expected

        Parameters
        ----------
        psf : `pfs.drp.stella.ImagePsf`
            Point-spread function object.
        """
        self.assertEqual(psf.imagePsf.getSigma(), self.sigma)
        self.assertEqual(psf.imagePsf.getDimensions(), lsst.geom.Extent2I(self.size, self.size))
        self.assertEqual(psf.detectorMap, self.detMap)

    def testBasic(self):
        """Test basic functionality"""
        self.assertImagePsf(self.psf)
        self.assertImagePsf(self.psf.clone())

    def testComputeShape(self):
        """Test computeShape method"""
        for fiberId in self.detMap.fiberId:
            for fraction in (0.1, 0.5, 0.9):
                wavelength = self.detMap.findWavelength(fiberId, self.synthConfig.height*fraction)
                shape = self.psf.computeShape(fiberId, wavelength)
                self.assertEqual(shape.getTraceRadius(), self.sigma)

    def testComputeImage(self):
        """Test computeImage and computeKernelImage methods"""
        for fiberId in self.detMap.fiberId:
            for fraction in (0.1, 0.5, 0.9):
                yy = self.synthConfig.height*fraction
                if yy == int(yy):
                    # Ensure we have a non-integer pixel position,
                    # so computeImage and computeKernelImage differ
                    yy += 0.5
                wavelength = self.detMap.findWavelength(fiberId, yy)
                image = self.psf.computeImage(fiberId, wavelength)
                kernel = self.psf.computeKernelImage(fiberId, wavelength)

                # Image should have xy0 set somewhere in the middle of the larger image
                self.assertNotEqual(image.getX0(), 0)
                self.assertNotEqual(image.getY0(), 0)

                # Kernel should have xy0 set to the half-size
                halfSize = (self.size - 1)//2
                self.assertEqual(kernel.getX0(), -halfSize)
                self.assertEqual(kernel.getY0(), -halfSize)

                # Centroid on image should be at the point of interest
                xx, yy = self.detMap.findPoint(fiberId, wavelength)
                centroid = calculateCentroid(image)
                self.assertFloatsAlmostEqual(xx, centroid.x, atol=2.0e-2)
                self.assertFloatsAlmostEqual(yy, centroid.y, atol=2.0e-2)

                # Centroid on kernel should be zero
                centroid = calculateCentroid(kernel)
                self.assertFloatsAlmostEqual(centroid.x, 0.0, atol=1.0e-7)
                self.assertFloatsAlmostEqual(centroid.y, 0.0, atol=1.0e-7)

    def testPickle(self):
        """Test pickling"""
        copy = pickle.loads(pickle.dumps(self.psf))
        self.assertImagePsf(copy)


@classParameters(oversampleFactor=[5, 6, 7])
class OversampledPsfTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.sigma = 2.345
        self.size = 21
        self.psf = GaussianOversampledPsf(self.sigma, self.oversampleFactor,
                                          lsst.geom.Extent2I(self.size, self.size))
        self.position = lsst.geom.Point2D(123.45, 67.89)

    def tearDown(self):
        del self.psf
        del self.position

    def assertGaussianOversampledPsfEqual(self, lhs, rhs):
        """Assert that two GaussianOversampledPsf objects are equal"""
        self.assertEqual(lhs.getSigma(), rhs.getSigma())
        self.assertEqual(lhs.getOversampleFactor(), rhs.getOversampleFactor())
        self.assertEqual(lhs.getTargetSize(), rhs.getTargetSize())

    def testBasic(self):
        """Test basic behaviour"""
        self.assertEqual(self.psf.getSigma(), np.float32(self.sigma))
        self.assertEqual(self.psf.getOversampleFactor(), self.oversampleFactor)
        self.assertEqual(self.psf.getTargetSize(), lsst.geom.Extent2I(self.size, self.size))

    def testKernel(self):
        """Test computeKernelImage"""
        kernel = self.psf.computeKernelImage(self.position)
        self.assertFloatsAlmostEqual(kernel.array.sum(), 1.0, atol=1.0e-14)

        # Values of kernel
        xx = np.arange(self.size) - self.size//2
        gaussian = np.exp(-0.5*(xx/self.sigma)**2)
        expected = gaussian[np.newaxis, :]*gaussian[:, np.newaxis]
        expected /= expected.sum()
        self.assertFloatsAlmostEqual(kernel.array, expected, atol=5.0e-4)

        bbox = self.psf.computeBBox(self.position)
        self.assertEqual(bbox.getWidth(), self.size)
        self.assertEqual(bbox.getHeight(), self.size)

        # Centroids
        centroid = calculateCentroid(kernel)
        self.assertFloatsAlmostEqual(centroid.x, 0.0, atol=1.0e-6)
        self.assertFloatsAlmostEqual(centroid.y, 0.0, atol=1.0e-6)

        # 2nd moments
        moments = calculateSecondMoments(kernel, centroid)
        self.assertFloatsAlmostEqual(moments.xx, self.sigma**2, rtol=2.0e-2)
        self.assertFloatsAlmostEqual(moments.yy, self.sigma**2, rtol=2.0e-2)
        self.assertFloatsAlmostEqual(moments.xy, 0.0, atol=1.0e-16)

    def testShape(self):
        """Test computeShape"""
        shape = self.psf.computeShape(self.position)
        self.assertFloatsAlmostEqual(shape.getIxx(), self.sigma**2, rtol=2.0e-2)
        self.assertFloatsAlmostEqual(shape.getIyy(), self.sigma**2, rtol=2.0e-2)
        self.assertFloatsAlmostEqual(shape.getIxy(), 0.0, atol=1.0e-8)

    @methodParameters(radius=[3.0, 4.5, 5.0, 7.5, 9.0])
    def testFlux(self, radius):
        """Test computeApertureFlux"""
        flux = self.psf.computeApertureFlux(radius, self.position)
        expected = 1.0 - np.exp(-0.5*(radius/self.sigma)**2)
        self.assertFloatsAlmostEqual(flux, expected, rtol=1.0e-2)

    def testImage(self):
        """Test computeImage"""
        rng = np.random.RandomState(12345)
        num = 10
        for xStar, yStar in rng.uniform(size=(num, 2))*4000:
            image = self.psf.computeImage(lsst.geom.Point2D(xStar, yStar))

            # Centroid
            centroid = calculateCentroid(image)
            self.assertFloatsAlmostEqual(centroid.x, xStar, atol=5.0e-3)
            self.assertFloatsAlmostEqual(centroid.y, yStar, atol=5.0e-3)

            # 2nd moments
            moments = calculateSecondMoments(image, centroid)
            self.assertFloatsAlmostEqual(moments.xx, self.sigma**2, rtol=4.0e-2)
            self.assertFloatsAlmostEqual(moments.yy, self.sigma**2, rtol=4.0e-2)
            self.assertFloatsAlmostEqual(moments.xy, 0.0, atol=1.0e-15)

            # Pixel values
            xx, yy = getIndices(image.getBBox())
            expected = np.exp(-0.5*((xx - xStar)**2 + (yy - yStar)**2)/self.sigma**2)
            expected /= expected.sum()
            self.assertFloatsAlmostEqual(image.array, expected, atol=5.0e-4)

    def testPersistence(self):
        """Test persistence of the PSF"""
        exposure = lsst.afw.image.ExposureF(1, 1)
        exposure.setPsf(self.psf)
        self.assertIsNotNone(exposure.getPsf())
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exposure.writeFits(filename)
            copy = lsst.afw.image.ExposureF(filename).getPsf()
        self.assertIsNotNone(copy)
        self.assertIsInstance(copy, GaussianOversampledPsf)
        self.assertGaussianOversampledPsfEqual(copy, self.psf)


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
