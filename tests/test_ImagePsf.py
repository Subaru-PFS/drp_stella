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

display = None


def calculateCentroid(image):
    """Calculate centroid for an image

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image on which to calculate centroid.

    Returns
    -------
    xCen, yCen : `float`
        Centroid coordinates.
    """
    bbox = image.getBBox()
    xx = np.arange(bbox.getMinX(), bbox.getMaxX() + 1, dtype=float)
    yy = np.arange(bbox.getMinY(), bbox.getMaxY() + 1, dtype=float)
    norm = np.sum(image.array.astype(float))
    xCen = np.sum(np.sum(image.array.astype(float), axis=0)*xx)/norm
    yCen = np.sum(np.sum(image.array.astype(float), axis=1)*yy)/norm
    return xCen, yCen


class ImagePsfTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.sigma = 2.345
        self.size = 2*int(4*self.sigma) + 1
        self.imagePsf = lsst.afw.detection.GaussianPsf(self.size, self.size, self.sigma)

        self.synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        self.detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(self.synthConfig)

        self.psf = pfs.drp.stella.ImagePsf(self.imagePsf, self.detMap)

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
        for fiberId in self.detMap.fiberIds:
            for fraction in (0.1, 0.5, 0.9):
                wavelength = self.detMap.findWavelength(fiberId, self.synthConfig.height*fraction)
                shape = self.psf.computeShape(fiberId, wavelength)
                self.assertEqual(shape.getTraceRadius(), self.sigma)

    def testComputeImage(self):
        """Test computeImage and computeKernelImage methods"""
        for fiberId in self.detMap.fiberIds:
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
                xCen, yCen = calculateCentroid(image)
                self.assertFloatsAlmostEqual(xx, xCen, atol=2.0e-2)
                self.assertFloatsAlmostEqual(yy, yCen, atol=2.0e-2)

                # Centroid on kernel should be zero
                xCen, yCen = calculateCentroid(kernel)
                self.assertFloatsAlmostEqual(xCen, 0.0, atol=1.0e-7)
                self.assertFloatsAlmostEqual(yCen, 0.0, atol=1.0e-7)

    def testPickle(self):
        """Test pickling"""
        copy = pickle.loads(pickle.dumps(self.psf))
        self.assertImagePsf(copy)


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
