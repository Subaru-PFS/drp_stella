import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils
from lsst.geom import Box2I, Point2I, Point2D, Extent2I
from lsst.afw.image import ImageD
from lsst.afw.detection import GaussianPsf
from lsst.afw.display import Display

from pfs.drp.stella import centroidImage
from pfs.drp.stella.images import getIndices
from pfs.drp.stella.tests import runTests, methodParameters

display = None


class CentroidImageTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.flux = 12.34567  # Flux of PSF in kernel
        self.sigma = 3.21  # Gaussian sigma of PSF in kernel (pixels)

    def makeImage(self, box, center):
        """Construct an image with a source to centroid

        Parameters
        ----------
        box : `lsst.geom.Box2I`
            Bounding box of image.
        center : `lsst.geom.Point2D`
            Center of source.
        """
        image = ImageD(box)
        xx, yy = getIndices(box)
        radius2 = (xx - center.getX())**2 + (yy - center.getY())**2
        norm = 1/(2*np.pi*self.sigma**2)
        image.array[:] = self.flux*norm*np.exp(-0.5*radius2/self.sigma**2)

        if display is not None:
            disp = Display(frame=1, backend=display)
            disp.mtv(image)

        return image

    def assertPoint(self, lhs, rhs, **kwargs):
        """Assert that a Point has the expected value

        Parameters
        ----------
        lhs, rhs : `lsst.geom.Point2D`
            Points to compare.
        **kwargs :
            Arguments for ``assertFloatsAlmostEqual``, e.g., ``atol``.
        """
        self.assertFloatsAlmostEqual(lhs.getX(), rhs.getX(), **kwargs)
        self.assertFloatsAlmostEqual(lhs.getY(), rhs.getY(), **kwargs)

    @methodParameters(
        xyMin=(Point2I(-24, -25), Point2I(1234, 5678)),
        dims=(Extent2I(47, 52), Extent2I(25, 47)),
        center=(Point2D(0, 0), Point2D(1245.67, 5699.01)),
        sigma=(2.31, 4.321),
    )
    def testCentroiding(self, xyMin, dims, center, sigma):
        """Test that centroidImage works

        We construct an image, and centroid the source in it.

        Parameters
        ----------
        xyMin : `lsst.geom.Point2I`
            Lower left-hand corner of image.
        dims : `lsst.geom.Extent2I`
            Dimensions of image.
        center : `lsst.geom.Point2D`
            Center of source.
        sigma : `float`
            Standard deviation of PSF to use in measuring centroid. The centroid
            should be fairly robust against changes to this. This is not the
            actual PSF sigma, just what is assumed in the centroid measurement.
        """
        box = Box2I(xyMin, dims)
        image = self.makeImage(box, center)
        psfSize = int(3*sigma) + 1
        psf = GaussianPsf(psfSize, psfSize, sigma)

        self.assertPoint(centroidImage(image, sigma), center, atol=1.0e-2)
        self.assertPoint(centroidImage(image, psf), center, atol=1.0e-2)
        self.assertPoint(centroidImage(image), center, atol=1.0e-2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
