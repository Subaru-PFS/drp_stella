import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils
from lsst.geom import Box2I, Point2I, Point2D
from lsst.afw.image import ImageD
from lsst.afw.display import Display

from pfs.drp.stella import resampleKernelImage, recenterOversampledKernelImage
from pfs.drp.stella.images import getIndices, calculateCentroid, calculateSecondMoments
from pfs.drp.stella.tests import runTests, methodParametersProduct

display = None


class PsfFunctionsTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.x = 1.2345  # x position of PSF in kernel; deliberately off-center
        self.y = -5.4321  # y position of PSF in kernel; deliberately off-center
        self.flux = 12.34567  # Flux of PSF in kernel
        self.sigma = 3.21  # Gaussian sigma of PSF in kernel (in native pixels)
        self.halfSize = 13  # Size of kernel

    def makeOversampledKernelImage(self, oversample):
        """Construct an oversampled kernel

        We insert a Gaussian PSF so we have something to center on.
        """
        halfSize = oversample*self.halfSize
        box = Box2I(Point2I(-halfSize, -halfSize), Point2I(halfSize, halfSize))
        image = ImageD(box)
        xx, yy = getIndices(box)
        sigma = self.sigma*oversample
        radius2 = (xx - self.x)**2 + (yy - self.y)**2
        norm = 1/(2*np.pi*sigma**2)
        image.array[:] = self.flux*norm*np.exp(-0.5*radius2/sigma**2)

        if display is not None:
            disp = Display(frame=1, backend=display)
            disp.mtv(image)

        centroid = calculateCentroid(image)
        self.assertFloatsAlmostEqual(centroid.x, self.x, atol=1.0e-2)
        self.assertFloatsAlmostEqual(centroid.y, self.y, atol=1.0e-2)
        moments = calculateSecondMoments(image, centroid)
        self.assertFloatsAlmostEqual(moments.xx, sigma**2, rtol=1.0e-2)
        self.assertFloatsAlmostEqual(moments.yy, sigma**2, rtol=1.0e-2)
        self.assertFloatsAlmostEqual(moments.xy, 0.0, atol=1.0e-12)

        return image

    @methodParametersProduct(
        oversample=(3, 4, 5, 10, 15),
        point=((0, 0),  # A kernel: like for computeKernelImage
               (987.654, 321.098)  # An image: like for computeImage
               ),
    )
    def testFunctions(self, oversample, point):
        """Test that recenterOversampledKernelImage and resampleKernelImage work

        We construct an image, recenter it and then resample it, checking each
        time that the image looks like what we expect.

        Parameters
        ----------
        oversample : `int`
            Binning factor.
        point : `tuple` of `float`, size 2
            Target point.
        """
        image = self.makeOversampledKernelImage(oversample)
        point = Point2D(point)

        # Recentering
        recentered = recenterOversampledKernelImage(image, oversample, point)
        if display is not None:
            disp = Display(frame=2, backend=display)
            disp.mtv(recentered)
        centroid = calculateCentroid(recentered)
        xExpect = oversample*point.getX() + (0.0 if oversample % 2 else 0.5)
        yExpect = oversample*point.getY() + (0.0 if oversample % 2 else 0.5)
        self.assertFloatsAlmostEqual(centroid.x, xExpect, atol=2.0e-2)
        self.assertFloatsAlmostEqual(centroid.y, yExpect, atol=2.0e-2)
        moments = calculateSecondMoments(recentered, centroid)
        sigma = self.sigma*oversample
        self.assertFloatsAlmostEqual(moments.xx, sigma**2, rtol=1.0e-2)
        self.assertFloatsAlmostEqual(moments.yy, sigma**2, rtol=1.0e-2)
        self.assertFloatsAlmostEqual(moments.xy, 0.0, atol=1.0e-12)
        flux = recentered.array.sum()
        self.assertFloatsAlmostEqual(flux, self.flux, rtol=5.0e-4)

        # Resampling
        bbox = Box2I(Point2I(-self.halfSize, -self.halfSize), Point2I(self.halfSize, self.halfSize))
        resampled = resampleKernelImage(recentered, oversample, bbox, point)
        if display is not None:
            disp = Display(frame=3, backend=display)
            disp.mtv(resampled)
        centroid = calculateCentroid(resampled)
        self.assertFloatsAlmostEqual(centroid.x, point.getX(), atol=2.0e-2)
        self.assertFloatsAlmostEqual(centroid.y, point.getY(), atol=2.0e-2)
        moments = calculateSecondMoments(resampled, centroid)
        self.assertFloatsAlmostEqual(moments.xx, self.sigma**2, rtol=1.0e-2)
        self.assertFloatsAlmostEqual(moments.yy, self.sigma**2, rtol=1.0e-2)
        self.assertFloatsAlmostEqual(moments.xy, 0.0, atol=1.0e-12)
        flux = resampled.array.sum()
        self.assertFloatsAlmostEqual(flux, 1.0, atol=1.0e-12)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
