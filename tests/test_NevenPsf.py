import os
import pickle
from functools import lru_cache

import numpy as np

from lsst.utils import getPackageDir
import lsst.utils.tests
import lsst.geom
import lsst.afw.detection
import lsst.afw.image.testUtils
from lsst.afw.display import Display

import pfs.drp.stella
from pfs.drp.stella.images import calculateCentroid
from pfs.drp.stella.tests.utils import methodParameters, runTests
import pfs.drp.stella.tests.nevenPsf

display = None


@lru_cache(None)
def getDetectorMap(filename):
    """Read the detectorMap

    The detectorMap gets cached, since reading some of the old detectorMaps can
    take a long time.

    Parameters
    ----------
    filename : `str`
        Filename for the detectorMap.

    Returns
    -------
    detectorMap : `pfs.drp.stella.DetectorMap`
        DetectorMap read from file.
    """
    detMap = pfs.drp.stella.DetectorMap.readFits(filename)
    detMap.metadata.markPersistent()
    return detMap


class NevenPsfTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        drpPfsData = getPackageDir("drp_pfs_data")
        detMapFilename = os.path.join(drpPfsData, "detectorMap", "detectorMap-2019Apr-r1.fits")
        self.detMap = getDetectorMap(detMapFilename)
        self.psf = pfs.drp.stella.NevenPsf.build(self.detMap, version="Apr1520_v3")
        self.fiberId = 339  # Needs to be a fiber for which we have a PSF

    def testBasic(self):
        """Test basic operation

        Not testing the actual values, since those are implementation-dependent,
        but testing the existence and functionality of the properties and
        getters.
        """
        self.assertFloatsEqual(self.psf.x, self.psf.getX())
        self.assertFloatsEqual(self.psf.y, self.psf.getY())
        self.assertEqual(len(self.psf.images), len(self.psf.getImages()))
        for image1, image2 in zip(self.psf.images, self.psf.getImages()):
            self.assertFloatsEqual(image1, image2)
        self.assertEqual(self.psf.oversampleFactor, self.psf.getOversampleFactor())
        self.assertEqual(self.psf.targetSize, self.psf.getTargetSize())
        self.assertEqual(self.psf.xMaxDistance, self.psf.getXMaxDistance())

    @methodParameters(wavelength=np.random.RandomState(12345).uniform(700, 900, size=10))
    def testPsf(self, wavelength):
        """Test PSF operations"""
        xPosition, yPosition = self.detMap.findPoint(self.fiberId, wavelength)
        kernel = self.psf.computeKernelImage(self.fiberId, wavelength)
        image = self.psf.computeImage(self.fiberId, wavelength)
        self.psf.computeShape(self.fiberId, wavelength)  # Not testing results, just that it works
        self.psf.computeApertureFlux(4.0, self.fiberId, wavelength)  # Not testing results

        kernelCentroid = calculateCentroid(kernel)
        self.assertFloatsAlmostEqual(kernelCentroid.x, 0, atol=1.0e-2)
        self.assertFloatsAlmostEqual(kernelCentroid.y, 0, atol=1.0e-2)

        imageCentroid = calculateCentroid(image)
        self.assertFloatsAlmostEqual(imageCentroid.x, xPosition + kernelCentroid.x, atol=1.0e-2)
        self.assertFloatsAlmostEqual(imageCentroid.y, yPosition + kernelCentroid.y, atol=1.0e-2)

    def assertNevenPsf(self, psf):
        """Test that the provided PSF is what is expected"""
        self.assertIsNotNone(psf)
        self.assertIsInstance(psf, pfs.drp.stella.NevenPsf)
        self.assertFloatsEqual(psf.x, self.psf.x)
        self.assertFloatsEqual(psf.y, self.psf.y)
        self.assertEqual(len(psf.images), len(self.psf.images))
        for image1, image2 in zip(psf.images, self.psf.images):
            self.assertFloatsEqual(image1, image2)
        self.assertEqual(psf.oversampleFactor, self.psf.oversampleFactor)
        self.assertEqual(psf.targetSize, self.psf.targetSize)
        self.assertEqual(psf.xMaxDistance, self.psf.xMaxDistance)

    def testPersistence(self):
        """Test persistence of the PSF"""
        exposure = lsst.afw.image.ExposureF(1, 1)
        exposure.setPsf(self.psf)
        self.assertIsNotNone(exposure.getPsf())
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exposure.writeFits(filename)
            copy = lsst.afw.image.ExposureF(filename).getPsf()
            self.assertNevenPsf(copy)

    def testPickle(self):
        """Test pickling of NevenPsf"""
        psf = pickle.loads(pickle.dumps(self.psf))
        self.assertNevenPsf(psf)

    def testInterpolationAlgorithm(self):
        """Test that our interpolation matches what Neven expects

        We check our results from C++ directly against what Neven produces with
        his python script. We use the `pfs.drp.stella.nevenPsf.NevenPsf`, which
        is a subclass of `pfs.drp.stella.NevenPsf` that exposes the protected
        ``doComputeOversampledKernelImage`` method.
        """
        directory = os.path.join(getPackageDir("drp_pfs_data"), "nevenPsf")
        positions = np.load(os.path.join(directory, "test_arrays_from_Neven_from_Python_positions.npy"))
        images = np.load(os.path.join(directory, "test_arrays_from_Neven_from_Python.npy"))
        psf = pfs.drp.stella.tests.nevenPsf.NevenPsf(self.psf)  # Expose the interpolation
        for pos, img in zip(positions, images):
            ours = psf.computeOversampledKernelImage(lsst.geom.Point2D(pos))
            self.assertFloatsAlmostEqual(ours.array/ours.array.max(), img/img.max(), atol=1.0e-8)

            if display is not None:
                diff = ours.array - img
                Display(backend=display, frame=1).mtv(ours)
                Display(backend=display, frame=2).mtv(lsst.afw.image.ImageD(img))
                Display(backend=display, frame=3).mtv(lsst.afw.image.ImageD(diff))
                input("Press ENTER to continue")


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
