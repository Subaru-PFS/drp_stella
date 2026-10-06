import numpy as np

import lsst.utils.tests
import lsst.geom
from lsst.afw.image import ExposureF

from pfs.drp.stella.synthetic import SyntheticPsfConfig, makeSyntheticDetectorMap
from pfs.drp.stella.psfProfiles import ParametricPsfModel
from pfs.drp.stella.psfSpline import PsfSplineBasis, SplineAxisConfig
from pfs.drp.stella.hybridPsfModel import HybridPsfModel
from pfs.drp.stella.pfsPsf import PfsPsf
from pfs.drp.stella.tests import runTests


class PfsPsfTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticPsfConfig()
        self.config.height = 120
        self.config.width = 60
        self.config.blockSize = 20
        self.detectorMap = makeSyntheticDetectorMap(self.config)

        truth = self.config.makeGroundTruthPsf()
        parametricModel = ParametricPsfModel(
            0, len(truth.wings), (0, self.config.numFibers - 1), (0, self.config.height - 1)
        )
        parametricModel.setInitialGuess(truth.sigmaX, truth.sigmaY, truth.tophatWidth, truth.wings)
        axisConfig = SplineAxisConfig(extent=6.0, fineRadius=1.0, mediumRadius=2.0, coarseSpacing=2.0)
        splineBasis = PsfSplineBasis(axisConfig, axisConfig)
        hybridModel = HybridPsfModel(parametricModel, splineBasis)

        self.halfSize = 8
        self.oversampling = 2
        self.psf = PfsPsf(hybridModel, self.detectorMap, self.halfSize, self.oversampling)
        self.position = self.psf.getAveragePosition()

    def testKernelImage(self):
        """The kernel image is the right shape and normalized to sum to one"""
        image = self.psf.computeKernelImage(self.position)
        size = 2 * self.halfSize + 1
        self.assertEqual(image.getBBox().getWidth(), size)
        self.assertEqual(image.getBBox().getHeight(), size)
        self.assertFloatsAlmostEqual(np.sum(image.array), 1.0, atol=1.0e-10)
        # Kernel image is in the local (position-independent) frame
        self.assertEqual(image.getBBox().getMinX(), -self.halfSize)
        self.assertEqual(image.getBBox().getMinY(), -self.halfSize)

    def testImage(self):
        """The detector-frame image is placed at the requested position"""
        image = self.psf.computeImage(self.position)
        indexX = int(np.floor(self.position.getX() + 0.5))
        indexY = int(np.floor(self.position.getY() + 0.5))
        self.assertEqual(image.getBBox().getMinX(), indexX - self.halfSize)
        self.assertEqual(image.getBBox().getMinY(), indexY - self.halfSize)
        self.assertFloatsAlmostEqual(np.sum(image.array), 1.0, atol=1.0e-10)

    def testBBox(self):
        """computeBBox/computeKernelBBox agree, and are position-independent"""
        size = 2 * self.halfSize + 1
        bbox = self.psf.computeBBox(self.position)
        self.assertEqual(bbox, self.psf.computeKernelBBox(self.position))
        self.assertEqual(bbox.getWidth(), size)
        self.assertEqual(bbox.getHeight(), size)
        other = lsst.geom.Point2D(self.position.getX() + 5.0, self.position.getY() - 3.0)
        self.assertEqual(self.psf.computeBBox(other), bbox)

    def testShape(self):
        """The computed shape has sane, positive second moments"""
        shape = self.psf.computeShape(self.position)
        self.assertGreater(shape.getIxx(), 0.0)
        self.assertGreater(shape.getIyy(), 0.0)

    def testApertureFlux(self):
        """Aperture flux increases with radius and approaches the full flux"""
        small = self.psf.computeApertureFlux(1.0, self.position)
        large = self.psf.computeApertureFlux(float(self.halfSize), self.position)
        self.assertGreater(large, small)
        self.assertLessEqual(large, 1.0 + 1.0e-8)
        self.assertGreater(large, 0.8)

    def testDeepCopy(self):
        """A deep copy is independent of the original"""
        import copy

        copied = copy.deepcopy(self.psf)
        original = self.psf.computeKernelImage(self.position)
        duplicate = copied.computeKernelImage(self.position)
        self.assertFloatsAlmostEqual(original.array, duplicate.array, atol=1.0e-12)

        copied.hybridModel.splineCoefficients[:] = 1.0
        unchanged = self.psf.hybridModel.splineCoefficients
        self.assertFloatsEqual(unchanged, 0.0)

    def testResized(self):
        """resized() rebuilds with a new stamp size, leaving the original untouched"""
        newHalfSize = self.halfSize + 2
        newSize = 2 * newHalfSize + 1
        resized = self.psf.resized(newSize, newSize)
        self.assertEqual(resized.halfSize, newHalfSize)
        self.assertEqual(self.psf.halfSize, self.halfSize)
        image = resized.computeKernelImage(self.position)
        self.assertEqual(image.getBBox().getWidth(), newSize)

    def testPersistence(self):
        """PfsPsf round-trips through FITS persistence via an Exposure"""
        exposure = ExposureF(self.detectorMap.bbox)
        exposure.setPsf(self.psf)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exposure.writeFits(filename)
            restored = ExposureF(filename)
        newPsf = restored.getPsf()
        self.assertIsInstance(newPsf, PfsPsf)
        original = self.psf.computeKernelImage(self.position)
        recovered = newPsf.computeKernelImage(self.position)
        self.assertFloatsAlmostEqual(original.array, recovered.array, atol=1.0e-10)
        self.assertEqual(newPsf.halfSize, self.psf.halfSize)
        self.assertEqual(newPsf.oversampling, self.psf.oversampling)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
