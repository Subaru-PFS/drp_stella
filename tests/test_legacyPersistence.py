import os

import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
from lsst.afw.detection import GaussianPsf
import lsst.geom

import pfs.drp.stella  # noqa: F401 (registers the persistence factories)
from pfs.drp.stella.tests.utils import runTests

display = None

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class LegacyPersistenceTestCase(lsst.utils.tests.TestCase):
    def testImagingSpectralPsf(self):
        """Test reading an exposure with an ImagingSpectralPsf

        The file was written by drp_stella w.2026.31 (before SpectralPsf was
        removed): a 16x16 exposure with an ``ImagingSpectralPsf`` wrapping a
        ``GaussianPsf(11, 11, 0.637)`` and a synthetic ``SplinedDetectorMap``,
        as attached to every calexp by ``checkPsf``. We should get back the
        ``GaussianPsf``.
        """
        exposure = afwImage.ExposureF.readFits(os.path.join(DATA_DIR, "legacyImagingSpectralPsf.fits"))
        self.assertFloatsEqual(exposure.image.array, np.arange(16 * 16, dtype=np.float32).reshape(16, 16))
        psf = exposure.getPsf()
        self.assertIsInstance(psf, GaussianPsf)
        self.assertEqual(psf.getSigma(), 0.637)
        self.assertEqual(psf.getDimensions(), lsst.geom.Extent2I(11, 11))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
