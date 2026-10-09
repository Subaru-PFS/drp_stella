import numpy as np

import lsst.utils.tests

from lsst.afw.detection import GaussianPsf
from lsst.afw.image import ImageD
from lsst.afw.math import FixedKernel

from pfs.drp.stella.psfBias import calculatePeakToCentroidBias
from pfs.drp.stella.tests import runTests


class PsfBiasTestCase(lsst.utils.tests.TestCase):
    """Tests of `pfs.drp.stella.psfBias.calculatePeakToCentroidBias`"""

    def testNoKernel(self):
        """For a symmetric Gaussian PSF, peak and centroid should agree almost exactly"""
        psf = GaussianPsf(25, 25, 2.0)

        # At exact integer positions, a symmetric PSF's peak and centroid coincide exactly
        # (up to floating-point round-off): a basic correctness check with a known answer.
        x = np.array([100.0, 200.0, 300.0])
        y = np.array([150.0, 250.0, 50.0])
        xBias, yBias = calculatePeakToCentroidBias(psf, x, y)
        self.assertFloatsAlmostEqual(xBias, 0.0, atol=1.0e-6)
        self.assertFloatsAlmostEqual(yBias, 0.0, atol=1.0e-6)

        # At generic sub-pixel positions, the sub-pixel resampling used to realize the PSF
        # introduces a tiny amount of real (if practically negligible) asymmetry, but it should
        # still be far smaller than the deliberately-asymmetric-kernel case in testKernel below.
        x = np.array([100.3, 200.7, 300.5])
        y = np.array([150.2, 250.6, 50.1])
        xBias, yBias = calculatePeakToCentroidBias(psf, x, y)
        self.assertTrue(np.all(np.abs(xBias) < 0.02))
        self.assertTrue(np.all(np.abs(yBias) < 0.02))

    def makeAsymmetricKernel(
        self,
        halfWidth: int,
        mainSigma: float = 0.8,
        wingSigma: float = 2.5,
        wingAmplitude: float = 0.2,
        wingOffset: float = 3.0,
    ) -> np.ndarray:
        """Make a deliberately asymmetric kernel: a narrow core plus a fainter, offset, broader wing

        The core dominates the peak location; the faint wing shifts the kernel's first moment
        without much moving the peak -- the same qualitative effect a real, non-Gaussian PSF's
        asymmetric wings have on `SdssCentroidAlgorithm` (see the module docstring).
        """
        offset = np.arange(-halfWidth, halfWidth + 1)
        xx, yy = np.meshgrid(offset, offset)
        core = np.exp(-0.5 * (xx**2 + yy**2) / mainSigma**2)
        wing = wingAmplitude * np.exp(-0.5 * ((xx - wingOffset) ** 2 + yy**2) / wingSigma**2)
        kernel = core + wing
        kernel /= kernel.sum()
        return kernel

    def makeKernel(self, array: np.ndarray) -> FixedKernel:
        """Make an `lsst.afw.math.Kernel` from an array"""
        image = ImageD(array.shape[1], array.shape[0])
        image.array[:] = array
        return FixedKernel(image)

    def testKernel(self):
        """Applying a deliberately asymmetric kernel shifts the measured bias"""
        halfWidth = 6
        array = self.makeAsymmetricKernel(halfWidth)
        offset = np.arange(-halfWidth, halfWidth + 1)
        firstMoment = np.sum(array.sum(axis=0) * offset) / array.sum()
        self.assertGreater(abs(firstMoment), 0.1)  # genuinely, usefully asymmetric

        psf = GaussianPsf(25, 25, 1.5)
        x = np.array([100.0])
        y = np.array([100.0])
        xBiasBare, _ = calculatePeakToCentroidBias(psf, x, y)
        xBiasKernel, _ = calculatePeakToCentroidBias(psf, x, y, kernel=self.makeKernel(array))

        # The kernel is applied as a correlation (the afw convention), so it shifts the full centroid
        # by (very close to) minus its first moment, but the peak-based estimate lags behind (shifts by
        # less) -- so the *bias* (peak - centroid) shifts substantially in the same direction as the
        # kernel's own asymmetry, by less than the full first moment.
        shift = xBiasKernel[0] - xBiasBare[0]
        self.assertGreater(shift * np.sign(firstMoment), 0.05)
        self.assertLess(abs(shift), abs(firstMoment))

    def testErrors(self):
        """Invalid kernels raise, rather than returning a bogus bias"""
        psf = GaussianPsf(25, 25, 1.5)
        x = np.array([100.0])
        y = np.array([100.0])
        array = self.makeAsymmetricKernel(6)
        with self.assertRaises(TypeError):
            calculatePeakToCentroidBias(psf, x, y, kernel=array)  # Not an lsst.afw.math.Kernel
        with self.assertRaises(ValueError):
            calculatePeakToCentroidBias(psf, x, y, kernel=self.makeKernel(array[:-1, :-1]))  # Even size


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
