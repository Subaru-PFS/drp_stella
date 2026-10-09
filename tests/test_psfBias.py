import numpy as np

import lsst.utils.tests

from lsst.afw.detection import GaussianPsf
from lsst.afw.image import MaskedImage
from lsst.geom import Box2I, Point2I, Extent2I

from pfs.drp.stella.AlardLupton import fitAlardLuptonKernel
from pfs.drp.stella.psfBias import calculatePeakToCentroidBias
from pfs.drp.stella.tests import runTests


class PsfBiasTestCase(lsst.utils.tests.TestCase):
    """Tests of `pfs.drp.stella.psfBias.calculatePeakToCentroidBias`"""

    def setUp(self):
        self.rng = np.random.default_rng(54321)
        self.width = 200
        self.height = 200
        self.readnoise = 5.0

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

    def makeSource(self) -> MaskedImage:
        """Make a random source image with a bunch of point sources on a background"""
        image = np.full((self.height, self.width), 100.0, dtype=np.float32)
        numStars = 200
        xx = self.rng.uniform(0, self.width, numStars)
        yy = self.rng.uniform(0, self.height, numStars)
        flux = self.rng.uniform(1000, 50000, numStars)
        sigma = 1.5
        radius = int(np.ceil(5 * sigma))
        for x0, y0, ff in zip(xx, yy, flux):
            xLow, xHigh = max(0, int(x0) - radius), min(self.width, int(x0) + radius + 1)
            yLow, yHigh = max(0, int(y0) - radius), min(self.height, int(y0) + radius + 1)
            if xLow >= xHigh or yLow >= yHigh:
                continue
            xGrid, yGrid = np.meshgrid(np.arange(xLow, xHigh), np.arange(yLow, yHigh))
            image[yLow:yHigh, xLow:xHigh] += (
                ff
                * np.exp(-0.5 * ((xGrid - x0) ** 2 + (yGrid - y0) ** 2) / sigma**2)
                / (2 * np.pi * sigma**2)
            )
        image += self.rng.normal(0, self.readnoise, (self.height, self.width))

        maskedImage = MaskedImage(Box2I(Point2I(0, 0), Extent2I(self.width, self.height)), dtype=np.float32)
        maskedImage.image.array[:] = image
        maskedImage.mask.array[:] = 0
        maskedImage.variance.array[:] = self.readnoise**2 + np.clip(image, 0, None)
        return maskedImage

    def convolve(self, image: MaskedImage, kernel: np.ndarray) -> MaskedImage:
        """Convolve a MaskedImage with a kernel (using a simple direct convolution)"""
        from scipy.signal import convolve2d

        result = image.clone()
        result.image.array[:] = convolve2d(image.image.array, kernel, mode="same", boundary="symm")
        result.variance.array[:] = convolve2d(image.variance.array, kernel**2, mode="same", boundary="symm")
        return result

    def testKernel(self):
        """Applying a deliberately asymmetric kernel shifts the measured bias"""
        kernelHalfWidth = 6
        trueKernel = self.makeAsymmetricKernel(kernelHalfWidth)
        source = self.makeSource()
        target = self.convolve(source, trueKernel)

        result = fitAlardLuptonKernel(source, target, kernelHalfWidth=kernelHalfWidth)
        solution = result.solutions[0]
        self.assertTrue(solution.success)
        firstMoment = solution.getFirstMoment()
        self.assertGreater(abs(firstMoment.getX()), 0.1)  # genuinely, usefully asymmetric

        psf = GaussianPsf(25, 25, 1.5)
        x = np.array([100.0])
        y = np.array([100.0])
        xBiasBare, _ = calculatePeakToCentroidBias(psf, x, y)
        xBiasKernel, _ = calculatePeakToCentroidBias(psf, x, y, kernel=result)

        # The kernel shifts the full centroid by (very close to) its own first moment, but the
        # peak-based estimate lags behind (shifts by less) -- so the *bias* (peak - centroid)
        # shifts substantially in the OPPOSITE direction to the kernel's own asymmetry, by less
        # than the full first moment.
        shift = xBiasKernel[0] - xBiasBare[0]
        self.assertLess(shift * np.sign(firstMoment.getX()), -0.05)
        self.assertLess(abs(shift), abs(firstMoment.getX()))

    def testErrors(self):
        """Invalid kernel/position combinations raise, rather than returning a bogus bias"""
        kernelHalfWidth = 6
        trueKernel = self.makeAsymmetricKernel(kernelHalfWidth)
        source = self.makeSource()
        target = self.convolve(source, trueKernel)
        result = fitAlardLuptonKernel(source, target, kernelHalfWidth=kernelHalfWidth)

        psf = GaussianPsf(25, 25, 1.5)

        # Position outside the kernel's fitted image (source/target are only 200x200).
        with self.assertRaises(ValueError):
            calculatePeakToCentroidBias(psf, np.array([1.0e5]), np.array([1.0e5]), kernel=result)

        # A minSignalToNoise so high that no pixel clears it makes the region's fit fail cleanly
        # (see test_AlardLupton.py's testMinSignalToNoise for the same technique).
        impossible = fitAlardLuptonKernel(
            source, target, kernelHalfWidth=kernelHalfWidth, minSignalToNoise=1.0e6
        )
        self.assertFalse(impossible.solutions[0].success)
        with self.assertRaises(ValueError):
            calculatePeakToCentroidBias(psf, np.array([100.0]), np.array([100.0]), kernel=impossible)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
