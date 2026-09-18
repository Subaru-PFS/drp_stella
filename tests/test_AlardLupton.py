import numpy as np

import lsst.utils.tests

from lsst.afw.image import MaskedImage
from lsst.geom import Box2I, Point2I, Extent2I

from pfs.drp.stella.AlardLupton import fitAlardLuptonKernel
from pfs.drp.stella.tests import runTests


class AlardLuptonTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12345)
        self.width = 200
        self.height = 200
        self.readnoise = 5.0
        self.kernelHalfWidth = 3
        self.kernelSize = 2 * self.kernelHalfWidth + 1

    def makeKernel(self, sigma: float = 1.2) -> np.ndarray:
        """Make a normalized Gaussian kernel"""
        offset = np.arange(-self.kernelHalfWidth, self.kernelHalfWidth + 1)
        xx, yy = np.meshgrid(offset, offset)
        kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
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
            xLow, xHigh = int(x0) - radius, int(x0) + radius + 1
            yLow, yHigh = int(y0) - radius, int(y0) + radius + 1
            xLow, xHigh = max(0, xLow), min(self.width, xHigh)
            yLow, yHigh = max(0, yLow), min(self.height, yHigh)
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

    def testBasic(self):
        """Recover a simple, spatially-constant kernel and background"""
        trueKernel = self.makeKernel(1.2)
        trueBackground = 12.34

        source = self.makeSource()
        target = self.convolve(source, trueKernel)
        target.image.array += trueBackground

        result = fitAlardLuptonKernel(source, target, kernelHalfWidth=self.kernelHalfWidth)
        self.assertEqual(len(result.solutions), 1)
        solution = result.solutions[0]
        self.assertTrue(solution.success)
        self.assertFloatsAlmostEqual(solution.kernel, trueKernel, atol=1.0e-2)
        self.assertFloatsAlmostEqual(solution.background[0], trueBackground, atol=1.0)
        self.assertGreater(solution.numFit, solution.getNumParams())
        self.assertEqual(solution.numRejected, 0)

        border = self.kernelHalfWidth
        interior = np.s_[border:-border, border:-border]
        self.assertFloatsAlmostEqual(result.difference.image.array[interior], 0.0, atol=10 * self.readnoise)
        self.assertFloatsAlmostEqual(
            np.std(result.difference.image.array[interior]), 0.0, atol=self.readnoise
        )

        # convolved is a first-class output: difference = target - convolved, by construction,
        # and the two images share the same mask (NO_DATA/DIFFIM_REJECTED/DIFFIM_PARTIAL).
        self.assertFloatsAlmostEqual(
            result.convolved.image.array[interior],
            target.image.array[interior] - result.difference.image.array[interior],
            atol=1.0e-3,
        )
        self.assertFloatsAlmostEqual(
            result.convolved.mask.array[interior], result.difference.mask.array[interior]
        )

        badBitMask = 1 << result.difference.mask.getMaskPlane("NO_DATA")
        self.assertTrue(np.all((result.difference.mask.array[:border, :] & badBitMask) != 0))
        self.assertTrue(np.all((result.convolved.mask.array[:border, :] & badBitMask) != 0))

        # Aggregate stats and single-solution stats should agree
        self.assertFloatsAlmostEqual(result.getChi2(), solution.chi2)
        self.assertEqual(result.getNumFit(), solution.numFit)
        self.assertEqual(result.getNumRejected(), solution.numRejected)

    def testRejection(self):
        """Bad pixels (e.g. a cosmic ray) should be rejected from the fit"""
        trueKernel = self.makeKernel(1.2)
        source = self.makeSource()
        target = self.convolve(source, trueKernel)

        # Inject some "cosmic rays": isolated, extreme outliers
        numBad = 30
        badX = self.rng.integers(self.kernelHalfWidth, self.width - self.kernelHalfWidth, numBad)
        badY = self.rng.integers(self.kernelHalfWidth, self.height - self.kernelHalfWidth, numBad)
        target.image.array[badY, badX] += 1.0e5

        result = fitAlardLuptonKernel(source, target, kernelHalfWidth=self.kernelHalfWidth, rejIter=3)
        solution = result.solutions[0]
        self.assertTrue(solution.success)
        self.assertFloatsAlmostEqual(solution.kernel, trueKernel, atol=2.0e-2)
        self.assertGreaterEqual(solution.numRejected, numBad - 2)  # allow for a couple of coincidences

        rejectedBitMask = 1 << result.difference.mask.getMaskPlane("DIFFIM_REJECTED")
        self.assertTrue(np.any((result.difference.mask.array[badY, badX] & rejectedBitMask) != 0))

    def testRegions(self):
        """Kernel varying discretely across regions should be recovered independently in each"""
        kernelA = self.makeKernel(0.9)
        kernelB = self.makeKernel(1.8)

        source = self.makeSource()
        convolvedA = self.convolve(source, kernelA)
        convolvedB = self.convolve(source, kernelB)

        target = source.clone()
        half = self.width // 2
        target.image.array[:, :half] = convolvedA.image.array[:, :half]
        target.image.array[:, half:] = convolvedB.image.array[:, half:]
        target.variance.array[:, :half] = convolvedA.variance.array[:, :half]
        target.variance.array[:, half:] = convolvedB.variance.array[:, half:]

        result = fitAlardLuptonKernel(
            source, target, kernelHalfWidth=self.kernelHalfWidth, numRegionsX=2, numRegionsY=1
        )
        self.assertEqual(len(result.solutions), 2)
        left, right = result.solutions
        self.assertTrue(left.success and right.success)
        self.assertFloatsAlmostEqual(left.kernel, kernelA, atol=2.0e-2)
        self.assertFloatsAlmostEqual(right.kernel, kernelB, atol=2.0e-2)

        leftSolution = result.getSolutionAt(10, 100)
        rightSolution = result.getSolutionAt(self.width - 10, 100)
        self.assertEqual(leftSolution.bbox, left.bbox)
        self.assertEqual(rightSolution.bbox, right.bbox)
        self.assertIsNone(result.getSolutionAt(-1, 100))
        self.assertIsNone(result.getSolutionAt(self.width, 100))

    def testCommonKernelSum(self):
        """A common kernel-sum constraint should stabilize the kernel sum in a low-signal region"""
        kernelA = self.makeKernel(0.9)
        kernelB = self.makeKernel(1.8)
        trueSum = 1.0  # makeKernel normalizes to sum 1, so both regions share the same true kernel sum

        source = self.makeSource()
        half = self.width // 2
        # Make the right half of the source low-signal (flat background + read noise, no point
        # sources), so the kernel *shape* there is poorly constrained by the data, analogous to a
        # continuum-only wavelength region with no strong emission lines.
        lowSignalShape = (self.height, self.width - half)
        source.image.array[:, half:] = 100.0 + self.rng.normal(0, self.readnoise, lowSignalShape)
        source.variance.array[:, half:] = self.readnoise**2 + np.clip(
            source.image.array[:, half:], 0, None
        )

        convolvedA = self.convolve(source, kernelA)
        convolvedB = self.convolve(source, kernelB)

        target = source.clone()
        target.image.array[:, :half] = convolvedA.image.array[:, :half]
        target.image.array[:, half:] = convolvedB.image.array[:, half:]
        target.variance.array[:, :half] = convolvedA.variance.array[:, :half]
        target.variance.array[:, half:] = convolvedB.variance.array[:, half:]

        unconstrained = fitAlardLuptonKernel(
            source, target, kernelHalfWidth=self.kernelHalfWidth, numRegionsX=2, numRegionsY=1
        )
        self.assertTrue(np.isnan(unconstrained.commonKernelSum))
        self.assertTrue(np.isnan(unconstrained.commonKernelSumScatter))
        leftUnconstrained, rightUnconstrained = unconstrained.solutions

        constrained = fitAlardLuptonKernel(
            source,
            target,
            kernelHalfWidth=self.kernelHalfWidth,
            numRegionsX=2,
            numRegionsY=1,
            commonKernelSum=True,
        )
        self.assertTrue(np.isfinite(constrained.commonKernelSum))
        self.assertTrue(np.isfinite(constrained.commonKernelSumScatter))
        leftConstrained, rightConstrained = constrained.solutions

        # The low-signal region's kernel sum should move closer to the true (shared) value...
        self.assertLess(
            abs(rightConstrained.getKernelSum() - trueSum),
            abs(rightUnconstrained.getKernelSum() - trueSum),
        )
        # ... while the well-constrained region is essentially undisturbed.
        self.assertFloatsAlmostEqual(
            leftConstrained.getKernelSum(), leftUnconstrained.getKernelSum(), atol=1.0e-2
        )

    def testBadPixels(self):
        """Pixels flagged as bad should not be used

        A pixel whose kernel footprint contains some bad source pixels can
        still be computed from the remaining good ones (and is flagged
        DIFFIM_PARTIAL); only a pixel whose entire footprint is bad (or that
        is otherwise too close to the masked region) is marked NO_DATA.
        """
        trueKernel = self.makeKernel(1.2)
        source = self.makeSource()
        target = self.convolve(source, trueKernel)

        badBitMask = (
            source.mask.getPlaneBitMask("BAD")
            if "BAD" in source.mask.getMaskPlaneDict()
            else 1 << source.mask.addMaskPlane("BAD")
        )
        source.mask.array[50:60, 50:60] |= badBitMask

        result = fitAlardLuptonKernel(
            source, target, kernelHalfWidth=self.kernelHalfWidth, badBitMask=badBitMask
        )
        solution = result.solutions[0]
        self.assertTrue(solution.success)
        self.assertFloatsAlmostEqual(solution.kernel, trueKernel, atol=2.0e-2)

        # The centre of the masked 10x10 block (indices 53-56 inclusive, for kernelHalfWidth=3) is far
        # enough from any good source pixel that its entire kernel footprint is bad, so it cannot be
        # computed even partially.
        noDataBitMask = 1 << result.difference.mask.getMaskPlane("NO_DATA")
        self.assertTrue(np.all((result.difference.mask.array[53:57, 53:57] & noDataBitMask) != 0))

        # Well away from the masked block, the kernel footprint doesn't overlap it at all, so pixels
        # there are computed normally (and not marked NO_DATA or DIFFIM_PARTIAL).
        partialBitMask = 1 << result.difference.mask.getMaskPlane("DIFFIM_PARTIAL")
        farAway = result.difference.mask.array[30, 30]
        self.assertEqual(farAway & (noDataBitMask | partialBitMask), 0)

        # A pixel just inside the edge of the masked block still has some good source pixels in its
        # kernel footprint (from outside the block), so it's computed from those (and flagged partial).
        self.assertNotEqual(result.difference.mask.array[50, 55] & partialBitMask, 0)
        self.assertEqual(result.difference.mask.array[50, 55] & noDataBitMask, 0)

    def testTooSmall(self):
        """An image too small for the requested kernel half-width should raise"""
        small = MaskedImage(Box2I(Point2I(0, 0), Extent2I(3, 3)), dtype=np.float32)
        small.image.array[:] = 1.0
        small.mask.array[:] = 0
        small.variance.array[:] = 1.0
        with self.assertRaises(Exception):
            fitAlardLuptonKernel(small, small, kernelHalfWidth=self.kernelHalfWidth)

    def testMismatchedBBox(self):
        """Source and target with different bounding boxes should raise"""
        source = self.makeSource()
        target = MaskedImage(Box2I(Point2I(1, 1), Extent2I(self.width, self.height)), dtype=np.float32)
        target.image.array[:] = 1.0
        target.mask.array[:] = 0
        target.variance.array[:] = 1.0
        with self.assertRaises(Exception):
            fitAlardLuptonKernel(source, target, kernelHalfWidth=self.kernelHalfWidth)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
