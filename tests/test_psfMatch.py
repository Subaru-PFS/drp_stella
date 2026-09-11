import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom

from pfs.drp.stella.psfMatch import PsfMatchTask
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticArc, addNoiseToImage
from pfs.drp.stella.tests.utils import runTests

display = None


def makeGaussianKernel(sigma, size=15):
    """Construct a normalized, circular Gaussian analytic kernel

    Parameters
    ----------
    sigma : `float`
        Gaussian sigma (pixels).
    size : `int`, optional
        Width and height of the kernel (pixels); should be odd.

    Returns
    -------
    kernel : `lsst.afw.math.AnalyticKernel`
        The Gaussian kernel.
    """
    function = afwMath.GaussianFunction2D(sigma, sigma)
    return afwMath.AnalyticKernel(size, size, function)


class PsfMatchTestCase(lsst.utils.tests.TestCase):
    """Test PsfMatchTask.run end-to-end

    Uses a pair of synthetic arc images derived from the same noiseless
    "true" image, but convolved with different-width Gaussian kernels (and
    given independent noise realizations) to stand in for a genuine PSF
    difference between two exposures. ``source`` is the sharper of the two,
    consistent with the requirement that ``source`` is convolved (never
    deconvolved) to match ``target``.
    """

    def setUp(self):
        synth = SyntheticConfig()
        synth.width = 256
        synth.height = 256
        synth.separation = 20.0
        synth.fwhm = 3.0
        synth.gain = 1.5
        synth.readnoise = 3.0
        self.synth = synth

        rng = np.random.RandomState(12345)
        trueImage = makeSyntheticArc(synth, numLines=30, fwhm=3.5, flux=3.0e5, addNoise=False).image

        sourceImage = afwImage.ImageF(trueImage.getBBox())
        targetImage = afwImage.ImageF(trueImage.getBBox())
        convolutionControl = afwMath.ConvolutionControl()
        convolutionControl.setDoNormalize(True)
        afwMath.convolve(sourceImage, trueImage, makeGaussianKernel(0.6), convolutionControl)
        afwMath.convolve(targetImage, trueImage, makeGaussianKernel(1.3), convolutionControl)

        # Convolution leaves non-finite values in the unconvolved border pixels.
        borderMask = ~np.isfinite(sourceImage.array) | ~np.isfinite(targetImage.array)
        sourceImage.array[~np.isfinite(sourceImage.array)] = 0.0
        targetImage.array[~np.isfinite(targetImage.array)] = 0.0

        addNoiseToImage(sourceImage, synth.gain, synth.readnoise, rng)
        addNoiseToImage(targetImage, synth.gain, synth.readnoise, rng)

        self.sourceExposure = self.makeExposure(sourceImage, borderMask)
        self.targetExposure = self.makeExposure(targetImage, borderMask)

        self.config = PsfMatchTask.ConfigClass()
        self.config.kernel["DF"].kernelSize = 15
        self.config.kernel["DF"].sizeCellX = 128
        self.config.kernel["DF"].sizeCellY = 128
        self.config.xStampSize = 48
        self.config.yStampSize = 48
        self.config.minPeakSignalToNoise = 5.0
        self.task = PsfMatchTask(config=self.config)
        self.task.log.setLevel(self.task.log.DEBUG)

    def tearDown(self):
        del self.sourceExposure
        del self.targetExposure
        del self.task

    def makeExposure(self, image, borderMask):
        """Wrap an Image in an Exposure with mask and variance planes

        Parameters
        ----------
        image : `lsst.afw.image.Image`
            Image to wrap.
        borderMask : `numpy.ndarray` of `bool`
            Pixels to flag with the ``EDGE`` mask plane.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            The resulting exposure.
        """
        mask = afwImage.Mask(image.getBBox())
        mask.set(0)
        mask.array[borderMask] = mask.getPlaneBitMask("EDGE")
        variance = afwImage.ImageF(image.getBBox())
        variance.array[:] = (
            np.where(image.array > 0, image.array, 0.0) / self.synth.gain
            + self.synth.readnoise**2 / self.synth.gain**2
        )
        maskedImage = afwImage.makeMaskedImage(image, mask, variance)
        return afwImage.makeExposure(maskedImage)

    def testBasic(self):
        """Test that the matched exposure reproduces the target"""
        result = self.task.run(self.sourceExposure, self.targetExposure, seed=1)
        self.assertGreater(result.selection.numUsed, 0)
        self.assertEqual(result.selection.numUsed, result.selection.numCandidates)

        select = (result.matchedExposure.mask.array == 0) & (self.targetExposure.mask.array == 0)
        self.assertGreater(select.sum(), 0.5 * select.size)  # Most of the image should be usable

        residual = result.matchedExposure.image.array - self.targetExposure.image.array
        chi2 = np.sum((residual[select] / np.sqrt(self.targetExposure.variance.array[select])) ** 2)
        self.assertLess(chi2 / select.sum(), 2.0)


class BuildCellSetTestCase(lsst.utils.tests.TestCase):
    """Test PsfMatchTask._buildCellSet's tiling and stamp-selection logic

    Uses simple, cheap-to-construct images so we can exercise stamp
    selection without paying for a full kernel solve.
    """

    def setUp(self):
        self.width = 240
        self.height = 240
        self.xStampSize = 40
        self.yStampSize = 40
        bbox = geom.Box2I(geom.Point2I(0, 0), geom.Extent2I(self.width, self.height))
        self.rng = np.random.RandomState(54321)

        source = afwImage.MaskedImageF(bbox)
        source.mask.array[:] = 0
        source.variance.array[:] = 100.0
        source.image.array[:] = 1000.0 + self.rng.normal(0.0, 10.0, source.image.array.shape)

        self.source = source
        self.target = source.clone()

        self.config = PsfMatchTask.ConfigClass()
        self.config.xStampSize = self.xStampSize
        self.config.yStampSize = self.yStampSize
        self.config.minPeakSignalToNoise = 5.0
        self.config.maxStamps = None
        self.task = PsfMatchTask(config=self.config)
        self.task.log.setLevel(self.task.log.DEBUG)

    def tearDown(self):
        del self.source
        del self.target
        del self.task

    def extractPositions(self, kernelCellSet):
        """Return the sorted (x, y) positions of all candidates in a cell set

        Parameters
        ----------
        kernelCellSet : `lsst.afw.math.SpatialCellSet`
            Cell set to inspect.

        Returns
        -------
        positions : `list` of `tuple` of `float`
            Sorted ``(x, y)`` candidate positions.
        """
        positions = []
        for cell in kernelCellSet.getCellList():
            for candidate in cell:
                positions.append((candidate.getXCenter(), candidate.getYCenter()))
        return sorted(positions)

    def testBasic(self):
        """All tiles should be kept: bright, uniform signal, no masking"""
        result = self.task._buildCellSet(self.source, self.target, seed=1)
        numTilesX = self.width // self.xStampSize
        numTilesY = self.height // self.yStampSize
        self.assertEqual(result.numTiles, numTilesX * numTilesY)
        self.assertEqual(result.numRejectedMask, 0)
        self.assertEqual(result.numRejectedSnr, 0)
        self.assertEqual(result.numCandidates, result.numTiles)
        self.assertEqual(result.numUsed, result.numCandidates)

    def testSignalToNoiseFiltering(self):
        """Stamps with low peak signal-to-noise should be rejected"""
        source = self.source.clone()
        target = self.target.clone()
        halfHeight = self.height // 2  # An exact multiple of yStampSize
        source.image.array[:halfHeight, :] = 0.0
        target.image.array[:halfHeight, :] = 0.0

        result = self.task._buildCellSet(source, target, seed=1)
        numTilesX = self.width // self.xStampSize
        numLowTilesY = halfHeight // self.yStampSize
        self.assertEqual(result.numRejectedSnr, numLowTilesY * numTilesX)
        self.assertEqual(result.numUsed, result.numTiles - result.numRejectedSnr)

        # Raising the threshold above the bright half's peak S/N rejects everything
        self.config.minPeakSignalToNoise = 1000.0
        task = PsfMatchTask(config=self.config)
        result2 = task._buildCellSet(source, target, seed=1)
        self.assertEqual(result2.numRejectedSnr, result2.numTiles)
        self.assertEqual(result2.numUsed, 0)

    def testMaskRejection(self):
        """Stamps with a bad mask plane set should be rejected"""
        baseline = self.task._buildCellSet(self.source, self.target, seed=1)
        source = self.source.clone()
        source.mask.array[: self.yStampSize, : self.xStampSize] |= source.mask.getPlaneBitMask("SAT")

        result = self.task._buildCellSet(source, self.target, seed=1)
        self.assertEqual(result.numTiles, baseline.numTiles)
        self.assertEqual(result.numRejectedMask, baseline.numRejectedMask + 1)
        self.assertEqual(result.numUsed, baseline.numUsed - 1)

    def testRandomSubsampling(self):
        """maxStamps limits the number of candidates, reproducibly"""
        self.config.maxStamps = 10
        task = PsfMatchTask(config=self.config)

        result1 = task._buildCellSet(self.source, self.target, seed=1)
        self.assertEqual(result1.numUsed, 10)
        positions1 = self.extractPositions(result1.kernelCellSet)

        result2 = task._buildCellSet(self.source, self.target, seed=1)
        positions2 = self.extractPositions(result2.kernelCellSet)
        self.assertEqual(positions1, positions2)  # Same seed: same stamps

        result3 = task._buildCellSet(self.source, self.target, seed=2)
        positions3 = self.extractPositions(result3.kernelCellSet)
        self.assertEqual(len(positions3), 10)
        self.assertNotEqual(positions1, positions3)  # Different seed: (almost certainly) different stamps

    def testMismatchedBoundingBoxes(self):
        """source/target with overlapping but non-identical bounding boxes"""
        shiftedBBox = geom.Box2I(geom.Point2I(20, 20), geom.Extent2I(self.width, self.height))
        target = afwImage.MaskedImageF(shiftedBBox)
        target.image.array[:] = self.target.image.array
        target.variance.array[:] = self.target.variance.array
        target.mask.array[:] = 0

        result = self.task._buildCellSet(self.source, target, seed=1)
        self.assertGreater(result.numUsed, 0)

        overlap = geom.Box2I(self.source.getBBox())
        overlap.clip(target.getBBox())
        expectedTilesX = overlap.getWidth() // self.xStampSize
        expectedTilesY = overlap.getHeight() // self.yStampSize
        self.assertEqual(result.numTiles, expectedTilesX * expectedTilesY)

    def testInsufficientArea(self):
        """Stamp size larger than the overlap region raises RuntimeError"""
        self.config.xStampSize = self.width + 10
        self.config.yStampSize = self.height + 10
        task = PsfMatchTask(config=self.config)
        with self.assertRaises(RuntimeError):
            task._buildCellSet(self.source, self.target, seed=1)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
