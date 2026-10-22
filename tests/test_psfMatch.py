import pickle

import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.pex.config as pexConfig

from pfs.drp.stella.psfMatch import PsfMatchTask, peakSignalToNoise, EXCLUDED_VARIANCE
from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticArc, addNoiseToImage
from pfs.drp.stella.synthetic import makeSpectrumImage, makeSyntheticDetectorMap
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


def makeExposure(image, borderMask, gain, readnoise):
    """Wrap an Image in an Exposure with mask and variance planes

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image to wrap.
    borderMask : `numpy.ndarray` of `bool`
        Pixels to flag with the ``EDGE`` mask plane.
    gain : `float`
        Detector gain (e/ADU).
    readnoise : `float`
        Detector read noise (e).

    Returns
    -------
    exposure : `lsst.afw.image.Exposure`
        The resulting exposure.
    """
    mask = afwImage.Mask(image.getBBox())
    mask.set(0)
    mask.array[borderMask] = mask.getPlaneBitMask("EDGE")
    variance = afwImage.ImageF(image.getBBox())
    variance.array[:] = np.where(image.array > 0, image.array, 0.0) / gain + readnoise**2 / gain**2
    maskedImage = afwImage.makeMaskedImage(image, mask, variance)
    return afwImage.makeExposure(maskedImage)


def convolveAndAddNoise(trueImage, sigma, synth, rng):
    """Convolve a noiseless image with a Gaussian and add noise

    Parameters
    ----------
    trueImage : `lsst.afw.image.Image`
        Noiseless image.
    sigma : `float`
        Gaussian sigma (pixels) of the convolution kernel.
    synth : `pfs.drp.stella.synthetic.SyntheticConfig`
        Synthetic spectrograph configuration, providing gain and read noise.
    rng : `numpy.random.RandomState`
        Random number generator.

    Returns
    -------
    image : `lsst.afw.image.Image`
        Convolved image with noise.
    borderMask : `numpy.ndarray` of `bool`
        Unconvolved border pixels.
    """
    image = afwImage.ImageF(trueImage.getBBox())
    convolutionControl = afwMath.ConvolutionControl()
    convolutionControl.setDoNormalize(True)
    afwMath.convolve(image, trueImage, makeGaussianKernel(sigma), convolutionControl)
    # Convolution leaves non-finite values in the unconvolved border pixels.
    borderMask = ~np.isfinite(image.array)
    image.array[borderMask] = 0.0
    addNoiseToImage(image, synth.gain, synth.readnoise, rng)
    return image, borderMask


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

        self.sourceExposure = makeExposure(sourceImage, borderMask, synth.gain, synth.readnoise)
        self.targetExposure = makeExposure(targetImage, borderMask, synth.gain, synth.readnoise)

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

    def testPickle(self):
        """Test that the result can be pickled"""
        result = self.task.run(self.sourceExposure, self.targetExposure, seed=1)
        copy = pickle.loads(pickle.dumps(result))

        self.assertImagesEqual(copy.matchedExposure.image, result.matchedExposure.image)
        self.assertIsNone(copy.spatialSolution)
        self.assertIsNone(copy.kernelCellSet)
        self.assertEqual(copy.selection.numUsed, result.selection.numUsed)

        kernel = result.psfMatchingKernel
        kernelCopy = copy.psfMatchingKernel
        self.assertEqual(type(kernelCopy), type(kernel))
        self.assertFloatsEqual(
            np.array(kernelCopy.getSpatialParameters()), np.array(kernel.getSpatialParameters())
        )
        for xx, yy in ((100.0, 200.0), (500.0, 1000.0)):
            image = afwImage.ImageD(kernel.getDimensions())
            imageCopy = afwImage.ImageD(kernelCopy.getDimensions())
            kernel.computeImage(image, False, xx, yy)
            kernelCopy.computeImage(imageCopy, False, xx, yy)
            self.assertImagesEqual(imageCopy, image)

        self.assertEqual(len(copy.basisList), len(result.basisList))
        for basis, basisCopy in zip(result.basisList, copy.basisList):
            self.assertEqual(type(basisCopy), type(basis))

        for xx, yy in ((100.0, 200.0), (500.0, 1000.0)):
            self.assertEqual(copy.backgroundModel(xx, yy), result.backgroundModel(xx, yy))


class LineExclusionTestCase(lsst.utils.tests.TestCase):
    """Test PsfMatchTask.run with a line list, excluding variable lines

    Mimics a pair of sky exposures: ``source`` and ``target`` contain the
    same lines, except that some (the "variable" lines, flagged ``BLEND``
    in the line list) are brighter in ``target``. Unless these are excluded,
    the fitted kernel sum is biased high.
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
        self.detectorMap = makeSyntheticDetectorMap(synth)

        numLines = 12
        flux = 3.0e5
        lineFwhm = 3.5
        self.variableFactor = 1.5
        lines = np.linspace(0, synth.height - 1, numLines + 2)[1:-1]
        isVariable = np.arange(numLines) % 3 == 1

        rows = np.arange(synth.height, dtype=float)
        sigma = lineFwhm / (2 * np.sqrt(2 * np.log(2)))
        profiles = np.exp(-0.5 * ((rows[np.newaxis, :] - lines[:, np.newaxis]) / sigma) ** 2)
        profiles *= flux / (sigma * np.sqrt(2 * np.pi))
        sourceSpectrum = profiles.sum(axis=0)
        targetFactor = np.where(isVariable, self.variableFactor, 1.0)
        targetSpectrum = (profiles * targetFactor[:, np.newaxis]).sum(axis=0)

        def makeTrueImage(spectrum):
            return makeSpectrumImage(
                spectrum.astype(np.float32), synth.dims, synth.traceCenters, synth.traceOffset, synth.fwhm
            )

        rng = np.random.RandomState(12345)
        sourceImage, sourceBorder = convolveAndAddNoise(makeTrueImage(sourceSpectrum), 0.6, synth, rng)
        targetImage, targetBorder = convolveAndAddNoise(makeTrueImage(targetSpectrum), 1.3, synth, rng)
        borderMask = sourceBorder | targetBorder
        self.sourceExposure = makeExposure(sourceImage, borderMask, synth.gain, synth.readnoise)
        self.targetExposure = makeExposure(targetImage, borderMask, synth.gain, synth.readnoise)

        fiberId = self.detectorMap.fiberId[len(self.detectorMap) // 2]
        wavelength = self.detectorMap.findWavelength(fiberId, lines)
        self.refLines = ReferenceLineSet.fromColumns(
            description=["Fake"] * numLines,
            wavelength=wavelength,
            intensity=np.full(numLines, flux),
            status=np.where(isVariable, ReferenceLineStatus.BLEND, ReferenceLineStatus.GOOD).astype(np.int32),
            transition=["UNKNOWN"] * numLines,
            source=np.zeros(numLines, dtype=np.int32),
            intensityErr=np.full(numLines, np.nan),
        )

        self.config = PsfMatchTask.ConfigClass()
        self.config.kernel["DF"].kernelSize = 15
        self.config.kernel["DF"].sizeCellX = 128
        self.config.kernel["DF"].sizeCellY = 128
        self.config.xStampSize = 48
        self.config.yStampSize = 48
        self.config.minPeakSignalToNoise = 5.0
        self.config.lineMaskRadius = 4
        self.task = PsfMatchTask(config=self.config)

    def tearDown(self):
        del self.sourceExposure
        del self.targetExposure
        del self.detectorMap
        del self.refLines
        del self.task

    def getKernelSums(self, kernel):
        """Return the kernel sum evaluated over a grid of positions

        Parameters
        ----------
        kernel : `lsst.afw.math.Kernel`
            Spatially varying kernel.

        Returns
        -------
        sums : `numpy.ndarray` of `float`
            Kernel sums.
        """
        image = afwImage.ImageD(kernel.getDimensions())
        positions = np.linspace(48, self.synth.width - 48, 4)
        return np.array([kernel.computeImage(image, False, xx, yy) for xx in positions for yy in positions])

    def testExclusion(self):
        """Excluding the variable lines gives an unbiased kernel"""
        sourceCopy = self.sourceExposure.maskedImage.clone()
        targetCopy = self.targetExposure.maskedImage.clone()

        result = self.task.run(
            self.sourceExposure,
            self.targetExposure,
            seed=1,
            refLines=self.refLines,
            detectorMap=self.detectorMap,
        )
        self.assertEqual(result.selection.numLinesExcluded, 4)
        self.assertGreater(result.selection.excludedFraction, 0.0)
        self.assertLess(result.selection.excludedFraction, 0.6)
        self.assertGreater(result.selection.numUsed, 0)
        sums = self.getKernelSums(result.psfMatchingKernel)
        self.assertFloatsAlmostEqual(sums, 1.0, atol=0.02)

        # Inputs are not modified
        self.assertMaskedImagesEqual(self.sourceExposure.maskedImage, sourceCopy)
        self.assertMaskedImagesEqual(self.targetExposure.maskedImage, targetCopy)
        excludeBit = self.sourceExposure.mask.getPlaneBitMask(self.config.lineMaskPlane)
        self.assertFalse(np.any(result.matchedExposure.mask.array & excludeBit))

        # Without the line list, the variable lines bias the kernel sum high. We disable clipping of
        # candidates, since otherwise all candidates are rejected (because they can't be fit well).
        self.config.kernel["DF"].singleKernelClipping = False
        self.config.kernel["DF"].kernelSumClipping = False
        self.config.kernel["DF"].spatialKernelClipping = False
        task = PsfMatchTask(config=self.config)
        biased = task.run(self.sourceExposure, self.targetExposure, seed=1)
        self.assertIsNone(biased.selection.numLinesExcluded)
        self.assertGreater(np.mean(self.getKernelSums(biased.psfMatchingKernel)), 1.05)

    def testNoExcludedLines(self):
        """Lines that don't match lineMaskStatus aren't excluded"""
        bbox = self.sourceExposure.getBBox()
        result = self.task._makeLineExclusion(bbox, self.refLines, self.detectorMap)
        self.assertEqual(result.numLines, 4)
        self.assertGreater(result.fraction, 0.0)
        self.assertEqual(result.exclude.shape, (bbox.getHeight(), bbox.getWidth()))

        self.config.lineMaskStatus = ["REJECTED"]
        task = PsfMatchTask(config=self.config)
        result = task._makeLineExclusion(bbox, self.refLines, self.detectorMap)
        self.assertEqual(result.numLines, 0)
        self.assertEqual(result.fraction, 0.0)
        self.assertFalse(result.exclude.any())

    def testMissingDetectorMap(self):
        """Providing refLines without a detectorMap raises"""
        with self.assertRaises(ValueError):
            self.task.run(self.sourceExposure, self.targetExposure, refLines=self.refLines)


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

    def testExclude(self):
        """Excluded pixels are flagged in copies of the stamps"""
        sourceCopy = self.source.clone()
        targetCopy = self.target.clone()
        exclude = np.zeros((self.height, self.width), dtype=bool)
        exclude[: self.yStampSize, : self.xStampSize] = True  # The whole of the first stamp
        exclude[self.yStampSize : self.yStampSize + 5, :] = True  # A strip through the second row of stamps
        # A bright pixel in the first stamp, which would pass the S/N cut if it weren't excluded
        self.source.image.array[10, 10] = 1.0e6
        self.target.image.array[10, 10] = 1.0e6
        sourceCopy.image.array[10, 10] = 1.0e6
        targetCopy.image.array[10, 10] = 1.0e6

        result = self.task._buildCellSet(self.source, self.target, seed=1, exclude=exclude)
        self.assertEqual(result.numRejectedSnr, 1)  # Only the fully-excluded stamp
        self.assertEqual(result.numUsed, result.numTiles - 1)

        # Inputs are not modified
        self.assertMaskedImagesEqual(self.source, sourceCopy)
        self.assertMaskedImagesEqual(self.target, targetCopy)

        excludeBit = self.source.mask.getPlaneBitMask(self.config.lineMaskPlane)
        numFlagged = 0
        for cell in result.kernelCellSet.getCellList():
            for candidate in cell:
                for image in (candidate.getTemplateMaskedImage(), candidate.getScienceMaskedImage()):
                    x0, y0 = image.getXY0()
                    height, width = image.getHeight(), image.getWidth()
                    expected = exclude[y0 : y0 + height, x0 : x0 + width]
                    flagged = (image.mask.array & excludeBit) != 0
                    self.assertTrue(np.array_equal(flagged, expected))
                    self.assertTrue(np.all(image.variance.array[expected] == EXCLUDED_VARIANCE))
                    self.assertTrue(np.all(image.variance.array[~expected] == 100.0))
                    numFlagged += expected.any()
        self.assertEqual(numFlagged, 2 * (self.width // self.xStampSize))

        with self.assertRaises(ValueError):
            self.task._buildCellSet(self.source, self.target, exclude=exclude[1:, :])

    def testPeakSignalToNoiseIgnore(self):
        """peakSignalToNoise skips ignored pixels"""
        image = self.source.clone()
        image.image.array[5, 7] = 1.0e6
        ignore = np.zeros(image.image.array.shape, dtype=bool)
        self.assertEqual(peakSignalToNoise(image), 1.0e5)
        ignore[5, 7] = True
        self.assertLess(peakSignalToNoise(image, ignore), 1.0e3)
        ignore[:] = True
        self.assertEqual(peakSignalToNoise(image, ignore), -np.inf)


class ConfigTestCase(lsst.utils.tests.TestCase):
    """Test PsfMatchConfig validation"""

    def testDefault(self):
        config = PsfMatchTask.ConfigClass()
        config.validate()
        self.assertIn(config.lineMaskPlane, config.kernel.active.badMaskPlanes)

    def testBadLineMaskPlane(self):
        config = PsfMatchTask.ConfigClass()
        config.lineMaskPlane = "SOMETHING_ELSE"
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()

        config = PsfMatchTask.ConfigClass()
        config.badMaskPlanes.append(config.lineMaskPlane)
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()

    def testBadLineMaskStatus(self):
        config = PsfMatchTask.ConfigClass()
        config.lineMaskStatus = ["NOT_A_FLAG"]
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
