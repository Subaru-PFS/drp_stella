import pickle

import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.ip.diffim.psfMatch import NoKernelCandidatesError

from pfs.drp.stella.psfMatch import PsfMatchTask, peakSignalToNoise, subtractRunningMedian, EXCLUDED_VARIANCE
from pfs.drp.stella.psfMatch import calculateFirstMoments, measureOffset, offsetKernel, shiftMaskedImage
from pfs.drp.stella.psfMatch import calculateWarpShifts, warpMaskedImage
from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticArc, addNoiseToImage
from pfs.drp.stella.synthetic import makeSpectrumImage, makeSyntheticDetectorMap, makeSyntheticPfsConfig
from pfs.drp.stella.tests.utils import runTests
from pfs.drp.stella import SplinedDetectorMap
from pfs.datamodel import FiberStatus, TargetType

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


def calculateKernelSums(kernel, synth):
    """Return the kernel sum evaluated over a grid of positions

    Parameters
    ----------
    kernel : `lsst.afw.math.Kernel`
        Spatially varying kernel.
    synth : `pfs.drp.stella.synthetic.SyntheticConfig`
        Synthetic spectrograph configuration, providing the image size.

    Returns
    -------
    sums : `numpy.ndarray` of `float`
        Kernel sums.
    """
    image = afwImage.ImageD(kernel.getDimensions())
    xPositions = np.linspace(48, synth.width - 48, 4)
    yPositions = np.linspace(48, synth.height - 48, 4)
    return np.array([kernel.computeImage(image, False, xx, yy) for xx in xPositions for yy in yPositions])


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
        # The synthetic lines are too dense for the running median to find the continuum, and there's no
        # continuum to remove anyway
        self.config.doContinuumFilter = False
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

    def testExcludeMaskPlanes(self):
        """Pixels with excludeMaskPlanes set are excluded, not their stamps"""
        baseline = self.task.run(self.sourceExposure, self.targetExposure, seed=1)
        self.assertIsNone(baseline.selection.excludedFraction)

        bbox = self.sourceExposure.getBBox()
        mask = self.sourceExposure.mask
        crBit = mask.getPlaneBitMask("CR")
        rng = np.random.RandomState(98765)
        xCr = rng.randint(20, bbox.getWidth() - 20, size=20)
        yCr = rng.randint(20, bbox.getHeight() - 20, size=20)
        mask.array[yCr, xCr] |= crBit
        self.sourceExposure.image.array[yCr, xCr] += 1.0e5  # Make them obvious

        exclusion = self.task._makeMaskExclusion(
            bbox, self.sourceExposure.maskedImage, self.targetExposure.maskedImage
        )
        radius = self.config.kernel["DF"].kernelSize // 2
        expected = np.zeros((bbox.getHeight(), bbox.getWidth()), dtype=bool)
        for xx, yy in zip(xCr, yCr):
            expected[yy - radius : yy + radius + 1, xx - radius : xx + radius + 1] = True
        self.assertTrue(np.array_equal(exclusion, expected))

        result = self.task.run(self.sourceExposure, self.targetExposure, seed=1)
        self.assertEqual(result.selection.numRejectedMask, baseline.selection.numRejectedMask)
        self.assertFloatsAlmostEqual(result.selection.excludedFraction, expected.mean(), atol=1.0e-12)
        sums = calculateKernelSums(result.psfMatchingKernel, self.synth)
        baselineSums = calculateKernelSums(baseline.psfMatchingKernel, self.synth)
        self.assertFloatsAlmostEqual(sums, baselineSums, rtol=0.01)

        # With CR in badMaskPlanes instead, the stamps are rejected
        self.config.excludeMaskPlanes.remove("CR")
        self.config.badMaskPlanes.append("CR")
        rejected = PsfMatchTask(config=self.config).run(self.sourceExposure, self.targetExposure, seed=1)
        self.assertGreater(rejected.selection.numRejectedMask, baseline.selection.numRejectedMask)

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
        # The synthetic lines are too dense for the running median to find the continuum, and there's no
        # continuum to remove anyway
        self.config.doContinuumFilter = False
        self.config.lineMaskRadius = 4
        self.task = PsfMatchTask(config=self.config)

    def tearDown(self):
        del self.sourceExposure
        del self.targetExposure
        del self.detectorMap
        del self.refLines
        del self.task

    def getKernelSums(self, kernel):
        """Return the kernel sum evaluated over a grid of positions"""
        return calculateKernelSums(kernel, self.synth)

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


class LineScaleTestCase(lsst.utils.tests.TestCase):
    """Test PsfMatchTask.run with config.doLineScale

    Mimics a pair of sky exposures in which the relative strengths of the
    lines differ: each line in ``target`` has its own flux ratio relative
    to ``source`` (common to all fibers), ramping along the dispersion
    direction with some line-to-line scatter. Without scaling the lines,
    the fitted kernel sum follows the ramp; with scaling, it is constant at
    the median ratio.
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
        lines = np.linspace(0, synth.height - 1, numLines + 2)[1:-1]
        self.factors = 0.7 * np.linspace(0.85, 1.15, numLines) * (1 + 0.05 * (-1) ** np.arange(numLines))

        rows = np.arange(synth.height, dtype=float)
        sigma = lineFwhm / (2 * np.sqrt(2 * np.log(2)))
        profiles = np.exp(-0.5 * ((rows[np.newaxis, :] - lines[:, np.newaxis]) / sigma) ** 2)
        profiles *= flux / (sigma * np.sqrt(2 * np.pi))
        sourceSpectrum = profiles.sum(axis=0)
        targetSpectrum = (profiles * self.factors[:, np.newaxis]).sum(axis=0)

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
            status=np.full(numLines, ReferenceLineStatus.GOOD, dtype=np.int32),
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
        # The synthetic lines are too dense for the running median to find the continuum, and there's no
        # continuum to remove anyway
        self.config.doContinuumFilter = False
        # The synthetic PSF change is large, so we need a wide box to capture all the flux
        self.config.lineRatioHalfWidth = 8
        # Without scaling, the line ratios can't be fit well, so we disable clipping of candidates, which
        # would otherwise reject them all
        self.config.kernel["DF"].singleKernelClipping = False
        self.config.kernel["DF"].kernelSumClipping = False
        self.config.kernel["DF"].spatialKernelClipping = False

    def tearDown(self):
        del self.sourceExposure
        del self.targetExposure
        del self.detectorMap
        del self.refLines

    def runTask(self, config):
        """Run PsfMatchTask with the line list"""
        return PsfMatchTask(config=config).run(
            self.sourceExposure,
            self.targetExposure,
            seed=1,
            refLines=self.refLines,
            detectorMap=self.detectorMap,
        )

    def testLineRatios(self):
        """The measured line ratios match the input"""
        sourceCopy = self.sourceExposure.maskedImage.clone()
        targetCopy = self.targetExposure.maskedImage.clone()
        result = self.runTask(self.config)

        ratios = result.lineRatios
        self.assertIsNotNone(ratios)
        self.assertFloatsEqual(ratios.wavelength, self.refLines.wavelength)
        self.assertTrue(np.all(ratios.good))
        self.assertTrue(np.all(ratios.signalToNoise > self.config.lineRatioMinSignalToNoise))
        self.assertFloatsAlmostEqual(ratios.ratio, self.factors, rtol=0.01)
        median = np.median(self.factors)
        self.assertFloatsAlmostEqual(ratios.median, median, rtol=0.01)
        self.assertFloatsAlmostEqual(ratios.relative, self.factors / median, rtol=0.01)

        # Inputs are not modified
        self.assertMaskedImagesEqual(self.sourceExposure.maskedImage, sourceCopy)
        self.assertMaskedImagesEqual(self.targetExposure.maskedImage, targetCopy)

    def testKernelSum(self):
        """Scaling the lines gives a constant kernel sum, equal to the median ratio"""
        median = np.median(self.factors)
        result = self.runTask(self.config)
        sums = calculateKernelSums(result.psfMatchingKernel, self.synth)
        self.assertFloatsAlmostEqual(sums, median, rtol=0.01)
        self.assertLess(np.ptp(sums), 0.01)

        self.config.doLineScale = False
        unscaled = self.runTask(self.config)
        self.assertIsNone(unscaled.lineRatios)
        sums = calculateKernelSums(unscaled.psfMatchingKernel, self.synth)
        self.assertGreater(np.ptp(sums), 0.1)

    def testNoLines(self):
        """Without a line list, no ratios are measured"""
        result = PsfMatchTask(config=self.config).run(self.sourceExposure, self.targetExposure, seed=1)
        self.assertIsNone(result.lineRatios)

    def testRejection(self):
        """Lines with low signal-to-noise or outlying ratios are rejected"""
        self.config.lineRatioMinSignalToNoise = 0.0
        task = PsfMatchTask(config=self.config)
        bbox = self.sourceExposure.getBBox()
        source = self.sourceExposure.maskedImage
        target = self.targetExposure.maskedImage.clone()

        # Make one line an outlier, by scaling the target around it
        fiberId = self.detectorMap.fiberId[len(self.detectorMap) // 2]
        outlier = 5
        row = int(self.detectorMap.findPoint(fiberId, self.refLines.wavelength[outlier]).getY() + 0.5)
        target.image.array[row - 8:row + 9] *= 3.0

        ratios = task._measureLineRatios(bbox, source, target, self.refLines, self.detectorMap)
        expected = np.ones(len(self.refLines), dtype=bool)
        expected[outlier] = False
        self.assertTrue(np.array_equal(ratios.good, expected))
        self.assertTrue(np.isnan(ratios.relative[outlier]))

        # A high signal-to-noise threshold rejects all lines
        config = PsfMatchTask.ConfigClass()
        config.lineRatioMinSignalToNoise = 1.0e10
        ratios = PsfMatchTask(config=config)._measureLineRatios(
            bbox, source, target, self.refLines, self.detectorMap
        )
        self.assertFalse(np.any(ratios.good))
        self.assertTrue(np.isnan(ratios.median))

    def testScaleLines(self):
        """Pixels near lines with good ratios are scaled; others are not"""
        task = PsfMatchTask(config=self.config)
        bbox = self.sourceExposure.getBBox()
        image = self.sourceExposure.maskedImage
        copy = image.clone()
        numLines = len(self.refLines)
        good = np.arange(numLines) != 3
        relative = np.where(good, 1.0 + 0.1 * np.arange(numLines), np.nan)
        lineRatios = pipeBase.Struct(wavelength=self.refLines.wavelength, good=good, relative=relative)

        scaled = task._scaleLines(bbox, image, lineRatios, self.detectorMap)
        self.assertMaskedImagesEqual(image, copy)  # Input not modified
        self.assertMasksEqual(scaled.mask, image.mask)

        # Expected scale for each row (traces are nearly vertical, so lines are at the same row in all fibers)
        fiberId = self.detectorMap.fiberId[len(self.detectorMap) // 2]
        lineRows = np.array([
            self.detectorMap.findPoint(fiberId, wl).getY() for wl in self.refLines.wavelength[good]
        ])
        rows = np.arange(bbox.getHeight(), dtype=float)
        distance = np.abs(rows[np.newaxis, :] - lineRows[:, np.newaxis])
        nearest = np.argmin(distance, axis=0)
        within = distance[nearest, np.arange(len(rows))] <= self.config.lineScaleHalfHeight
        expected = np.where(within, relative[good][nearest], 1.0)

        # Check a column on the central trace, away from the boundaries between scaled regions
        column = int(self.detectorMap.getXCenter(fiberId, bbox.getHeight() / 2) + 0.5)
        unambiguous = np.abs(np.abs(rows[np.newaxis, :] - lineRows[:, np.newaxis]).min(axis=0)
                             - self.config.lineScaleHalfHeight) > 2
        unambiguous &= np.isfinite(image.image.array[:, column])
        self.assertFloatsAlmostEqual(
            scaled.image.array[unambiguous, column],
            image.image.array[unambiguous, column] * expected[unambiguous],
            rtol=1.0e-6,
        )
        self.assertFloatsAlmostEqual(
            scaled.variance.array[unambiguous, column],
            image.variance.array[unambiguous, column] * expected[unambiguous] ** 2,
            rtol=1.0e-6,
        )


class ContinuumTestCase(lsst.utils.tests.TestCase):
    """Test PsfMatchTask with config.doContinuumFilter

    Mimics a pair of sky exposures in which the lines are the same, but the
    continuum is brighter in ``target``. Unless the continuum is removed
    before fitting, the fitted kernel sum is biased high.
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

        numLines = 6
        lines = np.linspace(0, synth.height - 1, numLines + 2)[1:-1]
        rows = np.arange(synth.height, dtype=float)
        sigma = 3.5 / (2 * np.sqrt(2 * np.log(2)))
        profiles = np.exp(-0.5 * ((rows[np.newaxis, :] - lines[:, np.newaxis]) / sigma) ** 2)
        lineSpectrum = profiles.sum(axis=0) * 3.0e5 / (sigma * np.sqrt(2 * np.pi))
        continuum = 2.0e4 * (1.0 + 0.5 * rows / synth.height)  # A gentle slope

        def makeTrueImage(spectrum):
            return makeSpectrumImage(
                spectrum.astype(np.float32), synth.dims, synth.traceCenters, synth.traceOffset, synth.fwhm
            )

        rng = np.random.RandomState(12345)
        sourceImage, sourceBorder = convolveAndAddNoise(
            makeTrueImage(lineSpectrum + continuum), 0.6, synth, rng
        )
        targetImage, targetBorder = convolveAndAddNoise(
            makeTrueImage(lineSpectrum + 1.5 * continuum), 1.3, synth, rng
        )
        borderMask = sourceBorder | targetBorder
        self.sourceExposure = makeExposure(sourceImage, borderMask, synth.gain, synth.readnoise)
        self.targetExposure = makeExposure(targetImage, borderMask, synth.gain, synth.readnoise)

        self.config = PsfMatchTask.ConfigClass()
        self.config.kernel["DF"].kernelSize = 15
        self.config.kernel["DF"].sizeCellX = 128
        self.config.kernel["DF"].sizeCellY = 128
        self.config.xStampSize = 48
        self.config.yStampSize = 48
        self.config.minPeakSignalToNoise = 5.0
        self.config.doContinuumFilter = False  # Enabled for specific tests
        # The continuum difference can't be fit well, so we disable clipping of candidates, which
        # otherwise rejects them all when not filtering
        self.config.kernel["DF"].singleKernelClipping = False
        self.config.kernel["DF"].kernelSumClipping = False
        self.config.kernel["DF"].spatialKernelClipping = False

    def tearDown(self):
        del self.sourceExposure
        del self.targetExposure

    def testSubtractRunningMedian(self):
        """subtractRunningMedian removes the continuum, leaving the lines"""
        image = afwImage.MaskedImageF(self.synth.width, self.synth.height)
        image.image.array[:] = 100.0 + np.arange(self.synth.width)[np.newaxis, :]  # Varies by column
        image.image.array[100:103, 50] += 1000.0  # A narrow "line"
        image.image.array[200, 60] = np.nan
        image.mask.array[:] = 0
        image.variance.array[:] = 1.0
        copy = image.clone()

        filtered = subtractRunningMedian(image, 21)
        self.assertMaskedImagesEqual(image, copy)  # Input not modified
        array = filtered.image.array
        self.assertTrue(np.isnan(array[200, 60]))
        array[200, 60] = 0.0
        expected = np.zeros_like(array)
        expected[100:103, 50] = 1000.0
        self.assertFloatsAlmostEqual(array, expected, atol=1.0e-3)
        self.assertImagesEqual(filtered.variance, image.variance)
        self.assertMasksEqual(filtered.mask, image.mask)

    def testContinuumFilter(self):
        """Fitting to continuum-subtracted images gives a nearly unbiased kernel"""
        biased = PsfMatchTask(config=self.config).run(self.sourceExposure, self.targetExposure, seed=1)
        self.assertGreater(np.mean(calculateKernelSums(biased.psfMatchingKernel, self.synth)), 1.2)

        sourceCopy = self.sourceExposure.maskedImage.clone()
        targetCopy = self.targetExposure.maskedImage.clone()
        self.config.doContinuumFilter = True
        result = PsfMatchTask(config=self.config).run(self.sourceExposure, self.targetExposure, seed=1)
        sums = calculateKernelSums(result.psfMatchingKernel, self.synth)
        # The running median is biased slightly by the wings of the lines, which are broader in the target
        # than in the source, so we get a few percent bias.
        self.assertFloatsAlmostEqual(sums, 1.0, atol=0.05)

        # Inputs are not modified
        self.assertMaskedImagesEqual(self.sourceExposure.maskedImage, sourceCopy)
        self.assertMaskedImagesEqual(self.targetExposure.maskedImage, targetCopy)


class FiberExclusionTestCase(lsst.utils.tests.TestCase):
    """Test PsfMatchTask.run with pfsConfigs, excluding masked fibers

    Mimics a pair of exposures in which the spectra of bright objects
    differ: ``source`` and ``target`` contain the same spectrum (continuum
    and lines) in each fiber, except that the ``FLUXSTD`` fibers (which are
    masked by default) are brighter in ``target``. Unless these are
    excluded, the fitted kernel sum is biased high. The ``SKY`` and
    ``SCIENCE`` fibers (which are not masked by default) are unchanged.
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

        self.pfsConfig = makeSyntheticPfsConfig(synth, 123, 456)
        index = np.arange(synth.numFibers)
        self.isBright = index % 3 == 1
        self.pfsConfig.targetType[:] = np.where(
            self.isBright, TargetType.FLUXSTD, np.where(index % 3 == 0, TargetType.SKY, TargetType.SCIENCE)
        )
        self.pfsConfig.fiberStatus[:] = FiberStatus.GOOD

        numLines = 12
        lines = np.linspace(0, synth.height - 1, numLines + 2)[1:-1]
        rows = np.arange(synth.height, dtype=float)
        sigma = 3.5 / (2 * np.sqrt(2 * np.log(2)))
        profiles = np.exp(-0.5 * ((rows[np.newaxis, :] - lines[:, np.newaxis]) / sigma) ** 2)
        spectrum = 1000.0 + profiles.sum(axis=0) * 3.0e5 / (sigma * np.sqrt(2 * np.pi))

        def makeTrueImage(select):
            return makeSpectrumImage(
                spectrum.astype(np.float32),
                synth.dims,
                synth.traceCenters[select],
                synth.traceOffset,
                synth.fwhm,
            )

        faintImage = makeTrueImage(~self.isBright)
        brightImage = makeTrueImage(self.isBright)
        trueSource = faintImage.clone()
        trueSource += brightImage
        trueTarget = faintImage.clone()
        trueTarget.scaledPlus(1.5, brightImage)

        rng = np.random.RandomState(12345)
        sourceImage, sourceBorder = convolveAndAddNoise(trueSource, 0.6, synth, rng)
        targetImage, targetBorder = convolveAndAddNoise(trueTarget, 1.3, synth, rng)
        borderMask = sourceBorder | targetBorder
        self.sourceExposure = makeExposure(sourceImage, borderMask, synth.gain, synth.readnoise)
        self.targetExposure = makeExposure(targetImage, borderMask, synth.gain, synth.readnoise)

        self.config = PsfMatchTask.ConfigClass()
        self.config.kernel["DF"].kernelSize = 15
        self.config.kernel["DF"].sizeCellX = 128
        self.config.kernel["DF"].sizeCellY = 128
        self.config.xStampSize = 48
        self.config.yStampSize = 48
        self.config.minPeakSignalToNoise = 5.0
        # The synthetic lines are too dense for the running median to find the continuum, and there's no
        # continuum to remove anyway
        self.config.doContinuumFilter = False
        self.config.fiberMaskRadius = 4
        self.task = PsfMatchTask(config=self.config)
        self.radius = self.config.fiberMaskRadius + self.config.kernel["DF"].kernelSize // 2

    def tearDown(self):
        del self.sourceExposure
        del self.targetExposure
        del self.detectorMap
        del self.pfsConfig
        del self.task

    def calculateExpected(self, fiberId):
        """Calculate the expected excluded pixels

        This is an independent implementation of the exclusion: integer
        pixels within ``self.radius`` (in x) of the trace center, rounded
        to the nearest pixel, in each row.

        Parameters
        ----------
        fiberId : iterable of `int`
            Fibers to exclude.

        Returns
        -------
        expected : `numpy.ndarray` of `bool`
            Pixels expected to be excluded.
        """
        expected = np.zeros((self.synth.height, self.synth.width), dtype=bool)
        for ff in fiberId:
            for yy in range(self.synth.height):
                center = int(np.floor(self.detectorMap.getXCenter(ff, float(yy)) + 0.5))
                for xx in range(center - self.radius, center + self.radius + 1):
                    if 0 <= xx < self.synth.width:
                        expected[yy, xx] = True
        return expected

    def assertExclusion(self, pfsConfigList, excludeFiberId):
        """Assert that _makeFiberExclusion excludes the expected fibers

        Parameters
        ----------
        pfsConfigList : iterable of `pfs.datamodel.PfsConfig`
            Top-end configurations to provide to ``_makeFiberExclusion``.
        excludeFiberId : iterable of `int`
            Fibers expected to be excluded.
        """
        bbox = self.sourceExposure.getBBox()
        result = self.task._makeFiberExclusion(bbox, self.detectorMap, pfsConfigList)
        self.assertEqual(result.numFibers, len(set(excludeFiberId)))
        expected = self.calculateExpected(excludeFiberId)
        self.assertTrue(np.array_equal(result.exclude, expected))
        self.assertFloatsAlmostEqual(result.fraction, expected.mean(), atol=1.0e-12)

        # The traces of the fibers that are used aren't excluded
        for ff in set(self.detectorMap.fiberId) - set(excludeFiberId):
            center = np.floor(self.detectorMap.getXCenter(ff) + 0.5).astype(int)
            self.assertFalse(np.any(result.exclude[np.arange(self.synth.height), center]))

    def testDefaults(self):
        """By default, FLUXSTD fibers are masked, whatever their status"""
        config = PsfMatchTask.ConfigClass()
        self.assertEqual(list(config.maskFibers.targetType), ["FLUXSTD"])
        self.assertEqual(list(config.maskFibers.fiberStatus), [])

    def testMakeFiberExclusion(self):
        """Excluded pixels follow the traces of the masked fibers"""
        bright = self.pfsConfig.fiberId[self.isBright]
        self.assertExclusion([self.pfsConfig], bright)

        # Masking SCIENCE fibers as well
        self.config.maskFibers.targetType = ["FLUXSTD", "SCIENCE"]
        self.task = PsfMatchTask(config=self.config)
        notSky = self.pfsConfig.fiberId[self.pfsConfig.targetType != TargetType.SKY]
        self.assertExclusion([self.pfsConfig], notSky)

    def testBothPfsConfigs(self):
        """Fibers masked in either pfsConfig are excluded"""
        bright = self.pfsConfig.fiberId[self.isBright]
        isScience = self.pfsConfig.targetType == TargetType.SCIENCE
        isSky = self.pfsConfig.targetType == TargetType.SKY
        sourceStar = self.pfsConfig.fiberId[isScience][0]
        targetStar = self.pfsConfig.fiberId[isSky][0]
        broken = self.pfsConfig.fiberId[isScience][1]

        # A FLUXSTD in the source, and a broken fiber (which isn't masked)
        targetType = self.pfsConfig.targetType.copy()
        targetType[self.pfsConfig.fiberId == sourceStar] = TargetType.FLUXSTD
        fiberStatus = self.pfsConfig.fiberStatus.copy()
        fiberStatus[self.pfsConfig.fiberId == broken] = FiberStatus.BROKENFIBER
        sourcePfsConfig = self.pfsConfig.copy(targetType=targetType, fiberStatus=fiberStatus)

        # A FLUXSTD in the target, which isn't GOOD (but is masked anyway)
        targetType = self.pfsConfig.targetType.copy()
        targetType[self.pfsConfig.fiberId == targetStar] = TargetType.FLUXSTD
        fiberStatus = self.pfsConfig.fiberStatus.copy()
        fiberStatus[self.pfsConfig.fiberId == targetStar] = FiberStatus.NOTCONVERGED
        targetPfsConfig = self.pfsConfig.copy(targetType=targetType, fiberStatus=fiberStatus)

        self.assertExclusion([sourcePfsConfig], np.append(bright, sourceStar))
        self.assertExclusion([targetPfsConfig], np.append(bright, targetStar))
        self.assertExclusion([sourcePfsConfig, targetPfsConfig], np.append(bright, [sourceStar, targetStar]))

        result = self.task.run(
            self.sourceExposure,
            self.targetExposure,
            seed=1,
            detectorMap=self.detectorMap,
            sourcePfsConfig=sourcePfsConfig,
            targetPfsConfig=targetPfsConfig,
        )
        self.assertEqual(result.selection.numFibersExcluded, len(bright) + 2)

    def testMissingFibers(self):
        """Fibers in the detectorMap but not in a pfsConfig aren't excluded"""
        missing = self.pfsConfig.fiberId[[0, 1]]
        self.assertTrue(self.isBright[1])  # So we're dropping a FLUXSTD
        keep = ~np.isin(self.pfsConfig.fiberId, missing)
        pfsConfig = self.pfsConfig.select(fiberId=self.pfsConfig.fiberId[keep])
        bright = self.pfsConfig.fiberId[self.isBright]
        self.assertExclusion([pfsConfig], bright[bright != missing[1]])
        # The other pfsConfig still identifies it as FLUXSTD
        self.assertExclusion([self.pfsConfig, pfsConfig], bright)

    def testExclusion(self):
        """Excluding the bright fibers gives an unbiased kernel"""
        sourceCopy = self.sourceExposure.maskedImage.clone()
        targetCopy = self.targetExposure.maskedImage.clone()

        result = self.task.run(
            self.sourceExposure,
            self.targetExposure,
            seed=1,
            detectorMap=self.detectorMap,
            sourcePfsConfig=self.pfsConfig,
            targetPfsConfig=self.pfsConfig,
        )
        self.assertEqual(result.selection.numFibersExcluded, np.sum(self.isBright))
        self.assertIsNone(result.selection.numLinesExcluded)
        self.assertGreater(result.selection.excludedFraction, 0.0)
        self.assertLess(result.selection.excludedFraction, 0.6)
        self.assertGreater(result.selection.numUsed, 0)
        sums = calculateKernelSums(result.psfMatchingKernel, self.synth)
        self.assertFloatsAlmostEqual(sums, 1.0, atol=0.02)

        # Inputs are not modified
        self.assertMaskedImagesEqual(self.sourceExposure.maskedImage, sourceCopy)
        self.assertMaskedImagesEqual(self.targetExposure.maskedImage, targetCopy)

        # Without the exclusion, the brighter objects bias the kernel sum high. We disable clipping of
        # candidates, since otherwise many candidates are rejected (because they can't be fit well).
        self.config.kernel["DF"].singleKernelClipping = False
        self.config.kernel["DF"].kernelSumClipping = False
        self.config.kernel["DF"].spatialKernelClipping = False
        task = PsfMatchTask(config=self.config)
        biased = task.run(self.sourceExposure, self.targetExposure, seed=1)
        self.assertIsNone(biased.selection.numFibersExcluded)
        self.assertIsNone(biased.selection.excludedFraction)
        self.assertGreater(np.mean(calculateKernelSums(biased.psfMatchingKernel, self.synth)), 1.05)

    def testMissingDetectorMap(self):
        """Providing a pfsConfig without a detectorMap raises"""
        with self.assertRaises(ValueError):
            self.task.run(self.sourceExposure, self.targetExposure, sourcePfsConfig=self.pfsConfig)
        with self.assertRaises(ValueError):
            self.task.run(self.sourceExposure, self.targetExposure, targetPfsConfig=self.pfsConfig)


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


class OffsetTestCase(lsst.utils.tests.TestCase):
    """Test removing an integer offset before fitting the kernel

    The target is the source (convolved to a broader PSF) shifted by more
    than a 9x9 kernel can span on its own.
    """

    def setUp(self):
        synth = SyntheticConfig()
        synth.width = 512
        synth.height = 512
        synth.separation = 20.0
        synth.fwhm = 3.0
        synth.gain = 1.5
        synth.readnoise = 3.0
        self.synth = synth
        self.offset = (2, -7)  # Offset of target relative to source

        rng = np.random.RandomState(12345)
        # Two sets of evenly-spaced lines, so the pattern isn't periodic over the offsets considered
        trueImage = makeSyntheticArc(synth, numLines=60, fwhm=3.5, flux=3.0e5, addNoise=False).image
        trueImage += makeSyntheticArc(synth, numLines=23, fwhm=3.5, flux=2.0e5, addNoise=False).image
        sourceImage = afwImage.ImageF(trueImage.getBBox())
        broadImage = afwImage.ImageF(trueImage.getBBox())
        convolutionControl = afwMath.ConvolutionControl()
        convolutionControl.setDoNormalize(True)
        afwMath.convolve(sourceImage, trueImage, makeGaussianKernel(0.6), convolutionControl)
        afwMath.convolve(broadImage, trueImage, makeGaussianKernel(1.3), convolutionControl)
        sourceImage.array[~np.isfinite(sourceImage.array)] = 0.0
        broadImage.array[~np.isfinite(broadImage.array)] = 0.0

        # target(x + offset) = broad(x)
        dx, dy = self.offset
        targetImage = afwImage.ImageF(trueImage.getBBox())
        targetImage.array[:] = np.roll(broadImage.array, (dy, dx), axis=(0, 1))

        borderMask = np.zeros(sourceImage.array.shape, dtype=bool)
        border = 7 + max(abs(dx), abs(dy))
        borderMask[:border] = True
        borderMask[-border:] = True
        borderMask[:, :border] = True
        borderMask[:, -border:] = True

        addNoiseToImage(sourceImage, synth.gain, synth.readnoise, rng)
        addNoiseToImage(targetImage, synth.gain, synth.readnoise, rng)
        self.sourceExposure = makeExposure(sourceImage, borderMask, synth.gain, synth.readnoise)
        self.targetExposure = makeExposure(targetImage, borderMask, synth.gain, synth.readnoise)

        self.config = PsfMatchTask.ConfigClass()
        self.config.kernel["DF"].kernelSize = 9
        self.config.kernel["DF"].sizeCellX = 128
        self.config.kernel["DF"].sizeCellY = 128
        self.config.xStampSize = 48
        self.config.yStampSize = 48
        self.config.doContinuumFilter = False  # Lines are too dense; see PsfMatchTestCase

    def tearDown(self):
        del self.sourceExposure
        del self.targetExposure

    def testMeasureOffset(self):
        """measureOffset finds the offset of the target"""
        result = measureOffset(self.sourceExposure.maskedImage, self.targetExposure.maskedImage, 10, 10)
        self.assertEqual((result.dx, result.dy), self.offset)
        self.assertGreater(result.peak, 0.5)

        # Restricted to smaller offsets, it can't
        result = measureOffset(self.sourceExposure.maskedImage, self.targetExposure.maskedImage, 3, 3)
        self.assertNotEqual((result.dx, result.dy), self.offset)

    def testShiftMaskedImage(self):
        """shiftMaskedImage(image, dx, dy)(x, y) = image(x + dx, y + dy)"""
        image = self.targetExposure.maskedImage
        dx, dy = self.offset
        shifted, missing = shiftMaskedImage(image, dx, dy)
        height, width = image.image.array.shape
        self.assertFloatsEqual(
            shifted.image.array[10:20, 30:40], image.image.array[10 + dy : 20 + dy, 30 + dx : 40 + dx]
        )
        self.assertEqual(missing.sum(), height * width - (height - abs(dy)) * (width - abs(dx)))
        self.assertTrue(np.all(missing[: abs(dy)]))  # dy < 0: bottom rows are missing
        self.assertTrue(np.all(missing[:, width - dx :]))  # dx > 0: right columns are missing
        self.assertTrue(np.all(shifted.mask.array[missing] & shifted.mask.getPlaneBitMask("NO_DATA")))
        self.assertTrue(np.all(shifted.image.array[missing] == 0))

    def testOffsetKernel(self):
        """offsetKernel shifts the convolution and the first moment"""
        basisList = [makeGaussianKernel(1.0, 9), makeGaussianKernel(2.0, 9)]
        spatialFunction = afwMath.PolynomialFunction2D(1)
        kernel = afwMath.LinearCombinationKernel(basisList, spatialFunction)
        kernel.setSpatialParameters([[1.0, 1.0e-3, 0.0], [0.5, 0.0, -1.0e-3]])
        dx, dy = self.offset
        offset = offsetKernel(kernel, dx, dy)
        self.assertEqual(offset.getDimensions(), geom.Extent2I(9 + 2 * abs(dx), 9 + 2 * abs(dy)))
        self.assertIs(offsetKernel(kernel, 0, 0), kernel)

        xx = np.array([50.0, 200.0])
        yy = np.array([100.0, 30.0])
        original = calculateFirstMoments(kernel, xx, yy)
        moved = calculateFirstMoments(offset, xx, yy)
        self.assertFloatsAlmostEqual(moved.kernelSum, original.kernelSum, rtol=1.0e-12)
        self.assertFloatsAlmostEqual(moved.x, original.x - dx, atol=1.0e-10)
        self.assertFloatsAlmostEqual(moved.y, original.y - dy, atol=1.0e-10)

        # Convolving with the offset kernel = convolving with the original, then shifting by the offset.
        # The spatial model is evaluated at the output pixel, which differs by the offset, so use a kernel
        # without spatial variation for this.
        kernel.setSpatialParameters([[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        offset = offsetKernel(kernel, dx, dy)
        image = self.sourceExposure.image
        convolutionControl = afwMath.ConvolutionControl()
        convolutionControl.setDoNormalize(False)
        convolved = afwImage.ImageF(image.getBBox())
        afwMath.convolve(convolved, image, kernel, convolutionControl)
        convolvedOffset = afwImage.ImageF(image.getBBox())
        afwMath.convolve(convolvedOffset, image, offset, convolutionControl)
        # convolvedOffset(x + offset) = convolved(x)
        self.assertFloatsAlmostEqual(
            convolvedOffset.array[40 + dy : 200 + dy, 40 + dx : 200 + dx],
            convolved.array[40:200, 40:200],
            rtol=1.0e-5,
        )

    def testRun(self):
        """The kernel includes the offset, and the matched image matches"""
        task = PsfMatchTask(config=self.config)
        result = task.run(self.sourceExposure, self.targetExposure, seed=1)
        self.assertEqual(result.offset, self.offset)

        dx, dy = self.offset
        xx = np.linspace(80, 432, 4)
        yy = np.linspace(80, 432, 4)
        moments = calculateFirstMoments(result.psfMatchingKernel, *np.meshgrid(xx, yy))
        # Centroid shift is the negative first moment
        self.assertFloatsAlmostEqual(moments.x, -dx, atol=0.1)
        self.assertFloatsAlmostEqual(moments.y, -dy, atol=0.1)

        select = (result.matchedExposure.mask.array == 0) & (self.targetExposure.mask.array == 0)
        select &= np.isfinite(result.matchedExposure.image.array)
        self.assertGreater(select.sum(), 0.5 * select.size)
        residual = result.matchedExposure.image.array - self.targetExposure.image.array
        chi2 = np.sum((residual[select] / np.sqrt(self.targetExposure.variance.array[select])) ** 2)
        self.assertLess(chi2 / select.sum(), 2.0)

        # Without the offset, a 9x9 kernel can't match: the fit either fails, or gets the shift wrong
        self.config.doOffset = False
        try:
            result = PsfMatchTask(config=self.config).run(self.sourceExposure, self.targetExposure, seed=1)
        except NoKernelCandidatesError:
            return
        self.assertEqual(result.offset, (0, 0))
        moments = calculateFirstMoments(result.psfMatchingKernel, *np.meshgrid(xx, yy))
        self.assertGreater(np.max(np.abs(moments.y + dy)), 1.0)


class CombOffsetTestCase(lsst.utils.tests.TestCase):
    """Test measuring the offset with a realistic comb of fibers

    The fibers are separated by 6 pixels, so the cross-correlation is nearly
    periodic in x. The comb has breaks of each kind recognized by
    `PsfMatchTask._makeOffsetWeights`: a broken fiber and an engineering
    fiber (from the pfsConfig), a fiber missing from the detectorMap (a jump
    in fiberId), and a chip gap (a larger jump in fiberId, with a gap).
    """

    def setUp(self):
        synth = SyntheticConfig()
        synth.width = 384
        synth.height = 256
        synth.fwhm = 2.5
        synth.slope = 0.0
        synth.gain = 1.5
        synth.readnoise = 3.0
        self.synth = synth
        self.offset = (2, -5)

        numFibers = 55
        centers = 12.0 + 6.0 * np.arange(numFibers)
        fiberId = 1 + np.arange(numFibers, dtype=np.int32)
        centers[45:] += 12.0  # Chip gap
        fiberId[45:] += 23
        self.broken = 10
        self.missing = 20  # Not in the detectorMap
        self.engineering = 30
        inMap = np.arange(numFibers) != self.missing
        lit = ~np.isin(np.arange(numFibers), [self.broken, self.missing, self.engineering])

        rows = np.arange(synth.height, dtype=float)
        self.detectorMap = SplinedDetectorMap(
            lsst.geom.Box2I(lsst.geom.Point2I(0, 0), synth.dims),
            fiberId[inMap],
            [rows] * inMap.sum(),
            [np.full(synth.height, cc) for cc in centers[inMap]],
            [rows] * inMap.sum(),
            [np.linspace(600.0, 900.0, synth.height)] * inMap.sum(),
        )
        synth.separation = 6.0  # Provides enough fibers for the pfsConfig
        self.pfsConfig = makeSyntheticPfsConfig(synth, 123, 456)[np.arange(numFibers)]
        self.pfsConfig.fiberId[:] = fiberId
        self.pfsConfig.fiberStatus[:] = FiberStatus.GOOD
        self.pfsConfig.targetType[:] = TargetType.SKY
        self.pfsConfig.fiberStatus[self.broken] = FiberStatus.BROKENFIBER
        self.pfsConfig.targetType[self.engineering] = TargetType.ENGINEERING

        sigma = 1.5
        lines = np.concatenate((np.linspace(10, 245, 17), np.linspace(17, 240, 7)))
        profiles = np.exp(-0.5 * ((rows[np.newaxis, :] - lines[:, np.newaxis]) / sigma) ** 2)
        spectrum = (profiles.sum(axis=0) * 2.0e4).astype(np.float32)
        trueImage = makeSpectrumImage(spectrum, synth.dims, centers[lit], np.zeros(synth.height), synth.fwhm)
        dx, dy = self.offset
        rng = np.random.RandomState(12345)
        sourceImage = trueImage.clone()
        targetImage = trueImage.clone()
        targetImage.array[:] = np.roll(trueImage.array, (dy, dx), axis=(0, 1))
        addNoiseToImage(sourceImage, synth.gain, synth.readnoise, rng)
        addNoiseToImage(targetImage, synth.gain, synth.readnoise, rng)
        borderMask = np.zeros(sourceImage.array.shape, dtype=bool)
        self.source = makeExposure(sourceImage, borderMask, synth.gain, synth.readnoise)
        self.target = makeExposure(targetImage, borderMask, synth.gain, synth.readnoise)
        self.centers = centers

        self.config = PsfMatchTask.ConfigClass()
        self.config.maxOffsetX = 8  # Includes aliases at dx +/- 6
        self.task = PsfMatchTask(config=self.config)

    def tearDown(self):
        del self.source
        del self.target
        del self.detectorMap
        del self.pfsConfig
        del self.task

    def getEdges(self, weight):
        """Return the indices of fibers with weight > 1"""
        row = self.synth.height // 2
        fiberWeight = weight[row, np.floor(self.centers + 0.5).astype(int)]
        return set(np.flatnonzero(fiberWeight > 1).tolist())

    def testWeights(self):
        """Fibers next to breaks are up-weighted, separately for each side"""
        bbox = self.source.getBBox()
        left, right = self.task._makeOffsetWeights(bbox, self.detectorMap, [self.pfsConfig])
        last = len(self.centers) - 1
        self.assertEqual(self.getEdges(left), {0, 11, 21, 31, 45})
        self.assertEqual(self.getEdges(right), {9, 19, 29, 44, last})

        # Without the pfsConfig, only the fiberId jump, the gap and the ends are recognized
        left, right = self.task._makeOffsetWeights(bbox, self.detectorMap)
        self.assertEqual(self.getEdges(left), {0, 21, 45})
        self.assertEqual(self.getEdges(right), {19, 44, last})

    def testOffset(self):
        """Up-weighting the edge fibers suppresses the aliases"""
        bbox = self.source.getBBox()
        unweighted = measureOffset(self.source.maskedImage, self.target.maskedImage, 8, 10)
        weights = self.task._makeOffsetWeights(bbox, self.detectorMap, [self.pfsConfig])
        weighted = measureOffset(self.source.maskedImage, self.target.maskedImage, 8, 10, weights=weights)
        self.assertEqual((weighted.dx, weighted.dy), self.offset)
        # The aliases are nearly as strong as the peak without the weights, and much weaker with
        self.assertGreater(unweighted.aliasRatio, 0.9)
        self.assertLess(weighted.aliasRatio, 0.65)

    def testTask(self):
        """The task uses the weights when given a detectorMap and pfsConfig"""
        bbox = self.source.getBBox()
        dx, dy = self.task._measureOffset(
            bbox, self.source.maskedImage, self.target.maskedImage, self.detectorMap, [self.pfsConfig], None
        )
        self.assertEqual((dx, dy), self.offset)
        self.assertLess(self.task.metadata["offsetAliasRatio"], 0.65)


def makeLinearDetectorMap(dims, centers, fiberId, dx, dy):
    """Make a detectorMap with straight traces, shifted by a function

    The trace of each fiber is at ``centers + dx(x, y)``, and the wavelength
    is ``600 + 0.1*(y - dy(x, y))`` nm.

    Parameters
    ----------
    dims : `lsst.geom.Extent2I`
        Dimensions of the image.
    centers : `numpy.ndarray` of `float`
        Unshifted x positions of the traces.
    fiberId : `numpy.ndarray` of `int`
        Fiber identifiers.
    dx, dy : callable
        Functions of (x, y) giving the shift.

    Returns
    -------
    detectorMap : `SplinedDetectorMap`
        Mapping of fiberId,wavelength to x,y.
    """
    rows = np.arange(dims.getY(), dtype=float)
    return SplinedDetectorMap(
        geom.Box2I(geom.Point2I(0, 0), dims),
        fiberId,
        [rows] * len(centers),
        [cc + dx(cc, rows) for cc in centers],
        [rows] * len(centers),
        [600.0 + 0.1 * (rows - dy(cc, rows)) for cc in centers],
    )


def renderLines(detectorMap, wavelengths, sigma, flux):
    """Render Gaussian lines at the positions given by a detectorMap

    Parameters
    ----------
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping of fiberId,wavelength to x,y.
    wavelengths : `numpy.ndarray` of `float`
        Wavelengths of lines (nm).
    sigma : `float`
        Gaussian sigma of the lines (pixels).
    flux : `float`
        Flux of each line.

    Returns
    -------
    image : `lsst.afw.image.ImageF`
        Image of the lines.
    """
    image = afwImage.ImageF(detectorMap.bbox)
    height, width = image.array.shape
    radius = int(6 * sigma) + 1
    norm = flux / (2 * np.pi * sigma**2)
    for fiberId in detectorMap.fiberId:
        points = detectorMap.findPoint(np.full(len(wavelengths), fiberId, dtype=np.int32), wavelengths)
        for xx, yy in points:
            if not np.isfinite(xx) or not np.isfinite(yy):
                continue
            xMin, xMax = max(int(xx) - radius, 0), min(int(xx) + radius + 1, width)
            yMin, yMax = max(int(yy) - radius, 0), min(int(yy) + radius + 1, height)
            if xMin >= xMax or yMin >= yMax:
                continue
            yIndex, xIndex = np.mgrid[yMin:yMax, xMin:xMax]
            image.array[yMin:yMax, xMin:xMax] += norm * np.exp(
                -0.5 * ((xIndex - xx) ** 2 + (yIndex - yy) ** 2) / sigma**2
            )
    return image


class WarpTestCase(lsst.utils.tests.TestCase):
    """Test warping the target onto the source before fitting the kernel

    The target has a broader PSF, and is shifted by several pixels, by an
    amount that varies over the image (as in reality for some cameras), which
    neither a 9x9 kernel nor an integer offset can represent.

    The fibers are widely separated: a realistic comb of fibers (all with the
    same spectrum) leaves the x first moment of the kernel poorly constrained
    (by about 0.1 pixel), whether or not we warp.
    """

    def setUp(self):
        self.size = 512
        self.sourceSigma = 1.3
        self.targetSigma = 1.6
        dims = geom.Extent2I(self.size, self.size)
        pitch = 16.0
        centers = 20.0 + pitch * np.arange(int((self.size - 40) / pitch))
        fiberId = 1 + np.arange(len(centers), dtype=np.int32)
        self.dx = lambda xx, yy: 2.0 + 3.5 * xx / self.size + 0.0 * yy  # noqa: E731
        self.dy = lambda xx, yy: 4.0 + 6.0 * yy / self.size + 0.0 * xx  # noqa: E731
        zero = lambda xx, yy: 0.0 * xx + 0.0 * yy  # noqa: E731
        self.sourceDetectorMap = makeLinearDetectorMap(dims, centers, fiberId, zero, zero)
        self.targetDetectorMap = makeLinearDetectorMap(dims, centers, fiberId, self.dx, self.dy)

        # Two sets of evenly-spaced lines, so the pattern isn't periodic
        wavelengths = np.concatenate((np.arange(602.0, 650.0, 2.37), np.arange(603.1, 650.0, 4.13)))
        gain = 1.5
        readnoise = 3.0
        rng = np.random.RandomState(12345)
        sourceImage = renderLines(self.sourceDetectorMap, wavelengths, self.sourceSigma, 3.0e4)
        targetImage = renderLines(self.targetDetectorMap, wavelengths, self.targetSigma, 3.0e4)
        addNoiseToImage(sourceImage, gain, readnoise, rng)
        addNoiseToImage(targetImage, gain, readnoise, rng)
        noBorder = np.zeros(sourceImage.array.shape, dtype=bool)
        self.source = makeExposure(sourceImage, noBorder, gain, readnoise)
        self.target = makeExposure(targetImage, noBorder, gain, readnoise)

        self.config = PsfMatchTask.ConfigClass()
        self.config.kernel["DF"].sizeCellX = 128
        self.config.kernel["DF"].sizeCellY = 128
        self.config.xStampSize = 48
        self.config.yStampSize = 48
        self.config.doContinuumFilter = False  # No continuum

    def tearDown(self):
        del self.source
        del self.target
        del self.sourceDetectorMap
        del self.targetDetectorMap

    def testCalculateWarpShifts(self):
        """calculateWarpShifts gives the shift of fixed fiberId,wavelength"""
        fiberId = np.repeat(self.sourceDetectorMap.fiberId[::10], 5)
        row = np.tile(np.linspace(50, 450, 5), len(self.sourceDetectorMap.fiberId[::10]))
        dx, dy = calculateWarpShifts(self.sourceDetectorMap, self.targetDetectorMap, fiberId, row)
        xx = self.sourceDetectorMap.getXCenter(fiberId.astype(np.int32), row)
        self.assertFloatsAlmostEqual(dx, self.dx(xx, row + dy), atol=1.0e-3)
        self.assertFloatsAlmostEqual(dy, self.dy(xx, row + dy), atol=1.0e-3)

    def testWarpMaskedImage(self):
        """Warping the target puts its lines where the source has them"""
        warped, missing = warpMaskedImage(
            self.target.maskedImage, self.targetDetectorMap, self.sourceDetectorMap
        )
        self.assertEqual(warped.getBBox(), self.target.getBBox())
        # The shift is 4 to 10 pixels in y and 2 to 5.5 pixels in x, so the top rows and the right
        # columns have no data. The warping kernel also needs a few pixels on each side.
        self.assertTrue(np.all(missing[-12:]))
        self.assertTrue(np.all(missing[:, -8:]))
        self.assertFalse(np.any(missing[3:-15, 3:-12]))
        noData = warped.mask.getPlaneBitMask("NO_DATA")
        self.assertTrue(np.all(warped.mask.array[missing] == noData))
        self.assertTrue(np.all(warped.variance.array[missing] == EXCLUDED_VARIANCE))
        self.assertTrue(np.all(warped.image.array[missing] == 0.0))
        # Bright pixels coincide: the correlation of the warped target with the source is high
        good = ~missing
        corr = np.corrcoef(warped.image.array[good], self.source.image.array[good])[0, 1]
        self.assertGreater(corr, 0.9)

    def testRun(self):
        """The kernel represents only the PSF change"""
        task = PsfMatchTask(config=self.config)
        with self.assertRaises(ValueError):
            task.run(self.source, self.target, seed=1, sourceDetectorMap=self.sourceDetectorMap)
        result = task.run(
            self.source,
            self.target,
            seed=1,
            sourceDetectorMap=self.sourceDetectorMap,
            targetDetectorMap=self.targetDetectorMap,
        )
        self.assertIsNotNone(result.warpedTarget)
        self.assertEqual(result.offset, (0, 0))

        xx, yy = np.meshgrid(np.linspace(80, 432, 4), np.linspace(80, 400, 4))
        moments = calculateFirstMoments(result.psfMatchingKernel, xx, yy)
        self.assertFloatsAlmostEqual(moments.x, 0.0, atol=0.05)
        self.assertFloatsAlmostEqual(moments.y, 0.0, atol=0.05)

        # Second moment of the kernel is the difference of the PSF second moments
        expectVariance = self.targetSigma**2 - self.sourceSigma**2
        image = afwImage.ImageD(result.psfMatchingKernel.getDimensions())
        for xPos, yPos in zip(xx.flatten(), yy.flatten()):
            result.psfMatchingKernel.computeImage(image, True, float(xPos), float(yPos))
            array = image.array
            indices = np.arange(array.shape[0]) - array.shape[0] // 2
            xVariance = np.sum(array.sum(axis=0) * indices**2)
            yVariance = np.sum(array.sum(axis=1) * indices**2)
            self.assertFloatsAlmostEqual(xVariance, expectVariance, rtol=0.15)
            self.assertFloatsAlmostEqual(yVariance, expectVariance, rtol=0.15)

        # The matched exposure matches the warped target
        warped = result.warpedTarget
        select = (result.matchedExposure.mask.array == 0) & (warped.mask.array == 0)
        select &= np.isfinite(result.matchedExposure.image.array)
        self.assertGreater(select.sum(), 0.5 * select.size)
        residual = result.matchedExposure.image.array - warped.image.array
        chi2 = np.sum((residual[select] / np.sqrt(warped.variance.array[select])) ** 2)
        self.assertLess(chi2 / select.sum(), 2.0)


class ConfigTestCase(lsst.utils.tests.TestCase):
    """Test PsfMatchConfig validation"""

    def testDefault(self):
        config = PsfMatchTask.ConfigClass()
        config.validate()
        self.assertIn(config.lineMaskPlane, config.kernel.active.badMaskPlanes)
        self.assertTrue(config.doContinuumFilter)
        self.assertTrue(config.doLineScale)

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

    def testBadExcludeMaskPlanes(self):
        config = PsfMatchTask.ConfigClass()
        config.badMaskPlanes.append("CR")
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()

        config = PsfMatchTask.ConfigClass()
        config.excludeMaskPlanes.append(config.lineMaskPlane)
        with self.assertRaises(pexConfig.FieldValidationError):
            config.validate()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
