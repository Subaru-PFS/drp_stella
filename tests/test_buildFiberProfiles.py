import matplotlib
matplotlib.use("Agg")  # noqa E402: disable showing plots
import matplotlib.pyplot as plt

import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils

from pfs.drp.stella.synthetic import (makeSpectrumImage, SyntheticConfig, makeSyntheticDetectorMap,
                                      makeSyntheticPfsConfig)
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.images import getIndices
from pfs.drp.stella.tests.utils import runTests, methodParameters

display = None


class BuildFiberProfilesTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Construct the basis for testing

        This builds a small image with five traces, with a mild linear slope.
        """
        self.synth = SyntheticConfig()
        self.synth.height = 256
        self.synth.width = 128
        self.synth.separation = 19.876
        self.synth.fwhm = 3.21
        self.synth.slope = 0.01

        self.flux = 1.0e5

        image = makeSpectrumImage(self.flux, self.synth.dims, self.synth.traceCenters,
                                  self.synth.traceOffset, self.synth.fwhm)
        mask = lsst.afw.image.Mask(image.getBBox())
        mask.set(0)
        variance = lsst.afw.image.ImageF(image.getBBox())
        variance.array[:] = image.array/self.synth.gain + self.synth.readnoise**2
        self.image = lsst.afw.image.makeMaskedImage(image, mask, variance)

        self.darkTime = 123.456
        self.metadata = ("FOO", "BAR")
        self.visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        self.exposure = self.makeExposure()

        self.detMap = makeSyntheticDetectorMap(self.synth)

        self.config = BuildFiberProfilesTask.ConfigClass()
        self.config.pruneMinLength = int(0.9*self.synth.height)
        self.config.profileSwath = 80  # Produces 5 swaths
        self.config.centerFit.order = 2
        self.task = BuildFiberProfilesTask(config=self.config)
        self.task.log.setLevel(self.task.log.DEBUG)

    def tearDown(self):
        del self.image
        del self.detMap
        del self.task

    def makeExposure(self, image=None):
        """Create an Exposure suitable as input to BuildFiberProfilesTask

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`, optional
            Image to use as the basis of the exposure. If not provided, will
            use ``self.image``.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Exposure with image, visitInfo and metadata set.
        """
        if image is None:
            image = self.image
        exposure = lsst.afw.image.makeExposure(image)
        exposure.getMetadata().set(*self.metadata)
        exposure.getInfo().setVisitInfo(self.visitInfo)
        return exposure

    def assertFiberProfiles(self, fiberProfiles, image=None, badBitMask=0, doCheckSpectra=True):
        """Assert that fiberProfiles are as expected

        We use the fiberProfile to extract the spectra, check that the spectra
        have the correct values, and then subtract the spectra from the 2D
        image; the resulting image should be close to zero.

        Parameters
        ----------
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Fiber profiles to check.
        image : `lsst.afw.image.Image`, optional
            Image containing the traces. If not provided, we'll use
            ``self.image``.
        badBitMask : `int`, optional
            Bitmask to use to ignore anomalous pixels.
        doCheckSpectra : `bool`, optional
            Check the spectra?
        """
        self.assertEqual(fiberProfiles.getVisitInfo().getDarkTime(), self.darkTime)
        self.assertEqual(fiberProfiles.visitInfo.getDarkTime(), self.darkTime)
        self.assertEqual(fiberProfiles.getMetadata().get(self.metadata[0]), self.metadata[1])
        self.assertEqual(fiberProfiles.metadata.get(self.metadata[0]), self.metadata[1])
        if image is None:
            image = self.image
        image = image.clone()  # We're going to modify it
        self.assertEqual(len(fiberProfiles), self.synth.numFibers)
        self.assertFloatsEqual(fiberProfiles.fiberId, sorted(self.synth.fiberId))

        for fiberId in fiberProfiles:
            fiberProfiles[fiberId].plot()
        plt.close("all")

        traces = fiberProfiles.makeFiberTracesFromDetectorMap(self.detMap)
        spectra = traces.extractSpectra(image, badBitMask)
        model = spectra.makeImage(image.getBBox(), traces)
        image -= model
        select = (image.mask.array & badBitMask) == 0
        chi2 = np.sum(image.image.array[select]**2/image.variance.array[select])
        dof = select.sum()  # Minus something from the model, which we'll ignore
        self.assertLess(chi2/dof, 0.5)  # Value chosen to reflect current state, and is < 1 which is good
        if not doCheckSpectra:
            return
        for ss in spectra:
            self.assertFloatsAlmostEqual(ss.flux, self.flux, rtol=3.0e-2)  # With bad pix, can get 2.7% diff

    def assertCenters(self, centers):
        """Assert that the centers are as expected

        We check that the center values that are produced match what went into
        the image.

        Parameters
        ----------
        centers : iterable of callables
            Functions that will produce the center of the trace for each row.
        """
        self.assertListEqual(sorted(centers.keys()), sorted(self.synth.fiberId))
        rows = np.arange(self.synth.height, dtype=float)
        for fiberId in centers:
            self.assertFloatsAlmostEqual(centers[fiberId](rows), self.detMap.getXCenter(fiberId), atol=5.0e-2)

    def assertNumTraces(self, result, numTraces=None):
        """Assert that the results have the correct number of traces

        Parameters
        ----------
        result : `lsst.pipe.base.Struct`
            Result from calling ``BuildFiberProfilesTask.run``.
        numTraces : `int`, optional
            Number of traces expected. Defaults to the number used to build the
            image.
        """
        if numTraces is None:
            numTraces = self.synth.numFibers
        self.assertEqual(len(result.centers), numTraces)
        self.assertEqual(len(result.profiles), numTraces)

    def testBasic(self):
        """Test basic operation

        * Right number of traces
        * Numbering of traces
        * Centers
        * Profiles are reasonable
        * Subtraction residuals are reasonable
        * Extracted spectra are as expected
        """
        results = self.task.run(self.exposure, self.detMap)
        self.assertFiberProfiles(results.profiles)
        self.assertCenters(results.centers)

    @methodParameters(oversample=(5, 7, 10, 15, 20))
    def testOversample(self, oversample):
        """Test different oversample factors

        Making it too small results in bad subtractions.
        """
        self.config.profileOversample = oversample
        results = self.task.run(self.makeExposure(self.image), self.detMap)
        self.assertFiberProfiles(results.profiles)
        self.assertCenters(results.centers)

    def testNonBlind(self):
        """Test that the non-blind method of finding traces works"""
        rng = np.random.RandomState(12345)
        pfsConfig = makeSyntheticPfsConfig(self.synth, 123456789, 54321, rng, fracSky=0.0, fracFluxStd=0.0)
        self.config.doBlindFind = False
        result = self.task.run(self.makeExposure(self.image), self.detMap, pfsConfig)
        self.assertFiberProfiles(result.profiles)
        self.assertCenters(result.centers)

    def testShortCosmics(self):
        """Test that we're robust to short cosmics

        Single pixels scattered all over should be ignored in the traces.
        """
        numCosmics = int(0.005*self.image.getWidth()*self.image.getHeight())
        cosmicValue = 10000.0
        rng = np.random.RandomState(12345)
        xx = rng.randint(0, self.image.getWidth(), numCosmics)
        yy = rng.randint(0, self.image.getHeight(), numCosmics)

        image = self.image.clone()
        image.image.array[yy, xx] += cosmicValue
        image.mask.array[yy, xx] |= image.mask.getPlaneBitMask("CR")
        result = self.task.run(self.makeExposure(image), self.detMap)
        self.assertFiberProfiles(result.profiles, image, image.mask.getPlaneBitMask("CR"))

        # Repeat without masking the CR pixels
        image.mask.array[yy, xx] &= ~image.mask.getPlaneBitMask("CR")
        result = self.task.run(self.makeExposure(image), self.detMap)
        image.mask.array[yy, xx] |= image.mask.getPlaneBitMask("CR")  # Reinstate mask for performance measure
        self.assertFiberProfiles(result.profiles, image, image.mask.getPlaneBitMask("CR"))

    @methodParameters(angle=(15, 30, 45, 60, 120, 135, 150, 165))
    def testLongCosmics(self, angle):
        """Test we can find the right number of traces in the presence of long
        cosmic rays linking traces at various angles
        """
        cosmicValue = 10000.0
        xx = np.arange(self.image.getWidth(), dtype=int)
        yy = np.rint(np.tan(np.radians(angle))*(xx - 0.5*self.image.getWidth()) +
                     0.5*self.image.getHeight()).astype(int)

        cosmics = np.zeros_like(self.image.image.array, dtype=bool)

        def setCosmics(xx, yy):
            select = (xx >= 0) & (xx < self.image.getWidth()) & (yy >= 0) & (yy < self.image.getHeight())
            cosmics[yy[select], xx[select]] = True

        setCosmics(xx, yy)
        setCosmics(xx + 1, yy)
        setCosmics(xx, yy + 1)
        setCosmics(xx - 1, yy)
        setCosmics(xx, yy - 1)

        image = self.image.clone()
        image.image.array[cosmics] += cosmicValue

        result = self.task.run(self.makeExposure(image), self.detMap)
        # The CR damages the profile in this small image; so just care about the number of traces
        self.assertNumTraces(result)

    def testShort(self):
        """Test that short traces are pruned"""
        rng = np.random.RandomState(12345)
        start = 50
        stop = 200
        self.image.image.array[:start, :] = rng.normal(0.0, self.synth.readnoise, (start, self.synth.width))
        self.image.image.array[stop:, :] = rng.normal(0.0, self.synth.readnoise,
                                                      (self.synth.height - stop, self.synth.width))
        # Expect that all traces are too short, hence pruned
        result = self.task.run(self.makeExposure(self.image), self.detMap)
        self.assertNumTraces(result, 0)

        # Set pruneMinLength to something smaller than the trace length, and they should all be there again
        self.config.pruneMinLength = int(0.9*(stop - start))
        result = self.task.run(self.makeExposure(self.image), self.detMap)
        self.assertNumTraces(result)

    def testBadRows(self):
        """Test that we can deal with a bad row or block of rows"""
        start = 100
        stop = 123
        self.image.image.array[start:stop] = np.nan
        self.image.mask.array[start:stop] = self.image.mask.getPlaneBitMask("BAD")
        result = self.task.run(self.exposure, self.detMap)
        self.assertFiberProfiles(result.profiles, self.image, self.image.mask.getPlaneBitMask("BAD"),
                                 doCheckSpectra=False)

    def testBadColumn(self):
        """Test that we can deal with a bad column"""
        for xx in self.synth.traceCenters.astype(np.int) - 2:
            self.image.image.array[:, xx] = -123.45
            self.image.mask.array[:, xx] = self.image.mask.getPlaneBitMask("BAD")
        result = self.task.run(self.exposure, self.detMap)
        # We're not going to have good profiles out of this, so not using self.assertFiberProfiles
        self.assertNumTraces(result)

    def testEarlyLongCosmic(self):
        """Test that we can deal with an early cosmic

        A cosmic ray could link all rows to a single trace before the proper
        fiber traces arise.
        """
        cosmicValue = 10000.0
        angle = 15
        xx = np.arange(self.image.getWidth(), dtype=int)
        yy = np.rint(np.tan(np.radians(angle))*(xx - 0.5*self.image.getWidth()) +
                     0.1*self.image.getHeight()).astype(int)
        crMaxRow = int(np.ceil(np.max(yy)))

        cosmics = np.zeros_like(self.image.image.array, dtype=bool)

        def setCosmics(xx, yy):
            select = (xx >= 0) & (xx < self.image.getWidth()) & (yy >= 0) & (yy < self.image.getHeight())
            cosmics[yy[select], xx[select]] = True

        setCosmics(xx, yy)

        image = self.image.clone()
        image.image.array[:crMaxRow + 1, :] = 0.0
        image.image.array[cosmics] += cosmicValue

        self.config.pruneMaxWidth = 1000  # Make sure the CR peaks are included
        self.config.pruneMinLength = self.synth.height - crMaxRow - 5  # Because we've stolen some image
        result = self.task.run(self.makeExposure(image), self.detMap)
        # The CR damages the profile in this small image; so just care about the number of traces
        self.assertNumTraces(result)

    def testSortaMaskedCosmic(self):
        """Test that a sorta-masked cosmic doesn't split a trace

        This is based on a failure mode discovered in a full fiber density sim.
        A cosmic cuts across multiple traces, but only one end is masked. That
        mask interrupts one of the traces, and the effect was that the trace
        was cut short and a new trace started, rather than continuing a single
        trace.
        """
        crValue = 1000
        colShift = 3
        maskRadius = 3
        middleRow = self.synth.height//2
        middleCol = self.synth.traceCenters[self.synth.numFibers//2]
        self.image.image.array[middleRow, :] += crValue
        xx, yy = getIndices(self.image.getBBox(), int)
        select = ((xx - middleCol - colShift)**2 + (yy - middleRow)**2) < maskRadius**2
        self.image.mask.array[select] |= self.image.mask.getPlaneBitMask("CR")

        self.config.pruneMinLength = self.synth.height//3  # So a half a trace counts as a trace
        result = self.task.run(self.makeExposure(self.image), self.detMap)
        # The CR damages the profile in this small image; so just care about the number of traces
        self.assertNumTraces(result)

    def testSpottyMaskedCosmic(self):
        """Test that a spotty-masked cosmic doesn't split a trace

        This is based on a failure mode discovered in half fiber density sims.
        A cosmic cuts across multiple traces, but is only masked between the
        traces. The mask starts immediately after the peak, and extends to a
        couple of pixels short of the next peak. The effect was that the trace
        was cut short and a new trace started, rather than continuing a single
        trace.

        We're going to take the three middle traces, and put masked pixels
        between them.
        """
        crValue = 1000  # Value to give CR pixels
        maskHalfRows = 3  # Half the number of rows to have masks
        leftOffset = 1  # Position of the left side of the CR mask relative to the trace
        rightOffset = 2  # Position of the right side of the CR mask relative to the trace

        middleTrace = self.synth.numFibers//2
        leftTrace = middleTrace - 1
        rightTrace = middleTrace + 1

        middleRow = self.synth.height//2
        rowSlice = slice(middleRow - maskHalfRows, middleRow + maskHalfRows)

        leftTracePeak = int(self.synth.traceCenters[leftTrace] + 0.5)
        middleTracePeak = int(self.synth.traceCenters[middleTrace] + 0.5)
        rightTracePeak = int(self.synth.traceCenters[rightTrace] + 0.5)

        leftSlice = slice(leftTracePeak + leftOffset, middleTracePeak - rightOffset)
        rightSlice = slice(middleTracePeak + leftOffset, rightTracePeak - rightOffset)

        for ss in (leftSlice, rightSlice):
            self.image.image.array[rowSlice, ss] += crValue
            self.image.mask.array[rowSlice, ss] |= self.image.mask.getPlaneBitMask("CR")

        self.config.pruneMinLength = self.synth.height//3  # So a half a trace counts as a trace
        result = self.task.run(self.makeExposure(self.image), self.detMap)
        # The CR damages the profile in this small image; so just care about the number of traces
        self.assertNumTraces(result)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
