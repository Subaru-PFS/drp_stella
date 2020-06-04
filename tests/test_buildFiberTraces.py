import matplotlib
matplotlib.use("Agg")  # noqa E402: disable showing plots
import matplotlib.pyplot as plt

import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils

from pfs.drp.stella.synthetic import (makeSpectrumImage, SyntheticConfig, makeSyntheticDetectorMap,
                                      makeSyntheticPfsConfig)
from pfs.drp.stella.buildFiberTraces import BuildFiberTracesTask
from pfs.drp.stella.tests.utils import runTests, methodParameters

display = None


class FiberTraceSetTestCase(lsst.utils.tests.TestCase):
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

        self.detMap = makeSyntheticDetectorMap(self.synth)

        self.config = BuildFiberTracesTask.ConfigClass()
        self.config.pruneMinLength = int(0.9*self.synth.height)
        self.config.profileSwath = 80  # Produces 5 swaths
        self.config.centerFit.order = 2
        self.task = BuildFiberTracesTask(config=self.config)
        self.task.log.setLevel(self.task.log.DEBUG)

    def tearDown(self):
        del self.image
        del self.detMap
        del self.task

    def assertFiberTraces(self, fiberTraces, image=None, badBitMask=0, doCheckSpectra=True):
        """Assert that fiberTraces are as expected

        We use the fiberTrace to extract the spectra, check that the spectra
        have the correct values, and then subtract the spectra from the 2D
        image; the resulting image should be close to zero.

        Parameters
        ----------
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces to check.
        image : `lsst.afw.image.Image`, optional
            Image containing the traces. If not provided, we'll use
            ``self.image``.
        badBitMask : `int`, optional
            Bitmask to use to ignore anomalous pixels.
        doCheckSpectra : `bool`, optional
            Check the spectra?
        """
        if image is None:
            image = self.image
        image = image.clone()  # We're going to modify it
        self.assertEqual(len(fiberTraces), self.synth.numFibers)
        self.assertFloatsEqual([tt.fiberId for tt in fiberTraces], self.synth.fiberId)
        spectra = fiberTraces.extractSpectra(image, badBitMask)
        model = spectra.makeImage(image.getBBox(), fiberTraces)
        image -= model
        select = (image.mask.array & badBitMask) == 0
        chi2 = np.sum(image.image.array[select]**2/image.variance.array[select])
        dof = select.sum()  # Minus something from the model, which we'll ignore
        self.assertLess(chi2/dof, 0.5)  # Value chosen to reflect current state, and is < 1 which is good
        if not doCheckSpectra:
            return
        for ss in spectra:
            self.assertFloatsAlmostEqual(ss.flux, self.flux, rtol=1.5e-2)  # With bad pix, can get 1.1% diff

    def assertCenters(self, centers):
        """Assert that the centers are as expected

        We check that the center values that are produced match what went into
        the image.

        Parameters
        ----------
        centers : iterable of callables
            Functions that will produce the center of the trace for each row.
        """
        self.assertEqual(len(centers), self.synth.numFibers)
        rows = np.arange(self.synth.height, dtype=float)
        for cen, fiberId in zip(centers, self.synth.fiberId):
            self.assertFloatsAlmostEqual(cen(rows), self.detMap.getXCenter(fiberId), atol=5.0e-2)

    def assertProfiles(self, profiles):
        """Assert that the profiles are as expected

        We check the number, and that the plotting function works. We leave
        checking the correctness of the values to ``assertFiberTraces``.

        Parameters
        ----------
        profiles : iterable of `pfs.drp.stella.FiberProfile`
            Profiles of each fiber.
        """
        self.assertEqual(len(profiles), self.synth.numFibers)
        for prof in profiles:
            prof.plot()
        plt.close("all")

    def assertNumTraces(self, result, numTraces=None):
        """Assert that the results have the correct number of traces

        Parameters
        ----------
        result : `lsst.pipe.base.Struct`
            Result from calling ``BuildFiberTracesTask.buildFiberTraces``.
        numTraces : `int`, optional
            Number of traces expected. Defaults to the number used to build the
            image.
        """
        if numTraces is None:
            numTraces = self.synth.numFibers
        self.assertEqual(len(result.fiberTraces), numTraces)
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
        results = self.task.buildFiberTraces(self.image, self.detMap)
        self.assertFiberTraces(results.fiberTraces)
        self.assertCenters(results.centers)
        self.assertProfiles(results.profiles)

    @methodParameters(oversample=(5, 7, 10, 15, 20))
    def testOversample(self, oversample):
        """Test different oversample factors

        Making it too small results in bad subtractions.
        """
        self.config.profileOversample = oversample
        results = self.task.buildFiberTraces(self.image, self.detMap)
        self.assertFiberTraces(results.fiberTraces)
        self.assertCenters(results.centers)
        self.assertProfiles(results.profiles)

    def testRunMethod(self):
        """Test that the 'run' method works

        That allows us to use this to retarget FindAndTraceAperturesTask.

        We only get the fiberTraces: no profiles or centers.
        """
        fiberTraces = self.task.run(self.image, self.detMap)
        self.assertFiberTraces(fiberTraces)

    def testNonBlind(self):
        """Test that the non-blind method of finding traces works"""
        rng = np.random.RandomState(12345)
        pfsConfig = makeSyntheticPfsConfig(self.synth, 123456789, 54321, rng, fracSky=0.0, fracFluxStd=0.0)
        self.config.doBlindFind = False
        result = self.task.buildFiberTraces(self.image, self.detMap, pfsConfig)
        self.assertFiberTraces(result.fiberTraces)
        self.assertCenters(result.centers)
        self.assertProfiles(result.profiles)

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
        result = self.task.buildFiberTraces(image, self.detMap)
        self.assertFiberTraces(result.fiberTraces, image, image.mask.getPlaneBitMask("CR"))

        # Repeat without masking the CR pixels
        image.mask.array[yy, xx] &= ~image.mask.getPlaneBitMask("CR")
        result = self.task.buildFiberTraces(image, self.detMap)
        image.mask.array[yy, xx] |= image.mask.getPlaneBitMask("CR")  # Reinstate mask for performance measure
        self.assertFiberTraces(result.fiberTraces, image, image.mask.getPlaneBitMask("CR"))

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

        result = self.task.buildFiberTraces(image, self.detMap)
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
        result = self.task.buildFiberTraces(self.image, self.detMap)
        self.assertNumTraces(result, 0)

        # Set pruneMinLength to something smaller than the trace length, and they should all be there again
        self.config.pruneMinLength = int(0.9*(stop - start))
        result = self.task.buildFiberTraces(self.image, self.detMap)
        self.assertNumTraces(result)

    def testBadRows(self):
        """Test that we can deal with a bad row or block of rows"""
        start = 100
        stop = 123
        self.image.image.array[start:stop] = np.nan
        self.image.mask.array[start:stop] = self.image.mask.getPlaneBitMask("BAD")
        result = self.task.buildFiberTraces(self.image, self.detMap)
        self.assertFiberTraces(result.fiberTraces, self.image, self.image.mask.getPlaneBitMask("BAD"),
                               doCheckSpectra=False)

    def testBadColumn(self):
        """Test that we can deal with a bad column"""
        for xx in self.synth.traceCenters.astype(np.int) - 2:
            self.image.image.array[:, xx] = -123.45
            self.image.mask.array[:, xx] = self.image.mask.getPlaneBitMask("BAD")
        result = self.task.buildFiberTraces(self.image, self.detMap)
        # We're not going to have good profiles out of this, so not using self.assertFiberTraces
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
        result = self.task.buildFiberTraces(image, self.detMap)
        # The CR damages the profile in this small image; so just care about the number of traces
        self.assertNumTraces(result)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
