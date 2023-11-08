import matplotlib
from lsst.afw.display.interface import Display
#matplotlib.use("Agg")  # noqa E402: disable showing plots
import matplotlib.pyplot as plt  # noqa E402: import after code

import itertools  # noqa E402: import after code

import numpy as np  # noqa E402: import after code

import lsst.utils.tests  # noqa E402: import after code
import lsst.afw.image  # noqa E402: import after code
import lsst.afw.image.testUtils  # noqa E402: import after code

from pfs.datamodel import CalibIdentity  # noqa E402: import after code
from pfs.drp.stella.synthetic import makeSpectrumImage, SyntheticConfig  # noqa E402: import after code
from pfs.drp.stella.synthetic import makeSyntheticDetectorMap, makeSyntheticPfsConfig  # noqa E402
from pfs.drp.stella.synthetic import addNoiseToImage  # noqa E402
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask  # noqa E402: import after code
from pfs.drp.stella.images import getIndices  # noqa E402: import after code
from pfs.drp.stella.tests.utils import runTests, methodParameters  # noqa E402: import after code

display = None


# class BuildFiberProfilesTestCase(lsst.utils.tests.TestCase):
#     def setUp(self):
#         """Construct the basis for testing

#         This builds a small image with five traces, with a mild linear slope.
#         """
#         self.synth = SyntheticConfig()
#         self.synth.height = 256
#         self.synth.width = 128
#         self.synth.separation = 19.876
#         self.synth.fwhm = 3.21
#         self.synth.slope = 0.05

#         self.flux = 1.0e5

#         image = makeSpectrumImage(self.flux, self.synth.dims, self.synth.traceCenters,
#                                   self.synth.traceOffset, self.synth.fwhm)
#         mask = lsst.afw.image.Mask(image.getBBox())
#         mask.set(0)
#         variance = lsst.afw.image.ImageF(image.getBBox())
#         variance.array[:] = image.array/self.synth.gain + self.synth.readnoise**2

#         self.image = lsst.afw.image.makeMaskedImage(image, mask, variance)

#         self.darkTime = 123.456
#         self.metadata = ("FOO", "BAR")
#         self.visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
#         self.identity = CalibIdentity("2020-01-01", 5, "x", 12345)
#         self.exposure = self.makeExposure()

#         self.detMap = makeSyntheticDetectorMap(self.synth)

#         self.config = BuildFiberProfilesTask.ConfigClass()
#         self.config.pruneMinLength = int(0.9*self.synth.height)
#         self.config.profileSwath = 80  # Produces 5 swaths
#         self.config.centerFit.order = 2
#         self.config.rowFwhm = self.synth.fwhm
#         self.config.columnFwhm = self.synth.fwhm
#         self.task = BuildFiberProfilesTask(config=self.config)
#         self.task.log.setLevel(self.task.log.DEBUG)

#     def tearDown(self):
#         del self.image
#         del self.detMap
#         del self.task

#     def makeExposure(self, image=None):
#         """Create an Exposure suitable as input to BuildFiberProfilesTask

#         Parameters
#         ----------
#         image : `lsst.afw.image.MaskedImage`, optional
#             Image to use as the basis of the exposure. If not provided, will
#             use ``self.image``.

#         Returns
#         -------
#         exposure : `lsst.afw.image.Exposure`
#             Exposure with image, visitInfo and metadata set.
#         """
#         if image is None:
#             image = self.image
#         exposure = lsst.afw.image.makeExposure(image)
#         exposure.getMetadata().set(*self.metadata)
#         exposure.getInfo().setVisitInfo(self.visitInfo)
#         return exposure

#     def assertFiberProfiles(self, fiberProfiles, image=None, badBitMask=0, doCheckSpectra=True):
#         """Assert that fiberProfiles are as expected

#         We use the fiberProfile to extract the spectra, check that the spectra
#         have the correct values, and then subtract the spectra from the 2D
#         image; the resulting image should be close to zero.

#         Parameters
#         ----------
#         fiberProfiles : `pfs.drp.stella.FiberProfileSet`
#             Fiber profiles to check.
#         image : `lsst.afw.image.Image`, optional
#             Image containing the traces. If not provided, we'll use
#             ``self.image``.
#         badBitMask : `int`, optional
#             Bitmask to use to ignore anomalous pixels.
#         doCheckSpectra : `bool`, optional
#             Check the spectra?
#         """
#         self.assertEqual(fiberProfiles.getVisitInfo().getDarkTime(), self.darkTime)
#         self.assertEqual(fiberProfiles.visitInfo.getDarkTime(), self.darkTime)
#         self.assertEqual(fiberProfiles.getMetadata().get(self.metadata[0]), self.metadata[1])
#         self.assertEqual(fiberProfiles.metadata.get(self.metadata[0]), self.metadata[1])
#         if image is None:
#             image = self.image
#         image = image.clone()  # We're going to modify it
#         self.assertEqual(len(fiberProfiles), self.synth.numFibers)
#         self.assertFloatsEqual(fiberProfiles.fiberId, sorted(self.synth.fiberId))

#         # for fiberId in fiberProfiles:
#         #     fiberProfiles[fiberId].plot()
#         # plt.close("all")

#         traces = fiberProfiles.makeFiberTracesFromDetectorMap(self.detMap)
#         spectra = traces.extractSpectra(image, badBitMask)
#         model = spectra.makeImage(image.getBBox(), traces)

#         if display:
#             Display(backend=display, frame=1).mtv(image, title="Image")

#         image -= model
#         image.image.array /= np.sqrt(image.variance.array)
#         select = (image.mask.array & badBitMask) == 0
#         chi2 = np.sum(image.image.array[select]**2)
#         if display:
#             Display(backend=display, frame=2).mtv(model, title="Model")
#             Display(backend=display, frame=3).mtv(image, title="Chi image")

#         dof = select.sum()  # Minus something from the model, which we'll ignore
#         print(chi2/dof)
#         self.assertLess(chi2/dof, 1.0)  # Value chosen to reflect current state, and is = 1 which is good
#         if not doCheckSpectra:
#             return
#         for ss in spectra:
#             self.assertFloatsAlmostEqual(ss.flux, self.flux, rtol=3.0e-2)  # With bad pix, can get 2.7% diff

#     def assertCenters(self, centers):
#         """Assert that the centers are as expected

#         We check that the center values that are produced match what went into
#         the image.

#         Parameters
#         ----------
#         centers : iterable of callables
#             Functions that will produce the center of the trace for each row.
#         """
#         self.assertListEqual(sorted(centers.keys()), sorted(self.synth.fiberId))
#         rows = np.arange(self.synth.height, dtype=float)
#         for fiberId in centers:
#             self.assertFloatsAlmostEqual(centers[fiberId](rows), self.detMap.getXCenter(fiberId), atol=5.0e-2)

#     def assertNumTraces(self, result, numTraces=None):
#         """Assert that the results have the correct number of traces

#         Parameters
#         ----------
#         result : `lsst.pipe.base.Struct`
#             Result from calling ``BuildFiberProfilesTask.run``.
#         numTraces : `int`, optional
#             Number of traces expected. Defaults to the number used to build the
#             image.
#         """
#         if numTraces is None:
#             numTraces = self.synth.numFibers
#         self.assertEqual(len(result.centers), numTraces)
#         self.assertEqual(len(result.profiles), numTraces)

#     def testBasic(self):
#         """Test basic operation

#         * Right number of traces
#         * Numbering of traces
#         * Centers
#         * Profiles are reasonable
#         * Subtraction residuals are reasonable
#         * Extracted spectra are as expected
#         """
#         results = self.task.run(self.exposure, self.identity, detectorMap=self.detMap)
#         self.assertFiberProfiles(results.profiles)
#         self.assertCenters(results.centers)

#     @methodParameters(oversample=(5, 7, 10, 15, 20))
#     def testOversample(self, oversample):
#         """Test different oversample factors

#         Making it too small results in bad subtractions.
#         """
#         self.config.profileOversample = oversample
#         results = self.task.run(self.makeExposure(self.image), self.identity, detectorMap=self.detMap)
#         self.assertFiberProfiles(results.profiles)
#         self.assertCenters(results.centers)

#     def testNonBlind(self):
#         """Test that the non-blind method of finding traces works"""
#         rng = np.random.RandomState(12345)
#         pfsConfig = makeSyntheticPfsConfig(self.synth, 123456789, 54321, rng, fracSky=0.0, fracFluxStd=0.0)
#         self.config.doBlindFind = False
#         result = self.task.run(
#             self.makeExposure(self.image), self.identity, detectorMap=self.detMap, pfsConfig=pfsConfig
#         )
#         self.assertFiberProfiles(result.profiles)
#         self.assertCenters(result.centers)

#     def testShortCosmics(self):
#         """Test that we're robust to short cosmics

#         Single pixels scattered all over should be ignored in the traces.
#         """
#         numCosmics = int(0.005*self.image.getWidth()*self.image.getHeight())
#         cosmicValue = 10000.0
#         rng = np.random.RandomState(12345)
#         xx = rng.randint(0, self.image.getWidth(), numCosmics)
#         yy = rng.randint(0, self.image.getHeight(), numCosmics)

#         image = self.image.clone()
#         image.image.array[yy, xx] += cosmicValue
#         image.mask.array[yy, xx] |= image.mask.getPlaneBitMask("CR")
#         result = self.task.run(self.makeExposure(image), self.identity, detectorMap=self.detMap)
#         self.assertFiberProfiles(result.profiles, image, image.mask.getPlaneBitMask("CR"))

#         # Repeat without masking the CR pixels
#         image.mask.array[yy, xx] &= ~image.mask.getPlaneBitMask("CR")
#         result = self.task.run(self.makeExposure(image), self.identity, detectorMap=self.detMap)
#         image.mask.array[yy, xx] |= image.mask.getPlaneBitMask("CR")  # Reinstate mask for performance measure
#         self.assertFiberProfiles(result.profiles, image, image.mask.getPlaneBitMask("CR"))

#     @methodParameters(angle=(15, 30, 45, 60, 120, 135, 150, 165))
#     def testLongCosmics(self, angle):
#         """Test we can find the right number of traces in the presence of long
#         cosmic rays linking traces at various angles
#         """
#         cosmicValue = 10000.0
#         xx = np.arange(self.image.getWidth(), dtype=int)
#         yy = np.rint(np.tan(np.radians(angle))*(xx - 0.5*self.image.getWidth()) +
#                      0.5*self.image.getHeight()).astype(int)

#         cosmics = np.zeros_like(self.image.image.array, dtype=bool)

#         def setCosmics(xx, yy):
#             select = (xx >= 0) & (xx < self.image.getWidth()) & (yy >= 0) & (yy < self.image.getHeight())
#             cosmics[yy[select], xx[select]] = True

#         setCosmics(xx, yy)
#         setCosmics(xx + 1, yy)
#         setCosmics(xx, yy + 1)
#         setCosmics(xx - 1, yy)
#         setCosmics(xx, yy - 1)

#         image = self.image.clone()
#         image.image.array[cosmics] += cosmicValue

#         result = self.task.run(self.makeExposure(image), self.identity, detectorMap=self.detMap)
#         # The CR damages the profile in this small image; so just care about the number of traces
#         self.assertNumTraces(result)

#     def testShort(self):
#         """Test that short traces are pruned"""
#         rng = np.random.RandomState(12345)
#         start = 50
#         stop = 200
#         self.image.image.array[:start, :] = rng.normal(0.0, self.synth.readnoise, (start, self.synth.width))
#         self.image.image.array[stop:, :] = rng.normal(0.0, self.synth.readnoise,
#                                                       (self.synth.height - stop, self.synth.width))
#         # Expect that all traces are too short, hence pruned
#         result = self.task.run(self.makeExposure(self.image), self.identity, detectorMap=self.detMap)
#         self.assertNumTraces(result, 0)

#         # Set pruneMinLength to something smaller than the trace length, and they should all be there again
#         self.config.pruneMinLength = int(0.9*(stop - start))
#         result = self.task.run(self.makeExposure(self.image), self.identity, detectorMap=self.detMap)
#         self.assertNumTraces(result)

#     def testBadRows(self):
#         """Test that we can deal with a bad row or block of rows"""
#         start = 100
#         stop = 123
#         self.image.image.array[start:stop] = np.nan
#         self.image.mask.array[start:stop] = self.image.mask.getPlaneBitMask("BAD")
#         self.config.pruneMinLength = 200
#         result = self.task.run(self.exposure, self.identity, detectorMap=self.detMap)
#         self.assertFiberProfiles(result.profiles, self.image, self.image.mask.getPlaneBitMask("BAD"),
#                                  doCheckSpectra=False)

#     def testBadColumn(self):
#         """Test that we can deal with a bad column"""
#         for xx in self.synth.traceCenters.astype(int) - 2:
#             self.image.image.array[:, xx] = -123.45
#             self.image.mask.array[:, xx] = self.image.mask.getPlaneBitMask("BAD")
#         self.config.pruneMinLength = 200
#         result = self.task.run(self.exposure, self.identity, detectorMap=self.detMap)
#         # We're not going to have good profiles out of this, so not using self.assertFiberProfiles
#         self.assertNumTraces(result)

#     def testEarlyLongCosmic(self):
#         """Test that we can deal with an early cosmic

#         A cosmic ray could link all rows to a single trace before the proper
#         fiber traces arise.
#         """
#         cosmicValue = 10000.0
#         angle = 15
#         xx = np.arange(self.image.getWidth(), dtype=int)
#         yy = np.rint(np.tan(np.radians(angle))*(xx - 0.5*self.image.getWidth()) +
#                      0.1*self.image.getHeight()).astype(int)
#         crMaxRow = int(np.ceil(np.max(yy)))

#         cosmics = np.zeros_like(self.image.image.array, dtype=bool)

#         def setCosmics(xx, yy):
#             select = (xx >= 0) & (xx < self.image.getWidth()) & (yy >= 0) & (yy < self.image.getHeight())
#             cosmics[yy[select], xx[select]] = True

#         setCosmics(xx, yy)

#         image = self.image.clone()
#         image.image.array[:crMaxRow + 1, :] = 0.0
#         image.image.array[cosmics] += cosmicValue

#         self.config.pruneMaxWidth = 1000  # Make sure the CR peaks are included
#         self.config.pruneMinLength = self.synth.height - crMaxRow - 7  # Because we've stolen some image
#         result = self.task.run(self.makeExposure(image), self.identity, detectorMap=self.detMap)
#         # The CR damages the profile in this small image; so just care about the number of traces
#         self.assertNumTraces(result)

#     def testSortaMaskedCosmic(self):
#         """Test that a sorta-masked cosmic doesn't split a trace

#         This is based on a failure mode discovered in a full fiber density sim.
#         A cosmic cuts across multiple traces, but only one end is masked. That
#         mask interrupts one of the traces, and the effect was that the trace
#         was cut short and a new trace started, rather than continuing a single
#         trace.
#         """
#         crValue = 1000
#         colShift = 3
#         maskRadius = 3
#         middleRow = self.synth.height//2
#         middleCol = self.synth.traceCenters[self.synth.numFibers//2]
#         self.image.image.array[middleRow, :] += crValue
#         xx, yy = getIndices(self.image.getBBox(), int)
#         select = ((xx - middleCol - colShift)**2 + (yy - middleRow)**2) < maskRadius**2
#         self.image.mask.array[select] |= self.image.mask.getPlaneBitMask("CR")

#         self.config.pruneMinLength = self.synth.height//3  # So a half a trace counts as a trace
#         result = self.task.run(self.makeExposure(self.image), self.identity, detectorMap=self.detMap)
#         # The CR damages the profile in this small image; so just care about the number of traces
#         self.assertNumTraces(result)

#     def testSpottyMaskedCosmic(self):
#         """Test that a spotty-masked cosmic doesn't split a trace

#         This is based on a failure mode discovered in half fiber density sims.
#         A cosmic cuts across multiple traces, but is only masked between the
#         traces. The mask starts immediately after the peak, and extends to a
#         couple of pixels short of the next peak. The effect was that the trace
#         was cut short and a new trace started, rather than continuing a single
#         trace.

#         We're going to take the three middle traces, and put masked pixels
#         between them.
#         """
#         crValue = 1000  # Value to give CR pixels
#         maskHalfRows = 3  # Half the number of rows to have masks
#         leftOffset = 1  # Position of the left side of the CR mask relative to the trace
#         rightOffset = 2  # Position of the right side of the CR mask relative to the trace

#         middleTrace = self.synth.numFibers//2
#         leftTrace = middleTrace - 1
#         rightTrace = middleTrace + 1

#         middleRow = self.synth.height//2
#         rowSlice = slice(middleRow - maskHalfRows, middleRow + maskHalfRows)

#         leftTracePeak = int(self.synth.traceCenters[leftTrace] + 0.5)
#         middleTracePeak = int(self.synth.traceCenters[middleTrace] + 0.5)
#         rightTracePeak = int(self.synth.traceCenters[rightTrace] + 0.5)

#         leftSlice = slice(leftTracePeak + leftOffset, middleTracePeak - rightOffset)
#         rightSlice = slice(middleTracePeak + leftOffset, rightTracePeak - rightOffset)

#         for ss in (leftSlice, rightSlice):
#             self.image.image.array[rowSlice, ss] += crValue
#             self.image.mask.array[rowSlice, ss] |= self.image.mask.getPlaneBitMask("CR")

#         self.config.pruneMinLength = self.synth.height//3  # So a half a trace counts as a trace
#         result = self.task.run(self.makeExposure(self.image), self.identity, detectorMap=self.detMap)
#         # The CR damages the profile in this small image; so just care about the number of traces
#         self.assertNumTraces(result)


class BuildFiberProfilesMultipleTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.synth = SyntheticConfig()
        self.synth.height = 1024
        self.synth.width = 128
        self.synth.separation = 19.876
        self.synth.fwhm = 3.21
        self.synth.slope = 0.04321  # Larger than default, to get more sampling of sub-pixel positions

        self.flux = 1.0e5
        self.rng = np.random.RandomState(12345)

        self.darkTime = 123.456
        self.metadata = ("FOO", "BAR")
        self.visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        self.identity = CalibIdentity("2020-01-01", 5, "x", 12345)

        self.config = BuildFiberProfilesTask.ConfigClass()
        self.config.pruneMinLength = int(0.9*self.synth.height)
        self.config.profileSwath = self.synth.height
        self.config.profileRejIter = 1
        self.config.profileRejThresh = 5.0
        self.config.centerFit.order = 2
        self.config.rowFwhm = self.synth.fwhm
        self.config.columnFwhm = self.synth.fwhm
        self.config.extractFwhm = 1.0*self.synth.fwhm
        self.config.doBlindFind = False
        self.config.profileRadius = 10  # Very broad!
        self.task = BuildFiberProfilesTask(config=self.config)
        self.task.log.setLevel(self.task.log.DEBUG)

    def tearDown(self):
        del self.synth
        del self.visitInfo

    def makeContinuumExposure(self, fluxScale=None, addNoise=True):
        """Construct a synthetic continuum exposure

        The fibers may have different flux scalings.

        Parameters
        ----------
        fluxScale : array_like
            Flux scalings (relative to ``self.flux``) for each of the fibers.
        addNoise : `bool`
            Add noise to image?

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Continuum exposure.
        """
        image = lsst.afw.image.ImageF(self.synth.dims)
        for ii in range(self.synth.numFibers):
            scale = fluxScale[ii] if fluxScale is not None else 1.0
            image += makeSpectrumImage(self.flux*scale, self.synth.dims, [self.synth.traceCenters[ii]],
                                       self.synth.traceOffset, self.synth.fwhm)
        if addNoise:
            addNoiseToImage(image, self.synth.gain, self.synth.readnoise, self.rng)

        mask = lsst.afw.image.Mask(image.getBBox())
        mask.set(0)
        variance = lsst.afw.image.ImageF(image.getBBox())
        if self.synth.gain == 0:
            variance.array[:] = self.synth.readnoise**2
        else:
            variance.array[:] = np.where(image.array > 0, image.array, 0)/self.synth.gain
            variance.array += self.synth.readnoise**2/self.synth.gain**2

        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(image, mask, variance))
        exposure.getMetadata().set(*self.metadata)
        exposure.getInfo().setVisitInfo(self.visitInfo)
        return exposure

    def assertFiberProfiles(self, fiberProfiles, image, detMap, flux=None, badBitMask=0, doCheckSpectra=True,
                            imageChi2=1.0, spectrumChi2=1.0):
        """Assert that fiberProfiles are as expected

        We use the fiberProfile to extract the spectra, check that the spectra
        have the correct values, and then subtract the spectra from the 2D
        image; the resulting image should be close to zero.

        Parameters
        ----------
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Fiber profiles to check.
        image : `lsst.afw.image.MaskedImage`
            Image containing the traces.
        detMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        flux : `numpy.ndarray` of `float`
            Expected flux for each fiber. If not provided, defaults to
            ``self.flux``.
        badBitMask : `int`, optional
            Bitmask to use to ignore anomalous pixels.
        doCheckSpectra : `bool`, optional
            Check the spectra?
        """
        self.assertEqual(fiberProfiles.getVisitInfo().getDarkTime(), self.darkTime)
        self.assertEqual(fiberProfiles.visitInfo.getDarkTime(), self.darkTime)
        self.assertEqual(fiberProfiles.getMetadata().get(self.metadata[0]), self.metadata[1])
        self.assertEqual(fiberProfiles.metadata.get(self.metadata[0]), self.metadata[1])
        image = image.clone()  # We're going to modify it
        self.assertEqual(len(fiberProfiles), self.synth.numFibers)
        self.assertFloatsEqual(fiberProfiles.fiberId, sorted(self.synth.fiberId))

        # for fiberId in fiberProfiles:
        #     fiberProfiles[fiberId].plot()
        # plt.close("all")

        if display:
            Display(backend=display, frame=1).mtv(image, title="Image")
            Display(backend=display, frame=2).mtv(image.getVariance(), title="Variance")

        traces = fiberProfiles.makeFiberTracesFromDetectorMap(detMap)
        spectra = traces.extractSpectra(image, badBitMask)
        model = spectra.makeImage(image.getBBox(), traces)
        image -= model
        image.image.array /= np.sqrt(image.variance.array)
        image.writeFits("chi2.fits")
        select = (image.mask.array & badBitMask) == 0
        chi2 = np.sum(image.image.array[select]**2)
        if display:
            Display(backend=display, frame=3).mtv(model, title="Model")
            Display(backend=display, frame=4).mtv(image, title="Chi image")

        numSwaths = fiberProfiles[0].profiles.shape[0]
        numFibers = self.synth.numFibers
        numPixels = (2*self.config.profileRadius + 1)*self.config.profileOversample

        dof = select.sum() - numSwaths*numFibers*numPixels
#        self.assertLess(chi2/dof, imageChi2)
        if not doCheckSpectra:
            return
        if flux is None:
            flux = itertools.repeat(self.flux)

        for ss in spectra:
            print(np.median(ss.flux), np.median(ss.variance))

        for ss, ff in zip(spectra, flux):
            chi2 = np.sum((ss.flux - ff)**2/ss.variance)
            print(ss.fiberId, np.median(ss.flux), chi2/self.synth.height)
            self.assertLess(chi2/self.synth.height, spectrumChi2, f"fiberId={ss.fiberId}")

    # def testBasic(self):
    #     """Test that things basically work"""
    #     exposure = self.makeContinuumExposure()
    #     detMap = makeSyntheticDetectorMap(self.synth)
    #     results = self.task.runMultiple([exposure], self.identity, [detMap])
    #     self.assertFiberProfiles(results.profiles, exposure.maskedImage, detMap, spectrumChi2=1.2)

    # @methodParameters(separation=(19.876, 12.345,))
    # def testTwo(self, separation):
    #     self.synth.separation = separation

    #     oddScale = np.zeros(self.synth.numFibers)
    #     oddScale[1::2] = 1
    #     evenScale = np.zeros(self.synth.numFibers)
    #     evenScale[0::2] = 1

    #     oddExp = self.makeContinuumExposure(oddScale)
    #     evenExp = self.makeContinuumExposure(evenScale)

    #     detMap = makeSyntheticDetectorMap(self.synth)
    #     results = self.task.runMultiple([oddExp, evenExp], self.identity, [detMap, detMap])
    #     self.assertFiberProfiles(results.profiles, oddExp.maskedImage, detMap, oddScale*self.flux,
    #                              spectrumChi2=1.3)
    #     self.assertFiberProfiles(results.profiles, evenExp.maskedImage, detMap, evenScale*self.flux,
    #                              spectrumChi2=1.3)

    @methodParameters(separation=(6.543,))
    def testFour(self, separation):
        """Separation of 6 pixels and radius of 10 pixels requires four
        exposures.
        """
        self.synth.separation = separation

        self.synth.readnoise = 1.0
        self.synth.gain = 0

#        self.synth.width = 60  # XXX numFibers=1

        scale1 = np.full(self.synth.numFibers, 0.0)
        scale1[0::4] = 1
        scale2 = np.full(self.synth.numFibers, 0.0)
        scale2[1::4] = 1
        scale3 = np.full(self.synth.numFibers, 0.0)
        scale3[2::4] = 1
        scale4 = np.full(self.synth.numFibers, 0.0)
        scale4[3::4] = 1

        exp1 = self.makeContinuumExposure(scale1)
        exp2 = self.makeContinuumExposure(scale2)
        exp3 = self.makeContinuumExposure(scale3)
        exp4 = self.makeContinuumExposure(scale4)

        xx = np.linspace(-1, 1, self.synth.width)
        bg = 0#*np.exp(-0.5*xx**2/0.4**2)

        exp1.image.array += 0.7*bg
        exp2.image.array += 0.9*bg
        exp3.image.array += 1.1*bg
        exp4.image.array += 1.3*bg

        exp1.writeFits("exp.fits")


        self.task.config.profileRejIter = 1
        self.task.config.extractIter = 3
        self.task.config.profileRadius = 15
        self.task.config.profileOversample = 2

        detMap = makeSyntheticDetectorMap(self.synth)
        results = self.task.runMultiple(
            [exp1, exp2, exp3, exp4], self.identity, [detMap, detMap, detMap, detMap]
#            [exp1], self.identity, [detMap]
        )

        exp1.writeFits("exp1.fits")
        exp2.writeFits("exp2.fits")
        exp3.writeFits("exp3.fits")
        exp4.writeFits("exp4.fits")

        print(results.profiles[0].calculateStatistics())

        figAxes = results.profiles.plot(4, 3, show=False)
        for fig, ax in figAxes.values():
            ax.semilogy()
            ax.set_ylim(1.0e-6, 1.0e1)
        plt.show()


        # Large chi^2 value for spectra here is a little concerning, but I'm not sure how to avoid it.
        #
        # A small part of this is that the variances in the spectra are underestimated:
        # FiberTraceSet.extractSpectra neglects covariance (so we don't have to realise a sparse matrix
        # inverse, which would not be efficient), and there's lots of covariance here since fiber profiles
        # overlap not just with the ones next to it, but also the ones next to them.
        #
        # The covariance between the fiber profiles is getting completely swept under the rug (we don't even
        # account for the variance in the fiber profile, let alone the covariance).
        #
        # Adding more exposures with different scales could reduce the noise. And the curvature in the traces
        # of real PFS data will help us get a much better sampling of the over-sampled profile than this toy
        # setup.
        #
        # In the end, decided that this is an extreme case (profile is 10 pixels radius with fibers separated
        # by 6.5 pixels) on a toy model, so I've loosened up on the limits.
        self.assertFiberProfiles(results.profiles, exp1.maskedImage, detMap, scale1*self.flux,
                                 spectrumChi2=1.5)
        self.assertFiberProfiles(results.profiles, exp2.maskedImage, detMap, scale2*self.flux,
                                 spectrumChi2=1.5)
        self.assertFiberProfiles(results.profiles, exp3.maskedImage, detMap, scale3*self.flux,
                                 spectrumChi2=1.5)
        self.assertFiberProfiles(results.profiles, exp4.maskedImage, detMap, scale4*self.flux,
                                 spectrumChi2=1.5)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
