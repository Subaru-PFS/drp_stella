from collections.abc import Mapping
import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils
from lsst.geom import Box2I, Point2I, Extent2I

from pfs.datamodel import CalibIdentity
from pfs.drp.stella.synthetic import makeSyntheticFlat, SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.fiberProfile import FiberProfile
from pfs.drp.stella.fiberProfileSet import FiberProfileSet
from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask
from pfs.drp.stella.tests import runTests

display = None


class ExtractSpectraTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)  # I have the same combination on my luggage
        self.synthConfig = SyntheticConfig()

        self.synthConfig.height = 256
        self.synthConfig.width = 128
        self.synthConfig.slope = 0.01
        self.identity = CalibIdentity("2020-01-01", 5, "x", 12345)
        self.setUpTraces(separation=12.3, fwhm=3.21)

        if display:
            from lsst.afw.display import Display
            Display(backend=display, frame=1).mtv(self.image, title="Synthetic image")
            Display(backend=display, frame=2).mtv(self.fiberTraces[4].trace, title="FiberTrace #4")

    def setUpTraces(self, separation, fwhm, measure=True):
        """Construct the synthetic flat and its fiber traces

        Parameters
        ----------
        separation : `float`
            Separation between the fibers (pixels).
        fwhm : `float`
            Full width at half maximum of the fiber profiles (pixels).
        measure : `bool`
            Measure the fiber profiles from the synthetic flat? Otherwise,
            use the (Gaussian) profiles with which it was made; the measured
            profiles are poor if the fibers overlap strongly.
        """
        self.synthConfig.separation = separation
        self.synthConfig.fwhm = fwhm
        self.detMap = makeSyntheticDetectorMap(self.synthConfig)
        self.flux = 1.0e5
        image = makeSyntheticFlat(self.synthConfig, flux=self.flux, addNoise=False, rng=self.rng)
        self.image = lsst.afw.image.makeMaskedImage(image)
        self.image.mask.array[:] = 0x0
        self.image.variance.array[:] = self.synthConfig.readnoise**2

        if measure:
            config = BuildFiberProfilesTask.ConfigClass()
            config.pruneMinLength = self.synthConfig.height//2
            task = BuildFiberProfilesTask(config=config)
            self.fiberProfiles = task.run(
                lsst.afw.image.makeExposure(self.image), self.identity, detectorMap=self.detMap
            ).profiles
        else:
            sigma = fwhm/(2*np.sqrt(2*np.log(2)))
            radius = int(np.ceil(4*sigma))
            self.fiberProfiles = FiberProfileSet.makeEmpty(self.identity)
            for fiberId in self.detMap.fiberId:
                self.fiberProfiles[fiberId] = FiberProfile.makeGaussian(
                    sigma, self.synthConfig.height, radius, 10
                )
        for fiberId in self.fiberProfiles:
            self.fiberProfiles[fiberId].norm = np.full(self.synthConfig.height, self.flux, dtype=np.float32)
        self.fiberTraces = self.fiberProfiles.makeFiberTracesFromDetectorMap(self.detMap)
        self.assertEqual(len(self.fiberTraces), self.synthConfig.numFibers)

    def assertSpectra(self, spectra, flux=None, mask=None, norm=None):
        """Assert that the extracted spectra are as expected

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        flux : `dict` (`int`: array_like) or array_like, optional
            Expected flux values. If a `dict`, provides the expected flux value
            indexed by the fiberId.
        mask : `dict` (`int`: array_like) or array_like, optional
            Expected mask values. If a `dict`, provides the expected mask value
            indexed by the fiberId.
        norm : `dict` (`int`: array_like) or array_like, optional
            Expected norm values. If a `dict`, provides the expected norm value
            indexed by the fiberId.
        """
        self.assertEqual(len(spectra), self.synthConfig.numFibers)
        self.assertEqual(spectra.getLength(), self.synthConfig.height)
        for ss, ft in zip(spectra, self.fiberTraces):
            if isinstance(flux, Mapping):
                expectFlux = flux.get(ss.fiberId, None)
            else:
                expectFlux = flux
            if expectFlux is None:
                expectFlux = self.flux

            if isinstance(mask, Mapping):
                expectMask = mask.get(ss.fiberId, None)
            else:
                expectMask = mask
            if expectMask is None:
                expectMask = 0

            if isinstance(norm, Mapping):
                expectNorm = norm.get(ss.fiberId, None)
            else:
                expectNorm = norm
            if expectNorm is None:
                bbox = ft.trace.getBBox()
                expectNorm = np.zeros(self.synthConfig.height, dtype=np.float32)
                expectNorm[bbox.getMinY():bbox.getMaxY() + 1] = ft.trace.image.array.sum(axis=1)

            self.assertEqual(ss.fiberId, ft.fiberId)
            self.assertEqual(len(ss), self.synthConfig.height)
            self.assertFloatsAlmostEqual(ss.flux, expectFlux, rtol=2.0e-3)
            self.assertFloatsAlmostEqual(ss.norm, expectNorm, rtol=1.0e-5)
            self.assertFloatsEqual(ss.mask.array[0], expectMask)
            self.assertTrue(np.all(ss.mask.array[0] | (ss.variance > 0)))
            self.assertTrue(np.all(np.isfinite(ss.variance)))

    def testBasic(self):
        """Test basic extraction"""
        spectra = self.fiberTraces.extractSpectra(self.image)
        self.assertSpectra(spectra)

    def testMasked(self):
        """Test extraction in the presence of masked pixels"""
        self.badRow = 123
        bitMask = self.image.mask.getPlaneBitMask("BAD")
        everyOther = np.arange(0, self.synthConfig.width, 2, dtype=int)
        self.image.mask.array[self.badRow][everyOther] |= bitMask

        # With mask supplied to extractSpectra and minFracMask=0
        # BAD doesn't get into the mask because it's excluded.
        spectra = self.fiberTraces.extractSpectra(self.image, bitMask, 0.0)
        expectMask = np.zeros(self.synthConfig.height, dtype=np.int32)
        expectMask[self.badRow] = 0
        self.assertSpectra(spectra, mask=expectMask)

        # No mask supplied to extractSpectra
        # BAD now gets into the mask because it's not explicitly excluded.
        spectra = self.fiberTraces.extractSpectra(self.image, 0, 0.0)
        expectMask[self.badRow] = bitMask
        self.assertSpectra(spectra, mask=expectMask)

    def testUndersizedFiberTrace(self):
        """Test extraction with undersized fiberTraces"""
        index = 3
        middle = self.synthConfig.height//2
        fiberId = self.fiberTraces[index].fiberId
        trace = self.fiberTraces[index]
        box = trace.trace.getBBox()

        # Start is missing
        newBox = Box2I(Point2I(box.getMinX(), middle), box.getMax())
        newTrace = type(trace)(trace.trace.Factory(trace.trace, newBox), trace.fiberId)
        self.fiberTraces[index] = newTrace
        spectra = self.fiberTraces.extractSpectra(self.image)
        expectFlux = np.zeros(self.synthConfig.height, dtype=np.float32)
        expectFlux[middle:] = self.flux
        expectMask = np.zeros(self.synthConfig.height, dtype=np.int32)
        expectMask[:middle] = trace.trace.mask.getPlaneBitMask("NO_DATA")
        self.assertSpectra(spectra, flux={fiberId: expectFlux}, mask={fiberId: expectMask})

        # End is missing
        newBox = Box2I(box.getMin(), Extent2I(box.getWidth(), middle))
        newTrace = type(trace)(trace.trace.Factory(trace.trace, newBox), trace.fiberId)
        self.fiberTraces[index] = newTrace
        spectra = self.fiberTraces.extractSpectra(self.image)
        expectFlux = np.zeros(self.synthConfig.height, dtype=np.float32)
        expectFlux[:middle] = self.flux
        expectMask = np.zeros(self.synthConfig.height, dtype=np.int32)
        expectMask[middle:] = trace.trace.mask.getPlaneBitMask("NO_DATA")
        self.assertSpectra(spectra, flux={fiberId: expectFlux}, mask={fiberId: expectMask})

    def testSingular(self):
        """Test behaviour with a singular matrix inversion

        We get a singular matrix when a trace has a row that is all zero and/or
        no pixels have the FIBERTRACE mask plane set. In that case, we expect
        to get good spectra for all the other fibers, and the mask planes
        BAD_FIBERTRACE|NO_DATA set on the appropriate row of the bad fiber.
        """
        index = self.synthConfig.numFibers//2
        row = self.synthConfig.height//2
        fiberId = self.synthConfig.fiberId[index]
        expectFlux = np.full(self.synthConfig.height, self.flux, dtype=np.float32)
        expectMask = np.zeros(self.synthConfig.height, dtype=np.int32)
        expectNorm = np.full(self.synthConfig.height, self.flux, dtype=np.float32)
        expectFlux[row] = 0.0  # We don't measure any flux for this row, because there's no fiberTrace data
        expectMask[row] = self.image.mask.getPlaneBitMask(["BAD_FIBERTRACE", "NO_DATA"])
        expectNorm[row] = 0.0  # There's no flux in the fiberTrace for this row

        self.fiberTraces[index].trace.image.array[row, :] = 0.0
        spectra = self.fiberTraces.extractSpectra(self.image)
        self.assertSpectra(
            spectra, flux={fiberId: expectFlux}, mask={fiberId: expectMask}, norm={fiberId: expectNorm}
        )

        # Remove the FIBERTRACE mask bit
        self.fiberTraces[index].trace.image.array[row, :] = 1000.0
        self.fiberTraces[index].trace.mask.array[row, :] = 0
        spectra = self.fiberTraces.extractSpectra(self.image)
        self.assertSpectra(
            spectra, flux={fiberId: expectFlux}, mask={fiberId: expectMask}, norm={fiberId: expectNorm}
        )

    def testMinFracMask(self):
        """Test behavior of the minFracMask parameter

        minFracMask is the minimum fractional contribution of a pixel for the
        mask to be accumulated.
        """
        index = self.synthConfig.numFibers//2
        row = self.synthConfig.height//2
        col = int(self.synthConfig.traceCenters[index] + self.synthConfig.traceOffset[index])

        bad = self.image.mask.getPlaneBitMask("BAD")
        self.image.mask.array[row][col] = bad

        masked = np.zeros(self.synthConfig.height, dtype=np.int32)
        masked[row] = bad

        fiberTrace = self.fiberTraces[index]
        x0 = fiberTrace.trace.getX0()
        ft = self.fiberTraces[index].trace.image.array[row]
        ft[col - x0] = 0.0
        norm = ft.sum()

        # With bad pixel at fraction=0 and minFracMask=0 --> unmasked
        spectra = self.fiberTraces.extractSpectra(self.image, bad, 0.0)
        self.assertFloatsEqual(spectra[index].mask.array[0], 0)
        self.assertFloatsAlmostEqual(spectra[index].normFlux, 1.0, rtol=1.0e-3)

        # With bad pixel at fraction=0 and minFracMask=0.5 --> unmasked
        spectra = self.fiberTraces.extractSpectra(self.image, bad, 0.0)
        self.assertFloatsEqual(spectra[index].mask.array[0], 0)
        self.assertFloatsAlmostEqual(spectra[index].normFlux, 1.0, rtol=1.0e-3)

        # With bad pixel at fraction=0.5 and minFracMask=0.3 --> masked but no sign of it
        ft[col - x0] = norm
        spectra = self.fiberTraces.extractSpectra(self.image, bad, 0.3)
        self.assertFloatsEqual(spectra[index].mask.array[0], 0)
        self.assertFloatsAlmostEqual(spectra[index].normFlux, 1.0, rtol=1.0e-3)

        # With bad pixel at fraction=0.5 and minFracMask = 0.7 --> unmasked
        spectra = self.fiberTraces.extractSpectra(self.image, bad, 0.7)
        self.assertFloatsEqual(spectra[index].mask.array[0], 0)
        self.assertFloatsAlmostEqual(spectra[index].normFlux, 1.0, rtol=1.0e-3)

    def makeNoisyImage(self, flux, lineRow, lineFactor):
        """Make a noisy image of the traces, with a bright 'line' in one row

        The line has the shape of the fiber profiles (it's just a brighter
        row), so the extraction should absorb it.
        """
        image = makeSyntheticFlat(self.synthConfig, flux=flux, addNoise=False, rng=self.rng)
        image.array[lineRow] *= lineFactor
        variance = image.array + self.synthConfig.readnoise**2
        image.array += self.rng.normal(size=image.array.shape)*np.sqrt(variance)
        maskedImage = lsst.afw.image.makeMaskedImage(image)
        maskedImage.mask.array[:] = 0
        maskedImage.variance.array[:] = variance
        return maskedImage

    def testRejectOutliers(self):
        """Test rejection of outliers (cosmic rays) in the extraction"""
        flux = 1000.0
        lineRow = 100
        lineFactor = 20.0
        image = self.makeNoisyImage(flux, lineRow, lineFactor)
        expectFlux = np.full(self.synthConfig.height, flux)
        expectFlux[lineRow] *= lineFactor

        # Cosmic rays: narrow (1-2 columns) deposits on the traces. N.b. a CR that resembles the profile
        # (e.g., 2 columns centered on the trace) is largely absorbed into the fitted flux, and since the
        # systematic error scales with the (raised) peak, it isn't rejected; so we use CRs that don't.
        crList = []  # (row, columns)
        for index, row, dx, width in ((3, 50, 0, 1), (5, 150, 2, 1), (2, 200, -3, 2)):
            col = int(np.round(self.detMap.getXCenter(self.fiberTraces[index].fiberId)[row])) + dx
            columns = np.arange(col, col + width)
            image.image.array[row, columns] += 3000.0
            crList.append((index, row, columns))

        config = ExtractSpectraTask.ConfigClass()
        config.doRejectOutliers = True
        task = ExtractSpectraTask(config=config)

        # Without rejection: the CRs bias the flux
        unrejected = image.clone()
        spectra = task.run(unrejected, self.fiberTraces, rejectOutliers=False).spectra
        crBit = unrejected.mask.getPlaneBitMask("CR")
        self.assertEqual(np.count_nonzero(unrejected.mask.array & crBit), 0)
        for index, row, _ in crList:
            self.assertGreater(spectra[index].flux[row] - flux, 300)

        # With rejection
        spectra = task.run(image, self.fiberTraces).spectra
        crBit = image.mask.getPlaneBitMask("CR")
        rejected = (image.mask.array & crBit) != 0
        for index, row, columns in crList:
            self.assertTrue(np.all(rejected[row, columns]), f"CR in row {row} not rejected")
        # N.b. excluded pixels don't propagate their mask to the spectrum (same as for CRs from the cosmicray
        # task); the spectrum is flagged SUSPECT only if too little of the profile remains.
        # Nothing rejected in the line, and only the CRs (grown) rejected elsewhere
        self.assertEqual(np.count_nonzero(rejected[lineRow]), 0)
        crRows = {row for _, row, _ in crList}
        self.assertEqual(set(np.where(rejected.any(axis=1))[0]), crRows)
        # The rejection is extended to the whole trace of the struck fiber in that row (config
        # rejectFiberFraction), but no further.
        halfSeparation = 0.5*self.synthConfig.separation
        columns = np.arange(self.synthConfig.width)
        for index, row, _ in crList:
            xCenter = self.detMap.getXCenter(self.fiberTraces[index].fiberId)[row]
            self.assertFalse(np.any(rejected[row] & (np.abs(columns - xCenter) > halfSeparation)))

        # Fluxes are recovered (within the noise), except where too little of the profile remains, which is
        # flagged.
        flagged = self.fiberTraces[0].trace.mask.getPlaneBitMask(["SUSPECT", "NO_DATA"])
        for ss in spectra:
            use = (ss.mask.array[0] & flagged) == 0
            self.assertLess(np.max(np.abs(ss.flux - expectFlux)[use]/np.sqrt(ss.variance[use])), 5.0)
        for index, row, _ in crList:
            ss = spectra[index]
            self.assertTrue(
                (ss.mask.array[0, row] & flagged) != 0
                or np.abs(ss.flux[row] - expectFlux[row]) < 5*np.sqrt(ss.variance[row])
            )

    def testRejectOutliersNonSeparable(self):
        """Test that a bright line with a different cross-section isn't rejected

        The PSF isn't separable into the fiber profile and the LSF, so a bright
        line's profile across the fiber differs from the continuum's. The
        residuals in the wings scale with the line's peak, and should not be
        rejected.
        """
        flux = 1000.0
        lineRow = 100
        lineFactor = 50.0
        broadConfig = SyntheticConfig()
        for name in ("height", "width", "separation", "slope"):
            setattr(broadConfig, name, getattr(self.synthConfig, name))
        broadConfig.fwhm = 1.2*self.synthConfig.fwhm
        image = makeSyntheticFlat(self.synthConfig, flux=flux, addNoise=False, rng=self.rng)
        broad = makeSyntheticFlat(broadConfig, flux=flux*lineFactor, addNoise=False, rng=self.rng)
        image.array[lineRow] = broad.array[lineRow]
        variance = image.array + self.synthConfig.readnoise**2
        image.array += self.rng.normal(size=image.array.shape)*np.sqrt(variance)
        maskedImage = lsst.afw.image.makeMaskedImage(image)
        maskedImage.mask.array[:] = 0
        maskedImage.variance.array[:] = variance

        config = ExtractSpectraTask.ConfigClass()
        config.doRejectOutliers = True
        task = ExtractSpectraTask(config=config)
        task.run(maskedImage, self.fiberTraces)
        rejected = (maskedImage.mask.array & maskedImage.mask.getPlaneBitMask("CR")) != 0
        self.assertEqual(np.count_nonzero(rejected), 0)

    def testRejectOutliersBackground(self):
        """Test that light between the fibers doesn't cause rejections

        The rejection is done before the scattered light is subtracted, so
        there's light that isn't in the model. In the wings of an empty fiber
        it isn't absorbed into the fitted flux, but it shouldn't be rejected;
        a cosmic ray on that fiber should be.
        """
        flux = 1000.0
        pedestal = 300.0
        emptyIndex = 3
        image = makeSyntheticFlat(self.synthConfig, flux=flux, addNoise=False, rng=self.rng)
        xCenter = self.detMap.getXCenter(self.fiberTraces[emptyIndex].fiberId)
        halfWidth = 0.5*self.synthConfig.separation
        columns = np.arange(self.synthConfig.width)
        for row in range(self.synthConfig.height):
            image.array[row, np.abs(columns - xCenter[row]) < halfWidth] = 0.0
        image.array += pedestal
        variance = image.array + self.synthConfig.readnoise**2
        image.array += self.rng.normal(size=image.array.shape)*np.sqrt(variance)
        maskedImage = lsst.afw.image.makeMaskedImage(image)
        maskedImage.mask.array[:] = 0
        maskedImage.variance.array[:] = variance

        config = ExtractSpectraTask.ConfigClass()
        config.doRejectOutliers = True
        config.rejectSystematicFraction = 0.03

        # Without the floor from the background, the wings of the empty fiber are rejected
        noFloorConfig = ExtractSpectraTask.ConfigClass()
        noFloorConfig.update(**config.toDict())
        noFloorConfig.rejectBackgroundMaxProfile = -1.0  # no background pixels
        noFloor = maskedImage.clone()
        ExtractSpectraTask(config=noFloorConfig).run(noFloor, self.fiberTraces)
        self.assertGreater(np.count_nonzero(noFloor.mask.array & noFloor.mask.getPlaneBitMask("CR")), 0)

        # With the floor, nothing is rejected
        clean = maskedImage.clone()
        ExtractSpectraTask(config=config).run(clean, self.fiberTraces)
        self.assertEqual(np.count_nonzero(clean.mask.array & clean.mask.getPlaneBitMask("CR")), 0)

        # ... except a cosmic ray on the empty fiber
        crRow = 120
        crColumn = int(np.round(xCenter[crRow])) + 2
        maskedImage.image.array[crRow, crColumn] += 3000.0
        ExtractSpectraTask(config=config).run(maskedImage, self.fiberTraces)
        rejected = (maskedImage.mask.array & maskedImage.mask.getPlaneBitMask("CR")) != 0
        self.assertTrue(rejected[crRow, crColumn])
        self.assertEqual(set(np.where(rejected)[0]), {crRow})

    def testRejectOutliersRejectedFiber(self):
        """Test that there are no further rejections in a fiber rejected in a row

        Once we have rejected enough of a fiber's trace in a row, its flux is
        fit from the remaining pixels, so the model within its trace is
        unreliable: the rest of that fiber's trace is rejected, but nothing
        else within its footprint (which would otherwise lead to a cascade of
        rejections in the neighbouring fibers). We simulate the unreliable fit
        by zeroing the flux of a fiber in two rows, one of which has part of
        the fiber already rejected. Pixels masked elsewhere (e.g., by the ISR)
        are trusted: they don't cause the rest of the fiber to be rejected.
        """
        flux = 1000.0
        image = self.makeNoisyImage(flux, 100, 1.0)
        index = 3
        maskedRow = 100
        controlRow = 150
        externalRow = 200
        config = ExtractSpectraTask.ConfigClass()
        config.doRejectOutliers = True
        image.mask.addMaskPlane("CR")
        image.mask.addMaskPlane(config.rejectOutlierMaskPlane)
        crBit = image.mask.getPlaneBitMask("CR")
        outlierBit = image.mask.getPlaneBitMask(config.rejectOutlierMaskPlane)
        xCenter = self.detMap.getXCenter(self.fiberTraces[index].fiberId)[maskedRow]
        columns = np.arange(self.synthConfig.width)
        select = (columns >= xCenter) & (columns < xCenter + 3)
        image.mask.array[maskedRow, select] |= crBit | outlierBit  # rejected by us earlier
        image.mask.array[externalRow, select] |= crBit  # e.g., from the ISR
        original = image.mask.array.copy()

        task = ExtractSpectraTask(config=config)
        badBitMask = image.mask.getPlaneBitMask(config.mask) | crBit
        spectra = self.fiberTraces.extractSpectra(image, badBitMask, config.minFracMask, config.minFracImage)
        for row in (maskedRow, controlRow):
            spectra[index].flux[row] = 0.0
        # A second iteration (without re-extracting, so the model remains wrong) mustn't spread the
        # rejection of the fiber's trace to its neighbours
        for _ in range(2):
            task.rejectOutliers(image, self.fiberTraces, spectra, badBitMask)
        rejected = ((image.mask.array & crBit) != 0) & ((original & crBit) == 0)
        self.assertGreater(np.count_nonzero(rejected[controlRow]), 0)
        self.assertEqual(set(np.where(rejected)[0]), {maskedRow, controlRow})
        # In the masked row, only the rest of the masked fiber is rejected, not its neighbours
        neighbours = [
            self.detMap.getXCenter(self.fiberTraces[ii].fiberId)[maskedRow] for ii in (index - 1, index + 1)
        ]
        halfway = 0.5*(np.array(neighbours) + xCenter)
        newColumns = columns[rejected[maskedRow]]
        self.assertGreater(len(newColumns), 0)
        self.assertTrue(np.all((newColumns > min(halfway)) & (newColumns < max(halfway))))

    def testRejectOutliersNoCascade(self):
        """Test that rejecting a fiber's trace doesn't propagate to its neighbours

        When the fibers are close together and their profiles broad (as in
        b4), the trace of a fiber rejected in a row covers much of its
        neighbours' profiles. Those pixels mustn't count towards the
        neighbours' rejected fractions, or each iteration rejects the next
        fibers out. We use a cosmic ray on a fiber that extends well to one
        side of its center (as in b4 row 0 of visit 155935), of which the
        part on the other side has been masked elsewhere (as by the ISR): we
        should reject the rest of it, and the rejections must stop spreading
        after the first full pass. A mask from elsewhere with no cosmic ray
        left in the data (as in another row) is trusted, and causes no
        rejections.
        """
        separation = 6.2
        self.setUpTraces(separation=separation, fwhm=5.0, measure=False)
        image = self.makeNoisyImage(1000.0, 100, 1.0)
        index = len(self.fiberTraces)//2
        row = 100
        cleanRow = 150
        image.mask.addMaskPlane("CR")
        crBit = image.mask.getPlaneBitMask("CR")
        columns = np.arange(self.synthConfig.width)
        for rr in (cleanRow, row):
            xCenter = self.detMap.getXCenter(self.fiberTraces[index].fiberId)[rr]
            image.mask.array[rr, (columns >= xCenter - 2) & (columns < xCenter)] |= crBit  # e.g., from ISR
        # Shape of the cosmic ray in b4 row 0 of visit 155935 (fiber 2149), scaled to our fiber's flux
        cosmicRay = (columns >= xCenter - 2) & (columns < xCenter + 5)
        image.image.array[row, cosmicRay] += 2*np.array([1350, 1450, 2300, 4600, 5100, 1300, 1300])
        original = image.mask.array.copy()

        config = ExtractSpectraTask.ConfigClass()
        config.doRejectOutliers = True
        task = ExtractSpectraTask(config=config)
        badBitMask = image.mask.getPlaneBitMask(config.mask) | crBit
        # As in the pipeline: a preliminary rejection, followed by the full rejection
        passes = [(config.preliminaryRejectThreshold, config.preliminaryRejectFiberFraction)]
        passes += [(config.rejectThreshold, config.rejectFiberFraction)]*config.rejectIterations
        numRejected = []
        for threshold, fiberFraction in passes:
            spectra = self.fiberTraces.extractSpectra(
                image, badBitMask, config.minFracMask, config.minFracImage
            )
            numRejected.append(
                task.rejectOutliers(image, self.fiberTraces, spectra, badBitMask, threshold, fiberFraction)
            )
        rejected = ((image.mask.array & crBit) != 0) & ((original & crBit) == 0)
        self.assertEqual(set(np.where(rejected)[0]), {row})
        self.assertTrue(np.all(rejected[row, cosmicRay & ((original[row] & crBit) == 0)]))
        self.assertEqual(numRejected[2:], [0]*(len(passes) - 2))  # no spreading after the first full pass

    def testRejectOutliersNoData(self):
        """Test that a fiber-row ruined by our rejection is flagged NO_DATA

        If our rejection leaves too little of a fiber's profile in a row, the
        flux would be fit from the faint edges of the profile, so we flag it
        NO_DATA. A fiber-row that is mostly masked for other reasons (e.g., by
        the ISR) keeps the extraction's usual treatment.
        """
        image = self.makeNoisyImage(1000.0, 100, 1.0)
        index = 3
        crRow = 100
        externalRow = 150
        fiberId = self.fiberTraces[index].fiberId
        columns = np.arange(self.synthConfig.width)
        image.mask.addMaskPlane("CR")
        crBit = image.mask.getPlaneBitMask("CR")
        xCenter = self.detMap.getXCenter(fiberId)[crRow]
        image.image.array[crRow, np.abs(columns - xCenter) < 2.5] += 20000.0
        xCenter = self.detMap.getXCenter(fiberId)[externalRow]
        image.mask.array[externalRow, np.abs(columns - xCenter) < 2.5] |= crBit  # e.g., from the ISR

        config = ExtractSpectraTask.ConfigClass()
        config.doRejectOutliers = True
        task = ExtractSpectraTask(config=config)
        spectra = task.run(image, self.fiberTraces, self.detMap).spectra
        spectrum = [ss for ss in spectra if ss.fiberId == fiberId][0]
        noData = spectrum.mask.getPlaneBitMask("NO_DATA")
        mask = spectrum.mask.array[0]
        self.assertTrue(mask[crRow] & noData)
        self.assertTrue(mask[crRow] & spectrum.mask.getPlaneBitMask("CR"))
        self.assertEqual(spectrum.flux[crRow], 0.0)
        self.assertEqual(spectrum.variance[crRow], 0.0)
        self.assertFalse(mask[externalRow] & noData)
        self.assertTrue(mask[externalRow] & spectrum.mask.getPlaneBitMask("SUSPECT"))
        self.assertEqual(np.count_nonzero(mask & noData), 1)  # nothing else

    def testRejectOutliersBoxcar(self):
        """Test that outlier rejection is not performed for boxcar extraction"""
        image = self.makeNoisyImage(1000.0, 100, 20.0)
        col = int(np.round(self.detMap.getXCenter(self.fiberTraces[3].fiberId)[50]))
        image.image.array[50, col] += 3000.0
        config = ExtractSpectraTask.ConfigClass()
        config.doRejectOutliers = True
        task = ExtractSpectraTask(config=config)
        boxcars = self.fiberProfiles.makeFiberTracesFromDetectorMap(self.detMap, 5)
        task.run(image, boxcars, isBoxcar=True)
        self.assertEqual(np.count_nonzero(image.mask.array & image.mask.getPlaneBitMask("CR")), 0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
