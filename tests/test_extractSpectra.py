from collections.abc import Mapping
import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils
from lsst.geom import Box2I, Point2I, Extent2I

from pfs.datamodel import CalibIdentity
from pfs.drp.stella.synthetic import makeSyntheticFlat, SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.tests import runTests

display = None


class ExtractSpectraTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(12345)  # I have the same combination on my luggage
        self.synthConfig = SyntheticConfig()

        self.synthConfig.height = 256
        self.synthConfig.width = 128
        self.synthConfig.fwhm = 3.21
        self.synthConfig.separation = 12.3
        self.synthConfig.slope = 0.01

        self.detMap = makeSyntheticDetectorMap(self.synthConfig)
        self.flux = 1.0e5
        image = makeSyntheticFlat(self.synthConfig, flux=self.flux, addNoise=False, rng=self.rng)
        self.image = lsst.afw.image.makeMaskedImage(image)
        self.image.mask.array[:] = 0x0
        self.image.variance.array[:] = self.synthConfig.readnoise**2
        self.identity = CalibIdentity("2020-01-01", 5, "x", 12345)

        config = BuildFiberProfilesTask.ConfigClass()
        config.pruneMinLength = self.synthConfig.height//2
        task = BuildFiberProfilesTask(config=config)
        self.fiberProfiles = task.run(
            lsst.afw.image.makeExposure(self.image), self.identity, detectorMap=self.detMap
        ).profiles
        for fiberId in self.fiberProfiles:
            self.fiberProfiles[fiberId].norm = np.full(self.synthConfig.height, self.flux, dtype=np.float32)
        self.fiberTraces = self.fiberProfiles.makeFiberTracesFromDetectorMap(self.detMap)
        self.assertEqual(len(self.fiberTraces), self.synthConfig.numFibers)

        if display:
            from lsst.afw.display import Display
            Display(backend=display, frame=1).mtv(self.image, title="Synthetic image")
            Display(backend=display, frame=2).mtv(self.fiberTraces[4].trace, title="FiberTrace #4")

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

    def maskCore(self, index, row, maxFracInformation):
        """Mask the core of a fiber's profile on a row as CR, until less than
        maxFracInformation of its information (sum of the squared profile)
        is unmasked

        Returns
        -------
        fracInformation : `float`
            Fraction of the information left unmasked.
        fracImage : `float`
            Fraction of the profile left unmasked.
        """
        trace = self.fiberTraces[index].trace
        x0 = trace.getX0()
        yy = row - trace.getY0()
        select = (trace.mask.array[yy] & trace.mask.getPlaneBitMask("FIBERTRACE")) != 0
        profile = np.where(select, trace.image.array[yy], 0.0)
        profile /= profile.sum()
        cr = self.image.mask.getPlaneBitMask("CR")
        unmasked = select.copy()
        for jj in np.argsort(profile)[::-1]:
            if np.sum(profile[unmasked]**2) < maxFracInformation*np.sum(profile**2):
                break
            unmasked[jj] = False
            self.image.mask.array[row, x0 + jj] |= cr
        return np.sum(profile[unmasked]**2)/np.sum(profile**2), profile[unmasked].sum()

    def testMaxErrorFactor(self):
        """Claude: a row whose unmasked pixels hold too little of the profile is NO_DATA

        A cosmic ray masking the core of a fiber's profile leaves only its far
        wings, whose estimate of the flux has a meaningless variance. If the
        error of the estimate would be more than maxErrorFactor times that of
        an unmasked extraction, the row has no data; otherwise (but with less
        than minFracImage of the profile) it is SUSPECT, with a finite
        variance. The other fibers are unaffected.
        """
        index = self.synthConfig.numFibers//2
        row = self.synthConfig.height//2
        fiberId = self.synthConfig.fiberId[index]
        badBitMask = self.image.mask.getPlaneBitMask(["CR"])
        noData = self.image.mask.getPlaneBitMask("NO_DATA")
        suspect = self.image.mask.getPlaneBitMask("SUSPECT")
        maxErrorFactor = 20.0
        minFracInformation = 1.0/maxErrorFactor**2

        fracInformation, fracImage = self.maskCore(index, row, minFracInformation)
        self.assertLess(fracInformation, minFracInformation)
        spectra = self.fiberTraces.extractSpectra(self.image, badBitMask, 0.2, 0.4, maxErrorFactor)
        spectrum = spectra[index]
        self.assertTrue(spectrum.mask.array[0, row] & noData)
        self.assertEqual(spectrum.flux[row], 0.0)
        self.assertEqual(spectrum.variance[row], 0.0)
        others = np.ones(self.synthConfig.height, dtype=bool)
        others[row] = False
        self.assertFloatsAlmostEqual(spectrum.flux[others], self.flux, rtol=2.0e-3)
        for ii, ss in enumerate(spectra):
            if ii != index:
                self.assertFloatsAlmostEqual(ss.flux, self.flux, rtol=2.0e-3, msg=f"fiber {ss.fiberId}")
                self.assertFalse(np.any(ss.mask.array[0] & noData))

        # Without the threshold, the old behaviour: a finite (but huge) variance, flagged SUSPECT
        spectra = self.fiberTraces.extractSpectra(self.image, badBitMask, 0.2, 0.4, np.inf)
        spectrum = spectra[index]
        self.assertFalse(spectrum.mask.array[0, row] & noData)
        self.assertTrue(spectrum.mask.array[0, row] & suspect)
        self.assertGreater(spectrum.variance[row], maxErrorFactor**2*np.median(spectrum.variance))

        # Within the threshold, even with only the wings left (an error of ~10x): SUSPECT, with the right flux
        # (to within the profile's accuracy in its wings)
        for keep, rtol in ((0.01, 1.0e-2), (0.2, 2.0e-3)):
            self.image.mask.array[:] = 0
            fracInformation, fracImage = self.maskCore(index, row, keep)
            self.assertGreater(fracInformation, minFracInformation)
            self.assertLess(fracImage, 0.4)
            spectra = self.fiberTraces.extractSpectra(self.image, badBitMask, 0.2, 0.4, maxErrorFactor)
            spectrum = spectra[index]
            self.assertFalse(spectrum.mask.array[0, row] & noData, msg=f"keep={keep}")
            self.assertTrue(spectrum.mask.array[0, row] & suspect, msg=f"keep={keep}")
            self.assertGreater(spectrum.variance[row], 0.0)
            self.assertLess(spectrum.variance[row], maxErrorFactor**2*np.median(spectrum.variance))
            self.assertFloatsAlmostEqual(spectrum.flux[row], self.flux, rtol=rtol)
            self.assertEqual(spectrum.fiberId, fiberId)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
