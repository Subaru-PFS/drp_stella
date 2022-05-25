from collections.abc import Mapping
import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils
from lsst.geom import Box2I, Point2I, Extent2I

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

        config = BuildFiberProfilesTask.ConfigClass()
        config.pruneMinLength = self.synthConfig.height//2
        task = BuildFiberProfilesTask(config=config)
        self.fiberProfiles = task.run(
            lsst.afw.image.makeExposure(self.image), detectorMap=self.detMap
        ).profiles
        for fiberId in self.fiberProfiles:
            self.fiberProfiles[fiberId].norm = np.full(self.synthConfig.height, self.flux, dtype=float)
        self.fiberTraces = self.fiberProfiles.makeFiberTracesFromDetectorMap(self.detMap)
        self.assertEqual(len(self.fiberTraces), self.synthConfig.numFibers)

        if display:
            from lsst.afw.display import Display
            Display(backend=display, frame=1).mtv(self.image, title="Synthetic image")
            Display(backend=display, frame=2).mtv(self.fiberTraces[4].trace, title="FiberTrace #4")

    def assertSpectra(self, spectra, flux=None, mask=None):
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

            bbox = ft.trace.getBBox()
            expectNorm = np.zeros(self.synthConfig.height, dtype=float)
            expectNorm[bbox.getMinY():bbox.getMaxY() + 1] = ft.trace.image.array.sum(axis=1)

            self.assertEqual(ss.fiberId, ft.fiberId)
            self.assertEqual(len(ss), self.synthConfig.height)
            self.assertFloatsAlmostEqual(ss.flux, expectFlux, rtol=2.0e-3)
            self.assertFloatsAlmostEqual(ss.norm, expectNorm, rtol=1.0e-6)
            self.assertFloatsEqual(ss.mask.array[0], expectMask)
            self.assertFloatsEqual(ss.background, 0.0)
            self.assertTrue(np.all(ss.mask.array[0] | (ss.variance > 0)))
            self.assertTrue(np.all(np.isfinite(ss.variance)))
            self.assertTrue(np.all(np.isfinite(ss.covariance)))

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

        # With mask supplied to extractSpectra
        spectra = self.fiberTraces.extractSpectra(self.image, bitMask)
        expectMask = np.zeros(self.synthConfig.height, dtype=np.int32)
        expectMask[self.badRow] = bitMask
        self.assertSpectra(spectra, mask=expectMask)

        # No mask supplied to extractSpectra
        spectra = self.fiberTraces.extractSpectra(self.image)
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
        expectFlux = np.zeros(self.synthConfig.height, dtype=float)
        expectFlux[middle:] = self.flux
        expectMask = np.zeros(self.synthConfig.height, dtype=np.int32)
        expectMask[:middle] = trace.trace.mask.getPlaneBitMask("NO_DATA")
        self.assertSpectra(spectra, flux={fiberId: expectFlux}, mask={fiberId: expectMask})

        # End is missing
        newBox = Box2I(box.getMin(), Extent2I(box.getWidth(), middle))
        newTrace = type(trace)(trace.trace.Factory(trace.trace, newBox), trace.fiberId)
        self.fiberTraces[index] = newTrace
        spectra = self.fiberTraces.extractSpectra(self.image)
        expectFlux = np.zeros(self.synthConfig.height, dtype=float)
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
        expectFlux = np.full(self.synthConfig.height, self.flux, dtype=float)
        expectMask = np.zeros(self.synthConfig.height, dtype=np.int32)
        expectFlux[row] = 0.0
        expectMask[row] = self.image.mask.getPlaneBitMask(["BAD_FIBERTRACE", "NO_DATA"])

        self.fiberTraces[index].trace.image.array[row, :] = 0.0
        spectra = self.fiberTraces.extractSpectra(self.image)
        self.assertSpectra(spectra, flux={fiberId: expectFlux}, mask={fiberId: expectMask})

        self.fiberTraces[index].trace.image.array[row, :] = 1000.0
        self.fiberTraces[index].trace.mask.array[row, :] = 0
        spectra = self.fiberTraces.extractSpectra(self.image)
        self.assertSpectra(spectra, flux={fiberId: expectFlux}, mask={fiberId: expectMask})

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

        # With bad pixel at fraction=0 and minFracMask=0.5 --> unmasked
        spectra = self.fiberTraces.extractSpectra(self.image, bad, 0.0)
        self.assertFloatsEqual(spectra[index].mask.array[0], 0)

        # With bad pixel at fraction=0.5 and minFracMask=0.3 --> masked
        ft[col - x0] = norm
        spectra = self.fiberTraces.extractSpectra(self.image, bad, 0.3)
        self.assertFloatsEqual(spectra[index].mask.array[0], masked)

        # With bad pixel at fraction=0.5 and minFracMask = 0.7 --> unmasked
        spectra = self.fiberTraces.extractSpectra(self.image, bad, 0.7)
        self.assertFloatsEqual(spectra[index].mask.array[0], 0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
