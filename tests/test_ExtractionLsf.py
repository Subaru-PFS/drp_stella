import pickle

import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.afw.image.testUtils
import lsst.afw.display

from pfs.drp.stella.lsf import ExtractionLsf
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.SpectralPsfContinued import ImagingSpectralPsf
from pfs.drp.stella.tests.utils import runTests
from pfs.drp.stella.utils.psf import fwhmToSigma

from pfs.drp.stella.synthetic import makeSpectrumImage, SyntheticConfig, makeSyntheticDetectorMap

display = None


class ExtractionLsfTestCase(lsst.utils.tests.TestCase):
    """Tests of pfs.drp.stella.ExtractionLsf"""
    def setUp(self):
        self.config = SyntheticConfig()
        self.config.height = 256
        self.config.width = 128
        self.config.separation = 21.09876
        self.config.fwhm = 3.21
        self.config.slope = 0.0

        flux = 1.0e5

        flat = makeSpectrumImage(flux, self.config.dims, self.config.traceCenters, self.config.traceOffset,
                                 self.config.fwhm)
        mask = lsst.afw.image.Mask(flat.getBBox())
        mask.set(0)
        variance = lsst.afw.image.ImageF(flat.getBBox())
        variance.set(10000.0)
        self.flat = lsst.afw.image.makeMaskedImage(flat, mask, variance)
        self.detMap = makeSyntheticDetectorMap(self.config)

        task = BuildFiberProfilesTask()
        task.config.pruneMinLength = 100
        task.config.findThreshold = 10
        profiles = task.run(lsst.afw.image.makeExposure(self.flat), self.detMap).profiles
        self.traces = profiles.makeFiberTracesFromDetectorMap(self.detMap)
        self.assertEqual(len(self.traces), self.config.numFibers)

        self.psfSigma = fwhmToSigma(self.config.fwhm)
        self.psfSize = 2*int(4*self.psfSigma) + 1
        imagePsf = lsst.afw.detection.GaussianPsf(self.psfSize, self.psfSize, self.psfSigma)

        self.psf = ImagingSpectralPsf(imagePsf, self.detMap)

    def tearDown(self):
        del self.flat
        del self.detMap
        del self.traces
        del self.psf

    def makeLsf(self, fiberId):
        """Construct an ExtractionLsf"""
        fiberTrace = self.traces.getByFiberId(fiberId)
        return ExtractionLsf(self.psf, fiberTrace, self.config.height)

    def checkLsf(self, lsf):
        """Check that the LSF produces the expected results"""
        for yy in (3.21, 0.5*self.config.height, 0.987*self.config.height):
            kernel = lsf.computeKernel(yy)
            xx = np.arange(-(self.psfSize//2), self.psfSize//2 + 1)
            outOfRange = (xx + int(yy + 0.5) < 0) | (xx + int(yy + 0.5) > self.config.height - 1)
            expect = np.exp(-0.5*(xx/self.psfSigma)**2)
            expect[outOfRange] = 0.0
            expect /= expect.sum()
            if not np.any(outOfRange):  # Otherwise, the calculation is biased by zeroes
                self.assertFloatsAlmostEqual(kernel.computeStdev(), self.psfSigma, atol=1.0e-1)
            self.assertFloatsAlmostEqual(kernel.values, expect, atol=1.0e-6)

            array = lsf.computeArray(yy)
            self.assertFloatsAlmostEqual(array.sum(), 1.0, atol=1.0e-6)
            indices = np.arange(self.config.height)
            centroid = (array*indices).sum()/array.sum()
            width = np.sqrt(np.sum(array*(indices - centroid)**2)/array.sum())
            self.assertFloatsAlmostEqual(centroid, yy, atol=1.0e-1)
            self.assertFloatsAlmostEqual(width, self.psfSigma, atol=1.0e-1)
            expect = np.exp(-0.5*((np.arange(self.config.height) - yy)/self.psfSigma)**2)
            expect /= expect.sum()
            self.assertFloatsAlmostEqual(array, expect, atol=1.0e-3)

    def testBasic(self):
        """Test that we can construct and use an ExtractionLsf"""
        if display:
            disp = lsst.afw.display.Display(frame=1, backend=display)
            disp.mtv(self.flat)
        for fiberId in self.config.fiberId:
            lsf = self.makeLsf(fiberId)
            self.checkLsf(lsf)

    def testPickle(self):
        """Test that we can pickle an ExtractionLsf"""
        for fiberId in self.config.fiberId:
            lsf = self.makeLsf(fiberId)
            copy = pickle.loads(pickle.dumps(lsf))
            self.checkLsf(copy)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
