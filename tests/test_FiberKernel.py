import numpy as np

import lsst.utils.tests

from pfs.datamodel import CalibIdentity
from pfs.drp.stella.fiberProfile import FiberProfile
from pfs.drp.stella.fiberProfileSet import FiberProfileSet
from pfs.drp.stella.FiberKernel import fitFiberKernel
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap, makeSyntheticFlat
from pfs.drp.stella.tests import runTests
from pfs.drp.stella.utils.psf import fwhmToSigma


class FiberKernelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticConfig()
        self.config.height = 128
        self.config.width = 4096
        self.config.fwhm = 3.21
        self.config.separation = 6.543

        self.flux = 1.0e4
        self.background = 123.45

        self.detMap = makeSyntheticDetectorMap(self.config)
        self.fiberTraces = self.makeFiberTraces()

    def makeFiberTraces(self):
        identity = CalibIdentity("2020-01-01", 5, "x", 12345)
        sigma = fwhmToSigma(self.config.fwhm)
        radius = int(np.ceil(3*sigma))
        oversample = 10
        profiles = FiberProfileSet.makeEmpty(identity)
        for fiberId in self.config.fiberId:
            profiles[fiberId] = FiberProfile.makeGaussian(sigma, self.config.height, radius, oversample)
        return profiles.makeFiberTracesFromDetectorMap(self.detMap)

    def makeImage(self, xOffset=0.0, background=0.0):
        image = makeSyntheticFlat(self.config, xOffset=xOffset, flux=self.flux, addNoise=False)
        maskedImage = lsst.afw.image.makeMaskedImage(image)
        maskedImage.image.array[:] += background
        maskedImage.mask.array[:] = 0
        maskedImage.variance.array[:] = self.config.readnoise**2
        return maskedImage

    def testFitPixelOffsetAndBackground(self):
        kernelHalfWidth = 2
        kernelOrder = 1
        bgWidth = 300
        bgHeight = 30
        xOffset = 1.0
        image = self.makeImage(xOffset=xOffset, background=self.background)

        kernel, background, flux = fitFiberKernel(
            image, self.fiberTraces, 0, kernelHalfWidth, kernelOrder, bgWidth, bgHeight
        )

        kernelImages = kernel.makeOffsetImages(10, 10)
        for ii, img in enumerate(kernelImages):
            offset = ii - kernelHalfWidth + (1 if ii >= kernelHalfWidth else 0)
            expect = 1.0 if offset == xOffset else 0.0
            self.assertFloatsAlmostEqual(img.array, expect, atol=1.0e-3)

        self.assertFloatsAlmostEqual(background.array, self.background, atol=0.2)
        self.assertFloatsAlmostEqual(flux, self.flux, rtol=2.0e-4)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
