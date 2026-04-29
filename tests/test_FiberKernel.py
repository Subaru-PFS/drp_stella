import numpy as np

import lsst.utils.tests

from lsst.afw.image import Image, makeMaskedImage, MaskedImage

from pfs.datamodel import CalibIdentity
from pfs.drp.stella.fiberProfile import FiberProfile
from pfs.drp.stella.fiberProfileSet import FiberProfileSet
from pfs.drp.stella.FiberTraceSetContinued import FiberTraceSet
from pfs.drp.stella.FiberKernel import fitFiberKernel, FiberKernel
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap, makeSyntheticFlat
from pfs.drp.stella.tests import runTests
from pfs.drp.stella.utils.psf import fwhmToSigma


class FiberKernelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticConfig()
        self.config.height = 100
        self.config.width = 4000
        self.config.fwhm = 3.21
        self.config.separation = 6.543

        self.flux = 1.0e4
        self.background = 123.45

        self.detMap = makeSyntheticDetectorMap(self.config)

    def makeFiberTraces(self, fwhm: float | None = None) -> FiberTraceSet:
        identity = CalibIdentity("2020-01-01", 5, "x", 12345)
        if fwhm is None:
            fwhm = self.config.fwhm
        sigma = fwhmToSigma(fwhm)
        radius = int(np.ceil(4*sigma))
        oversample = 10
        profiles = FiberProfileSet.makeEmpty(identity)
        for fiberId in self.config.fiberId:
            profiles[fiberId] = FiberProfile.makeGaussian(sigma, self.config.height, radius, oversample)
        return profiles.makeFiberTracesFromDetectorMap(self.detMap)

    def makeImage(self, xOffset: float = 0.0, background: float = 0.0) -> MaskedImage:
        image = makeSyntheticFlat(self.config, xOffset=xOffset, flux=self.flux, addNoise=False)
        maskedImage = makeMaskedImage(image)
        maskedImage.image.array[:] += background
        maskedImage.mask.array[:] = 0
        maskedImage.variance.array[:] = self.config.readnoise**2
        return maskedImage

    def assertResidual(
        self,
        image: MaskedImage,
        kernel: FiberKernel,
        background: Image,
        fiberTraces: FiberTraceSet,
    ) -> None:
        """Check that the residual image is zero"""
        resid = image.clone()
        resid -= np.mean(background.array)
        convolvedTraces = kernel.convolve(fiberTraces, image.getBBox())
        spectra = convolvedTraces.extractSpectra(resid)
        model = spectra.makeImage(image.getBBox(), convolvedTraces)
        resid -= model
        self.assertFloatsAlmostEqual(np.std(resid.image.array), 0.0, atol=1.0)

    def testIntegerOffset(self):
        kernelHalfWidth = 2
        kernelNum = 1
        bgNum = 1
        xOffset = 1.0
        image = self.makeImage(xOffset=xOffset, background=self.background)
        fiberTraces = self.makeFiberTraces()

        kernel, background, flux = fitFiberKernel(
            image, fiberTraces, 0, kernelHalfWidth, kernelNum, kernelNum, bgNum, bgNum
        )

        self.assertFloatsAlmostEqual(background.array, self.background, atol=0.2)
        self.assertFloatsAlmostEqual(flux, self.flux, rtol=2.0e-4)
        self.assertResidual(image, kernel, background, fiberTraces)

        kernelImages = kernel.makeOffsetImages(10, 10)
        for offset, img in enumerate(kernelImages, -kernelHalfWidth):
            expect = 1.0 if offset == xOffset else 0.0
            self.assertFloatsAlmostEqual(img, expect, atol=1.0e-3)

    def testOffset(self):
        """We apply a subpixel offset"""
        kernelHalfWidth = 3
        kernelNum = 1
        bgNum = 1
        xOffset = -0.5
        image = self.makeImage(xOffset=xOffset, background=self.background)
        fiberTraces = self.makeFiberTraces()

        kernel, background, flux = fitFiberKernel(
            image, fiberTraces, 0, kernelHalfWidth, kernelNum, kernelNum, bgNum, bgNum
        )
        self.assertFloatsAlmostEqual(background.array, self.background, atol=1.0)
        self.assertFloatsAlmostEqual(flux, self.flux, rtol=2.0e-3)
        self.assertResidual(image, kernel, background, fiberTraces)

        # Check that the kernel gives the expected offset
        kernelImages = kernel.makeOffsetImages(1, 1)
        kernelValues = np.array([kk[0, 0] for kk in kernelImages])
        offsetValues = np.arange(-kernelHalfWidth, kernelHalfWidth + 1)
        self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-9)
        centroid = np.sum(kernelValues*offsetValues)
        self.assertFloatsAlmostEqual(centroid, xOffset, atol=1.0e-2)

    def testWidth(self):
        kernelHalfWidth = 3
        kernelNum = 1
        bgNum = 1
        image = self.makeImage(background=self.background)
        fiberTraces = self.makeFiberTraces(3.33)

        kernel, background, flux = fitFiberKernel(
            image, fiberTraces, 0, kernelHalfWidth, kernelNum, kernelNum, bgNum, bgNum
        )
        self.assertFloatsAlmostEqual(background.array, self.background, atol=1.0)
        self.assertFloatsAlmostEqual(flux, self.flux, rtol=2.0e-3)
        self.assertResidual(image, kernel, background, fiberTraces)


class ImageKernelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticConfig()
        self.config.height = 100
        self.config.width = 4000
        self.config.fwhm = 3.21
        self.config.separation = 6.543

        self.flux = 1.0e4
        self.background = 123.45

        self.detMap = makeSyntheticDetectorMap(self.config)

    def makeFiberTraces(self, fwhm: float | None = None) -> FiberTraceSet:
        identity = CalibIdentity("2020-01-01", 5, "x", 12345)
        if fwhm is None:
            fwhm = self.config.fwhm
        sigma = fwhmToSigma(fwhm)
        radius = int(np.ceil(4*sigma))
        oversample = 10
        profiles = FiberProfileSet.makeEmpty(identity)
        for fiberId in self.config.fiberId:
            profiles[fiberId] = FiberProfile.makeGaussian(sigma, self.config.height, radius, oversample)
        return profiles.makeFiberTracesFromDetectorMap(self.detMap)

    def makeImage(self, xOffset: float = 0.0, background: float = 0.0) -> MaskedImage:
        image = makeSyntheticFlat(self.config, xOffset=xOffset, flux=self.flux, addNoise=False)
        maskedImage = makeMaskedImage(image)
        maskedImage.image.array[:] += background
        maskedImage.mask.array[:] = 0
        maskedImage.variance.array[:] = self.config.readnoise**2
        return maskedImage

    def assertResidual(
        self,
        source: MaskedImage,
        target: MaskedImage,
        kernel: FiberKernel,
        background: Image,
    ) -> None:
        """Check that the residual image is zero"""
        resid = target.clone()
        resid -= np.mean(background.array[0, 0])
        resid -= kernel.convolve(source.image)
        resid.writeFits("resid.fits")
        self.assertFloatsAlmostEqual(np.std(resid.image.array), 0.0, atol=2.0)

    def assertSpectra(
        self,
        source: MaskedImage,
        target: MaskedImage,
        kernel: FiberKernel,
        background: Image,
        fiberTraces: FiberTraceSet,
    ) -> None:
        target = target.clone()
        target -= np.mean(background.array)
        sourceSpectra = fiberTraces.extractSpectra(source)
        targetSpectra = kernel.convolve(fiberTraces, target.getBBox()).extractSpectra(target)
        sourceFlux = sourceSpectra.getAllFluxes()
        targetFlux = targetSpectra.getAllFluxes()
        self.assertFloatsAlmostEqual(sourceFlux, targetFlux, rtol=2e-3)

    def testIntegerOffset(self):
        kernelHalfWidth = 2
        kernelNum = 1
        bgNum = 1
        xOffset = 1.0
        source = self.makeImage()
        target = self.makeImage(xOffset=xOffset, background=self.background)

        kernel, background = fitFiberKernel(
            source, target, 0, kernelHalfWidth, kernelNum, kernelNum, bgNum, bgNum
        )
        self.assertResidual(source, target, kernel, background)
        self.assertSpectra(source, target, kernel, background, self.makeFiberTraces())

        kernelImages = kernel.makeOffsetImages(1, 1)
        for offset, img in enumerate(kernelImages, -kernelHalfWidth):
            expect = 1.0 if offset == xOffset else 0.0
            self.assertFloatsAlmostEqual(img, expect, atol=1.0e-2)

    def testOffset(self):
        """We apply a subpixel offset"""
        kernelHalfWidth = 3
        kernelNum = 1
        bgNum = 1
        xOffset = -0.5
        source = self.makeImage()
        target = self.makeImage(xOffset=xOffset, background=self.background)

        kernel, background = fitFiberKernel(
            source, target, 0, kernelHalfWidth, kernelNum, kernelNum, bgNum, bgNum
        )
        self.assertResidual(source, target, kernel, background)
        self.assertSpectra(source, target, kernel, background, self.makeFiberTraces())

        # Check that the kernel gives the expected offset
        kernelImages = kernel.makeOffsetImages(1, 1)
        kernelValues = np.array([kk[0, 0] for kk in kernelImages])
        offsetValues = np.arange(-kernelHalfWidth, kernelHalfWidth + 1)
        self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-3)
        centroid = np.sum(kernelValues*offsetValues)
        self.assertFloatsAlmostEqual(centroid, xOffset, atol=1.0e-2)

    def testWidth(self):
        kernelHalfWidth = 3
        kernelNum = 1
        bgNum = 1
        source = self.makeImage()
        self.config.fwhm = 3.33
        target = self.makeImage(background=self.background)

        kernel, background = fitFiberKernel(
            source, target, 0, kernelHalfWidth, kernelNum, kernelNum, bgNum, bgNum
        )
        self.assertResidual(source, target, kernel, background)
        self.assertSpectra(source, target, kernel, background, self.makeFiberTraces())


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
