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

    def makeImage(self, xOffset: float = 0.0) -> MaskedImage:
        image = makeSyntheticFlat(self.config, xOffset=xOffset, flux=self.flux, addNoise=False)
        maskedImage = makeMaskedImage(image)
        maskedImage.mask.array[:] = 0
        maskedImage.variance.array[:] = self.config.readnoise**2
        return maskedImage

    def assertResidual(
        self,
        image: MaskedImage,
        kernel: FiberKernel,
        fiberTraces: FiberTraceSet,
    ) -> None:
        """Check that the residual image is zero"""
        resid = image.clone()
        convolvedTraces = kernel.convolve(fiberTraces, image.getBBox())
        spectra = convolvedTraces.extractSpectra(resid)
        model = spectra.makeImage(image.getBBox(), convolvedTraces)
        resid -= model
        self.assertFloatsAlmostEqual(np.std(resid.image.array), 0.0, atol=1.0)

    def testIntegerOffset(self):
        kernelHalfWidth = 2
        kernelNum = 3
        xOffset = 1.0
        image = self.makeImage(xOffset=xOffset)
        fiberTraces = self.makeFiberTraces()

        kernel = fitFiberKernel(
            image, fiberTraces, 0, kernelHalfWidth, kernelNum, kernelNum
        )
        print(kernel.coefficients)

        kernelImages = kernel.makeOffsetImages(kernelNum, kernelNum)
        for ii in range(kernelNum):
            for jj in range(kernelNum):
                kernelValues = kernelImages[:, ii, jj]
                self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-3)
                for offset, value in enumerate(kernelValues, -kernelHalfWidth):
                    expect = 1.0 if offset == xOffset else 0.0
                    self.assertFloatsAlmostEqual(value, expect, atol=1.0e-2)

        self.assertResidual(image, kernel, fiberTraces)

    def testOffset(self):
        """We apply a subpixel offset"""
        kernelHalfWidth = 3
        kernelNum = 3
        xOffset = -0.5
        image = self.makeImage(xOffset=xOffset)
        fiberTraces = self.makeFiberTraces()

        kernel = fitFiberKernel(
            image, fiberTraces, 0, kernelHalfWidth, kernelNum, kernelNum
        )

        # Check that the kernel gives the expected offset
        kernelImages = kernel.makeOffsetImages(kernelNum, kernelNum)
        for ii in range(kernelNum):
            for jj in range(kernelNum):
                kernelValues = kernelImages[:, ii, jj]
                offsetValues = np.arange(-kernelHalfWidth, kernelHalfWidth + 1)
                self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-3)
                centroid = np.sum(kernelValues*offsetValues)
                print(f"Kernel {ii}, {jj}={kernelImages[:, ii, jj]}, centroid={centroid}")
                self.assertFloatsAlmostEqual(centroid, xOffset, atol=1.0e-2)

        self.assertResidual(image, kernel, fiberTraces)

    def testWidth(self):
        kernelHalfWidth = 3
        kernelNum = 3
        image = self.makeImage()
        fiberTraces = self.makeFiberTraces(3.33)

        kernel = fitFiberKernel(image, fiberTraces, 0, kernelHalfWidth, kernelNum, kernelNum)
        self.assertResidual(image, kernel, fiberTraces)


class ImageKernelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticConfig()
        self.config.height = 100
        self.config.width = 4000
        self.config.fwhm = 3.21
        self.config.separation = 6.543

        self.flux = 1.0e4

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

    def makeImage(self, xOffset: float = 0.0) -> MaskedImage:
        image = makeSyntheticFlat(self.config, xOffset=xOffset, flux=self.flux, addNoise=False)
        maskedImage = makeMaskedImage(image)
        maskedImage.mask.array[:] = 0
        maskedImage.variance.array[:] = self.config.readnoise**2
        return maskedImage

    def assertResidual(
        self,
        source: MaskedImage,
        target: MaskedImage,
        kernel: FiberKernel,
    ) -> None:
        """Check that the residual image is zero"""
        resid = target.clone()
        resid -= kernel.convolve(source.image)

        source.writeFits("source.fits")
        target.writeFits("target.fits")
        kernel.convolve(source.image).writeFits("convolved.fits")
        resid.writeFits("resid.fits")

        self.assertFloatsAlmostEqual(np.std(resid.image.array), 0.0, atol=2.0)

    def assertSpectra(
        self,
        source: MaskedImage,
        target: MaskedImage,
        kernel: FiberKernel,
        fiberTraces: FiberTraceSet,
    ) -> None:
        target = target.clone()
        sourceSpectra = fiberTraces.extractSpectra(source)
        targetSpectra = kernel.convolve(fiberTraces, target.getBBox()).extractSpectra(target)
        sourceFlux = sourceSpectra.getAllFluxes()
        targetFlux = targetSpectra.getAllFluxes()
        self.assertFloatsAlmostEqual(sourceFlux, targetFlux, rtol=2e-3)

    def testIntegerOffset(self):
        kernelHalfWidth = 2
        kernelNum = 3
        xOffset = 1.0
        source = self.makeImage()
        target = self.makeImage(xOffset=xOffset)

        kernel = fitFiberKernel(
            source, target, 0, kernelHalfWidth, kernelNum, kernelNum
        )
        self.assertResidual(source, target, kernel)
        self.assertSpectra(source, target, kernel, self.makeFiberTraces())

        kernelImages = kernel.makeOffsetImages(kernelNum, kernelNum)
        for ii in range(kernelNum):
            for jj in range(kernelNum):
                kernelValues = kernelImages[:, ii, jj]
                self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-3)
                for offset, value in enumerate(kernelValues, -kernelHalfWidth):
                    expect = 1.0 if offset == xOffset else 0.0
                    self.assertFloatsAlmostEqual(value, expect, atol=1.0e-2)

    def testOffset(self):
        """We apply a subpixel offset"""
        kernelHalfWidth = 3
        kernelNum = 3
        xOffset = -0.5
        source = self.makeImage()
        target = self.makeImage(xOffset=xOffset)

        kernel = fitFiberKernel(
            source, target, 0, kernelHalfWidth, kernelNum, kernelNum
        )

        # Check that the kernel gives the expected offset
        kernelImages = kernel.makeOffsetImages(kernelNum, kernelNum)
        for ii in range(kernelNum):
            for jj in range(kernelNum):
                kernelValues = kernelImages[:, ii, jj]
                offsetValues = np.arange(-kernelHalfWidth, kernelHalfWidth + 1)
                self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-3)
                centroid = np.sum(kernelValues*offsetValues)
                print(f"Kernel {ii}, {jj}={kernelImages[:, ii, jj]}, centroid={centroid}")
                self.assertFloatsAlmostEqual(centroid, xOffset, atol=1.0e-2)

        self.assertResidual(source, target, kernel)
        self.assertSpectra(source, target, kernel, self.makeFiberTraces())


    def testWidth(self):
        kernelHalfWidth = 3
        kernelNum = 3
        source = self.makeImage()
        self.config.fwhm = 3.33
        target = self.makeImage()

        kernel = fitFiberKernel(
            source, target, 0, kernelHalfWidth, kernelNum, kernelNum
        )
        self.assertResidual(source, target, kernel)
        self.assertSpectra(source, target, kernel, self.makeFiberTraces())


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
