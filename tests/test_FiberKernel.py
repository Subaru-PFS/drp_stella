import numpy as np

import lsst.utils.tests

from lsst.afw.image import makeMaskedImage, MaskedImage
from lsst.geom import Extent2I

from pfs.datamodel import CalibIdentity
from pfs.drp.stella.fiberProfile import FiberProfile
from pfs.drp.stella.fiberProfileSet import FiberProfileSet
from pfs.drp.stella.FiberTraceSetContinued import FiberTraceSet
from pfs.drp.stella.FiberKernel import fitFiberKernel, LinearInterpolationHelper
from pfs.drp.stella.FiberKernelContinued import FiberKernel
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap, makeSyntheticFlat
from pfs.drp.stella.tests import runTests
from pfs.drp.stella.utils.psf import fwhmToSigma


class LinearInterpolationHelperTestCase(lsst.utils.tests.TestCase):
    def testLinearInterpolationHelper(self):
        print("Testing LinearInterpolationHelper")
        rng = np.random.default_rng(12345)
        num = 5
        length = 100

        values = np.empty(num, dtype=float)
        for ii in range(num):
            xMin = ii*length//num
            xMax = (ii + 1)*length//num
            values[ii] = 0.5*(xMin + xMax)

        helper = LinearInterpolationHelper(values, length)
        for xx in rng.integers(0, length, 10):
            result = helper(xx)
            leftIndex = result[0][0]
            leftWeight = result[0][1]
            rightIndex = result[1][0]
            rightWeight = result[1][1]
            interpolated = leftWeight*values[leftIndex] + rightWeight*values[rightIndex]
            self.assertFloatsAlmostEqual(interpolated, xx, atol=1.0e-10)

    def testConstant(self):
        length = 100
        constant = 123.45
        values = np.full(1, constant, dtype=float)
        helper = LinearInterpolationHelper(values, length)
        for xx in range(length):
            result = helper(xx)
            leftIndex = result[0][0]
            leftWeight = result[0][1]
            rightIndex = result[1][0]
            rightWeight = result[1][1]
            interpolated = leftWeight*values[leftIndex] + rightWeight*values[rightIndex]
            self.assertFloatsEqual(interpolated, constant)


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
        resid.writeFits("resid.fits")
        kernelSize = 2*kernel.halfWidth + 1
        self.assertFloatsAlmostEqual(np.std(resid.image.array[:, kernelSize:-kernelSize]), 0.0, atol=5.0)

    def testIntegerOffset(self):
        kernelHalfWidth = 2
        xKernelNum = 3
        yKernelNum = 1
        xOffset = 1.0
        image = self.makeImage(xOffset=xOffset)
        fiberTraces = self.makeFiberTraces()

        kernel = fitFiberKernel(
            image, fiberTraces, 0, kernelHalfWidth, xKernelNum, yKernelNum
        )
        print(kernel.values)

        # Check that the kernel gives the expected offset
        # This is allowed to vary rather more than I'm comfortable with, but the
        # kernel is not incredibly well constrained, and what we really care about
        # is the spectra and the extraction-subtracted image.
        kernelImages = kernel.makeOffsetImages(xKernelNum, yKernelNum)
        for ii in range(xKernelNum):
            for jj in range(yKernelNum):
                kernelValues = kernelImages[:, jj, ii]
                self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-3)
                for offset, value in enumerate(kernelValues, -kernelHalfWidth):
                    expect = 1.0 if offset == xOffset else 0.0
                    self.assertFloatsAlmostEqual(value, expect, rtol=0.3, atol=0.4)

        self.assertResidual(image, kernel, fiberTraces)

    def testOffset(self):
        """We apply a subpixel offset"""
        kernelHalfWidth = 3
        xKernelNum = 3
        yKernelNum = 1
        xOffset = -0.5
        image = self.makeImage(xOffset=xOffset)
        fiberTraces = self.makeFiberTraces()

        kernel = fitFiberKernel(
            image, fiberTraces, 0, kernelHalfWidth, xKernelNum, yKernelNum
        )

        # Check that the kernel gives the expected offset
        # This is allowed to vary rather more than I'm comfortable with, but the
        # kernel is not incredibly well constrained, and what we really care about
        # is the spectra and the extraction-subtracted image.
        kernelImages = kernel.makeOffsetImages(xKernelNum, yKernelNum)
        for ii in range(xKernelNum):
            for jj in range(yKernelNum):
                kernelValues = kernelImages[:, jj, ii]
                offsetValues = np.arange(-kernelHalfWidth, kernelHalfWidth + 1)
                self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-3)
                centroid = np.sum(kernelValues*offsetValues)
                print(f"Kernel {ii}, {jj}={kernelImages[:, jj, ii]}, centroid={centroid}")
                self.assertFloatsAlmostEqual(centroid, xOffset, rtol=0.3, atol=0.3)

        self.assertResidual(image, kernel, fiberTraces)

    def testWidth(self):
        kernelHalfWidth = 3
        xKernelNum = 3
        yKernelNum = 1
        image = self.makeImage()
        fiberTraces = self.makeFiberTraces(3.33)

        kernel = fitFiberKernel(image, fiberTraces, 0, kernelHalfWidth, xKernelNum, yKernelNum)
        self.assertResidual(image, kernel, fiberTraces)

    def testIO(self):
        rng = np.random.default_rng(12345)
        dims = Extent2I(4321, 5678)
        halfWidth = 12
        xNumBlocks = 3
        yNumBlocks = 4
        values = rng.uniform(size=(2*halfWidth)*xNumBlocks*yNumBlocks)

        kernel = FiberKernel(dims, halfWidth, xNumBlocks, yNumBlocks, values)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            kernel.writeFits(filename)
            newKernel = FiberKernel.readFits(filename)
            self.assertEqual(newKernel.dims, kernel.dims)
            self.assertEqual(newKernel.halfWidth, kernel.halfWidth)
            self.assertEqual(newKernel.xNumBlocks, kernel.xNumBlocks)
            self.assertEqual(newKernel.yNumBlocks, kernel.yNumBlocks)
            self.assertFloatsEqual(newKernel.values, kernel.values)


class ImageKernelTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.config = SyntheticConfig()
        self.config.height = 1000
        self.config.width = 1000
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

        denom = 0.5*(source.image.array + target.image.array)
        resid.image.array /= np.sqrt(denom + self.config.readnoise**2)  # variance-like
        chi2 = np.sum(resid.image.array**2)
        print(f"Chi2={chi2}, dof={resid.image.array.size}")
        kernelSize = 2*kernel.halfWidth + 1
        self.assertFloatsAlmostEqual(np.std(resid.image.array[:, kernelSize:-kernelSize]), 0.0, atol=1.0)

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
        diff = (targetFlux - sourceFlux)/sourceFlux
        rms = np.std(diff)
        self.assertFloatsAlmostEqual(rms, 0.0, atol=2.0e-3)

    def testIntegerOffset(self):
        kernelHalfWidth = 3
        kernelNum = 3
        xOffset = 1.0
        source = self.makeImage()
        target = self.makeImage(xOffset=xOffset)

        kernel = fitFiberKernel(
            source, target, 0, kernelHalfWidth, kernelNum, kernelNum
        )

        # Check that the kernel gives the expected offset
        # This is allowed to vary rather more than I'm comfortable with, but the
        # kernel is not incredibly well constrained, and what we really care about
        # is the spectra and the extraction-subtracted image.
        kernelImages = kernel.makeOffsetImages(kernelNum, kernelNum)
        for ii in range(kernelNum):
            for jj in range(kernelNum):
                kernelValues = kernelImages[:, jj, ii]
                self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-3)
                print(ii, jj, kernelValues)
                for offset, value in enumerate(kernelValues, -kernelHalfWidth):
                    expect = 1.0 if offset == xOffset else 0.0
                    self.assertFloatsAlmostEqual(value, expect, rtol=0.3, atol=0.35)

        self.assertResidual(source, target, kernel)
        self.assertSpectra(source, target, kernel, self.makeFiberTraces())

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

        print("Repeating with a single box selected via the mask")
        source.mask.array[:] = 0xFFFF
        target.mask.array[:] = 0xFFFF
        source.mask.array[0:100, 0:100] = 0
        target.mask.array[0:100, 0:100] = 0
        fitFiberKernel(
            source, target, 0xFFFF, kernelHalfWidth, 1, 1
        )

        # Check that the kernel gives the expected offset
        # This is allowed to vary rather more than I'm comfortable with, but the
        # kernel is not incredibly well constrained, and what we really care about
        # is the spectra and the extraction-subtracted image.
        kernelImages = kernel.makeOffsetImages(kernelNum, kernelNum)
        for ii in range(kernelNum):
            for jj in range(kernelNum):
                kernelValues = kernelImages[:, jj, ii]
                offsetValues = np.arange(-kernelHalfWidth, kernelHalfWidth + 1)
                self.assertFloatsAlmostEqual(np.sum(kernelValues), 1.0, atol=1.0e-3)
                centroid = np.sum(kernelValues*offsetValues)
                print(f"Kernel {ii}, {jj}={kernelImages[:, ii, jj]}, centroid={centroid}")
                self.assertFloatsAlmostEqual(centroid, xOffset, rtol=0.3, atol=0.3)

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
