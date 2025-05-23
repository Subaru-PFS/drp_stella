import sys
import unittest
import pickle

import numpy as np
from scipy.special import erf

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.afw.image
import lsst.afw.image.testUtils

import pfs.drp.stella as drpStella
from pfs.drp.stella.tests.utils import methodParameters
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticFlat, makeSyntheticDetectorMap
from pfs.drp.stella.utils.psf import fwhmToSigma

display = None


class FiberTraceTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Create a FiberTrace from a image of random values."""
        self.width = 11  # Width of trace
        self.height = 111  # Height of trace
        self.xy0 = lsst.geom.Point2I(123, 123)  # Origin for trace
        self.bbox = lsst.geom.Box2I(
            self.xy0, lsst.geom.Extent2I(self.width, self.height)
        )  # Bounding box for trace
        self.fiberId = 456  # Fiber identifier
        self.variance = 2.0  # Value of variance in image

        self.trace = lsst.afw.image.MaskedImageF(self.width, self.height)  # Image to use for trace
        self.trace.setXY0(self.xy0)
        rng = np.random.RandomState(12345)
        self.trace.image.array[:] = rng.uniform(size=self.trace.image.array.shape)
        self.trace.variance.array[:] = self.variance
        self.trace.mask.addMaskPlane(drpStella.fiberMaskPlane)
        self.maskVal = self.trace.mask.getPlaneBitMask(drpStella.fiberMaskPlane)  # Mask value for trace
        self.bad = self.trace.mask.getPlaneBitMask("BAD")  # Mask value for bad pixels
        self.noData = self.trace.mask.getPlaneBitMask("NO_DATA")  # Mask value for empty pixels
        self.trace.mask.array[:] = self.maskVal

        self.fiberTrace = drpStella.FiberTrace(self.trace, self.fiberId)  # The fiber trace

        self.fullDims = lsst.geom.Extent2I(2*self.width + self.bbox.getMinX(),
                                           2*self.height + self.bbox.getMinY())  # Extent of full image
        self.image = lsst.afw.image.MaskedImageF(self.fullDims)  # Image for extraction
        self.image.image.array[:] = np.nan
        self.image.mask.array[:] = 0xFFFF
        self.image.variance.array[:] = np.nan
        self.subimage = self.image[self.bbox, lsst.afw.image.PARENT]  # Image overlapping trace
        self.subimage[:] = self.trace
        self.subimage.mask.array[:] = 0

        self.nonOptimalVariance = self.width*self.variance  # Expected variance for non-optimal extraction
        # Expected variance for optimal extraction
        self.optimalVariance = 1.0/np.sum(self.trace.image.array**2/self.variance, axis=1)

    def tearDown(self):
        del self.trace
        del self.fiberTrace

    def assertFiberTracesEqual(self, ft1, ft2):
        """Assert that two FiberTrace objects are equal"""
        self.assertMaskedImagesEqual(ft1.trace, ft2.trace)
        self.assertEqual(ft1.fiberId, ft2.fiberId)

    def assertCopy(self, image1, image2, deep):
        """Assert that the two images are copies

        If the images are a shallow copy (``deep=False``), they should vary
        together; otherwise they should not.

        Parameters
        ----------
        image1, image2 : `lsst.afw.image.Image`
            Images to test. ``image1`` will be modified.
        deep : `bool`
            Is this a deep copy?
        """
        self.assertImagesEqual(image1, image2)
        image1.array += 1
        if not deep:
            self.assertImagesEqual(image1, image2)
        else:
            self.assertFloatsNotEqual(image1.array, image2.array)

    def testConstructors(self):
        """Test that the constructors work

        Includes the copy constructor and its deep copy version.

        In the process, also exercises the python properties and
        getter/setters.
        """
        self.assertMaskedImagesEqual(self.fiberTrace.getTrace(), self.trace)
        self.assertEqual(self.fiberTrace.getFiberId(), self.fiberId)
        # Check properties; these are more convenient to use
        self.assertMaskedImagesEqual(self.fiberTrace.trace, self.trace)
        self.assertEqual(self.fiberTrace.fiberId, self.fiberId)
        self.assertCopy(self.fiberTrace.trace.image, self.trace.image, False)

        # Copy constructor
        copy = drpStella.FiberTrace(self.fiberTrace)
        self.assertFiberTracesEqual(copy, self.fiberTrace)
        self.assertCopy(copy.trace.image, self.fiberTrace.trace.image, False)

        # Deep copy constructor
        deepCopy = drpStella.FiberTrace(self.fiberTrace, True)
        self.assertFiberTracesEqual(deepCopy, self.fiberTrace)
        self.assertCopy(deepCopy.trace.image, self.fiberTrace.trace.image, True)

    def testPickle(self):
        """Test pickling

        Round trips the fiber trace object through pickling.
        """
        copy = pickle.loads(pickle.dumps(self.fiberTrace))
        self.assertFiberTracesEqual(copy, self.fiberTrace)
        self.assertCopy(copy.trace.image, self.fiberTrace.trace.image, True)

    def assertSpectrumValues(self, spectrum, image=None, variance=None, mask=0, norm=None,
                             imageTol=0.0, varianceTol=0.0):
        """Assert that the spectrum has the expected values

        Parameters
        ----------
        spectrum : `pfs.drp.stella.Spectrum`
            Spectrum to test.
        image : `numpy.ndarray` or scalar
            Expected values for the image part of the spectrum
            (``spectrum.flux``).
        variance : `numpy.ndarray` or scalar
            Expected values for the variance part of the spectrum.
        mask : `numpy.ndarray` or scalar
            Expected values for the mask part of the spectrum.
        norm : `numpy.ndarray` or scalar
            Expected values for the norm part of the spectrum.
        imageTol : `float`
            Relative tolerance for image comparison.
        varianceTol : `float`
            Relative tolerance for variance comparison.
        """
        beforeSlice = slice(0, self.xy0.getY())
        specSlice = slice(self.xy0.getY(), self.xy0.getY() + self.height)
        afterSlice = slice(self.xy0.getY() + self.height, self.fullDims.getY())
        if norm is None:
            norm = np.sum(self.trace.image.array, axis=1)

        self.assertFloatsEqual(spectrum.norm[beforeSlice], 0.0)
        self.assertFloatsAlmostEqual(spectrum.norm[specSlice], norm, atol=1.0e-6)
        self.assertFloatsEqual(spectrum.norm[beforeSlice], 0.0)
        if image is not None:
            self.assertFloatsEqual(spectrum.flux[beforeSlice], 0.0)
            self.assertFloatsAlmostEqual(spectrum.flux[specSlice], image*spectrum.norm[specSlice],
                                         rtol=imageTol)
            self.assertFloatsEqual(spectrum.flux[afterSlice], 0.0)
        if variance is not None:
            self.assertFloatsEqual(spectrum.variance[beforeSlice], 0.0)
            self.assertFloatsAlmostEqual(spectrum.variance[specSlice], variance*spectrum.norm[specSlice]**2,
                                         rtol=varianceTol)
            self.assertFloatsEqual(spectrum.variance[beforeSlice], 0.0)
        if mask is not None:
            self.assertFloatsEqual(spectrum.mask.array[0, beforeSlice], self.noData)
            self.assertFloatsEqual(spectrum.mask.array[0, specSlice], mask)
            self.assertFloatsEqual(spectrum.mask.array[0, beforeSlice], self.noData)
        self.assertEqual(spectrum.fiberId, self.fiberId)

    def testExtractOptimal(self):
        """Vanilla optimal extraction"""
        spectrum = self.fiberTrace.extractSpectrum(self.image, self.bad)
        self.assertSpectrumValues(spectrum, image=1.0, variance=self.optimalVariance,
                                  imageTol=1.0e-14, varianceTol=1.0e-6)

    def testExtractOptimalMissing(self):
        """Optimal extraction with some pixels removed from the trace

        Checks that the ``FIBERTRACE`` mask is being respected.
        """
        badCol = 5
        self.fiberTrace.trace.mask.array[:, badCol] = 0
        spectrum = self.fiberTrace.extractSpectrum(self.image, self.bad)
        expectVariance = 1.0/(np.sum(self.subimage.image.array**2/self.variance, axis=1) -
                              self.subimage.image.array[:, badCol]**2/self.variance)
        expectNorm = self.fiberTrace.trace.image.array.sum(axis=1)
        expectNorm -= self.fiberTrace.trace.image.array[:, badCol]
        self.assertSpectrumValues(spectrum, image=1.0, variance=expectVariance, norm=expectNorm,
                                  imageTol=1.0e-14, varianceTol=1.0e-6)

    def testExtractOptimalMasked(self):
        """Optimal extraction with some input pixels masked

        Checks that the image mask is being respected.
        """
        badCol = 5
        self.subimage.mask.array[:, badCol] = self.bad
        spectrum = self.fiberTrace.extractSpectrum(self.image, self.bad)
        expectVariance = 1.0/(np.sum(self.subimage.image.array**2/self.variance, axis=1) -
                              self.subimage.image.array[:, badCol]**2/self.variance)
        # BAD doesn't get propagated to the spectrum mask because it's excluded
        self.assertSpectrumValues(spectrum, image=1.0, variance=expectVariance, mask=0,
                                  imageTol=1.0e-14, varianceTol=1.0e-6)

    def testExtractOptimalBadRow(self):
        """Optimal extraction with a bad row of pixels

        Checks that bad spectral elements are masked.
        """
        badRow = 23
        self.subimage.mask.array[badRow, :] = self.bad
        spectrum = self.fiberTrace.extractSpectrum(self.image, self.bad, 0.0)
        expectImage = np.ones(self.height, dtype=float)
        expectImage[badRow] = 0.0
        expectVariance = self.optimalVariance
        expectVariance[badRow] = 0.0
        expectMask = np.zeros(self.height, dtype=np.int32)
        # The mask should be NO_DATA|BAD: NO_DATA because the value is zero, BAD tells us why
        expectMask[badRow] = spectrum.mask.getPlaneBitMask(["NO_DATA", "BAD"])
        self.assertSpectrumValues(spectrum, image=expectImage, variance=expectVariance,
                                  mask=expectMask, imageTol=1.0e-14, varianceTol=1.0e-6)

    def testConstructImage(self):
        """Test construction of image from the spectrum

        Reverse process of extraction.
        """
        spectrum = drpStella.Spectrum(self.fullDims.getY(), self.fiberId)
        spectrum.flux[self.xy0.getY():self.xy0.getY() + self.height] = 1.0
        image = self.fiberTrace.constructImage(spectrum, self.bbox)
        self.assertEqual(image.getBBox(), self.bbox)

        expected = self.subimage.image.array
        self.assertFloatsAlmostEqual(image[self.bbox, lsst.afw.image.PARENT].array, expected, atol=1.0e-7)
        image[self.bbox, lsst.afw.image.PARENT].set(0.0)
        self.assertFloatsEqual(image.array, 0.0)

        # Modify provided image
        image.set(0.0)
        self.fiberTrace.constructImage(image, spectrum)
        self.assertFloatsAlmostEqual(image[self.bbox, lsst.afw.image.PARENT].array, expected, atol=1.0e-7)
        # Check that we're actually adding, not simply setting
        self.fiberTrace.constructImage(image, spectrum)
        self.assertFloatsAlmostEqual(image[self.bbox, lsst.afw.image.PARENT].array, 2*expected, atol=1.0e-7)
        image[self.bbox, lsst.afw.image.PARENT].set(0.0)
        self.assertFloatsEqual(image.array, 0.0)

    def testApplyToMask(self):
        """Test application of trace mask to image"""
        self.fiberTrace.applyToMask(self.image.mask)
        self.assertImagesEqual(self.image.mask[self.bbox, lsst.afw.image.PARENT], self.trace.mask)


class BoxcarFiberTraceTestCase(lsst.utils.tests.TestCase):
    @methodParameters(radius=(2.5, 3.5, 5.0),
                      rtol=(2.0e-2, 1.0e-2, 5.0e-3))
    def testBoxcar(self, radius, rtol):
        """Extraction with a boxcar fiberTrace"""
        flux = 100000
        rng = np.random.RandomState(12345)
        badBitmask = 0x0
        config = SyntheticConfig()
        image = lsst.afw.image.makeMaskedImage(makeSyntheticFlat(config, flux=flux, addNoise=False, rng=rng))
        detMap = makeSyntheticDetectorMap(config)
        sigma = fwhmToSigma(config.fwhm)
        factor = erf(radius/sigma/np.sqrt(2))
        expectFlux = flux*factor/(2*radius)

        for fiberId in config.fiberId:
            trace = drpStella.FiberTrace.boxcar(fiberId, config.dims, radius, detMap.getXCenter(fiberId))
            spectrum = trace.extractAperture(image, badBitmask)
            self.assertFloatsAlmostEqual(spectrum.norm, np.sum(trace.trace.image.array, axis=1), atol=1.0e-6)
            self.assertFloatsAlmostEqual(spectrum.flux, expectFlux/spectrum.norm, rtol=rtol)
            self.assertEqual(spectrum.fiberId, fiberId)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    from argparse import ArgumentParser
    parser = ArgumentParser(__file__)
    parser.add_argument("--display", help="Display backend")
    args, argv = parser.parse_known_args()
    display = args.display
    unittest.main(failfast=True, argv=[__file__] + argv)
