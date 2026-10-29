import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.afw.detection import GaussianPsf
from lsst.geom import Box2D, Point2I

from pfs.datamodel import CalibIdentity
from pfs.drp.stella import PolynomialDistortion
from pfs.drp.stella.fiberProfile import FiberProfile
from pfs.drp.stella.fiberProfileSet import FiberProfileSet
from pfs.drp.stella.pipelines.psfMatchCalibs import PsfMatchCalibsTask, addOpticsDistortion
from pfs.drp.stella.pipelines.psfMatchCalibs import convolveFiberProfile
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.tests.utils import runTests

display = None


def makeDeltaKernel(size: int, dx: int, dy: int) -> afwMath.Kernel:
    """Make a kernel that shifts the image

    Parameters
    ----------
    size : `int`
        Width and height of kernel (odd).
    dx, dy : `int`
        Offset of the delta function from the kernel center.

    Returns
    -------
    kernel : `lsst.afw.math.Kernel`
        Delta-function kernel.
    """
    image = afwImage.ImageD(size, size)
    image.array[size // 2 + dy, size // 2 + dx] = 1.0
    return afwMath.FixedKernel(image)


def makeGaussianKernel(width: int, height: int, sigma: float) -> afwMath.Kernel:
    """Make a kernel that is Gaussian in x and a delta function in y

    Parameters
    ----------
    width, height : `int`
        Dimensions of kernel (odd).
    sigma : `float`
        Gaussian sigma in x (pixels).

    Returns
    -------
    kernel : `lsst.afw.math.Kernel`
        Gaussian kernel.
    """
    image = afwImage.ImageD(width, height)
    xx = np.arange(width) - width // 2
    image.array[height // 2, :] = np.exp(-0.5 * (xx / sigma) ** 2)
    image.array /= image.array.sum()
    return afwMath.FixedKernel(image)


def makeVaryingKernel(size: int, bbox) -> afwMath.Kernel:
    """Make a spatially-varying kernel

    The kernel is a combination of a central delta function and a delta
    function offset by one pixel in x and y, with the relative weights
    varying linearly over the image.

    Parameters
    ----------
    size : `int`
        Width and height of kernel (odd).
    bbox : `lsst.geom.Box2I`
        Bounding box of image.

    Returns
    -------
    kernel : `lsst.afw.math.LinearCombinationKernel`
        Spatially-varying kernel.
    """
    center = size // 2
    basisList = [
        afwMath.DeltaFunctionKernel(size, size, Point2I(center, center)),
        afwMath.DeltaFunctionKernel(size, size, Point2I(center + 1, center + 1)),
    ]
    # Weights vary from 0.1 to 0.3 for the offset delta function
    width = bbox.getWidth()
    height = bbox.getHeight()
    offsetFunc = afwMath.PolynomialFunction2D([0.1, 0.1 / width, 0.1 / height])
    centerFunc = afwMath.PolynomialFunction2D([0.9, -0.1 / width, -0.1 / height])
    return afwMath.LinearCombinationKernel(basisList, [centerFunc, offsetFunc])


class PsfMatchCalibsTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.synthConfig = SyntheticConfig()
        self.detectorMap = makeSyntheticDetectorMap(self.synthConfig, 650.0, 950.0)
        self.psf = GaussianPsf(17, 17, 1.3)
        self.identity = CalibIdentity("2026-01-01", 1, "r", 12345)
        config = PsfMatchCalibsTask.ConfigClass()
        config.referenceVisit = 1
        config.numFibers = 10
        config.numRows = 10
        config.distortionOrder = 2
        self.task = PsfMatchCalibsTask(config=config)

    def adjustDetectorMap(self, kernel):
        """Measure peak shifts and adjust the detectorMap"""
        shifts = self.task.measurePeakShifts(kernel, self.psf, self.detectorMap)
        fit = self.task.fitDistortion(self.detectorMap, shifts)
        newDetectorMap = addOpticsDistortion(self.detectorMap, fit.distortion)
        self.task.checkDetectorMap(self.detectorMap, newDetectorMap, shifts, fit)
        return newDetectorMap, shifts

    def calculatePositionChange(self, newDetectorMap, shifts):
        """Calculate change in position for fixed fiberId,wavelength"""
        wavelength = self.detectorMap.findWavelength(shifts.fiberId, shifts.row).astype(float)
        oldPoints = self.detectorMap.findPoint(shifts.fiberId, wavelength)
        newPoints = newDetectorMap.findPoint(shifts.fiberId, wavelength)
        return newPoints - oldPoints

    def testAddOpticsDistortion(self):
        """Test that addOpticsDistortion shifts the positions"""
        bbox = Box2D(self.detectorMap.bbox)
        numCoeff = PolynomialDistortion.getNumDistortionForOrder(1)
        xCoeff = np.zeros(numCoeff, dtype=float)
        yCoeff = np.zeros(numCoeff, dtype=float)
        xCoeff[0] = 0.5
        yCoeff[0] = -0.25
        distortion = PolynomialDistortion(1, bbox, xCoeff, yCoeff)
        newDetectorMap = addOpticsDistortion(self.detectorMap, distortion)

        fiberId = self.detectorMap.fiberId
        for ff in fiberId:
            wavelength = self.detectorMap.getWavelength(ff)[::100].astype(float)
            oldPoints = self.detectorMap.findPoint(ff, wavelength)
            newPoints = newDetectorMap.findPoint(ff, wavelength)
            self.assertFloatsAlmostEqual(newPoints[:, 0] - oldPoints[:, 0], 0.5, atol=1.0e-3)
            self.assertFloatsAlmostEqual(newPoints[:, 1] - oldPoints[:, 1], -0.25, atol=1.0e-3)

    def testDeltaKernel(self):
        """Test that a delta-function kernel shifts the detectorMap

        A delta function offset by +1 pixel in x moves the image by -1 pixel
        in x (because convolution is actually a correlation). The PSF is
        unchanged, so the peak-to-centroid bias is unchanged.
        """
        kernel = makeDeltaKernel(5, 1, 0)
        newDetectorMap, shifts = self.adjustDetectorMap(kernel)
        self.assertFloatsAlmostEqual(shifts.dxCentroid, -1.0, atol=1.0e-12)
        self.assertFloatsAlmostEqual(shifts.dyCentroid, 0.0, atol=1.0e-12)
        self.assertFloatsAlmostEqual(shifts.dx, -1.0, atol=1.0e-3)
        self.assertFloatsAlmostEqual(shifts.dy, 0.0, atol=1.0e-3)
        diff = self.calculatePositionChange(newDetectorMap, shifts)
        self.assertFloatsAlmostEqual(diff[:, 0], -1.0, atol=2.0e-3)
        self.assertFloatsAlmostEqual(diff[:, 1], 0.0, atol=2.0e-3)

    def testVaryingKernel(self):
        """Test that a spatially-varying kernel shifts the detectorMap

        The kernel is asymmetric, so the peak-to-centroid bias changes.
        """
        kernel = makeVaryingKernel(5, self.detectorMap.bbox)
        newDetectorMap, shifts = self.adjustDetectorMap(kernel)

        # Centroid shift is minus the weight of the offset delta function
        weight = 0.1 + 0.1 * shifts.xx / self.detectorMap.bbox.getWidth()
        weight += 0.1 * shifts.yy / self.detectorMap.bbox.getHeight()
        self.assertFloatsAlmostEqual(shifts.dxCentroid, -weight, atol=1.0e-12)
        self.assertFloatsAlmostEqual(shifts.dyCentroid, -weight, atol=1.0e-12)

        # Peak moves less than the centroid: the peak is dominated by the central delta function
        self.assertTrue(np.all(np.abs(shifts.dx) < np.abs(shifts.dxCentroid)))

        # The bias has some high-frequency structure (presumably from the pixel phase) that the
        # polynomial distortion cannot follow.
        diff = self.calculatePositionChange(newDetectorMap, shifts)
        self.assertFloatsAlmostEqual(diff[:, 0], shifts.dx, atol=1.0e-2)
        self.assertFloatsAlmostEqual(diff[:, 1], shifts.dy, atol=1.0e-2)
        self.assertLess(np.std(diff[:, 0] - shifts.dx), 4.0e-3)
        self.assertLess(np.std(diff[:, 1] - shifts.dy), 4.0e-3)

    def testProfileDeltaKernel(self):
        """Test that a shifted profile with a delta-function kernel is unchanged

        The delta-function kernel shifts the profile by -1 pixel, and the trace
        moves by -1 pixel, so the profile relative to the trace is unchanged.
        """
        profile = FiberProfile.makeGaussian(1.5, self.synthConfig.height, 5, 10)
        kernel = makeDeltaKernel(5, 1, 0)
        xOld = np.full(len(profile.rows), 100.0)
        xNew = xOld - 1.0
        convolved = convolveFiberProfile(profile, kernel, xOld, xNew)
        # Values within a pixel of the edge are lost by the shift
        interior = np.abs(profile.index) < profile.radius
        self.assertFloatsAlmostEqual(
            convolved.profiles.data[:, interior], profile.profiles.data[:, interior], atol=1.0e-12
        )

    def testProfileGaussianKernel(self):
        """Test convolving a Gaussian profile with a Gaussian kernel"""
        sigma = 1.5
        kernelSigma = 1.0
        shift = 0.3
        profile = FiberProfile.makeGaussian(sigma, self.synthConfig.height, 8, 10)
        kernel = makeGaussianKernel(9, 3, kernelSigma)
        xOld = np.full(len(profile.rows), 100.0)
        xNew = xOld - shift
        convolved = convolveFiberProfile(profile, kernel, xOld, xNew)
        for values in convolved.profiles:
            values = values.filled(0.0)
            norm = values.sum()
            centroid = np.sum(values * profile.index) / norm
            variance = np.sum(values * (profile.index - centroid) ** 2) / norm
            # Profile is relative to the new trace, which moved by -shift
            self.assertFloatsAlmostEqual(centroid, shift, atol=1.0e-3)
            self.assertFloatsAlmostEqual(variance, sigma**2 + kernelSigma**2, rtol=1.0e-2)

    def testAdjustFiberProfiles(self):
        """Test adjustFiberProfiles"""
        profile = FiberProfile.makeGaussian(1.5, self.synthConfig.height, 5, 10)
        profiles = FiberProfileSet({ff: profile.copy() for ff in self.detectorMap.fiberId}, self.identity)
        kernel = makeDeltaKernel(5, 1, 0)
        newDetectorMap, _ = self.adjustDetectorMap(kernel)
        visitInfo = afwImage.VisitInfo()
        newProfiles = self.task.adjustFiberProfiles(
            profiles, kernel, self.detectorMap, newDetectorMap, self.identity, visitInfo
        )
        self.assertEqual(set(newProfiles.fiberId), set(profiles.fiberId))
        for ff in profiles:
            # Not exact, because the detectorMap shift is fit
            self.assertFloatsAlmostEqual(
                newProfiles[ff].profiles.data, profiles[ff].profiles.data, atol=2.0e-3
            )

    def testConfigDefaults(self):
        """Test the default configuration"""
        config = PsfMatchCalibsTask.ConfigClass()
        self.assertTrue(config.doAdjustReference)
        self.assertFalse(config.adjustReference.doSlitOffsets)
        self.assertEqual(config.psfMatch.dfKernel.spatialKernelOrder, 3)

    def testConfigValidate(self):
        """Test that referenceVisit is required"""
        config = PsfMatchCalibsTask.ConfigClass()
        with self.assertRaises(ValueError):
            config.validate()
        config.referenceVisit = 123
        config.validate()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
