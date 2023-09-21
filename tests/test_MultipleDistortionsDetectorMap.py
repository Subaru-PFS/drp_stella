import pickle
from itertools import product

import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils
from lsst.afw.detection import GaussianPsf
from lsst.afw.image import ExposureF
from lsst.geom import Point2D, Box2D
from lsst.pex.exceptions import DomainError

from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella import DetectorMap, DoubleDistortion
from pfs.drp.stella import MultipleDistortionsDetectorMap, ReferenceLineStatus, ImagingSpectralPsf
from pfs.drp.stella.arcLine import ArcLine, ArcLineSet
from pfs.drp.stella.fitDistortedDetectorMap import FitDistortedDetectorMapTask
from pfs.drp.stella.tests.utils import runTests, methodParameters
from pfs.drp.stella.referenceLine import ReferenceLineSource


display = None


class MultipleDistortionsDetectorMapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Construct a ``SplinedDetectorMap`` to play with"""
        self.synthConfig = SyntheticConfig()
        self.minWl = 650.0
        self.maxWl = 950.0
        self.base = makeSyntheticDetectorMap(self.synthConfig, self.minWl, self.maxWl)
        self.metadata = 123456
        self.darkTime = 12345.6

    def makeMultipleDistortionsDetectorMap(self, likeBase: bool) -> MultipleDistortionsDetectorMap:
        """Construct a `MultipleDistortionsDetectorMap`

        Parameters
        ----------
        likeBase : `bool`
            Should the output have similar positions as the ``self.base``
            `SplinedDetectorMap`? This allows the use of ``assertPositions``,
            but the internal values are not random. Use ``True`` if you want to
            check calculations; use ``False`` if you want to check persistence.

        Returns
        -------
        detMap : `pfs.drp.stella.MultipleDistortionsDetectorMap`
            `MultipleDistortionsDetectorMap` to be used in tests.
        """
        base = self.base.clone()

        if likeBase:
            distortionOrder = 1
            numCoeffs = DoubleDistortion.getNumDistortionForOrder(distortionOrder)
            xLeft = np.zeros(numCoeffs, dtype=float)
            yLeft = np.zeros(numCoeffs, dtype=float)
            xRight = np.zeros(numCoeffs, dtype=float)
            yRight = np.zeros(numCoeffs, dtype=float)

            # Introduce a non-zero distortion field that replicates the base detectorMap
            offset = 0.5
            xLeft[0] = offset
            yLeft[0] = offset
            xRight[0] = offset
            yRight[0] = offset
            slitOffsets = np.full(base.getNumFibers(), -offset, dtype=float)
            base.setSlitOffsets(slitOffsets, slitOffsets)
        else:
            distortionOrder = 3
            numCoeffs = DoubleDistortion.getNumDistortionForOrder(distortionOrder)

            # Introduce a random distortion field; will likely get weird values, but good for testing I/O
            rng = np.random.RandomState(12345)
            xLeft = rng.uniform(size=numCoeffs)
            yLeft = rng.uniform(size=numCoeffs)
            xRight = rng.uniform(size=numCoeffs)
            yRight = rng.uniform(size=numCoeffs)

        distortion = DoubleDistortion(distortionOrder, Box2D(base.bbox), xLeft, yLeft, xRight, yRight)

        visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        metadata = lsst.daf.base.PropertyList()
        metadata.set("METADATA", self.metadata)
        return MultipleDistortionsDetectorMap(base, [distortion], visitInfo, metadata)

    def assertPositions(self, detMap: MultipleDistortionsDetectorMap):
        """Check that the detectorMap reproduces the results of base"""
        for fiberId in self.base.getFiberId():
            self.assertFloatsAlmostEqual(detMap.getXCenter(fiberId), self.base.getXCenter(fiberId),
                                         atol=1.0e-3)
            self.assertFloatsAlmostEqual(detMap.getWavelength(fiberId), self.base.getWavelength(fiberId),
                                         atol=5.0e-4)

        # Metadata: we only care that what we planted is there;
        # there may be other stuff that we don't care about.
        self.assertTrue(detMap.getMetadata() is not None)
        self.assertTrue(detMap.metadata is not None)
        self.assertIn("METADATA", detMap.metadata.names())
        self.assertEqual(detMap.metadata.get("METADATA"), self.metadata)
        # VisitInfo: only checking one element, assuming the rest are protected by afw unit tests
        self.assertTrue(detMap.visitInfo is not None)
        self.assertTrue(detMap.getVisitInfo() is not None)
        self.assertEqual(detMap.visitInfo.getDarkTime(), self.darkTime)

    def assertMultipleDistortionsDetectorMapsEqual(self, lhs, rhs):
        """Assert that the ``MultipleDistortionsDetectorMap``s are the same"""
        self.assertFloatsEqual(lhs.fiberId, rhs.fiberId)

        # Base
        self.assertFloatsEqual(lhs.base.fiberId, rhs.base.fiberId)
        for ff in lhs.base.fiberId:
            self.assertFloatsEqual(lhs.base.getXCenterSpline(ff).getX(),
                                   rhs.base.getXCenterSpline(ff).getX())
            self.assertFloatsEqual(lhs.base.getXCenterSpline(ff).getY(),
                                   rhs.base.getXCenterSpline(ff).getY())
            self.assertFloatsEqual(lhs.base.getWavelengthSpline(ff).getX(),
                                   rhs.base.getWavelengthSpline(ff).getX())
            self.assertFloatsEqual(lhs.base.getWavelengthSpline(ff).getY(),
                                   rhs.base.getWavelengthSpline(ff).getY())
            self.assertEqual(lhs.base.getSpatialOffset(ff), rhs.base.getSpatialOffset(ff))
            self.assertEqual(lhs.base.getSpectralOffset(ff), rhs.base.getSpectralOffset(ff))

        # Distortions
        self.assertEqual(len(lhs.distortions), len(rhs.distortions))
        for ll, rr in zip(lhs.distortions, rhs.distortions):
            self.assertFloatsEqual(ll.getXLeftCoefficients(), rr.getXLeftCoefficients())
            self.assertFloatsEqual(ll.getYLeftCoefficients(), rr.getYLeftCoefficients())
            self.assertFloatsEqual(ll.getXRightCoefficients(), rr.getXRightCoefficients())
            self.assertFloatsEqual(ll.getYRightCoefficients(), rr.getYRightCoefficients())

        # Metadata
        for name in lhs.metadata.names():
            self.assertEqual(lhs.metadata.get(name), rhs.metadata.get(name))

        # VisitInfo: only checking one element, assuming the rest are protected by afw unit tests
        self.assertEqual(lhs.visitInfo.getDarkTime(), rhs.visitInfo.getDarkTime())

    def testBasic(self):
        """Test basic functionality"""
        detMap = self.makeMultipleDistortionsDetectorMap(True)
        self.assertPositions(detMap)

        if display is not None:
            from pfs.drp.stella.synthetic import makeSyntheticFlat
            from lsst.afw.display import Display
            disp = Display(frame=1, backend=display)
            disp.mtv(makeSyntheticFlat(self.synthConfig))
            detMap.display(disp)

    def testSlitOffsets(self):
        """Test different value for one of the slit offsets"""
        self.synthConfig.slope = 0.0  # Straighten the traces to avoid coupling x,y offsets
        detMap = self.makeMultipleDistortionsDetectorMap(True)
        middle = len(self.base)//2

        spatial = detMap.getSpatialOffsets()
        spectral = detMap.getSpectralOffsets()
        spatial[middle] += 0.54321
        spectral[middle] -= 0.54321
        detMap.setSlitOffsets(spatial, spectral)

        spatial = self.base.getSpatialOffsets()
        spectral = self.base.getSpectralOffsets()
        spatial[middle] += 0.54321
        spectral[middle] -= 0.54321
        self.base.setSlitOffsets(spatial, spectral)

        self.assertPositions(detMap)

    def testFinds(self):
        """Test the various ``find*`` methods

        We throw down a random array of points on the image, run the
        ``find*`` methods and check that the answers are consistent.
        """
        num = 1000
        detMap = self.makeMultipleDistortionsDetectorMap(True)
        indices = np.arange(0, detMap.bbox.getHeight())
        xCenter = np.array([detMap.getXCenter(ff) for ff in detMap.fiberId])
        numFibers = len(detMap)
        rng = np.random.RandomState(54321)
        for ii, (xx, yy) in enumerate(zip(rng.uniform(0, detMap.bbox.getWidth() - 1, size=num),
                                          rng.uniform(0, detMap.bbox.getHeight() - 1, size=num))):
            fiberId = detMap.findFiberId(lsst.geom.Point2D(xx, yy))

            cols = np.arange(numFibers, dtype=int)
            rows = (yy + 0.5).astype(int)
            distances = [np.abs(xCenter[cc][rows] - xx) for cc in cols]
            closestIndex = np.argmin(distances)
            first, second = np.partition(distances, 1)[0:2]  # Two smallest values
            if np.fabs(first - second) < 1.0e-2:
                # We're right on the threshold, and the code could be forgiven for choosing a different
                # index than we did (e.g., spline vs linear interpolation); but that choice has large
                # consequences on the rest of the tests below, so skip this one.
                continue
            self.assertEqual(fiberId, detMap.fiberId[closestIndex])

            wavelength = detMap.findWavelength(fiberId, yy)
            wavelengthExpect = np.interp(yy, indices, detMap.getWavelength(fiberId))
            self.assertFloatsAlmostEqual(wavelength, wavelengthExpect, atol=1.0e-3)

            point = detMap.findPoint(fiberId, wavelength)
            self.assertFloatsAlmostEqual(point.getY(), yy, atol=1.0e-3)

    def testReadWriteFits(self):
        """Test reading and writing to/from FITS"""
        detMap = self.makeMultipleDistortionsDetectorMap(False)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = MultipleDistortionsDetectorMap.readFits(filename)
            self.assertMultipleDistortionsDetectorMapsEqual(detMap, copy)
        # Read with parent class
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = DetectorMap.readFits(filename)
            self.assertMultipleDistortionsDetectorMapsEqual(detMap, copy)

    def testPickle(self):
        """Test round-trip pickle"""
        detMap = self.makeMultipleDistortionsDetectorMap(False)
        copy = pickle.loads(pickle.dumps(detMap))
        self.assertMultipleDistortionsDetectorMapsEqual(detMap, copy)

    def testPersistable(self):
        """Test behaviour as a Persistable

        This involves sticking it in a Psf, which gets stuck in an Exposure
        """
        size = 21
        sigma = 3
        detMap = self.makeMultipleDistortionsDetectorMap(False)
        imagePsf = GaussianPsf(size, size, sigma)
        psf = ImagingSpectralPsf(imagePsf, detMap)
        exposure = ExposureF(size, size)
        exposure.setPsf(psf)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exposure.writeFits(filename)
            copy = ExposureF(filename).getPsf().getDetectorMap()
            copy.metadata.set("METADATA", self.metadata)  # Persistence doesn't preserve the metadata
            self.assertMultipleDistortionsDetectorMapsEqual(detMap, copy)

    @methodParameters(arm=("r", "m"))
    def testFit(self, arm):
        """Test FitDistortedDetectorMapTask

        Parameters
        ----------
        arm : `str`
            Spectrograph arm; affects behaviour of
            `FitDistortedDetectorMapTask`.
        """
        flux = 1000.0
        fluxErr = 1.0
        bbox = self.base.bbox
        lines = []
        for ff in self.synthConfig.fiberId:
            for yy in range(bbox.getMinY(), bbox.getMaxY()):
                lines.append(ArcLine(ff, self.base.getWavelength(ff, yy), self.base.getXCenter(ff, yy),
                             float(yy), 0.01, 0.01, np.nan, np.nan, np.nan, flux, fluxErr, np.nan, False,
                             ReferenceLineStatus.GOOD, "Fake", None, ReferenceLineSource.NONE))
        lines = ArcLineSet.fromRows(lines)
        config = FitDistortedDetectorMapTask.ConfigClass()
        config.order = 1
        config.doSlitOffsets = True
        config.exclusionRadius = 1.0  # We've got a lot of close lines, but no real fear of confusion
        task = FitDistortedDetectorMapTask(name="fitDistortedDetectorMap", config=config)
        task.log.setLevel(task.log.DEBUG)
        dataId = dict(visit=12345, arm=arm, spectrograph=1)
        detMap = task.run(dataId, bbox, lines, self.base.visitInfo, base=self.base).detectorMap
        self.assertEqual(len(detMap.distortions), 1)
        self.assertFloatsAlmostEqual(detMap.distortions[0].getCoefficients(), 0.0, atol=1.0e-7)
        self.assertFloatsAlmostEqual(detMap.getSpatialOffsets(), 0.0, atol=1.0e-7)
        self.assertFloatsAlmostEqual(detMap.getSpectralOffsets(), 0.0, atol=1.0e-7)

    def testOutOfRange(self):
        """Test that inputs that are out-of-range produce NaNs"""
        detMap = self.makeMultipleDistortionsDetectorMap(True)

        goodFiberId = (self.synthConfig.fiberId[0], self.synthConfig.fiberId[-1])
        goodWavelength = (self.minWl + 0.1, 0.5*(self.minWl + self.maxWl), self.maxWl - 0.1)
        badFiberId = (-1, 12345)
        badWavelength = (self.minWl - 5, self.maxWl + 5)

        for ff, wl in product(goodFiberId, goodWavelength):
            point = detMap.findPoint(ff, wl)
            self.assertTrue(np.all(np.isfinite(point)))
        for ff, wl in product(goodFiberId, badWavelength):
            point = detMap.findPoint(ff, wl)
            self.assertFalse(np.any(np.isfinite(point)))
        for ff, wl in product(badFiberId, goodWavelength):
            point = detMap.findPoint(ff, wl)
            self.assertFalse(np.any(np.isfinite(point)))
        for ff, wl in product(badFiberId, badWavelength):
            point = detMap.findPoint(ff, wl)
            self.assertFalse(np.any(np.isfinite(point)))

    def testFindFiberIdOutOfRange(self):
        """Test that findFiberId works with out-of-range input"""
        detMap = self.makeMultipleDistortionsDetectorMap(True)
        self.assertRaises(DomainError, detMap.findFiberId, Point2D(6000, -20000))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
