import pickle

import numpy as np

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.daf.base
import lsst.afw.image
import lsst.afw.image.testUtils
import lsst.afw.display

import pfs.drp.stella.synthetic
from pfs.drp.stella import GlobalDetectorMap, GlobalDetectorModel, BaseDetectorMap
from pfs.drp.stella.tests.utils import runTests, methodParameters

display = None


class GlobalDetectorMapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Construct a ``SplinedDetectorMap`` to play with"""
        self.synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        self.original = None  # Create with makeSyntheticDetectorMap
        self.metadata = 123456
        self.darkTime = 12345.6
        self.xCenterTol = 1.0e-4
        self.wavelengthTol = 1.0e-4

    def makeSyntheticDetectorMap(self):
        self.original = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(self.synthConfig)
        return self.original

    def makeGlobalDetectorMap(self, original=None, order=1, dualDetector=True):
        if original is None:
            original = self.makeSyntheticDetectorMap()
        fiberId, yy = np.meshgrid(original.getFiberId(),
                                  np.arange(original.bbox.getMinY(), original.bbox.getMaxY()).astype(float))
        fiberId = fiberId.flatten()
        yy = yy.flatten()
        wavelength = np.array([original.getWavelength(ff, row) for ff, row in zip(fiberId, yy)])
        xx = np.array([original.getXCenter(ff, row) for ff, row in zip(fiberId, yy)])
        xErr = np.full_like(xx, 0.01)
        yErr = np.full_like(yy, 0.01)
        model = GlobalDetectorModel.fit(original.bbox, order, dualDetector,
                                        fiberId.astype(np.int32), wavelength,
                                        xx, yy, xErr, yErr)
        visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        metadata = lsst.daf.base.PropertyList()
        metadata.set("METADATA", self.metadata)
        return GlobalDetectorMap(model, visitInfo, metadata)

    def assertGlobalDetectorMap(self, detMap):
        """Assert that a ``GlobalDetectorMap`` matches expectations"""
        self.assertEqual(detMap.bbox, self.original.bbox)
        self.assertFloatsEqual(detMap.fiberId, self.original.fiberId)
        for fiberId in detMap.fiberId:
            self.assertFloatsAlmostEqual(detMap.getXCenter(fiberId), self.original.getXCenter(fiberId),
                                         atol=1.0e-3)
            self.assertFloatsAlmostEqual(detMap.getWavelength(fiberId), self.original.getWavelength(fiberId),
                                         atol=1.0e-3)

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

    def testBasic(self):
        """Test basic functionality

        Constructor, getters, properties.
        """
        detMap = self.makeGlobalDetectorMap()
        self.assertGlobalDetectorMap(detMap)

        self.assertEqual(detMap.getBBox(), self.original.bbox)
        self.assertFloatsEqual(detMap.getFiberId(), self.original.fiberId)
        self.assertEqual(detMap.getNumFibers(), len(self.original.fiberId))
        self.assertEqual(len(detMap), len(self.original.fiberId))
        self.assertIsNotNone(detMap.getModel())
        self.assertFloatsAlmostEqual(detMap.getSpatialOffsets(), self.original.getSpatialOffsets(),
                                     atol=1.0e-7)
        self.assertFloatsAlmostEqual(detMap.getSpectralOffsets(), self.original.getSpectralOffsets(),
                                     atol=1.0e-7)

    @methodParameters(order=(1, 1, 2, 5), slope=(0.0, 0.1, 0.1, 0.1))
    def testFitting(self, order, slope):
        """Test basic functionality

        Constructor, getters, properties.
        """
        self.synthConfig.slope = slope
        detMap = self.makeGlobalDetectorMap(order=order)
        self.assertGlobalDetectorMap(detMap)

    def testSlitOffsets(self):
        """Test different value for one of the slit offsets"""
        self.synthConfig.slope = 0.0  # Straighten the traces to avoid coupling x,y offsets
        detMap = self.makeGlobalDetectorMap()
        middle = len(self.original)//2
        spatial = self.original.getSpatialOffsets()
        spectral = self.original.getSpectralOffsets()
        spatial[middle] += 0.54321
        spectral[middle] -= 0.54321
        self.original.setSlitOffsets(spatial, spectral)
        detMap.setSlitOffsets(spatial, spectral)
        self.assertGlobalDetectorMap(detMap)

    def testChangeSlitOffsets(self):
        """Test that changing the slit offsets changes the x,wavelength

        Check spatial and spectral separately, because there's a coupling due
        to the tilted traces.
        """
        spatialOffset = 1.2345
        spectralOffset = -3.21
        detMap = self.makeGlobalDetectorMap()
        self.assertGlobalDetectorMap(detMap)
        middle = len(self.original)//2
        fiberId = self.original.fiberId[middle]

        wavelength = detMap.getWavelength(fiberId)[5:-5:5]  # Don't go to the very ends, due to the shift
        before = np.array([detMap.findPoint(fiberId, wl) for wl in wavelength])
        spatial = detMap.getSpatialOffsets()
        spectral = detMap.getSpectralOffsets()

        spatial[middle] += spatialOffset
        detMap.setSlitOffsets(spatial, spectral)
        self.assertFloatsEqual(detMap.getSpatialOffsets(), spatial)
        self.assertFloatsEqual(detMap.getSpectralOffsets(), spectral)
        after = np.array([detMap.findPoint(fiberId, wl) for wl in wavelength])
        self.assertFloatsAlmostEqual(after[:, 0], before[:, 0] + spatialOffset, atol=1.0e-7)

        spatial[middle] -= spatialOffset  # Restore
        spectral[middle] += spectralOffset
        detMap.setSlitOffsets(spatial, spectral)
        self.assertFloatsEqual(detMap.getSpatialOffsets(), spatial)
        self.assertFloatsEqual(detMap.getSpectralOffsets(), spectral)
        after = np.array([detMap.findPoint(fiberId, wl) for wl in wavelength])
        self.assertFloatsAlmostEqual(after[:, 1], before[:, 1] + spectralOffset, atol=1.0e-7)

    def testHoldByValue(self):
        """Test that we hold arrays by value

        After using an array to construct the ``GlobalDetectorMap``, changing the
        array does NOT change the ``GlobalDetectorMap``.
        """
        detMap = self.makeGlobalDetectorMap()
        bbox = detMap.bbox
        bbox.grow(12)
        self.assertNotEqual(detMap.getBBox(), bbox)
        with self.assertRaises(AttributeError):  # "AttributeError: can't set attribute"
            detMap.fiberId += 123

    def testFinds(self):
        """Test the various ``find*`` methods

        We throw down a random array of points on the image, run the
        ``find*`` methods and check that the answers are consistent.
        """
        num = 1000
        detMap = self.makeGlobalDetectorMap()
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
        detMap = self.makeGlobalDetectorMap()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = GlobalDetectorMap.readFits(filename)
            self.assertGlobalDetectorMap(copy)
        # Read with parent class
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = BaseDetectorMap.readFits(filename)
            self.assertGlobalDetectorMap(copy)

    def testPickle(self):
        """Test round-trip pickle"""
        detMap = self.makeGlobalDetectorMap()
        copy = pickle.loads(pickle.dumps(detMap))
        self.assertGlobalDetectorMap(copy)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
