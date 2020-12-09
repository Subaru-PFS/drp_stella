import pickle

import numpy as np
import scipy.interpolate

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.geom
import lsst.afw.image
import lsst.afw.image.testUtils
import lsst.afw.display

import pfs.drp.stella.synthetic
from pfs.drp.stella import SplinedDetectorMap
from pfs.drp.stella.tests.utils import runTests

display = None


def makeSpline(xx, yy):
    """Construct a spline with an alternate implementation

    Spline interpolates from ``xx`` to ``yy``.

    Parameters
    ----------
    xx, yy : `numpy.ndarray`
        Arrays for interpolation.

    Returns
    -------
    spline : `scipy.interpolate.CubicSpline`
        Spline object.
    """
    return scipy.interpolate.CubicSpline(xx, yy, bc_type='not-a-knot')


class SplinedDetectorMapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Construct a ``SplinedDetectorMap`` to play with"""
        synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(synthConfig)
        self.bbox = detMap.bbox
        self.fiberId = detMap.fiberId
        self.numFibers = len(self.fiberId)
        self.centerKnots = [detMap.getXCenterSpline(ff).getX() for ff in self.fiberId]
        self.xCenter = [detMap.getXCenterSpline(ff).getY() for ff in self.fiberId]
        self.wavelengthKnots = [detMap.getWavelengthSpline(ff).getX() for ff in self.fiberId]
        self.wavelength = [detMap.getWavelengthSpline(ff).getY() for ff in self.fiberId]
        self.rng = np.random.RandomState(54321)
        self.spatialOffsets = self.rng.uniform(size=self.numFibers).astype(np.float32)
        self.spectralOffsets = self.rng.uniform(size=self.numFibers).astype(np.float32)
        self.calculateExpectations(detMap)
        self.metadata = 123456
        self.darkTime = 12345.6

    def calculateExpectations(self, detMap):
        """Calculate what to expect for the center and wavelength

        Sets ``self.centerExpect`` and ``self.wavelengthExpect``.
        This accounts for the random slitOffsets.
        """
        self.xCenterExpect = np.zeros((self.numFibers, self.bbox.getHeight()), dtype=np.float32)
        self.wavelengthExpect = np.zeros((self.numFibers, self.bbox.getHeight()), dtype=np.float32)
        for ii, ff in enumerate(self.fiberId):
            yOffset = self.spectralOffsets[ii]
            xOffset = self.spatialOffsets[ii]
            xCenterSpline = makeSpline(detMap.getXCenterSpline(ff).getX(), detMap.getXCenterSpline(ff).getY())
            self.xCenterExpect[ii, :] = xCenterSpline(np.arange(self.bbox.getHeight()) - yOffset) + xOffset
            wlSpline = makeSpline(detMap.getWavelengthSpline(ff).getX(),
                                  detMap.getWavelengthSpline(ff).getY())
            self.wavelengthExpect[ii, :] = wlSpline(np.arange(self.bbox.getHeight()) - yOffset)
        self.xCenterTol = 1.0e-4
        self.wavelengthTol = 2.0e-4

    def makeSplinedDetectorMap(self):
        """Construct a ``SplinedDetectorMap``

        Returns
        -------
        detMap : `pfs.drp.stella.SplinedDetectorMap`
            Detector map.
        """
        detMap = SplinedDetectorMap(self.bbox, self.fiberId, self.centerKnots, self.xCenter,
                                    self.wavelengthKnots, self.wavelength,
                                    self.spatialOffsets, self.spectralOffsets)
        detMap.visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        detMap.metadata.set("METADATA", self.metadata)
        self.assertSplinedDetectorMap(detMap)
        return detMap

    def assertSplinedDetectorMap(self, detMap):
        """Assert that a ``SplinedDetectorMap`` matches expectations"""
        self.assertEqual(detMap.bbox, self.bbox)
        self.assertFloatsEqual(detMap.fiberId, self.fiberId)
        xCenter = np.array([detMap.getXCenter(ff) for ff in self.fiberId])
        wavelength = np.array([detMap.getWavelength(ff) for ff in self.fiberId])
        self.assertFloatsAlmostEqual(xCenter, self.xCenterExpect, atol=self.xCenterTol)
        self.assertFloatsAlmostEqual(wavelength, self.wavelengthExpect, atol=self.wavelengthTol)
        self.assertFloatsEqual(detMap.getSpatialOffsets(), self.spatialOffsets)
        self.assertFloatsEqual(detMap.getSpectralOffsets(), self.spectralOffsets)
        # Metadata: we only care that what we planted is there;
        # there may be other stuff that we don't care about.
        self.assertTrue(detMap.getMetadata() is not None)
        self.assertTrue(detMap.metadata is not None)
        self.assertIn("METADATA", detMap.metadata.names())
        self.assertEqual(detMap.metadata.get("METADATA"), self.metadata)
        # VisitInfo; only checking one element, assuming the rest are protected by afw unit tests
        self.assertTrue(detMap.visitInfo is not None)
        self.assertTrue(detMap.getVisitInfo() is not None)
        self.assertEqual(detMap.visitInfo.getDarkTime(), self.darkTime)

    def testBasic(self):
        """Test basic functionality

        Constructor, getters, setters, properties.
        """
        detMap = self.makeSplinedDetectorMap()
        self.assertSplinedDetectorMap(detMap)

        # Check accessor functions work as well as properties
        self.assertEqual(detMap.getBBox(), self.bbox)
        self.assertFloatsEqual(detMap.getFiberId(), self.fiberId)
        xCenter = np.array([detMap.getXCenter(ff) for ff in self.fiberId])
        wavelength = np.array([detMap.getWavelength(ff) for ff in self.fiberId])
        self.assertFloatsAlmostEqual(xCenter, self.xCenterExpect, atol=self.xCenterTol)
        self.assertFloatsAlmostEqual(wavelength, self.wavelengthExpect, atol=self.wavelengthTol)
        self.assertFloatsEqual(detMap.getSpatialOffsets(), self.spatialOffsets)
        self.assertFloatsEqual(detMap.getSpectralOffsets(), self.spectralOffsets)
        self.assertEqual(detMap.getNumFibers(), len(self.fiberId))
        self.assertEqual(len(detMap), len(self.fiberId))

        # Check setters
        self.spatialOffsets += 1.2345
        self.spectralOffsets += 1.2345
        detMap.setSlitOffsets(self.spatialOffsets, self.spectralOffsets)
        self.calculateExpectations(detMap)
        self.assertSplinedDetectorMap(detMap)

        self.spatialOffsets -= 2.34567
        self.spectralOffsets -= 2.34567
        for ii, fiber in enumerate(self.fiberId):
            detMap.setSlitOffsets(fiber, self.spatialOffsets[ii], self.spectralOffsets[ii])
        self.calculateExpectations(detMap)
        self.assertSplinedDetectorMap(detMap)

        for xCenter in self.xCenter:
            xCenter -= 2.3456
        for wavelength in self.wavelength:
            wavelength += 3.4567
        for ii, fiber in enumerate(self.fiberId):
            detMap.setXCenter(fiber, self.centerKnots[ii], self.xCenter[ii])
            detMap.setWavelength(fiber, self.wavelengthKnots[ii], self.wavelength[ii])
        self.calculateExpectations(detMap)
        self.assertSplinedDetectorMap(detMap)

    def testBadCtor(self):
        """Test that we cannot create a SplinedDetectorMap with invalid arguments"""
        short = 3  # Number of values for short array

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between fiberId and center/wavelength
            SplinedDetectorMap(self.bbox, self.fiberId[:short], self.centerKnots, self.xCenter,
                               self.wavelengthKnots, self.wavelength)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between the center and wavelength
            SplinedDetectorMap(self.bbox, self.fiberId, self.centerKnots[:short], self.xCenter,
                               self.wavelengthKnots, self.wavelength)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between the center and wavelength
            SplinedDetectorMap(self.bbox, self.fiberId, self.centerKnots, self.xCenter[:short],
                               self.wavelengthKnots, self.wavelength)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between the center and wavelength
            SplinedDetectorMap(self.bbox, self.fiberId, self.centerKnots, self.xCenter,
                               self.wavelengthKnots[:short], self.wavelength)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between the center and wavelength
            SplinedDetectorMap(self.bbox, self.fiberId, self.centerKnots, self.xCenter,
                               self.wavelengthKnots, self.wavelength[:short])

        # Various mismatches in the second constructor
        detMap = self.makeSplinedDetectorMap()
        centerKnots = np.array([detMap.getXCenterSpline(ff).getX() for ff in self.fiberId])
        centerValues = np.array([detMap.getXCenterSpline(ff).getY() for ff in self.fiberId])
        wavelengthKnots = np.array([detMap.getWavelengthSpline(ff).getX() for ff in self.fiberId])
        wavelengthValues = np.array([detMap.getWavelengthSpline(ff).getY() for ff in self.fiberId])
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            SplinedDetectorMap(detMap.bbox, detMap.fiberId[:short],
                               centerKnots, centerValues,
                               wavelengthKnots, wavelengthValues,
                               detMap.getSpatialOffsets(), detMap.getSpectralOffsets())
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            SplinedDetectorMap(detMap.bbox, detMap.fiberId,
                               centerKnots[:short], centerValues,
                               wavelengthKnots, wavelengthValues,
                               detMap.getSpatialOffsets(), detMap.getSpectralOffsets())
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            SplinedDetectorMap(detMap.bbox, detMap.fiberId,
                               centerKnots, centerValues[:short],
                               wavelengthKnots, wavelengthValues,
                               detMap.getSpatialOffsets(), detMap.getSpectralOffsets())
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            SplinedDetectorMap(detMap.bbox, detMap.fiberId,
                               centerKnots, centerValues,
                               wavelengthKnots[:short], wavelengthValues,
                               detMap.getSpatialOffsets(), detMap.getSpectralOffsets())
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            SplinedDetectorMap(detMap.bbox, detMap.fiberId,
                               centerKnots, centerValues,
                               wavelengthKnots, wavelengthValues[:short],
                               detMap.getSpatialOffsets(), detMap.getSpectralOffsets())
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            SplinedDetectorMap(detMap.bbox, detMap.fiberId,
                               centerKnots, centerValues,
                               wavelengthKnots, wavelengthValues,
                               detMap.getSpatialOffsets()[:short], detMap.getSpectralOffsets())
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            SplinedDetectorMap(detMap.bbox, detMap.fiberId,
                               centerKnots, centerValues,
                               wavelengthKnots, wavelengthValues,
                               detMap.getSpatialOffsets(), detMap.getSpectralOffsets()[:short])
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            SplinedDetectorMap(detMap.bbox, detMap.fiberId,
                               centerKnots[:short], centerValues[:short],
                               wavelengthKnots, wavelengthValues,
                               detMap.getSpatialOffsets(), detMap.getSpectralOffsets())
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            SplinedDetectorMap(detMap.bbox, detMap.fiberId,
                               centerKnots, centerValues,
                               wavelengthKnots[:short], wavelengthValues[:short],
                               detMap.getSpatialOffsets(), detMap.getSpectralOffsets())

    def testHoldByValue(self):
        """Test that we hold arrays by value

        After using an array to construct the ``SplinedDetectorMap``, changing the
        array does NOT change the ``SplinedDetectorMap``.
        """
        detMap = self.makeSplinedDetectorMap()
        self.bbox.grow(12)
        self.spatialOffsets += 1.2345
        self.spectralOffsets -= 1.2345

        self.assertNotEqual(detMap.getBBox(), self.bbox)
        self.assertFloatsNotEqual(detMap.getSpatialOffsets(), self.spatialOffsets)
        self.assertFloatsNotEqual(detMap.getSpectralOffsets(), self.spectralOffsets)

        with self.assertRaises(AttributeError):  # "AttributeError: can't set attribute"
            detMap.fiberId += 123

    def testFinds(self):
        """Test the various ``find*`` methods

        We throw down a random array of points on the image, run the
        ``find*`` methods and check that the answers are consistent.
        """
        detMap = self.makeSplinedDetectorMap()
        if display:
            image = pfs.drp.stella.synthetic.makeSyntheticFlat()
            dd = lsst.afw.display.Display(1, display)
            dd.mtv(image)

        num = 1000
        indices = np.arange(0, detMap.bbox.getHeight())
        xCenter = np.array([detMap.getXCenter(ff) for ff in self.fiberId])
        numFibers = len(detMap)
        for ii, (xx, yy) in enumerate(zip(self.rng.uniform(0, self.bbox.getWidth() - 1, size=num),
                                      self.rng.uniform(0, self.bbox.getHeight() - 1, size=num))):
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
        detMap = self.makeSplinedDetectorMap()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = SplinedDetectorMap.readFits(filename)
            self.assertSplinedDetectorMap(copy)

    def testPickle(self):
        """Test round-trip pickle"""
        detMap = self.makeSplinedDetectorMap()
        copy = pickle.loads(pickle.dumps(detMap))
        self.assertSplinedDetectorMap(copy)

    def testOffDetector(self):
        """Test that a wavelength off the detector

        It should produce a x,y position off the detector: PIPE2D-675.
        """
        detMap = self.makeSplinedDetectorMap()
        fiberId = detMap.getFiberId()[len(detMap)//2]
        wavelength = detMap.getWavelength(fiberId)
        for wl in (wavelength.min() - 10, wavelength.max() + 10):
            point = detMap.findPoint(fiberId, wl)
            self.assertFalse(detMap.bbox.contains(lsst.geom.Point2I(point)))


class SplinedDetectorMapSlitOffsetsTestCase(lsst.utils.tests.TestCase):
    """Test that SplinedDetectorMap slit offsets work as expected

    We create a dummy SplinedDetectorMap, vary the offsets and verify that the
    xCenter and wavelength move as expected.
    """
    def setUp(self):
        self.box = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(2048, 4096))
        self.fiberId = np.array([123, 456]).astype(np.int32)
        self.left = np.poly1d((-0.0e-5, 1.0e-2, 0.0))  # xCenter for left trace
        self.left0 = 750
        self.right = np.poly1d((0.0e-5, -1.0e-2, 0.0))  # xCenter for right trace
        self.right0 = 1250
        self.rows = np.arange(self.box.getMinY(), self.box.getMaxY() + 1, dtype=np.float32)
        self.middle = 0.5*(self.box.getMinY() + self.box.getMaxY())
        self.wavelength = np.poly1d((0.0e-5, 0.1, 500.0))
        self.spatialOffsets = np.zeros(2, dtype=np.float32)
        self.spectralOffsets = np.zeros(2, dtype=np.float32)

        if False:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots()
            axes.plot(self.calculateXCenter()[0], self.rows, "r-")
            axes.plot(self.calculateXCenter()[1], self.rows, "b-")
            axes.set_xlabel("x")
            axes.set_ylabel("y")
            plt.show()
            fig, axes = plt.subplots()
            axes.plot(self.rows, self.wavelength(self.rows), "k-")
            axes.set_xlabel("y")
            axes.set_ylabel("Wavelength")
            plt.show()

        self.detMap = SplinedDetectorMap(self.box, self.fiberId,
                                         [self.rows, self.rows], self.calculateXCenter(),
                                         [self.rows, self.rows], self.calculateWavelength(),
                                         self.spatialOffsets, self.spectralOffsets)

    def calculateXCenter(self):
        """Calculate the expected xCenter"""
        result = [(self.left(self.rows - self.spectralOffsets[0]) + self.left0 -
                   self.left(self.middle - self.spectralOffsets[0]) + self.spatialOffsets[0]),
                  (self.right(self.rows - self.spectralOffsets[1]) + self.right0 -
                   self.right(self.middle - self.spectralOffsets[1]) + self.spatialOffsets[1])]
        return result

    def calculateWavelength(self):
        """Calculate the expected wavelength"""
        return [self.wavelength(self.rows - self.spectralOffsets[0]),
                self.wavelength(self.rows - self.spectralOffsets[1])]

    def assertPositions(self, atol=0.07):
        """Check that the xCenter and wavelength match expected values"""
        self.assertFloatsEqual(self.detMap.getSpatialOffsets(), self.spatialOffsets)
        self.assertFloatsEqual(self.detMap.getSpectralOffsets(), self.spectralOffsets)
        xCenter = np.array([self.detMap.getXCenter(ff) for ff in self.fiberId])
        wavelength = np.array([self.detMap.getWavelength(ff) for ff in self.fiberId])
        self.assertFloatsAlmostEqual(xCenter, np.array(self.calculateXCenter()), atol=atol)
        self.assertFloatsAlmostEqual(wavelength, np.array(self.calculateWavelength()), atol=atol)

    def testVanilla(self):
        """No slit offset"""
        for ff in self.fiberId:
            self.assertFloatsEqual(self.detMap.getXCenter(ff),
                                   np.array([self.detMap.getXCenterSpline(ff)(yy) for yy in self.rows]))
            self.assertFloatsEqual(self.detMap.getWavelength(ff),
                                   np.array([self.detMap.getWavelengthSpline(ff)(yy) for yy in self.rows]))
        self.assertPositions(0.0)

    def testX(self):
        """Slit offset in x"""
        self.spatialOffsets[:] = 5.0
        self.detMap.setSlitOffsets(self.spatialOffsets, self.spectralOffsets)
        self.assertPositions()

    def testY(self):
        """Slit offset in y"""
        self.spectralOffsets[:] = -5.0
        self.detMap.setSlitOffsets(self.spatialOffsets, self.spectralOffsets)
        self.assertPositions()

    def testXY(self):
        """Slit offset in x and y"""
        self.spatialOffsets[:] = 5.0
        self.spectralOffsets[:] = -5.0
        self.detMap.setSlitOffsets(self.spatialOffsets, self.spectralOffsets)
        self.assertPositions()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
