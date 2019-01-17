import sys
import unittest
import pickle

import numpy as np
import scipy.interpolate

import lsst.utils.tests
import lsst.pex.exceptions
import lsst.daf.base
import lsst.geom
import lsst.afw.image
import lsst.afw.image.testUtils
import lsst.afw.display

import pfs.drp.stella as drpStella
import pfs.drp.stella.synthetic

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


class SplineTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Create a spline to play with"""
        self.num = 100
        self.rng = np.random.RandomState(12345)
        self.xx = np.arange(0.0, self.num, dtype=np.float32)
        self.yy = self.rng.uniform(size=self.num).astype(np.float32)
        self.spline = drpStella.SplineF(self.xx, self.yy)

    def testBasic(self):
        """Test basic properties"""
        self.assertFloatsEqual(self.spline.getX(), self.xx)
        self.assertFloatsEqual(self.spline.getY(), self.yy)
        values = np.array([self.spline(x) for x in self.xx])
        self.assertFloatsEqual(values, self.yy)

    def testCompare(self):
        """Compare with alternate implementation"""
        alternate = makeSpline(self.xx, self.yy)
        rand = self.rng.uniform(size=self.num)*(self.num - 1)
        ours = np.array([self.spline(x) for x in rand])
        theirs = alternate(rand)
        self.assertFloatsAlmostEqual(ours, theirs, atol=3.0e-6)


class DetectorMapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Construct a ``DetectorMap`` to play with"""
        synthConfig = pfs.drp.stella.synthetic.SyntheticConfig()
        detMap = pfs.drp.stella.synthetic.makeSyntheticDetectorMap(synthConfig)
        self.bbox = detMap.bbox
        self.fiberIds = detMap.fiberIds
        self.numFibers = len(self.fiberIds)
        self.centerKnots = [detMap.getCenterSpline(ii).getX() for ii in range(self.numFibers)]
        self.xCenter = [detMap.getCenterSpline(ii).getY() for ii in range(self.numFibers)]
        self.wavelengthKnots = [detMap.getWavelengthSpline(ii).getX() for ii in range(self.numFibers)]
        self.wavelength = [detMap.getWavelengthSpline(ii).getY() for ii in range(self.numFibers)]
        self.rng = np.random.RandomState(54321)
        self.slitOffsets = self.rng.uniform(size=(3, self.numFibers)).astype(np.float32)
        self.calculateExpectations(detMap)
        self.metadata = 123456
        self.darkTime = 12345.6

    def calculateExpectations(self, detMap):
        """Calculate what to expect for the center and wavelength

        Sets ``self.centerExpect`` and ``self.wavelengthExpect``.
        This accounts for the random slitOffsets.
        """
        DX = int(drpStella.DetectorMap.DX)
        DY = int(drpStella.DetectorMap.DY)
        self.xCenterExpect = np.zeros((self.numFibers, self.bbox.getHeight()), dtype=np.float32)
        self.wavelengthExpect = np.zeros((self.numFibers, self.bbox.getHeight()), dtype=np.float32)
        for ii in range(self.numFibers):
            yOffset = self.slitOffsets[DY, ii]
            xOffset = self.slitOffsets[DX, ii]
            xCenterSpline = makeSpline(detMap.getCenterSpline(ii).getX(), detMap.getCenterSpline(ii).getY())
            self.xCenterExpect[ii, :] = xCenterSpline(np.arange(self.bbox.getHeight()) - yOffset) + xOffset
            wlSpline = makeSpline(detMap.getWavelengthSpline(ii).getX(),
                                  detMap.getWavelengthSpline(ii).getY())
            self.wavelengthExpect[ii, :] = wlSpline(np.arange(self.bbox.getHeight()) - yOffset)
        self.xCenterTol = 1.0e-4
        self.wavelengthTol = 2.0e-4

    def makeDetectorMap(self):
        """Construct a ``DetectorMap``

        Returns
        -------
        detMap : `pfs.drp.stella.DetectorMap`
            Detector map.
        """
        detMap = drpStella.DetectorMap(self.bbox, self.fiberIds, self.centerKnots, self.xCenter,
                                       self.wavelengthKnots, self.wavelength, self.slitOffsets)
        detMap.visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        detMap.metadata.set("METADATA", self.metadata)
        self.assertDetectorMap(detMap)
        return detMap

    def assertDetectorMap(self, detMap):
        """Assert that a ``DetectorMap`` matches expectations"""
        self.assertEqual(detMap.bbox, self.bbox)
        self.assertFloatsEqual(detMap.fiberIds, self.fiberIds)
        self.assertFloatsAlmostEqual(detMap.xCenter, self.xCenterExpect, atol=self.xCenterTol)
        self.assertFloatsAlmostEqual(detMap.wavelength, self.wavelengthExpect, atol=self.wavelengthTol)
        self.assertFloatsEqual(detMap.slitOffsets, self.slitOffsets)
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
        detMap = self.makeDetectorMap()
        self.assertDetectorMap(detMap)

        # Check accessor functions work as well as properties
        self.assertEqual(detMap.getBBox(), self.bbox)
        self.assertFloatsEqual(detMap.getFiberIds(), self.fiberIds)
        self.assertFloatsAlmostEqual(detMap.getXCenter(), self.xCenterExpect, atol=self.xCenterTol)
        self.assertFloatsAlmostEqual(detMap.getWavelength(), self.wavelengthExpect, atol=self.wavelengthTol)
        self.assertFloatsEqual(detMap.getSlitOffsets(), self.slitOffsets)
        self.assertEqual(detMap.getNumFibers(), len(self.fiberIds))
        self.assertEqual(len(detMap), len(self.fiberIds))

        # Check setters
        self.slitOffsets += 1.2345
        detMap.setSlitOffsets(self.slitOffsets)
        self.calculateExpectations(detMap)
        self.assertDetectorMap(detMap)

        self.slitOffsets -= 2.34567
        for ii, fiber in enumerate(self.fiberIds):
            detMap.setSlitOffsets(fiber, self.slitOffsets[:, ii])
        self.calculateExpectations(detMap)
        self.assertDetectorMap(detMap)

        for xCenter in self.xCenter:
            xCenter -= 2.3456
        for wavelength in self.wavelength:
            wavelength += 3.4567
        for ii, fiber in enumerate(self.fiberIds):
            detMap.setXCenter(fiber, self.centerKnots[ii], self.xCenter[ii])
            detMap.setWavelength(fiber, self.wavelengthKnots[ii], self.wavelength[ii])
        self.calculateExpectations(detMap)
        self.assertDetectorMap(detMap)

    def testBadCtor(self):
        """Test that we cannot create a DetectorMap with invalid arguments"""
        smallBox = lsst.geom.BoxI(lsst.geom.PointI(0, 0), lsst.geom.ExtentI(100, 100))
        short = 3  # Number of values for short array

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between bbox and center/wavelength
            drpStella.DetectorMap(smallBox, self.fiberIds, self.centerKnots, self.xCenter,
                                  self.wavelengthKnots, self.wavelength)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between fiberIds and center/wavelength
            drpStella.DetectorMap(self.bbox, self.fiberIds[:short], self.centerKnots, self.xCenter,
                                  self.wavelengthKnots, self.wavelength)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between the center and wavelength
            drpStella.DetectorMap(self.bbox, self.fiberIds, self.centerKnots[:short], self.xCenter,
                                  self.wavelengthKnots, self.wavelength)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between the center and wavelength
            drpStella.DetectorMap(self.bbox, self.fiberIds, self.centerKnots, self.xCenter[:short],
                                  self.wavelengthKnots, self.wavelength)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between the center and wavelength
            drpStella.DetectorMap(self.bbox, self.fiberIds, self.centerKnots, self.xCenter,
                                  self.wavelengthKnots[:short], self.wavelength)

        with self.assertRaises(lsst.pex.exceptions.LengthError):
            # Mismatch between the center and wavelength
            drpStella.DetectorMap(self.bbox, self.fiberIds, self.centerKnots, self.xCenter,
                                  self.wavelengthKnots, self.wavelength[:short])

        # Various mismatches in the second constructor
        detMap = self.makeDetectorMap()
        centerKnots = np.array([detMap.getCenterSpline(ii).getX() for ii in range(len(detMap))])
        centerValues = np.array([detMap.getCenterSpline(ii).getY() for ii in range(len(detMap))])
        wavelengthKnots = np.array([detMap.getWavelengthSpline(ii).getX() for ii in range(len(detMap))])
        wavelengthValues = np.array([detMap.getWavelengthSpline(ii).getY() for ii in range(len(detMap))])
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            drpStella.DetectorMap(detMap.bbox, detMap.fiberIds[:short],
                                  centerKnots, centerValues,
                                  wavelengthKnots, wavelengthValues,
                                  detMap.slitOffsets)
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            drpStella.DetectorMap(detMap.bbox, detMap.fiberIds,
                                  centerKnots[:short], centerValues,
                                  wavelengthKnots, wavelengthValues,
                                  detMap.slitOffsets)
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            drpStella.DetectorMap(detMap.bbox, detMap.fiberIds,
                                  centerKnots, centerValues[:short],
                                  wavelengthKnots, wavelengthValues,
                                  detMap.slitOffsets)
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            drpStella.DetectorMap(detMap.bbox, detMap.fiberIds,
                                  centerKnots, centerValues,
                                  wavelengthKnots[:short], wavelengthValues,
                                  detMap.slitOffsets)
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            drpStella.DetectorMap(detMap.bbox, detMap.fiberIds,
                                  centerKnots, centerValues,
                                  wavelengthKnots, wavelengthValues[:short],
                                  detMap.slitOffsets)
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            drpStella.DetectorMap(detMap.bbox, detMap.fiberIds,
                                  centerKnots, centerValues,
                                  wavelengthKnots, wavelengthValues,
                                  detMap.slitOffsets[:2, :])
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            drpStella.DetectorMap(detMap.bbox, detMap.fiberIds,
                                  centerKnots, centerValues,
                                  wavelengthKnots, wavelengthValues,
                                  detMap.slitOffsets[:, :short])
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            drpStella.DetectorMap(detMap.bbox, detMap.fiberIds,
                                  centerKnots[:short], centerValues[:short],
                                  wavelengthKnots, wavelengthValues,
                                  detMap.slitOffsets)
        with self.assertRaises(lsst.pex.exceptions.LengthError):
            drpStella.DetectorMap(detMap.bbox, detMap.fiberIds,
                                  centerKnots, centerValues,
                                  wavelengthKnots[:short], wavelengthValues[:short],
                                  detMap.slitOffsets)

    def testHoldByValue(self):
        """Test that we hold arrays by value

        After using an array to construct the ``DetectorMap``, changing the
        array does NOT change the ``DetectorMap``.
        """
        detMap = self.makeDetectorMap()
        self.bbox.grow(12)
        self.fiberIds += 123
        self.slitOffsets += 1.2345

        self.assertNotEqual(detMap.getBBox(), self.bbox)
        self.assertFloatsNotEqual(detMap.getFiberIds(), self.fiberIds)
        self.assertFloatsNotEqual(detMap.getSlitOffsets(), self.slitOffsets)

    def testFinds(self):
        """Test the various ``find*`` methods

        We throw down a random array of points on the image, run the
        ``find*`` methods and check that the answers are consistent.
        """
        detMap = self.makeDetectorMap()
        if display:
            image = pfs.drp.stella.synthetic.makeSyntheticFlat()
            dd = lsst.afw.display.Display(1, display)
            dd.mtv(image)

        num = 1000
        indices = np.arange(0, detMap.bbox.getHeight())
        xCenter = detMap.xCenter
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
            self.assertEqual(fiberId, detMap.fiberIds[closestIndex])

            wavelength = detMap.findWavelength(fiberId, yy)
            wavelengthExpect = np.interp(yy, indices, detMap.getWavelength(fiberId))
            self.assertFloatsAlmostEqual(wavelength, wavelengthExpect, atol=1.0e-3)

            point = detMap.findPoint(fiberId, wavelength)
            self.assertFloatsAlmostEqual(point.getY(), yy, atol=1.0e-3)

    def testReadWriteFits(self):
        """Test reading and writing to/from FITS"""
        detMap = self.makeDetectorMap()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = drpStella.DetectorMap.readFits(filename)
            self.assertDetectorMap(copy)

    def testPickle(self):
        """Test round-trip pickle"""
        detMap = self.makeDetectorMap()
        copy = pickle.loads(pickle.dumps(detMap))
        self.assertDetectorMap(copy)


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
