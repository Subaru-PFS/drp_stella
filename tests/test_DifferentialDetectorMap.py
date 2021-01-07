import pickle

import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils

from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella import DifferentialDetectorMap, GlobalDetectorModel, GlobalDetectorModelScaling
from pfs.drp.stella import DetectorMap, ReferenceLine
from pfs.drp.stella.arcLine import ArcLineSet
from pfs.drp.stella.fitDifferentialDetectorMap import FitDifferentialDetectorMapTask
from pfs.drp.stella.tests.utils import runTests, methodParameters


display = None


class DifferentialDetectorMapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Construct a ``SplinedDetectorMap`` to play with"""
        self.synthConfig = SyntheticConfig()
        self.minWl = 650.0
        self.maxWl = 950.0
        self.base = makeSyntheticDetectorMap(self.synthConfig, self.minWl, self.maxWl)
        self.metadata = 123456
        self.darkTime = 12345.6

    def makeDifferentialDetectorMap(self):
        """Construct a `DifferentialDetectorMap`

        Based on the ``self.base`` `SplinedDetectorMap`.

        Returns
        -------
        detMap : `pfs.drp.stella.DifferentialDetectorMap`
            `DifferentialDetectorMap` to be used in tests.
        """
        base = self.base.clone()
        bbox = self.base.bbox
        distortionOrder = 1
        fiberId = self.base.getFiberId()
        numCoeffs = GlobalDetectorModel.getNumDistortion(distortionOrder)
        xDistortion = np.zeros(numCoeffs, dtype=float)
        yDistortion = np.zeros(numCoeffs, dtype=float)
        rightCcd = np.zeros(6, dtype=float)

        # Introduce a non-zero distortion field
        offset = 0.5
        xDistortion[0] = offset
        yDistortion[0] = offset
        slitOffsets = np.full(base.getNumFibers(), -offset, dtype=float)
        base.setSlitOffsets(slitOffsets, slitOffsets)

        fiberPitch = self.synthConfig.separation
        dispersion = (self.maxWl - self.minWl)/self.synthConfig.height
        wavelengthCenter = 0.5*(self.minWl + self.maxWl)
        scaling = GlobalDetectorModelScaling(fiberPitch, dispersion, wavelengthCenter,
                                             fiberId.min(), fiberId.max(), self.synthConfig.height)

        model = GlobalDetectorModel(bbox, distortionOrder, fiberId, scaling, 0,
                                    xDistortion, yDistortion, rightCcd)

        visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        metadata = lsst.daf.base.PropertyList()
        metadata.set("METADATA", self.metadata)
        return DifferentialDetectorMap(base, model, visitInfo, metadata)

    def assertDifferentialDetectorMap(self, detMap):
        """Assert that a ``DifferentialDetectorMap`` matches expectations"""
        # Positions
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

    def testBasic(self):
        """Test basic functionality"""
        detMap = self.makeDifferentialDetectorMap()
        self.assertDifferentialDetectorMap(detMap)

        if display is not None:
            from pfs.drp.stella.synthetic import makeSyntheticFlat
            from lsst.afw.display import Display
            disp = Display(frame=1, backend=display)
            disp.mtv(makeSyntheticFlat(self.synthConfig))
            detMap.display(disp)

    def testSlitOffsets(self):
        """Test different value for one of the slit offsets"""
        self.synthConfig.slope = 0.0  # Straighten the traces to avoid coupling x,y offsets
        detMap = self.makeDifferentialDetectorMap()
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

        self.assertDifferentialDetectorMap(detMap)

    def testFinds(self):
        """Test the various ``find*`` methods

        We throw down a random array of points on the image, run the
        ``find*`` methods and check that the answers are consistent.
        """
        num = 1000
        detMap = self.makeDifferentialDetectorMap()
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
        detMap = self.makeDifferentialDetectorMap()
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = DifferentialDetectorMap.readFits(filename)
            self.assertDifferentialDetectorMap(copy)
        # Read with parent class
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = DetectorMap.readFits(filename)
            self.assertDifferentialDetectorMap(copy)

    def testPickle(self):
        """Test round-trip pickle"""
        detMap = self.makeDifferentialDetectorMap()
        copy = pickle.loads(pickle.dumps(detMap))
        self.assertDifferentialDetectorMap(copy)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
