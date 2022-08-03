import pickle
from itertools import product

import numpy as np

import lsst.utils.tests
import lsst.afw.image.testUtils
from lsst.afw.detection import GaussianPsf
from lsst.afw.image import ExposureF
from lsst.geom import Point2D
from lsst.pex.exceptions import DomainError

from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella import DifferentialDetectorMap, GlobalDetectorModel, GlobalDetectorModelScaling
from pfs.drp.stella import DetectorMap, ReferenceLineStatus, ImagingSpectralPsf
from pfs.drp.stella.arcLine import ArcLine, ArcLineSet
from pfs.drp.stella.fitDifferentialDetectorMap import FitDifferentialDetectorMapTask
from pfs.drp.stella.tests.utils import runTests, methodParameters
from pfs.drp.stella.referenceLine import ReferenceLineSource


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

    def makeDifferentialDetectorMap(self, likeBase):
        """Construct a `DifferentialDetectorMap`

        Parameters
        ----------
        likeBase : `bool`
            Should the output have similar positions as the ``self.base``
            `SplinedDetectorMap`? This allows the use of ``assertPositions``,
            but the internal values are not random. Use ``True`` if you want to
            check calculations; use ``False`` if you want to check persistence.

        Returns
        -------
        detMap : `pfs.drp.stella.DifferentialDetectorMap`
            `DifferentialDetectorMap` to be used in tests.
        """
        base = self.base.clone()
        fiberId = self.base.getFiberId()

        if likeBase:
            distortionOrder = 1
            numCoeffs = GlobalDetectorModel.getNumDistortion(distortionOrder)
            xDistortion = np.zeros(numCoeffs, dtype=float)
            yDistortion = np.zeros(numCoeffs, dtype=float)
            rightCcd = np.zeros(6, dtype=float)

            # Introduce a non-zero distortion field that replicates the base detectorMap
            offset = 0.5
            xDistortion[0] = offset
            yDistortion[0] = offset
            slitOffsets = np.full(base.getNumFibers(), -offset, dtype=float)
            base.setSlitOffsets(slitOffsets, slitOffsets)
        else:
            distortionOrder = 3
            numCoeffs = GlobalDetectorModel.getNumDistortion(distortionOrder)

            # Introduce a random distortion field; will likely get weird values, but good for testing I/O
            rng = np.random.RandomState(12345)
            xDistortion = rng.uniform(size=numCoeffs)
            yDistortion = rng.uniform(size=numCoeffs)
            rightCcd = rng.uniform(size=6)

        fiberPitch = self.synthConfig.separation
        dispersion = (self.maxWl - self.minWl)/self.synthConfig.height
        wavelengthCenter = 0.5*(self.minWl + self.maxWl)
        buffer = 0.123
        scaling = GlobalDetectorModelScaling(fiberPitch, dispersion, wavelengthCenter,
                                             fiberId.min(), fiberId.max(), self.synthConfig.height, buffer)

        model = GlobalDetectorModel(distortionOrder, scaling, 0, xDistortion, yDistortion, rightCcd)

        visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        metadata = lsst.daf.base.PropertyList()
        metadata.set("METADATA", self.metadata)
        return DifferentialDetectorMap(base, model, visitInfo, metadata)

    def assertPositions(self, detMap):
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

    def assertDifferentialDetectorMapsEqual(self, lhs, rhs):
        """Assert that the ``DifferentialDetectorMap``s are the same"""
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

        # Model
        self.assertFloatsEqual(lhs.fiberId, rhs.fiberId)
        self.assertFloatsEqual(lhs.model.getXCoefficients(), rhs.model.getXCoefficients())
        self.assertFloatsEqual(lhs.model.getYCoefficients(), rhs.model.getYCoefficients())
        self.assertFloatsEqual(lhs.model.getHighCcdCoefficients(), rhs.model.getHighCcdCoefficients())

        # Scaling
        self.assertEqual(lhs.model.getScaling().fiberPitch, rhs.model.getScaling().fiberPitch)
        self.assertEqual(lhs.model.getScaling().dispersion, rhs.model.getScaling().dispersion)
        self.assertEqual(lhs.model.getScaling().wavelengthCenter, rhs.model.getScaling().wavelengthCenter)
        self.assertEqual(lhs.model.getScaling().minFiberId, rhs.model.getScaling().minFiberId)
        self.assertEqual(lhs.model.getScaling().maxFiberId, rhs.model.getScaling().maxFiberId)
        self.assertEqual(lhs.model.getScaling().height, rhs.model.getScaling().height)
        self.assertEqual(lhs.model.getScaling().buffer, rhs.model.getScaling().buffer)

        # Metadata
        for name in lhs.metadata.names():
            self.assertEqual(lhs.metadata.get(name), rhs.metadata.get(name))

        # VisitInfo: only checking one element, assuming the rest are protected by afw unit tests
        self.assertEqual(lhs.visitInfo.getDarkTime(), rhs.visitInfo.getDarkTime())

    def testBasic(self):
        """Test basic functionality"""
        detMap = self.makeDifferentialDetectorMap(True)
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
        detMap = self.makeDifferentialDetectorMap(True)
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
        detMap = self.makeDifferentialDetectorMap(True)
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
        detMap = self.makeDifferentialDetectorMap(False)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = DifferentialDetectorMap.readFits(filename)
            self.assertDifferentialDetectorMapsEqual(detMap, copy)
        # Read with parent class
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            copy = DetectorMap.readFits(filename)
            self.assertDifferentialDetectorMapsEqual(detMap, copy)

    def testPickle(self):
        """Test round-trip pickle"""
        detMap = self.makeDifferentialDetectorMap(False)
        copy = pickle.loads(pickle.dumps(detMap))
        self.assertDifferentialDetectorMapsEqual(detMap, copy)

    def testPersistable(self):
        """Test behaviour as a Persistable

        This involves sticking it in a Psf, which gets stuck in an Exposure
        """
        size = 21
        sigma = 3
        detMap = self.makeDifferentialDetectorMap(False)
        imagePsf = GaussianPsf(size, size, sigma)
        psf = ImagingSpectralPsf(imagePsf, detMap)
        exposure = ExposureF(size, size)
        exposure.setPsf(psf)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            exposure.writeFits(filename)
            copy = ExposureF(filename).getPsf().getDetectorMap()
            copy.metadata.set("METADATA", self.metadata)  # Persistence doesn't preserve the metadata
            self.assertDifferentialDetectorMapsEqual(detMap, copy)

    @methodParameters(arm=("r", "m"))
    def testFit(self, arm):
        """Test FitDifferentialDetectorMapTask

        Parameters
        ----------
        arm : `str`
            Spectrograph arm; affects behaviour of
            `FitDifferentialDetectorMapTask`.
        """
        flux = 1000.0
        fluxErr = 1.0
        bbox = self.base.bbox
        lines = []
        for ff in self.synthConfig.fiberId:
            for yy in range(bbox.getMinY(), bbox.getMaxY()):
                lines.append(ArcLine(ff, self.base.getWavelength(ff, yy), self.base.getXCenter(ff, yy),
                             float(yy), 0.01, 0.01, flux, fluxErr, False, ReferenceLineStatus.GOOD, "Fake",
                             None, ReferenceLineSource.NONE))
        lines = ArcLineSet.fromRows(lines)
        config = FitDifferentialDetectorMapTask.ConfigClass()
        config.order = 1
        config.doSlitOffsets = True
        task = FitDifferentialDetectorMapTask(name="fitDifferentialDetectorMap", config=config)
        dataId = dict(visit=12345, arm=arm, spectrograph=1)
        detMap = task.run(dataId, bbox, lines, self.base.visitInfo, base=self.base).detectorMap
        self.assertFloatsEqual(detMap.model.getXCoefficients(), 0.0)
        self.assertFloatsEqual(detMap.model.getYCoefficients(), 0.0)
        self.assertFloatsEqual(detMap.model.getHighCcdCoefficients(), 0.0)
        self.assertFloatsEqual(detMap.getSpatialOffsets(), 0.0)
        self.assertFloatsEqual(detMap.getSpectralOffsets(), 0.0)

    def testOutOfRange(self):
        """Test that inputs that are out-of-range produce NaNs"""
        detMap = self.makeDifferentialDetectorMap(True)

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
        detMap = self.makeDifferentialDetectorMap(True)
        self.assertRaises(DomainError, detMap.findFiberId, Point2D(6000, -20000))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
