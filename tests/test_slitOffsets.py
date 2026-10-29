"""Tests of slit offsets in the OpticalModelDetectorMap

The slit offsets are applied in the "preslit" frame (pixels), and converted to
the slit frame by dividing by the fiberPitch (spatial) or wavelengthDispersion
(spectral). Older detectorMaps may have a negative fiberPitch (or
wavelengthDispersion); these are converted to positive values in some
operations, and the slit offsets need to change sign at the same time in order
to preserve the positions. These tests check that the positions are preserved
through all such operations, for all combinations of the signs, and that
fitting and evaluating the slit offsets are consistent.
"""

import pickle
from itertools import product

import numpy as np

import lsst.afw.image
import lsst.daf.base
import lsst.geom
import lsst.utils.tests
from lsst.geom import Box2D, Point2D

from pfs.drp.stella import PolynomialDistortion, ReferenceLineStatus, SplinedDetectorMap
from pfs.drp.stella.adjustDetectorMap import AdjustDetectorMapTask
from pfs.drp.stella.arcLine import ArcLine, ArcLineSet
from pfs.drp.stella.fitDetectorMap import FitDetectorMapTask
from pfs.drp.stella.OpticalModel import DetectorModel, OpticsModel, SlitModel
from pfs.drp.stella.OpticalModelDetectorMapContinued import OpticalModelDetectorMap
from pfs.drp.stella.referenceLine import ReferenceLineSource
from pfs.drp.stella.synthetic import SyntheticConfig
from pfs.drp.stella.tests.utils import methodParameters, runTests

display = None

SIGNS = list(product((1, -1), (1, -1)))  # fiberPitch sign, wavelengthDispersion sign


class SlitOffsetsTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.synthConfig = SyntheticConfig()
        self.synthConfig.slope = 0.0  # Straight traces, so x and y are independent
        self.minWl = 650.0
        self.maxWl = 950.0
        self.numFibers = self.synthConfig.numFibers
        # Consecutive fiberIds, so that the fiberPitch is the same as pixels per fiberId
        self.fiberId = np.arange(1, self.numFibers + 1, dtype=np.int32)
        self.rng = np.random.RandomState(12345)
        self.spatialOffsets = self.rng.normal(0.0, 0.3, self.numFibers)
        self.spectralOffsets = self.rng.normal(0.0, 0.3, self.numFibers)
        self.visitInfo = lsst.afw.image.VisitInfo(darkTime=123.45)

    def makeSplinedDetectorMap(self, xFlip: bool = False) -> SplinedDetectorMap:
        """Make a `SplinedDetectorMap` with consecutive fiberIds

        The `SplinedDetectorMap` requires the wavelength to increase with row,
        so it can't have a negative wavelengthDispersion.

        Parameters
        ----------
        xFlip : `bool`
            Make x decrease with fiberId (negative fiberPitch)?

        Returns
        -------
        detMap : `pfs.drp.stella.SplinedDetectorMap`
            Mapping from fiberId,wavelength to x,y, with no slit offsets.
        """
        config = self.synthConfig
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), config.dims)
        knots = np.arange(config.height, dtype=float)
        centers = config.traceCenters
        if xFlip:
            centers = centers[::-1]
        wavelength = np.linspace(self.minWl, self.maxWl, config.height, dtype=float)
        return SplinedDetectorMap(
            bbox,
            self.fiberId,
            [knots] * self.numFibers,
            [(cc + config.traceOffset).astype(float) for cc in centers],
            [knots] * self.numFibers,
            [wavelength] * self.numFibers,
        )

    def makeOpticalModelDetectorMap(
        self,
        pitchSign: int,
        dispersionSign: int,
        spatialOffsets: np.ndarray | None = None,
        spectralOffsets: np.ndarray | None = None,
        withSlitDistortion: bool = False,
    ) -> OpticalModelDetectorMap:
        """Make an `OpticalModelDetectorMap` with the specified signs

        The zero-offset positions are independent of the signs, but the effect
        of the slit offsets changes sign with the scales.

        Parameters
        ----------
        pitchSign, dispersionSign : `int`
            Sign of the fiberPitch and wavelengthDispersion.
        spatialOffsets, spectralOffsets : `numpy.ndarray`, optional
            Slit offsets (pixels in the preslit frame). Defaults to zero.
        withSlitDistortion : `bool`
            Include a slit distortion?

        Returns
        -------
        detMap : `pfs.drp.stella.OpticalModelDetectorMap`
            Mapping from fiberId,wavelength to x,y.
        """
        base = self.makeSplinedDetectorMap()
        slit = SlitModel(base)
        if spatialOffsets is None:
            spatialOffsets = np.zeros(self.numFibers, dtype=float)
        if spectralOffsets is None:
            spectralOffsets = np.zeros(self.numFibers, dtype=float)
        distortions = []
        if withSlitDistortion:
            order = 2
            numCoeffs = PolynomialDistortion.getNumDistortionForOrder(order)
            xi = slit.getFiberPitch() * (self.fiberId - slit.getFiberMin()) * pitchSign
            eta = np.array([self.minWl, self.maxWl]) - slit.getWavelengthMin()
            eta /= slit.getWavelengthDispersion() * dispersionSign
            box = Box2D(Point2D(xi.min(), eta.min()), Point2D(xi.max(), eta.max()))
            coeffs = np.zeros(2 * numCoeffs, dtype=float)
            coeffs[0] = 0.1
            coeffs[1] = 0.05
            coeffs[numCoeffs] = -0.1
            coeffs[numCoeffs + 2] = 0.05
            distortions.append(PolynomialDistortion(order, box, coeffs))
        newSlit = SlitModel(
            self.fiberId,
            pitchSign * slit.getFiberPitch(),
            slit.getFiberMin(),
            dispersionSign * slit.getWavelengthDispersion(),
            slit.getWavelengthMin(),
            np.asarray(spatialOffsets, dtype=float),
            np.asarray(spectralOffsets, dtype=float),
            distortions,
        )
        metadata = lsst.daf.base.PropertyList()
        metadata.set("METADATA", 12345)
        return OpticalModelDetectorMap(
            newSlit, OpticsModel(base), DetectorModel(base.bbox), self.visitInfo, metadata
        )

    def getPoints(self, detMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return positions on a grid of fiberId,wavelength"""
        wavelength = np.linspace(self.minWl + 20, self.maxWl - 20, 25)
        fiberId, wavelength = (arr.flatten() for arr in np.meshgrid(self.fiberId, wavelength))
        points = detMap.findPoint(fiberId.astype(np.int32), wavelength)
        return fiberId, wavelength, points

    def assertSamePositions(self, lhs, rhs, atol=1.0e-6):
        """Assert that two detectorMaps have the same positions"""
        _, _, lhsPoints = self.getPoints(lhs)
        _, _, rhsPoints = self.getPoints(rhs)
        self.assertTrue(np.all(np.isfinite(lhsPoints)))
        self.assertFloatsAlmostEqual(lhsPoints, rhsPoints, atol=atol)
        rows = np.full(self.numFibers, 0.5 * lhs.bbox.getHeight())
        self.assertFloatsAlmostEqual(
            lhs.getXCenter(self.fiberId, rows), rhs.getXCenter(self.fiberId, rows), atol=atol
        )

    def assertPositionsDiffer(self, lhs, rhs, minDiff=0.01):
        """Assert that two detectorMaps have different positions"""
        _, _, lhsPoints = self.getPoints(lhs)
        _, _, rhsPoints = self.getPoints(rhs)
        self.assertGreater(np.max(np.abs(lhsPoints - rhsPoints)), minDiff)

    def makeLines(self, detMap) -> ArcLineSet:
        """Make perfect line measurements from a detectorMap"""
        fiberId, wavelength, points = self.getPoints(detMap)
        lines = [
            ArcLine(
                int(ff),
                float(wl),
                float(xx),
                float(yy),
                0.01,
                0.01,
                np.nan,
                np.nan,
                np.nan,
                1000.0,
                1.0,
                np.nan,
                False,
                ReferenceLineStatus.GOOD,
                "Fake",
                None,
                ReferenceLineSource.NONE,
            )
            for ff, wl, (xx, yy) in zip(fiberId, wavelength, points)
        ]
        return ArcLineSet.fromRows(lines)

    def assertMatchesLines(self, detMap, lines, atol):
        """Assert that a detectorMap reproduces the line positions"""
        points = detMap.findPoint(lines.fiberId, lines.wavelength.astype(float))
        self.assertFloatsAlmostEqual(points[:, 0], lines.x, atol=atol)
        self.assertFloatsAlmostEqual(points[:, 1], lines.y, atol=atol)

    def readWrite(self, detMap):
        """Round-trip a detectorMap through FITS"""
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            detMap.writeFits(filename)
            return OpticalModelDetectorMap.readFits(filename)

    def testSignEquivalence(self):
        """Changing the sign of a scale and its offsets preserves positions"""
        canonical = self.makeOpticalModelDetectorMap(1, 1, self.spatialOffsets, self.spectralOffsets)
        zero = self.makeOpticalModelDetectorMap(1, 1)
        self.assertPositionsDiffer(canonical, zero)  # The offsets have an effect
        for pitchSign, dispersionSign in SIGNS:
            detMap = self.makeOpticalModelDetectorMap(
                pitchSign,
                dispersionSign,
                pitchSign * self.spatialOffsets,
                dispersionSign * self.spectralOffsets,
            )
            self.assertSamePositions(detMap, canonical)

    @methodParameters(signs=SIGNS)
    def testWithoutDistortion(self, signs):
        """SlitModel.withoutDistortion preserves positions (when there's no distortion)"""
        pitchSign, dispersionSign = signs
        detMap = self.makeOpticalModelDetectorMap(
            pitchSign, dispersionSign, self.spatialOffsets, self.spectralOffsets
        )
        slit = detMap.slitModel.withoutDistortion()
        self.assertGreater(slit.getFiberPitch(), 0)
        self.assertGreater(slit.getWavelengthDispersion(), 0)
        copy = OpticalModelDetectorMap(slit, detMap.opticsModel.copy(), detMap.detectorModel.copy())
        self.assertSamePositions(copy, detMap)

    @methodParameters(signs=SIGNS * 2, withSlitDistortion=[False] * len(SIGNS) + [True] * len(SIGNS))
    def testPersistence(self, signs, withSlitDistortion):
        """FITS and pickle round trips preserve positions"""
        pitchSign, dispersionSign = signs
        detMap = self.makeOpticalModelDetectorMap(
            pitchSign, dispersionSign, self.spatialOffsets, self.spectralOffsets, withSlitDistortion
        )
        # The slit distortion isn't persisted with full precision
        atol = 1.0e-3 if withSlitDistortion else 1.0e-6
        self.assertSamePositions(self.readWrite(detMap), detMap, atol=atol)
        self.assertSamePositions(pickle.loads(pickle.dumps(detMap)), detMap, atol=atol)
        self.assertSamePositions(detMap.clone(), detMap)

    @methodParameters(xFlip=(False, True))
    def testFromSplinedDetectorMap(self, xFlip):
        """Conversion from a SplinedDetectorMap with slit offsets preserves positions"""
        splined = self.makeSplinedDetectorMap(xFlip)
        splined.setSlitOffsets(self.spatialOffsets, self.spectralOffsets)
        detMap = OpticalModelDetectorMap(
            SlitModel(splined), OpticsModel(splined), DetectorModel(splined.bbox), self.visitInfo
        )
        self.assertGreater(detMap.slitModel.getFiberPitch(), 0)
        self.assertGreater(detMap.slitModel.getWavelengthDispersion(), 0)
        self.assertSamePositions(detMap, splined, atol=1.0e-3)

    @methodParameters(signs=SIGNS)
    def testAdjustPreservesOffsets(self, signs):
        """Adjusting a detectorMap without fitting slit offsets preserves them

        The adjustment replaces the slit distortion (making the scales
        positive), but should leave the positions alone when the lines are
        already reproduced.
        """
        pitchSign, dispersionSign = signs
        truth = self.makeOpticalModelDetectorMap(
            pitchSign, dispersionSign, self.spatialOffsets, self.spectralOffsets
        )
        lines = self.makeLines(truth)
        config = AdjustDetectorMapTask.ConfigClass()
        config.doSlitOffsets = False
        config.exclusionRadius = 1.0
        task = AdjustDetectorMapTask(config=config)
        adjusted = task.run(truth.clone(), lines, "r", self.visitInfo).detectorMap
        self.assertMatchesLines(adjusted, lines, atol=1.0e-3)
        self.assertMatchesLines(self.readWrite(adjusted), lines, atol=1.0e-3)

    def checkFitOffsets(self, fitOffsets, atol):
        """Check that fitting slit offsets recovers positions, independent of sign

        We use realistic slit offsets (0.05 pixels RMS). The fitting leaves a
        small error that is proportional to the size of the offsets, but a
        sign error produces errors of about twice the offsets. We require that
        the results are the same regardless of the signs of the scales, and
        that the fitted detectorMap reproduces the lines, both directly and
        after a FITS round trip.

        Parameters
        ----------
        fitOffsets : callable
            Function that takes a starting detectorMap (with zero offsets) and
            the lines, and returns the fitted detectorMap.
        atol : `float`
            Tolerance for reproducing the line positions (pixels).
        """
        spatialOffsets = 0.05 / 0.3 * self.spatialOffsets
        spectralOffsets = 0.05 / 0.3 * self.spectralOffsets
        reference = None
        for pitchSign, dispersionSign in SIGNS:
            with self.subTest(pitchSign=pitchSign, dispersionSign=dispersionSign):
                # The truth has the same positions for all signs
                truth = self.makeOpticalModelDetectorMap(
                    pitchSign,
                    dispersionSign,
                    pitchSign * spatialOffsets,
                    dispersionSign * spectralOffsets,
                )
                start = self.makeOpticalModelDetectorMap(pitchSign, dispersionSign)
                lines = self.makeLines(truth)
                fitted = fitOffsets(start, lines)
                self.assertMatchesLines(fitted, lines, atol=atol)
                self.assertMatchesLines(self.readWrite(fitted), lines, atol=atol)
                if reference is None:
                    reference = fitted
                else:
                    self.assertSamePositions(fitted, reference, atol=1.0e-5)

    def testAdjustFitsOffsets(self):
        """Fitting slit offsets in the adjustment recovers the positions"""
        config = AdjustDetectorMapTask.ConfigClass()
        config.doSlitOffsets = True
        config.exclusionRadius = 1.0
        task = AdjustDetectorMapTask(config=config)

        def fitOffsets(start, lines):
            return task.run(start, lines, "r", self.visitInfo).detectorMap

        self.checkFitOffsets(fitOffsets, atol=0.01)

    def testFitOffsets(self):
        """Fitting slit offsets from scratch recovers the positions"""
        config = FitDetectorMapTask.ConfigClass()
        config.order = 1
        config.doSlitOffsets = True
        config.exclusionRadius = 1.0
        task = FitDetectorMapTask(name="fitDetectorMap", config=config)
        dataId = dict(visit=12345, arm="r", spectrograph=1)

        def fitOffsets(start, lines):
            return task.run(dataId, start.bbox, lines, self.visitInfo, base=start).detectorMap

        self.checkFitOffsets(fitOffsets, atol=0.005)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
