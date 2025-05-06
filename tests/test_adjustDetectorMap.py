import logging
import lsst.log

import numpy as np

import lsst.utils.tests
from lsst.geom import Box2D, AffineTransform
import lsst.afw.image
from lsst.afw.display import Display
import lsst.log

from pfs.datamodel import CalibIdentity
from pfs.drp.stella.adjustDetectorMap import AdjustDetectorMapTask
from pfs.drp.stella.arcLine import ArcLineSet
from pfs.drp.stella.buildFiberProfiles import BuildFiberProfilesTask
from pfs.drp.stella.referenceLine import ReferenceLine, ReferenceLineSet, ReferenceLineStatus
from pfs.drp.stella.referenceLine import ReferenceLineSource
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap, makeSyntheticPfsConfig
from pfs.drp.stella.synthetic import makeSyntheticArc, makeSyntheticFlat, makeSpectrumImage, addNoiseToImage
from pfs.drp.stella.centroidLines import CentroidLinesTask
from pfs.drp.stella.centroidTraces import CentroidTracesTask, tracesToLines
from pfs.drp.stella import PolynomialDistortion, LayeredDetectorMap, SplinedDetectorMap
from pfs.drp.stella.tests.utils import runTests, methodParameters
from pfs.drp.stella.utils.math import robustRms

display = None


class AdjustDetectorMapTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        """Construct a ``SplinedDetectorMap`` to play with"""
        self.synthConfig = SyntheticConfig()
        self.synthConfig.separation = 100  # Avoid having a fiber go down the middle
        self.minWl = 650.0
        self.maxWl = 950.0
        self.base = makeSyntheticDetectorMap(self.synthConfig, self.minWl, self.maxWl)
        self.metadata = 123456
        self.darkTime = 12345.6
        self.identity = CalibIdentity("2020-01-01", 5, "x", 12345)

        distortionOrder = 5
        numCoeffs = PolynomialDistortion.getNumDistortionForOrder(distortionOrder)
        xDistortion = np.zeros(numCoeffs, dtype=float)
        yDistortion = np.zeros(numCoeffs, dtype=float)

        # Introduce a low-order inaccuracy that we will correct with AdjustDetectorMapTask
        scale = 3.21e-4
        theta = 3.21e-3
        xNorm = 0.5*self.synthConfig.width
        yNorm = 0.5*self.synthConfig.height
        xDistortion[0] = 1.5
        xDistortion[1] = (np.cos(theta) - 1 + scale) * xNorm
        xDistortion[2] = -np.sin(theta) * yNorm
        yDistortion[0] = -1.5
        yDistortion[1] = np.sin(theta) * xNorm
        yDistortion[2] = (np.cos(theta) - 1 + scale) * yNorm

        distortion = PolynomialDistortion(
            distortionOrder, Box2D(self.base.bbox), np.concatenate((xDistortion, yDistortion))
        )

        visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        metadata = lsst.daf.base.PropertyList()
        metadata.set("METADATA", self.metadata)
        slitOffsets = np.zeros(len(self.base), dtype=float)
        rightCcd = AffineTransform()
        self.distorted = LayeredDetectorMap(
            self.base.bbox,
            slitOffsets,
            slitOffsets,
            self.base.clone(),
            [distortion],
            False,
            rightCcd,
            visitInfo,
            metadata,
        )

    def assertExpected(self, detMap, checkWavelengths=True):
        """Assert that the ``DistortedDetectorMap``s is as expected

        Parameters
        ----------
        detMap : `pfs.drp.stella.DistortedDetectorMap`
            DetectorMap to check.
        checkWavelengths : `bool`
            Check that wavelengths are correct? If there are no lines, the
            wavelength may be inaccurate, since the straight traces we have in
            the synthetic data don't give any handle on the wavelength fit.
        """
        # Positions as for the original detectorMap
        for fiberId in self.base.getFiberId():
            self.assertFloatsAlmostEqual(detMap.getXCenter(fiberId), self.base.getXCenter(fiberId),
                                         atol=1.5e-2)
            if checkWavelengths:
                self.assertFloatsAlmostEqual(detMap.getWavelength(fiberId), self.base.getWavelength(fiberId),
                                             atol=2.0e-3)

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

        # Base hasn't been modified
        self.assertFloatsEqual(detMap.fiberId, self.base.fiberId)
        for ff in detMap.fiberId:
            self.assertFloatsEqual(detMap.base.getXCenterSpline(ff).getX(),
                                   self.base.getXCenterSpline(ff).getX())
            self.assertFloatsEqual(detMap.base.getXCenterSpline(ff).getY(),
                                   self.base.getXCenterSpline(ff).getY())
            self.assertFloatsEqual(detMap.base.getWavelengthSpline(ff).getX(),
                                   self.base.getWavelengthSpline(ff).getX())
            self.assertFloatsEqual(detMap.base.getWavelengthSpline(ff).getY(),
                                   self.base.getWavelengthSpline(ff).getY())
            self.assertEqual(detMap.base.getSpatialOffset(ff), self.base.getSpatialOffset(ff))
            self.assertEqual(detMap.base.getSpectralOffset(ff), self.base.getSpectralOffset(ff))

    @methodParameters(flatFlux=(10, 200000, 1000),
                      numLines=(50, 0, 50),
                      traceTol=(0.05, 0.005, 0.07),
                      )
    def testAdjustment(self, flatFlux=1000, numLines=0, arcFlux=10000, traceTol=1.0e-2):
        """Test adjustment of a detectorMap

        We construct a synthetic image with both trace and lines, and check that
        we can adjust the detectorMap back to what it should be.

        Parameters
        ----------
        flatFlux : `float`
            Flux per row in the trace.
        numLines : `int`
            Number of lines per fiber.
        arcFlux : `float`
            Flux in each line.
        """
        rng = np.random.RandomState(12345)
        pfsConfig = makeSyntheticPfsConfig(self.synthConfig, 1234567890, 12345)
        image = makeSyntheticFlat(self.synthConfig, flux=flatFlux, addNoise=True, rng=rng)
        arc = makeSyntheticArc(self.synthConfig, numLines=numLines, flux=arcFlux, fwhm=self.synthConfig.fwhm,
                               addNoise=True, rng=rng)
        image += arc.image
        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(image))
        exposure.mask.set(0)

        readnoise = self.synthConfig.readnoise
        gain = self.synthConfig.gain
        exposure.variance.array = readnoise**2/gain + image.array*gain

        refLines = []
        fiberId = self.synthConfig.fiberId[self.synthConfig.numFibers//2]
        for yy in arc.lines:
            refLines.append(ReferenceLine("Fake", self.base.findWavelength(fiberId, yy), arcFlux,
                                          ReferenceLineStatus.GOOD,
                                          "UNKNOWN",
                                          ReferenceLineSource.NONE))
        refLines = ReferenceLineSet.fromRows(refLines)

        profilesConfig = BuildFiberProfilesTask.ConfigClass()
        profilesConfig.pruneMinLength = self.synthConfig.height//2
        profilesConfig.profileRadius = 3
        profilesConfig.doBlindFind = False

        buildFiberProfiles = BuildFiberProfilesTask(config=profilesConfig)
        fiberProfiles = buildFiberProfiles.run(exposure, self.identity, detectorMap=self.base).profiles
        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(self.base)

        centroidLines = CentroidLinesTask()
        centroidTraces = CentroidTracesTask()
        centroidLines.config.fwhm = self.synthConfig.fwhm
        centroidLines.config.doSubtractContinuum = False  # Can bias the measurements if we're off by a lot
        centroidTraces.config.fwhmCol = self.synthConfig.fwhm
        centroidTraces.config.fwhmRow = 0  # Don't smooth rows: the slope is too large
        centroidTraces.config.threshold = 20  # The traceTol values were set using this
        centroidTraces.config.searchRadius = 5  # Our distorted detectorMap can be a bit off
        if numLines > 0:
            lines = centroidLines.run(exposure, refLines, self.distorted, pfsConfig, fiberTraces)
        else:
            lines = ArcLineSet.empty()
        traces = centroidTraces.run(exposure, self.distorted, pfsConfig)
        for ff in traces:
            rows = np.array([pp.row for pp in traces[ff]], dtype=float)
            centers = np.array([pp.peak for pp in traces[ff]])
            expected = self.base.getXCenter(ff, rows)
            self.assertLess(robustRms(centers - expected), traceTol)

        logger = logging.getLogger()
        logger.getChild("adjustDetectorMap").setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        config = AdjustDetectorMapTask.ConfigClass()
        config.order = 1
        task = AdjustDetectorMapTask(log=logger, config=config)

        if display is not None:
            disp = Display(frame=1, backend=display)
            disp.mtv(exposure)
            self.distorted.display(disp, ctype="red", wavelengths=refLines.wavelength)
            with disp.Buffering():
                for ll in lines:
                    disp.dot("o", ll.x, ll.y, ctype="blue")
                for ff in traces:
                    for tt in traces[ff]:
                        disp.dot("+", tt.peak, tt.row, ctype="blue")

        lines += tracesToLines(self.distorted, traces, 10.0)
        try:
            # Wavelengths are actually for r, but arm=n triggers (single) RotScaleDistortion, for simplicity
            adjusted = task.run(
                self.distorted, lines, "n", self.distorted.visitInfo, self.distorted.metadata
            )
        finally:
            logger.removeHandler(handler)

        if display is not None:
            adjusted.detectorMap.display(disp, ctype="green", wavelengths=refLines.wavelength)

        self.assertExpected(adjusted.detectorMap, checkWavelengths=(numLines > 0))


class AdjustDetectorMapQuartzTestCase(lsst.utils.tests.TestCase):
    """Test the AdjustDetectorMapTask on a quartz lamp image

    There are no wavelength references, but the traces are curved and the scale
    change is identical in x and y, so we can still get a great measurement of
    the wavelengths.
    """
    def setUp(self):
        """Construct a ``SplinedDetectorMap`` to play with"""
        self.minWl = 650.0
        self.maxWl = 950.0

        self.synthConfig = SyntheticConfig()
        self.synthConfig.separation = 100  # Avoid having a fiber go down the middle
        self.synthConfig.slope = 0.0

        height = self.synthConfig.height
        dims = self.synthConfig.dims

        traceCenters = self.synthConfig.traceCenters
        numFibers = self.synthConfig.numFibers

        secondOrderMax = 43.21
        flatFlux = 2.0e5
        secondOrder = np.linspace(-secondOrderMax, secondOrderMax, numFibers, True)

        self.image = None
        xCenter = []
        xx = np.linspace(-1, 1, height, True)
        spectrum = np.full(height, flatFlux)
        for ii, (tt, order2) in enumerate(zip(traceCenters, secondOrder)):
            traceOffsets = order2*xx**2
            xCenter.append(tt + traceOffsets)

            fiberImage = makeSpectrumImage(
                spectrum, self.synthConfig.dims, [tt], traceOffsets, self.synthConfig.fwhm
            )

            if self.image is None:
                self.image = fiberImage
            else:
                self.image += fiberImage

        rng = np.random.RandomState(12345)
        addNoiseToImage(self.image, self.synthConfig.gain, self.synthConfig.readnoise, rng)

        self.metadata = 123456
        self.darkTime = 12345.6

        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), dims)
        fiberId = self.synthConfig.fiberId
        knots = [np.arange(height, dtype=float)]*numFibers
        wavelength = np.linspace(self.minWl, self.maxWl, height, dtype=float)

        self.base = SplinedDetectorMap(bbox, fiberId, knots, xCenter, knots, [wavelength]*numFibers)
        self.identity = CalibIdentity("2020-01-01", 5, "x", 12345)
        self.pfsConfig = makeSyntheticPfsConfig(self.synthConfig, 1234567890, 12345)

        distortionOrder = 1
        numCoeffs = PolynomialDistortion.getNumDistortionForOrder(distortionOrder)
        xDistortion = np.zeros(numCoeffs, dtype=float)
        yDistortion = np.zeros(numCoeffs, dtype=float)

        # Introduce a distortion that we will correct with AdjustDetectorMapTask
        xNorm = 0.5*self.synthConfig.width
        yNorm = 0.5*self.synthConfig.height
        xDistortion[0] = 1.5
        xDistortion[1] = 1.23e-2 * xNorm
        xDistortion[2] = -7.65e-4 * yNorm
        yDistortion[0] = -1.5
        yDistortion[1] = 8.76e-5 * xNorm
        yDistortion[2] = 1.23e-2 * yNorm

        distortion = PolynomialDistortion(
            distortionOrder, Box2D(self.base.bbox), np.concatenate((xDistortion, yDistortion))
        )

        visitInfo = lsst.afw.image.VisitInfo(darkTime=self.darkTime)
        metadata = lsst.daf.base.PropertyList()
        metadata.set("METADATA", self.metadata)
        slitOffsets = np.zeros(len(self.base), dtype=float)
        rightCcd = AffineTransform()
        self.distorted = LayeredDetectorMap(
            self.base.bbox,
            slitOffsets,
            slitOffsets,
            self.base.clone(),
            [distortion],
            False,
            rightCcd,
            visitInfo,
            metadata,
        )

    def testAdjustDetectorMap(self):
        exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(self.image))
        exposure.mask.set(0)

        readnoise = self.synthConfig.readnoise
        gain = self.synthConfig.gain
        exposure.variance.array = readnoise**2/gain + self.image.array*gain

        profilesConfig = BuildFiberProfilesTask.ConfigClass()
        profilesConfig.pruneMinLength = self.synthConfig.height//2
        profilesConfig.profileRadius = 3
        profilesConfig.doBlindFind = False

        centroidTraces = CentroidTracesTask()
        centroidTraces.config.fwhmCol = self.synthConfig.fwhm
        centroidTraces.config.searchRadius = 13  # We've made a fairly large distortion
        lines = ArcLineSet.empty()
        traces = centroidTraces.run(exposure, self.distorted, self.pfsConfig)
        self.assertEqual(len(traces), self.synthConfig.numFibers)

        logger = logging.getLogger()
        logger.getChild("adjustDetectorMap").setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        logger.addHandler(handler)

        config = AdjustDetectorMapTask.ConfigClass()
        config.order = 1
        task = AdjustDetectorMapTask(log=logger, config=config)

        if display is not None:
            disp = Display(frame=1, backend=display)
            disp.mtv(exposure)
            self.distorted.display(disp, ctype="red")
            self.base.display(disp, ctype="blue")
            if False:
                with disp.Buffering():
                    for ff in traces:
                        for tt in traces[ff]:
                            disp.dot("+", tt.peak, tt.row, ctype="orange")

        lines += tracesToLines(self.distorted, traces, 10.0)
        try:
            # Wavelengths are actually for r, but arm=n triggers (single) RotScaleDistortion, for simplicity
            adjusted = task.run(self.distorted, lines, "n", self.distorted.visitInfo)
        finally:
            logger.removeHandler(handler)

        if display is not None:
            adjusted.detectorMap.display(disp, ctype="green")

        # xCenter
        expected = self.base.getXCenter()
        actual = adjusted.detectorMap.getXCenter()
        diff = actual - expected
        self.assertFloatsAlmostEqual(diff, 0.0, atol=0.2)
        self.assertFloatsAlmostEqual(np.median(diff), 0.0, atol=0.01)
        self.assertLess(robustRms(diff), 0.07)

        # Wavelength
        expected = self.base.getWavelength()
        actual = adjusted.detectorMap.getWavelength()
        diff = actual - expected
        self.assertFloatsAlmostEqual(diff, 0.0, atol=0.02)
        self.assertFloatsAlmostEqual(np.median(diff), 0.0, atol=0.005)
        self.assertLess(robustRms(diff), 0.008)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
