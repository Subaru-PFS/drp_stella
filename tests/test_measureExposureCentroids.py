import types

import numpy as np

import lsst.utils.tests
from lsst.daf.base import DateTime
from lsst.afw.image import VisitInfo
from lsst.pipe.base import Struct

from pfs.drp.stella.arcLine import ArcLine, ArcLineSet
from pfs.drp.stella.fitDetectorMap import FittingError
from pfs.drp.stella.pipelines.measureCentroids import MeasureExposureCentroidsTask
from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineSource, ReferenceLineStatus
from pfs.drp.stella.synthetic import SyntheticConfig, makeSyntheticDetectorMap
from pfs.drp.stella.tests import runTests


def makeArcLine(fiberId, wavelength, description):
    """Construct a single synthetic `ArcLine` row"""
    return ArcLine(
        fiberId,
        wavelength,
        100.0,
        200.0,
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
        description,
        None,
        ReferenceLineSource.NONE,
    )


class FakeExposure:
    """Minimal stand-in for an ISR-corrected `Exposure`

    Only exposes what `MeasureExposureCentroidsTask` actually touches:
    `getMetadata()`/`metadata` (dict-like, driving `getLamps`), `visitInfo`,
    and `maskedImage` (an opaque placeholder, since extraction is faked).
    """

    def __init__(self, metadata):
        self.metadata = metadata
        self.visitInfo = VisitInfo(
            id=12345, date=DateTime(2026, 8, 25, 10, 0, 0, DateTime.UTC), exposureTime=30.0
        )
        self.maskedImage = None

    def getMetadata(self):
        return self.metadata


class FakeRun:
    """Stand-in for a subtask exposing only `.run(...)`: records calls and
    either returns a fixed result or raises a fixed exception."""

    def __init__(self, result=None, exception=None):
        self.result = result
        self.exception = exception
        self.calls = []

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.exception is not None:
            raise self.exception
        return self.result


class FakeFiberProfiles:
    """Stand-in for a `FiberProfileSet`: records the detectorMap it was
    asked to make fiberTraces from."""

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def makeFiberTracesFromDetectorMap(self, detectorMap):
        self.calls.append(detectorMap)
        return self.result


class FakePfsArm:
    def __init__(self, fiberId):
        self.fiberId = fiberId


class FakeSpectra:
    def __init__(self, fiberId):
        self.fiberId = fiberId

    def toPfsArm(self, identity):
        return FakePfsArm(self.fiberId)


class MeasureExposureCentroidsTestCase(lsst.utils.tests.TestCase):
    """Test MeasureExposureCentroidsTask.run()'s lamp-dependent dispatch

    Subtasks are replaced with lightweight fakes so we can exercise the
    branch logic (including the twilight adjust/extract/centroid chain and
    its `FittingError` fallback) without needing real image data.
    """

    def setUp(self):
        self.synthConfig = SyntheticConfig()
        self.synthConfig.separation = 100  # Avoid having a fiber go down the middle
        self.detectorMap = makeSyntheticDetectorMap(self.synthConfig, 650.0, 950.0)
        self.arm = "r"
        self.spectrograph = 1
        self.pfsConfig = types.SimpleNamespace(fiberId=np.array(self.synthConfig.fiberId), pfsDesignId=12345)

    def makeTask(self, requireAdjustDetectorMap=False):
        config = MeasureExposureCentroidsTask.ConfigClass()
        config.doApplyCrMask = False
        config.requireAdjustDetectorMap = requireAdjustDetectorMap
        task = MeasureExposureCentroidsTask(config=config)
        task.centroidTraces = FakeRun(ArcLineSet.empty())
        task.readLineList = FakeRun(ReferenceLineSet.empty())
        task.centroidLines = FakeRun(
            ArcLineSet.fromRows([makeArcLine(self.synthConfig.fiberId[0], 700.0, "Fake")])
        )
        task.extractSpectra = FakeRun(Struct(spectra=FakeSpectra(np.array(self.synthConfig.fiberId))))
        task.centroidSolar = FakeRun(
            Struct(lines=ArcLineSet.fromRows([makeArcLine(self.synthConfig.fiberId[0], 700.0, "solar")]))
        )
        task.adjustDetectorMap = FakeRun(Struct(detectorMap=self.detectorMap))
        return task

    def testQuartz(self):
        """lamps == {"Quartz"} dispatches straight to centroidTraces"""
        task = self.makeTask()
        traces = ArcLineSet.fromRows([makeArcLine(self.synthConfig.fiberId[0], np.nan, "Trace")])
        task.centroidTraces = FakeRun(traces)

        exposure = FakeExposure({"DATA-TYP": "flat"})
        result = task.run(exposure, self.pfsConfig, self.detectorMap, self.arm, self.spectrograph)

        self.assertIs(result.centroids, traces)
        self.assertEqual(len(task.centroidTraces.calls), 1)
        self.assertEqual(len(task.readLineList.calls), 0)
        self.assertEqual(len(task.adjustDetectorMap.calls), 0)

    def testArc(self):
        """Non-empty, non-Quartz lamps dispatch to the line-list/centroidLines path"""
        task = self.makeTask()
        exposure = FakeExposure({"W_AITHGA": True})
        result = task.run(exposure, self.pfsConfig, self.detectorMap, self.arm, self.spectrograph)

        self.assertEqual(len(task.readLineList.calls), 1)
        self.assertEqual(len(task.centroidLines.calls), 1)
        self.assertEqual(len(task.centroidTraces.calls), 1)  # doForceTraces defaults True
        self.assertEqual(len(task.adjustDetectorMap.calls), 0)
        self.assertTrue(np.all(result.centroids.description == "Fake"))

    def testTwilight(self):
        """Empty lamps dispatch to the twilight (solar-absorption) branch"""
        task = self.makeTask()
        fiberProfiles = FakeFiberProfiles(result=object())
        exposure = FakeExposure({"DATA-TYP": "object"})

        result = task.run(
            exposure,
            self.pfsConfig,
            self.detectorMap,
            self.arm,
            self.spectrograph,
            fiberProfiles=fiberProfiles,
        )

        self.assertEqual(len(task.centroidTraces.calls), 1)
        self.assertEqual(len(task.adjustDetectorMap.calls), 1)
        self.assertEqual(fiberProfiles.calls, [self.detectorMap])
        self.assertEqual(len(task.extractSpectra.calls), 1)
        self.assertEqual(len(task.centroidSolar.calls), 1)
        self.assertTrue(np.all(result.centroids.description == "solar"))

    def testTwilightAdjustDetectorMapFailureFallsBack(self):
        """A FittingError from adjustDetectorMap falls back to the unadjusted map"""
        task = self.makeTask(requireAdjustDetectorMap=False)
        task.adjustDetectorMap = FakeRun(exception=FittingError("mock failure"))
        fiberProfiles = FakeFiberProfiles(result=object())
        exposure = FakeExposure({"DATA-TYP": "object"})

        result = task.run(
            exposure,
            self.pfsConfig,
            self.detectorMap,
            self.arm,
            self.spectrograph,
            fiberProfiles=fiberProfiles,
        )

        self.assertEqual(fiberProfiles.calls, [self.detectorMap])
        self.assertTrue(np.all(result.centroids.description == "solar"))

    def testTwilightRequireAdjustDetectorMapRaises(self):
        """requireAdjustDetectorMap=True re-raises the FittingError"""
        task = self.makeTask(requireAdjustDetectorMap=True)
        task.adjustDetectorMap = FakeRun(exception=FittingError("mock failure"))
        fiberProfiles = FakeFiberProfiles(result=object())
        exposure = FakeExposure({"DATA-TYP": "object"})

        with self.assertRaises(FittingError):
            task.run(
                exposure,
                self.pfsConfig,
                self.detectorMap,
                self.arm,
                self.spectrograph,
                fiberProfiles=fiberProfiles,
            )


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
