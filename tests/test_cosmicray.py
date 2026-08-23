import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.afw.cameraGeom.testUtils import DetectorWrapper

from pfs.drp.stella.cosmicray import CosmicRayConfig, CosmicRayTask
from pfs.drp.stella.tests import runTests


def makeExposure(detectorName: str) -> afwImage.ExposureF:
    """Build a tiny exposure on a named detector, with no metadata"""
    exposure = afwImage.ExposureF(8, 8)
    exposure.setDetector(DetectorWrapper(name=detectorName, id=1).detector)
    return exposure


class CosmicRayConfigTestCase(lsst.utils.tests.TestCase):
    """The H4RG morphological CR pass is part of the default control surface.

    The up-the-ramp CR detector in ISR works one pixel at a time along the
    ramp, so it cannot use a CR's spatial footprint. The morphological pass
    is what supplies that, and it runs by default on the NIR arms too.
    """

    def testMorphologicalCRsOnByDefault(self):
        self.assertTrue(CosmicRayConfig().doH4MorphologicalCRs)

    def testNoMinimumReadsGate(self):
        """H4RG CR rejection no longer applies a minimum-reads threshold."""
        self.assertFalse(hasattr(CosmicRayConfig(), "crMinReadsH4"))


class CosmicRayNirTestCase(lsst.utils.tests.TestCase):
    """Which exposures reach the CR finder.

    The exposures here carry no metadata at all: a gate that consults the
    ramp header would raise rather than skip.
    """

    def setUp(self):
        self.config = CosmicRayConfig()
        self.exposure = makeExposure("n1")

    def makeTask(self):
        """Make a task whose single-exposure CR finder only records its calls"""
        task = CosmicRayTask(config=self.config, name="cosmicray")
        self.called = []
        task.runSingle = self.called.append
        return task

    def testNirIsRepairedByDefault(self):
        task = self.makeTask()
        task.run([self.exposure])
        self.assertEqual(self.called, [self.exposure])

    def testNirIsSkippedWhenMorphologicalCRsDisabled(self):
        self.config.doH4MorphologicalCRs = False
        task = self.makeTask()
        result = task.run([self.exposure])
        self.assertEqual(self.called, [])
        self.assertEqual(result.exposures, [self.exposure])

    def testNothingHappensWithoutDoCosmicRay(self):
        self.config.doCosmicRay = False
        task = self.makeTask()
        task.run([self.exposure])
        self.assertEqual(self.called, [])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    runTests(globals())
