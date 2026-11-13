import numpy as np

import lsst.utils.tests
import lsst.afw.geom

from pfs.drp.stella.applyExclusionZone import applyExclusionZone, getExclusionZone
from pfs.drp.stella.maskNearby import maskNearby
from pfs.drp.stella.readLineList import ReadLineListTask
from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus, ReferenceLineSource
from pfs.drp.stella.tests import runTests

display = None


class ApplyExclusionZoneTestCase(lsst.utils.tests.TestCase):
    """Test functionality of applyExclusionZone"""
    def setUp(self):
        self.num = 301
        self.wlMin = 600.0  # nm
        self.wlMax = 900.0  # nm
        self.spacing = (self.wlMax - self.wlMin)/(self.num - 1)
        self.wavelength = np.linspace(self.wlMin, self.wlMax, self.num, dtype=float)
        self.lines = ReferenceLineSet.fromColumns(wavelength=self.wavelength, intensity=1.0, intensityErr=0.1,
                                                  status=ReferenceLineStatus.GOOD, description="Fake",
                                                  transition=None, source=ReferenceLineSource.NONE)

    def testBasic(self):
        """Test basic functionality"""
        # Exclusion zone smaller than spacing: nothing flagged
        applyExclusionZone(self.lines, 0.5*self.spacing, ReferenceLineStatus.BLEND)
        self.assertFloatsEqual(self.lines.status, ReferenceLineStatus.GOOD)
        # Exclusion zone larger than spacing: everything flagged
        applyExclusionZone(self.lines, 1.1*self.spacing, ReferenceLineStatus.BLEND)
        self.assertFloatsEqual(self.lines.status, ReferenceLineStatus.BLEND)

    def testSingle(self):
        """Test that it works on just a single set of lines in the zone"""
        index = self.num//2
        wavelength = self.wavelength[index]
        self.lines.wavelength[index - 1] = wavelength - 0.25*self.spacing
        self.lines.wavelength[index + 1] = wavelength + 0.25*self.spacing
        applyExclusionZone(self.lines, 0.5*self.spacing, ReferenceLineStatus.BLEND)
        expect = np.full(self.num, ReferenceLineStatus.GOOD, dtype=int)
        expect[index - 1: index + 2] = ReferenceLineStatus.BLEND
        self.assertFloatsEqual(self.lines.status, expect)

    def testOrStatusFlags(self):
        """Test that we're applying the flags with OR"""
        origFlag = ReferenceLineStatus.DETECTORMAP_USED
        self.lines.status[:] = origFlag
        applyExclusionZone(self.lines, 1.1*self.spacing, ReferenceLineStatus.BLEND)
        self.assertFloatsEqual(self.lines.status, origFlag | ReferenceLineStatus.BLEND)

    def testUnsorted(self):
        """Test that it works even if the wavelengths aren't sorted"""
        indices = np.argsort(np.random.uniform(size=self.num))
        self.lines.wavelength[:] = self.lines.wavelength[indices]
        applyExclusionZone(self.lines, 1.1*self.spacing, ReferenceLineStatus.BLEND)
        self.assertFloatsEqual(self.lines.status, ReferenceLineStatus.BLEND)

    def makePair(self, ratio: float, separation: float = 0.25):
        """Move the neighbors of the middle line close to it, with relative
        intensity ``ratio``

        Returns the index of the middle line.
        """
        index = self.num//2
        self.lines.wavelength[index - 1] = self.wavelength[index] - separation*self.spacing
        self.lines.intensity[index - 1] = ratio
        self.lines.intensity[index] = 1.0
        return index

    def testRelativeIntensity(self):
        """Test that faint neighbors don't exclude a bright line"""
        radius = 0.5*self.spacing
        minRelativeIntensity = 0.05
        for ratio, brightFlagged in ((0.01, False), (0.2, True), (np.nan, True)):
            with self.subTest(ratio=ratio):
                self.setUp()
                index = self.makePair(ratio)
                applyExclusionZone(self.lines, radius, ReferenceLineStatus.BLEND, minRelativeIntensity)
                expect = np.full(self.num, ReferenceLineStatus.GOOD, dtype=int)
                expect[index - 1] = ReferenceLineStatus.BLEND  # The neighbor is always flagged
                if brightFlagged:
                    expect[index] = ReferenceLineStatus.BLEND
                self.assertFloatsEqual(self.lines.status, expect)

        # Disabling the relative intensity flags both
        self.setUp()
        index = self.makePair(0.01)
        applyExclusionZone(self.lines, radius, ReferenceLineStatus.BLEND, 0.0)
        self.assertEqual(self.lines.status[index], ReferenceLineStatus.BLEND)
        self.assertEqual(self.lines.status[index - 1], ReferenceLineStatus.BLEND)

        # A comparable line on the other side still excludes the bright line
        self.setUp()
        index = self.makePair(0.01)
        self.lines.wavelength[index + 1] = self.wavelength[index] + 0.25*self.spacing
        applyExclusionZone(self.lines, radius, ReferenceLineStatus.BLEND, minRelativeIntensity)
        self.assertEqual(self.lines.status[index], ReferenceLineStatus.BLEND)

        # The ReferenceLineSet method passes the threshold through
        self.setUp()
        index = self.makePair(0.01)
        self.lines.applyExclusionZone(radius, minRelativeIntensity=minRelativeIntensity)
        self.assertEqual(self.lines.status[index], ReferenceLineStatus.GOOD)
        self.assertEqual(self.lines.status[index - 1], ReferenceLineStatus.BLEND)

    def testRelativeIntensityProtected(self):
        """Test that PROTECTED and NOT_VISIBLE work with relative intensities"""
        radius = 0.5*self.spacing
        index = self.makePair(1.0)
        self.lines.status[index] = ReferenceLineStatus.PROTECTED
        applyExclusionZone(self.lines, radius, ReferenceLineStatus.BLEND, 0.05)
        self.assertEqual(self.lines.status[index], ReferenceLineStatus.PROTECTED)
        self.assertEqual(self.lines.status[index - 1], ReferenceLineStatus.BLEND)

        self.setUp()
        index = self.makePair(1.0)
        self.lines.status[index - 1] = ReferenceLineStatus.NOT_VISIBLE
        applyExclusionZone(self.lines, radius, ReferenceLineStatus.BLEND, 0.05)
        self.assertEqual(self.lines.status[index], ReferenceLineStatus.GOOD)
        self.assertEqual(self.lines.status[index - 1], ReferenceLineStatus.NOT_VISIBLE)

    def testRelativeIntensityMatchesMaskNearby(self):
        """Test that equal intensities give the same result as maskNearby"""
        rng = np.random.default_rng(12345)
        wavelength = np.sort(rng.uniform(self.wlMin, self.wlMax, 1000))
        wavelength[10] = wavelength[11]  # Duplicates don't exclude each other
        status = np.zeros(wavelength.size, dtype=int)
        for radius in (0.05, 0.3, 1.0):
            with self.subTest(radius=radius):
                expect = getExclusionZone(wavelength, radius, status)
                result = getExclusionZone(wavelength, radius, status, np.ones_like(wavelength), 0.05)
                np.testing.assert_array_equal(result, expect)
                np.testing.assert_array_equal(expect, maskNearby(wavelength, radius))

    def testReadLineList(self):
        """Test the exclusion zone in ReadLineListTask uses intensities"""
        config = ReadLineListTask.ConfigClass()
        config.lightSources = ["HgCd"]
        config.exclusionRadius = 0.3
        lines = ReadLineListTask(config=config).run(arm="b")
        hg = lines.status[np.isclose(lines.wavelength, 404.7708)]
        self.assertEqual(len(hg), 1)
        self.assertEqual(hg[0] & ReferenceLineStatus.BLEND, 0)
        ar = lines.status[np.isclose(lines.wavelength, 404.5561)]
        self.assertTrue(np.all(ar & ReferenceLineStatus.BLEND))


class MaskNearbyTestCase(lsst.utils.tests.TestCase):
    """Test maskNearby"""
    def setUp(self):
        config = ReadLineListTask.ConfigClass()
        config.lightSources = ["sky"]
        task = ReadLineListTask(config=config)
        self.lines = task.run(arm="r")

    def assertMask(self, wavelength, mask, exclusionRadius):
        """Assert that the mask is as expected

        We calculate the mask with the old, slow code and compare.
        """
        expect = np.zeros_like(wavelength, dtype=bool)
        for wl in wavelength:
            distance = wavelength - wl
            expect |= (np.abs(distance) < exclusionRadius) & (distance != 0)
        self.assertTrue(np.all(mask == expect))

    def testBasic(self):
        """Test basic functionality"""
        exclusionZone = 0.5  # nm
        mask = maskNearby(self.lines.wavelength, exclusionZone)
        self.assertMask(self.lines.wavelength, mask, exclusionZone)

    def testUnsorted(self):
        """Test that it works even if the wavelengths aren't sorted"""
        rng = np.random.RandomState(12345)
        indices = np.argsort(rng.uniform(size=len(self.lines)))
        wavelength = self.lines.wavelength[indices]
        exclusionZone = 0.5
        mask = maskNearby(wavelength, exclusionZone)
        self.assertMask(wavelength, mask, exclusionZone)

    def testMultiple(self):
        """Test that it works with multiple copies of the wavelengths"""
        exclusionZone = 0.5
        wavelength = np.concatenate([self.lines.wavelength]*3)
        mask = maskNearby(wavelength, exclusionZone)
        self.assertMask(wavelength, mask, exclusionZone)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()
    np.random.seed(12345)


if __name__ == "__main__":
    runTests(globals())
