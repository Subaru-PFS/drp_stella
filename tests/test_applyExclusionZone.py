import numpy as np

import lsst.utils.tests
import lsst.afw.geom

from pfs.drp.stella.applyExclusionZone import applyExclusionZone
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
        self.lines = ReferenceLineSet.fromColumns(wavelength=self.wavelength, intensity=1.0,
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
        self.lines.status[:] = ReferenceLineStatus.NOT_VISIBLE
        applyExclusionZone(self.lines, 1.1*self.spacing, ReferenceLineStatus.BLEND)
        self.assertFloatsEqual(self.lines.status, ReferenceLineStatus.NOT_VISIBLE | ReferenceLineStatus.BLEND)

    def testUnsorted(self):
        """Test that it works even if the wavelengths aren't sorted"""
        indices = np.argsort(np.random.uniform(size=self.num))
        self.lines.wavelength[:] = self.lines.wavelength[indices]
        applyExclusionZone(self.lines, 1.1*self.spacing, ReferenceLineStatus.BLEND)
        self.assertFloatsEqual(self.lines.status, ReferenceLineStatus.BLEND)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()
    np.random.seed(12345)


if __name__ == "__main__":
    runTests(globals())
