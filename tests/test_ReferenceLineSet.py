import sys
import pickle
import unittest

import numpy as np

import lsst.utils.tests
from pfs.drp.stella import ReferenceLine, ReferenceLineSet, ReferenceLineStatus, ReferenceLineSource


class ReferenceLineSetTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.num = 7
        self.description = "line"
        self.status = ReferenceLineStatus.BROAD
        self.wavelength = 1234.5678
        self.intensity = 543.21
        self.transition = "UNKNOWN"
        self.source = ReferenceLineSource.NONE
        lines = []
        for _ in range(self.num):
            lines.append(ReferenceLine(self.description, self.wavelength, self.intensity, self.status,
                                       self.transition, self.source))
        self.lines = ReferenceLineSet.fromRows(lines)

    def assertReferenceLines(self, lines):
        """Assert that the values are as expected"""
        self.assertTrue(np.all(lines.description == np.array([self.description]*self.num)))
        self.assertFloatsEqual(lines.wavelength, np.full(self.num, self.wavelength, dtype=float))
        self.assertFloatsEqual(lines.intensity, np.full(self.num, self.intensity, dtype=float))
        self.assertFloatsEqual(lines.status, np.full(self.num, self.status, dtype=np.int32))

    def testBasics(self):
        """Test construction, getters, setters and arithmetic with ``Status``"""
        self.assertReferenceLines(self.lines)

    def testPickle(self):
        """Test that ``ReferenceLineSet`` can be pickled"""
        copy = pickle.loads(pickle.dumps(self.lines))
        self.assertReferenceLines(copy)

    def testReadWrite(self):
        """Test that we can write and read correctly"""
        with lsst.utils.tests.getTempFilePath("txt") as filename:
            self.lines.writeLineList(filename)
            new = ReferenceLineSet.fromLineList(filename)
            self.assertReferenceLines(new)

    def testSort(self):
        """Test we can sort the data correctly"""
        lines = []
        lines.append(ReferenceLine('OH', 3456.0, 1.234, 0, "UNKNOWN", ReferenceLineSource.NONE))
        lines.append(ReferenceLine('NaI', 1234.0, 3.0, 0, "UNKNOWN", ReferenceLineSource.NONE))
        lineSet = ReferenceLineSet.fromRows(lines)
        self.assertEqual('OH', lineSet[0].description)
        self.assertEqual('NaI', lineSet[1].description)
        lineSet.sort()
        self.assertEqual('NaI', lineSet[0].description)
        self.assertEqual('OH', lineSet[1].description)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main()
