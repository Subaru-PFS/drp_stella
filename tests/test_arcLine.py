import numpy as np

import lsst.utils.tests
import lsst.afw.geom

from pfs.drp.stella.arcLine import ArcLine, ArcLineSet
from pfs.drp.stella.tests import runTests
from pfs.drp.stella.ReferenceLine import ReferenceLine

display = None


class ArcLineTestCase(lsst.utils.tests.TestCase):
    """Test functionality of ArcLine, ArcLineSet"""
    def setUp(self):
        self.length = 128
        self.fiberId = np.arange(123, 456, 78, dtype=np.int32)

        self.wlMin = 600.0  # nm
        self.wlMax = 900.0  # nm
        scale = self.wlMax - self.wlMin

        self.x = np.random.uniform(size=self.fiberId.shape)*(self.length - 1)
        self.xErr = 0.01*np.random.uniform(size=self.fiberId.shape)
        self.y = np.random.uniform(size=self.fiberId.shape)*(self.length - 1)
        self.yErr = 0.01*np.random.uniform(size=self.fiberId.shape)
        self.wavelength = np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin
        self.flag = np.random.uniform(0, 10, size=self.fiberId.shape).astype(int)
        self.status = np.full_like(self.fiberId, int(ReferenceLine.Status.FIT), dtype=int)
        self.description = np.array([chr(ord("A") + ii) for ii in range(len(self.fiberId))])

        self.lines = [ArcLine(*args) for args in zip(self.fiberId, self.wavelength,
                                                     self.x, self.y, self.xErr, self.yErr,
                                                     self.flag, self.status, self.description)]

    def assertArcLineSet(self, lines, atol=0.0):
        """Check that the ArcLineSet is what we expect

        Parameters
        ----------
        lines : `pfs.drp.stella.arcLine.ArcLineSet`
            Arc line data.
        atol : `float`, optional
            Absolute tolerance.
        """
        self.assertEqual(len(lines), len(self.fiberId))
        self.assertFloatsEqual(lines.fiberId, self.fiberId)
        self.assertFloatsAlmostEqual(lines.wavelength, self.wavelength, atol=atol)
        self.assertFloatsAlmostEqual(lines.x, self.x, atol=atol)
        self.assertFloatsAlmostEqual(lines.xErr, self.xErr, atol=atol)
        self.assertFloatsAlmostEqual(lines.y, self.y, atol=atol)
        self.assertFloatsAlmostEqual(lines.yErr, self.yErr, atol=atol)
        self.assertFloatsEqual(lines.flag, self.flag)
        self.assertFloatsEqual(lines.status, self.status)
        self.assertListEqual(lines.description.tolist(), self.description.tolist())

    def testBasic(self):
        """Test basic functionality"""
        lines = ArcLineSet(self.lines)
        self.assertArcLineSet(lines)

    def testExtend(self):
        """Test ArcLineSet.extend"""
        lines1 = ArcLineSet.empty()
        lines1.extend(self.lines)
        self.assertArcLineSet(lines1)

        lines2 = ArcLineSet.empty()
        lines2 += lines1
        self.assertArcLineSet(lines2)

        lines3 = ArcLineSet.empty() + lines2
        self.assertArcLineSet(lines3)

    def testAppend(self):
        """Test ArcLineSet.append"""
        lines = ArcLineSet.empty()
        for ii, ll in enumerate(self.lines):
            lines.append(self.fiberId[ii], self.wavelength[ii], self.x[ii], self.y[ii],
                         self.xErr[ii], self.yErr[ii], self.flag[ii], self.status[ii], self.description[ii])
        self.assertArcLineSet(lines)

    def testIteration(self):
        """Test iteration over ArcLineSet"""
        lines = ArcLineSet(self.lines)
        for ii, ll in enumerate(lines):
            self.assertEqual(ll.fiberId, self.fiberId[ii])
            self.assertEqual(ll.wavelength, self.wavelength[ii])
            self.assertEqual(ll.x, self.x[ii])
            self.assertEqual(ll.xErr, self.xErr[ii])
            self.assertEqual(ll.y, self.y[ii])
            self.assertEqual(ll.yErr, self.yErr[ii])
            self.assertEqual(ll.flag, self.flag[ii])
            self.assertEqual(ll.status, self.status[ii])
            self.assertEqual(ll.description, self.description[ii])

    def testPersistence(self):
        """Test persistence"""
        lines = ArcLineSet(self.lines)
        with lsst.utils.tests.getTempFilePath(".fits") as filename:
            lines.writeFits(filename)
            copy = ArcLineSet.readFits(filename)
            self.assertArcLineSet(copy)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()
    np.random.seed(12345)


if __name__ == "__main__":
    runTests(globals())
