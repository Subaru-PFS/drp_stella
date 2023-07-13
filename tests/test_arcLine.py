import numpy as np

import lsst.utils.tests
import lsst.afw.geom

from pfs.drp.stella.arcLine import ArcLine, ArcLineSet
from pfs.drp.stella.tests import runTests
from pfs.drp.stella.referenceLine import ReferenceLineStatus
from pfs.drp.stella.referenceLine import ReferenceLineSource

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
        self.xx = np.random.uniform(size=self.fiberId.shape)*(self.length - 1)
        self.yy = np.random.uniform(size=self.fiberId.shape)*(self.length - 1)
        self.xy = np.random.uniform(size=self.fiberId.shape)*(self.length - 1)
        self.wavelength = np.random.uniform(size=self.fiberId.shape)*scale + self.wlMin
        self.flux = 1000.0*np.random.uniform(size=self.fiberId.shape)
        self.fluxErr = 10.0*np.random.uniform(size=self.fiberId.shape)
        self.flag = np.random.choice((True, False), size=self.fiberId.shape)
        self.status = np.full_like(self.fiberId, int(ReferenceLineStatus.GOOD), dtype=int)
        self.description = np.array([chr(ord("A") + ii) for ii in range(len(self.fiberId))])
        self.transition = np.full_like(self.fiberId, "", dtype=str)
        self.source = np.full_like(self.fiberId, int(ReferenceLineSource.NONE), dtype=int)

        self.lines = [ArcLine(*args) for args in zip(self.fiberId, self.wavelength,
                                                     self.x, self.y, self.xErr, self.yErr,
                                                     self.xx, self.yy, self.xy,
                                                     self.flux, self.fluxErr,
                                                     self.flag, self.status, self.description,
                                                     self.transition, self.source)]

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
        self.assertFloatsAlmostEqual(lines.flux, self.flux, atol=atol)
        self.assertFloatsAlmostEqual(lines.fluxErr, self.fluxErr, atol=atol)
        self.assertTrue(np.all(lines.flag == self.flag))
        self.assertFloatsEqual(lines.status, self.status)
        self.assertListEqual(lines.description.tolist(), self.description.tolist())

    def testBasic(self):
        """Test basic functionality"""
        lines = ArcLineSet.fromRows(self.lines)
        self.assertArcLineSet(lines)

    def testExtend(self):
        """Test ArcLineSet.extend"""
        lines1 = ArcLineSet.fromRows(self.lines)
        self.assertArcLineSet(lines1)

        lines2 = ArcLineSet.empty()
        lines2 += lines1
        self.assertArcLineSet(lines2)

        lines3 = ArcLineSet.empty() + lines2
        self.assertArcLineSet(lines3)

    def testIteration(self):
        """Test iteration over ArcLineSet"""
        lines = ArcLineSet.fromRows(self.lines)
        for ii, ll in enumerate(lines):
            self.assertEqual(ll.fiberId, self.fiberId[ii])
            self.assertEqual(ll.wavelength, self.wavelength[ii])
            self.assertEqual(ll.x, self.x[ii])
            self.assertEqual(ll.xErr, self.xErr[ii])
            self.assertEqual(ll.y, self.y[ii])
            self.assertEqual(ll.yErr, self.yErr[ii])
            self.assertEqual(ll.flux, self.flux[ii])
            self.assertEqual(ll.fluxErr, self.fluxErr[ii])
            self.assertEqual(ll.flag, self.flag[ii])
            self.assertEqual(ll.status, self.status[ii])
            self.assertEqual(ll.description, self.description[ii])

    def testPersistence(self):
        """Test persistence"""
        lines = ArcLineSet.fromRows(self.lines)
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
