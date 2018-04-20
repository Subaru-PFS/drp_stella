#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python test_detectorMap.py
or
   python
   >>> import test_detectorMap; test_detectorMap.run()
"""
import os
import sys
import pickle
import unittest

import numpy as np

import lsst.utils.tests as tests
import lsst.pex.exceptions as pexExcept
import lsst.afw.geom as afwGeom

import pfs.drp.stella.utils as drpStellaUtils
import pfs.drp.stella.detectorMap as detectorMap

def read_fiberIdsxCentersWavelengths():
    wLenFile = os.path.join(os.environ["OBS_PFS_DIR"], "pfs/RedFiberPixels.fits.gz")
    xCenters, wavelengths, fiberIds = drpStellaUtils.readWavelengthFile(wLenFile)
    #
    # N.b. Andreas calls these "traceIds" but I'm assuming that they are actually be 1-indexed fiberIds
    #
    nFiber = len(set(fiberIds))
    fiberIds = fiberIds.reshape(nFiber, len(fiberIds)//nFiber)
    xCenters = xCenters.reshape(fiberIds.shape)
    wavelengths = wavelengths.reshape(fiberIds.shape)
    fiberIds = fiberIds.reshape(fiberIds.shape)[:, 0].copy()

    missing = (wavelengths == 0)
    xCenters[missing] = np.nan
    wavelengths[missing] = np.nan

    return fiberIds, xCenters, wavelengths
    
class DetectorMapTestCase(tests.TestCase):
    """A test case for measuring DetectorMap quantities"""

    @classmethod
    def setUpClass(cls):
        cls.fiberIds, cls.xCenters, cls.wavelengths = read_fiberIdsxCentersWavelengths()
        cls.bbox = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.ExtentI(400, cls.xCenters.shape[1]))
        cls.nKnot = 25

    @classmethod
    def tearDownClass(cls):
        del cls.bbox
        del cls.fiberIds
        del cls.xCenters
        del cls.wavelengths

    def setUp(self):
        """Set things up.  This ctor should be fine, and is checked again in testCtor"""
        self.ftMap = detectorMap.DetectorMap(self.bbox, self.fiberIds, self.xCenters,
                                             self.wavelengths, nKnot=self.nKnot)

    def tearDown(self):
        del self.ftMap

    def testCtor(self):
        """Test that we can create a DetectorMap"""
        # N.b. this must be identical to the call in setUp() (which is wrapped in a try-block)
        ftMap = detectorMap.DetectorMap(self.bbox, self.fiberIds, self.xCenters, self.wavelengths,
                                            nKnot=100)

        self.assertTrue((ftMap.getFiberIds() == self.fiberIds).all())
        self.assertTrue((ftMap.getSlitOffsets() == 0.0).all())

    def testBadCtor(self):
        """Test that we cannot create a DetectorMap with invalid arguments"""
        with self.assertRaises(pexExcept.LengthError):
            bbox = afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.ExtentI(100, 100))
            detectorMap.DetectorMap(bbox, self.fiberIds, self.xCenters, self.wavelengths)

        with self.assertRaises(pexExcept.LengthError):
            detectorMap.DetectorMap(self.bbox, self.fiberIds[:-10], self.xCenters, self.wavelengths)

        with self.assertRaises(pexExcept.LengthError):
            detectorMap.DetectorMap(self.bbox, self.fiberIds, self.xCenters[0:10], self.wavelengths)

    def testFiberId(self):
        """Test that we set fiberId correctly"""
        self.assertTrue((self.ftMap.getFiberIds() == self.fiberIds).all())

    def testSlitOffset(self):
        """Test that we read/set fiberOffset correctly"""
        nFiber = len(self.ftMap.getFiberIds())

        val = 666.0
        offsets = np.empty((3, nFiber), dtype=np.float32)
        offsets[:] = val
        self.ftMap.setSlitOffsets(offsets);

        slitOffsets = self.ftMap.getSlitOffsets()
        self.assertTrue(slitOffsets.shape[0] == 3) # dx, dy, dfocus

        for i in (self.ftMap.FIBER_DX, self.ftMap.FIBER_DY, self.ftMap.FIBER_DFOCUS):
            self.assertTrue((self.ftMap.getSlitOffsets()[i] == val).all())
        
        with self.assertRaises(pexExcept.LengthError):
            self.ftMap.setSlitOffsets(np.empty((3, 1), dtype=np.float32))

        with self.assertRaises(pexExcept.LengthError):
            self.ftMap.setSlitOffsets(np.empty((2, nFiber), dtype=np.float32))

    def testSlitOffset2(self):
        """Test that we can set slitOffsets in ctor"""
        slitOffsets = np.ones((3, len(self.fiberIds)), dtype=np.float32)

        ftMap = detectorMap.DetectorMap(self.bbox, self.fiberIds, self.xCenters, self.wavelengths,
                                            slitOffsets=slitOffsets)

        self.assertTrue((ftMap.getSlitOffsets() == 1.0).all())

    def testGetWavelength(self):
        """Test that we recover Wavelength correctly"""
        with self.assertRaises(pexExcept.RangeError):
            self.ftMap.getWavelength(0)      # fiberIds start at 0

        fid = 10
        fiberId = self.fiberIds[fid]
        delta = self.ftMap.getWavelength(fiberId) - self.wavelengths[fid]

        if display:
            import matplotlib.pyplot as plt
            y = np.arange(self.bbox.getHeight())
            
            plt.plot(y, delta)
            plt.show()

        delta = delta[np.isfinite(self.wavelengths[fid])]

        self.assertLess(np.max(np.abs(delta)), 2e-3)

    def testGetXCenter(self):
        """Test that we recover XCenter correctly"""
        with self.assertRaises(pexExcept.RangeError):
            self.ftMap.getXCenter(0)      # fiberIds start at 0

        y = np.arange(0, self.bbox.getMaxY(), dtype=np.float32)
        fid = 10

        fiberId = self.fiberIds[fid]
        delta = self.ftMap.getXCenter(fiberId) - self.xCenters[fid]

        delta = delta[np.isfinite(self.xCenters[fid])]
        
        self.assertLess(np.max(np.abs(delta)), 2.5e-4)

    def testFindFiberId(self):
        """Test that we can find a fiber ID"""
        pos = afwGeom.PointD(10, 100)
        self.ftMap.findFiberId(pos)

    def testPickle(self):
        """Test that we can successfully round-trip pickle"""
        other = pickle.loads(pickle.dumps(self.ftMap))
        self.assertEqual(self.ftMap.getBBox(),other.getBBox())
        self.assertFloatsEqual(self.ftMap.getFiberIds(), other.getFiberIds())
        # xCenter and wavelength are inexact because they are not stored in the object but have
        # to be calculated; the result is that the two objects have slightly different splines
        # defined.
        self.assertFloatsAlmostEqual(self.ftMap.getXCenter(), other.getXCenter(), rtol=1.0e-5)
        self.assertFloatsAlmostEqual(self.ftMap.getWavelength(), other.getWavelength(), rtol=1.0e-6)
        self.assertEqual(self.ftMap.getNKnot(), other.getNKnot())
        self.assertFloatsEqual(self.ftMap.getSlitOffsets(), other.getSlitOffsets())
        self.assertFloatsEqual(self.ftMap.getThroughput(), other.getThroughput())


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(DetectorMapTestCase)
    return unittest.TestSuite(suites)

def run(exit=False):
    #Run the tests
    tests.run(suite(), exit)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--display", '-d', default=False, action="store_true", help="Activate display?")
    parser.add_argument("--verbose", '-v', type=int, default=0, help="Verbosity level")
    args = parser.parse_args()
    display = args.display
    verbose = args.verbose
    run(True)
