#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python FiberTrace.py
or
   python
   >>> import FiberTrace; FiberTrace.run()
"""

import unittest
import numpy as np
import lsst.utils.tests as tests
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import pfs.drp.stella as drpStella

try:
    type(display)
except NameError:
    display = False

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

class FiberTraceTestCase(tests.TestCase):
    """A test case for measuring FiberTrace quantities"""

    def setUp(self):
        self.nAperture, self.traceWidth = 4, 11 # number of apertures, and width of a trace

        self.im = afwImage.ImageF(self.nAperture*self.traceWidth, 100)
        ima = self.im.getArray()
        X, Y = np.meshgrid(np.arange(self.im.getWidth()), np.arange(self.im.getHeight()))
        ima[Y, X] = np.sin(X.astype(float)*np.pi/self.traceWidth)**2
        ima /= np.sum(ima)/(self.nAperture*self.im.getHeight())
        del ima
        
    def tearDown(self):
        del self.im

    def testFiberTrace(self):
        """Test that we can't create a FiberTrace"""

        fts = drpStella.FiberTraceSet() # not virtual

        with self.assertRaises(AttributeError) as e:
            drpStella.FiberTrace(afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.ExtentI(self.traceWidth, 10)))
            
        self.assertTrue("class is abstract" in str(e.exception))

    def testImageFiberTrace(self):
        """Test creating ImageFiberTrace"""
        x0, traceHeight = 10, 100

        traceExtent = afwGeom.ExtentI(self.traceWidth, traceHeight)
        trace = drpStella.ImageFiberTrace(afwGeom.BoxI(afwGeom.PointI(x0, 0), traceExtent))

        trace.setXCenters(x0 + 0.5*self.traceWidth + np.zeros(traceHeight))
        
        bbox = trace.getBBox()
        self.assertEqual(x0, bbox.getBeginX())
        self.assertEqual(self.traceWidth, bbox.getWidth())
        self.assertEqual(traceHeight, bbox.getHeight())

        self.assertEqual(trace.getXCenters()[traceHeight//2], x0 + 0.5*self.traceWidth)

        trace.setBBox(afwGeom.BoxI(afwGeom.PointI(0, 0), traceExtent))
        bbox = trace.getBBox()
        self.assertEqual(0, trace.getBBox().getBeginX())        

    def testImageFiberTraceSet(self):
        """Test that we can set and manipulate a set of ImageFiberTrace objects"""
        fts = drpStella.FiberTraceSet(self.im)

        if display:
            ds9.mtv(self.im)

        traceExtent = afwGeom.ExtentI(self.traceWidth, self.im.getHeight())
        for i in range(self.im.getWidth()//self.traceWidth):
            trace = drpStella.ImageFiberTrace(afwGeom.BoxI(afwGeom.PointI(i*self.traceWidth, 0), traceExtent))
            fts.setFiberTrace(i, trace)

            trace.setXCenters((i + 0.5)*self.traceWidth + np.zeros(self.im.getHeight()))

        for i in range(fts.getNAperture()):
            trace = fts.getFiberTrace(i)
            bbox = trace.getBBox()                                          
            prof = trace.getProfile(self.im.getHeight()//2)

            self.assertEqual(len(prof), self.traceWidth)
            self.assertAlmostEqual(sum(prof), 1.0, 4) # relies on correct normalisation of im.self

            if display:
                displayUtils.drawBBox(bbox, borderWidth=0.4)
                for y in range(bbox.getHeight()):
                    ds9.dot('+', trace.getXCenters()[y], y, size=0.45)
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(FiberTraceTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
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
