#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python FiberTraces.py
or
   python
   >>> import FiberTraces; FiberTraces.run()
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

class FiberTracesTestCase(tests.TestCase):
    """A test case for measuring FiberTraces quantities"""

    def setUp(self):
        pass
        
    def tearDown(self):
        pass

    def testFiberTrace(self):
        """Test that we can set and manipulate a FiberTrace"""
        traceWidth = 10                 # width of a trace
        traceHeight = 100

        x0 = 20
        traceExtent = afwGeom.ExtentI(traceWidth, traceHeight)
        
        trace = drpStella.FiberTrace(afwGeom.BoxI(afwGeom.PointI(x0, 0), traceExtent))
        trace.setXCenters(x0 + 0.5*traceWidth + np.zeros(traceHeight))
        
        bbox = trace.getBBox()
        self.assertEqual(x0, bbox.getBeginX())
        self.assertEqual(traceWidth, bbox.getWidth())
        self.assertEqual(traceHeight, bbox.getHeight())

        self.assertEqual(trace.getXCenters()[traceHeight//2], x0 + 0.5*traceWidth)

        trace.setBBox(afwGeom.BoxI(afwGeom.PointI(0, 0), traceExtent))
        bbox = trace.getBBox()
        self.assertEqual(0, trace.getBBox().getBeginX())        

    def testFiberTraces(self):
        """Test that we can set and manipulate a set of FiberTrace objects"""
        fts = drpStella.FiberTraceSet()

        traceWidth = 10                 # width of a trace

        im = afwImage.ImageF(40, 100)
        ima = im.getArray()
        X, Y = np.meshgrid(np.arange(im.getWidth()), np.arange(im.getHeight()))
        ima[Y, X] = np.sin(X.astype(float)*np.pi/traceWidth)**2
        del ima

        if display:
            ds9.mtv(im)

        traceExtent = afwGeom.ExtentI(traceWidth, im.getHeight())
        for i in range(im.getWidth()//traceWidth):
            trace = drpStella.FiberTrace(afwGeom.BoxI(afwGeom.PointI(i*traceWidth, 0), traceExtent))
            fts.setFiberTrace(i, trace)

            trace.setXCenters((i + 0.5)*traceWidth + np.zeros(im.getHeight()))

        for i in range(fts.getNAperture()):
            trace = fts.getFiberTrace(i)
            bbox = trace.getBBox()                                          

            print trace.getProfile(10)

            if display:
                displayUtils.drawBBox(bbox, borderWidth=0.4)
                for y in range(bbox.getHeight()):
                    ds9.dot('+', trace.getXCenters()[y], y, size=0.45)
        pass
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(FiberTracesTestCase)
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
