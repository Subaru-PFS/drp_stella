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
        self.nAperture, self.traceWidth, self.traceHeight = 4, 11, 200 # number of apertures, and width of a trace

        self.im = afwImage.ImageF(self.nAperture*self.traceWidth, 100)
        ima = self.im.getArray()
        X, Y = np.meshgrid(np.arange(self.im.getWidth()), np.arange(self.im.getHeight()))
        ima[Y, X] = np.sin(X.astype(float)*np.pi/self.traceWidth)**2
        ima /= np.sum(ima)/(self.nAperture*self.im.getHeight())
        del ima
        
    def tearDown(self):
        del self.im

    def testFiberTrace(self):
        """Test that we can create a FiberTraceFunctionFindingControl"""
        ftffc = drpStella.FiberTraceFunctionFindingControl()
        
        """Test that we can set the parameters of the FiberTraceFunctionFindingControl"""
        ftffc.fiberTraceFunctionControl.interpolation = "POLYNOMIAL"
        ftffc.fiberTraceFunctionControl.order = 3
        ftffc.fiberTraceFunctionControl.xLow = -4.2
        ftffc.fiberTraceFunctionControl.xHigh = 4.2
        ftffc.apertureFWHM = 3.2
        ftffc.signalThreshold = 100.
        ftffc.nTermsGaussFit = 3
        ftffc.saturationLevel = 65500.
        
#        import pdb; pdb.set_trace()
#        print(ftfc)
#        self.assertEqual(ftfc.xLow, ftf.fiberTraceFunctionControl.xLow)

        """Test that we can create a FiberTrace given width and height"""
        width = 5
        height = 100
        ft = drpStella.FiberTraceF(width,height)
        if ft.getImage().getWidth() == width:
            print("ft.getImage.getWidth() returned ", ft.getImage().getWidth())
        if ft.getImage().getHeight() == height:
            print("ft.getImage.getHeight() returned ", ft.getImage().getHeight())

        """Test that we can create a FiberTrace given a MaskedImage"""
        """Flat"""
        mif = afwImage.MaskedImageF("/home/azuri/spectra/pfs/IR-23-0-centerFlatx2.fits")
        print("mif created")
        
        fti = drpStella.FiberTraceF(mif)
        print("fti created")
        
        """Test that we can create a FiberTraceSet"""
        fts = drpStella.FiberTraceSetF()
        print("fts created")
        
        """Test that we can add a FiberTrace to the FiberTraceSet"""
        fts.addFiberTrace(fti)
        print("fti added to fts")
        
        """Test that we can trace fibers"""
        msi = drpStella.MaskedSpectrographImageF(mif)
        print("msi created")
#        ds9.mtv(msi.getMaskedImage())
        
        msi.findAndTraceApertures(ftffc, 0, mif.getHeight(), 10)
        print("msi.findAndTraceApertures finished")
        print("msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().xCenter = ",msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().xCenter)
        print("msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().yCenter = ",msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().yCenter)
        print("msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().yLow = ",msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().yLow)
        print("msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().yHigh = ",msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().yHigh)
        print("msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().coefficients(0) = ",msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().coefficients[0])
        print("msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().coefficients(1) = ",msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().coefficients[1])
        print("msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().coefficients(2) = ",msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().coefficients[2])
        print("msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().coefficients(3) = ",msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction().coefficients[3])

        ftec = drpStella.FiberTraceExtractionControl();
        ft = msi.getFiberTraceSet().getFiberTrace(0);
        ft.setFiberTraceExtractionControl(ftec);
        print("ft.getFiberTraceExtractionControl().swathWidth = ");
        print(ft.getFiberTraceExtractionControl().swathWidth);
        print("msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceExtractionControl().swathWidth = ",msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceExtractionControl().swathWidth);

        """calculate profile and extract from flat"""
        ft.MkSlitFunc();
        print("ft.MkSlitFunc finished");

        """write msi.getFiberTraceSet().getFiberTrace(0).getProfile() to fits file"""
#        imageDirectory = "/home/azuri/spectra/pfs/";
        profile = msi.getFiberTraceSet().getFiberTrace(0).getProfile();
        print("profile copied");
        print("profile.getDimensions() = ",profile.getDimensions());
#        print("profile")
#        profile.writeFits("/home/azuri/spectra/pfs/trace0_profile.fits");
#        print("profile written to /home/azuri/spectra/pfs/trace0_profile.fits");
        
        """extract sky spectrum"""
#        mis = drpStella.MaskedImageF("/home/azuri/spectra/pfs/IR-23-0-centerSkyx2.fits")
        
        
#        ft = drpStella.FiberTrace(self.traceWidth, self.traceHeight)
#        fts = drpStella.FiberTraceSet(self.traceWidth, self.traceHeight)
#
#        with self.assertRaises(AttributeError) as e:
#            drpStella.FiberTrace(afwGeom.BoxI(afwGeom.PointI(0, 0), afwGeom.ExtentI(self.traceWidth, 10)))
#            
#        self.assertTrue("class is abstract" in str(e.exception))
#
#    def testImageFiberTrace(self):
#        """Test creating ImageFiberTrace"""
#        x0, traceHeight = 10, 100
#
#        traceExtent = afwGeom.ExtentI(self.traceWidth, traceHeight)
#        trace = drpStella.ImageFiberTrace(traceExtent))
#
#        trace.setXCenters(x0 + 0.5*self.traceWidth + np.zeros(traceHeight))
#        
#        bbox = trace.getBBox()
#        self.assertEqual(x0, bbox.getBeginX())
#        self.assertEqual(self.traceWidth, bbox.getWidth())
#        self.assertEqual(traceHeight, bbox.getHeight())
#
#        self.assertEqual(trace.getXCenters()[traceHeight//2], x0 + 0.5*self.traceWidth)
#
#        trace.setBBox(afwGeom.BoxI(afwGeom.PointI(0, 0), traceExtent))
#        bbox = trace.getBBox()
#        self.assertEqual(0, trace.getBBox().getBeginX())        
#
#    def testImageFiberTraceSet(self):
#        """Test that we can set and manipulate a set of ImageFiberTrace objects"""
#        fts = drpStella.FiberTraceSet(self.im)
#
#        if display:
#            ds9.mtv(self.im)
#
#        traceExtent = afwGeom.ExtentI(self.traceWidth, self.im.getHeight())
#        for i in range(self.im.getWidth()//self.traceWidth):
#            trace = drpStella.ImageFiberTrace(afwGeom.BoxI(afwGeom.PointI(i*self.traceWidth, 0), traceExtent))
#            fts.setFiberTrace(i, trace)
#
#            trace.setXCenters((i + 0.5)*self.traceWidth + np.zeros(self.im.getHeight()))
#
#        for i in range(fts.getNAperture()):
#            trace = fts.getFiberTrace(i)
#            bbox = trace.getBBox() 
#            
##            import pdb; pdb.set_trace()
#            prof = trace.getProfile(self.im.getHeight()//2)
#
#            self.assertEqual(len(prof), self.traceWidth)
#            self.assertAlmostEqual(sum(prof), 1.0, 4) # relies on correct normalisation of im.self
#
#            if display:
#                displayUtils.drawBBox(bbox, borderWidth=0.4)
#                for y in range(bbox.getHeight()):
#                    ds9.dot('+', trace.getXCenters()[y], y, size=0.45)
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(FiberTraceTestCase)
#    suites += unittest.makeSuite(tests.MemoryTestCase)
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
