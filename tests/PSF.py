#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python PSF.py
or
   python
   >>> import PSF; PSF.run()
"""

import os
import unittest
import sys
import numpy as np
import lsst.utils
import lsst.utils.tests as tests
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import pfs.drp.stella as drpStella
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

try:
    type(display)
except NameError:
    display = False

class PSFTestCase(tests.TestCase):
    """A test case for measuring PSF quantities"""

    def setUp(self):
        drpStellaDataDir = lsst.utils.getPackageDir("drp_stella_data")
        flatfile = os.path.join(drpStellaDataDir,"tests/data/PFS/CALIB/FLAT/r/flat/pfsFlat-2016-01-12-0-2r.fits")
        self.flat = afwImage.ExposureF(flatfile)

        arcfile = os.path.join(drpStellaDataDir,"tests/data/PFS/postISRCCD/2016-01-12/v0000004/PFFAr2.fits")
        self.arc = afwImage.ExposureF(arcfile)
        
        self.ftffc = drpStella.FiberTraceFunctionFindingControl()
        self.ftffc.fiberTraceFunctionControl.order = 4
        self.ftffc.fiberTraceFunctionControl.xLow = -5
        self.ftffc.fiberTraceFunctionControl.xHigh = 5
        
        self.ftpfc = drpStella.FiberTraceProfileFittingControl()
        
        self.tdpsfc = drpStella.TwoDPSFControl()
        
    def tearDown(self):
        del self.flat
        del self.arc
        del self.ftffc
        del self.ftpfc
        del self.tdpsfc

    def testPSFConstructors(self):
        if False:
            iTrace = 1
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            ft = fiberTraceSet.getFiberTrace(iTrace)
            ft.setFiberTraceProfileFittingControl(self.ftpfc.getPointer())
            self.assertTrue(ft.calcProfile())
            ft.createTrace(self.arc.getMaskedImage())
            spec = ft.extractFromProfile()
            psfSet = drpStella.calculate2dPSFPerBin(ft, spec, self.tdpsfc.getPointer())
            if True:

                """Test that we can create a PSF with the standard constructor"""
                psf = drpStella.PSFF()
                self.assertEqual(psf.getITrace(), 0)
                self.assertEqual(psf.getIBin(), 0)

                psf = drpStella.PSFF(1, 2)
                self.assertEqual(psf.getITrace(), 1)
                self.assertEqual(psf.getIBin(), 2)

                """Test copy constructors"""
                """shallow copy"""
                iPSF = 2
                psf = psfSet.getPSF(iPSF)
                psfCopy = drpStella.PSFF(psf)
                psf.getTwoDPSFControl().swathWidth = 250
                self.assertEqual(psf.getTwoDPSFControl().swathWidth, psfCopy.getTwoDPSFControl().swathWidth)
                self.assertEqual(psf.getITrace(), iTrace)
                self.assertEqual(psf.getIBin(), iPSF)

                """deep copy"""
                psfCopy = drpStella.PSFF(psf, True)
                psf.getTwoDPSFControl().swathWidth = 350
                self.assertNotEqual(psf.getTwoDPSFControl().swathWidth, psfCopy.getTwoDPSFControl().swathWidth)

                """Init Constructor"""
                psf = drpStella.PSFF(350, 750,self.tdpsfc.getPointer(),1,2)
                self.assertTrue(psf.getYLow(), 350)
                self.assertTrue(psf.getYHigh(), 750)
                self.assertTrue(psf.getITrace(), 1)
                self.assertTrue(psf.getIBin(), 2)
 
    def testCalculate2DPSFPerBin(self):
        if False:
            iTrace = 1
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            fiberTrace = fiberTraceSet.getFiberTrace(iTrace)
            self.assertTrue(fiberTrace.setFiberTraceProfileFittingControl(self.ftpfc.getPointer()))
            self.assertTrue(fiberTrace.calcProfile())
            ftComb = drpStella.FiberTraceF(fiberTrace)
            ftComb.createTrace(self.arc.getMaskedImage())
            spec = ftComb.extractFromProfile()
            psfSet = drpStella.calculate2dPSFPerBin(fiberTrace, spec, self.tdpsfc.getPointer())
            print "psfSet.size() = ",psfSet.size()
            self.assertGreater(psfSet.size(),0)
            for i in range(psfSet.size()):
                psf = psfSet.getPSF(i)
                self.assertGreater(psf.getYHigh(), psf.getYLow())
                self.assertEqual(psf.getITrace(), iTrace)
                self.assertEqual(psf.getIBin(), i)
        
    def testPFSGet(self):
        if False:
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            iTrace = 0
            fiberTrace = fiberTraceSet.getFiberTrace(iTrace)
            self.assertTrue(fiberTrace.setFiberTraceProfileFittingControl(self.ftpfc.getPointer()))
            self.assertTrue(fiberTrace.calcProfile())
            ftComb = drpStella.FiberTraceF(fiberTrace)
            ftComb.createTrace(self.arc.getMaskedImage())
            spec = ftComb.extractFromProfile()
            psfSet = drpStella.calculate2dPSFPerBin(fiberTrace, spec, self.tdpsfc.getPointer())

            """test getYLow and getYHigh"""
            swathWidth = self.tdpsfc.swathWidth;
            ndArr = fiberTrace.calcSwathBoundY(swathWidth);
            print "ndArr = ",ndArr[:]

            for i in range(psfSet.size()):
                self.assertEqual(psfSet.getPSF(i).getYLow(), ndArr[i,0])
                self.assertEqual(psfSet.getPSF(i).getYHigh(), ndArr[i,1])
                self.assertGreater(len(psfSet.getPSF(i).getImagePSF_XTrace()), 0)
            for i in range(psfSet.size()-2):
                self.assertEqual(psfSet.getPSF(i+2).getYLow(), psfSet.getPSF(i).getYHigh()+1)
            for i in range(2, psfSet.size()):
                self.assertEqual(psfSet.getPSF(i).getYLow(), psfSet.getPSF(i-2).getYHigh()+1)

            """test get..."""
            for i in range(psfSet.size()):
                size = len(psfSet.getPSF(i).getImagePSF_XTrace())
                self.assertGreater(size, 0)
                self.assertEqual(len(psfSet.getPSF(i).getImagePSF_YTrace()), size)
                self.assertEqual(len(psfSet.getPSF(i).getImagePSF_ZTrace()), size)
                self.assertEqual(len(psfSet.getPSF(i).getImagePSF_XRelativeToCenter()), size)
                self.assertEqual(len(psfSet.getPSF(i).getImagePSF_YRelativeToCenter()), size)
                self.assertEqual(len(psfSet.getPSF(i).getImagePSF_ZNormalized()), size)
                self.assertEqual(len(psfSet.getPSF(i).getImagePSF_Weight()), size)
                self.assertEqual(psfSet.getPSF(i).getITrace(), iTrace)
                self.assertEqual(psfSet.getPSF(i).getIBin(), i)

            """test isTwoDPSFControlSet"""
            psf = drpStella.PSFF()
            self.assertFalse(psf.isTwoDPSFControlSet())
            self.assertTrue(psf.setTwoDPSFControl(self.tdpsfc.getPointer()))
            self.assertTrue(psf.isTwoDPSFControlSet())

            """test isPSFsExtracted"""
            self.assertFalse(psf.isPSFsExtracted())
            self.assertTrue(psfSet.getPSF(2).isPSFsExtracted())

            """test extractPSFs"""
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            fiberTrace = fiberTraceSet.getFiberTrace(0)
            self.assertTrue(fiberTrace.setFiberTraceProfileFittingControl(self.ftpfc.getPointer()))
            self.assertTrue(fiberTrace.calcProfile())
            spec = fiberTrace.extractFromProfile()
            ftComb = drpStella.FiberTraceF(fiberTrace)
            ftComb.createTrace(self.arc.getMaskedImage())
            spec = ftComb.extractFromProfile()
            psf = drpStella.PSFF(350, 750,self.tdpsfc.getPointer(),1,2)
            self.assertTrue(psf.setTwoDPSFControl(self.tdpsfc.getPointer()))
            
            if False:
                psf.extractPSFs(ftComb, spec)
                #self.assertTrue(psf.extractPSFs(ftComb, spec))

                self.assertGreater(len(psf.getImagePSF_XTrace()), 0)
                self.assertTrue(psf.isPSFsExtracted())

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(PSFTestCase)
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
