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
        flatfile = "/Users/azuri/spectra/pfs/2014-10-28/sampledFlatx2-IR-0-23-5-10-nonoise.fits"
        self.flat = afwImage.ImageF(flatfile)
        self.flat = afwImage.makeExposure(afwImage.makeMaskedImage(self.flat))
        self.bias = afwImage.ImageF(flatfile, 3)
        self.flat.getMaskedImage()[:] -= self.bias

        combfile = "/Users/azuri/spectra/pfs/2014-12-14/sampledCombx2-IR-0-23.fits"
        self.comb = afwImage.ImageF(combfile)
        self.comb = afwImage.makeExposure(afwImage.makeMaskedImage(self.comb))
        bias = afwImage.ImageF(combfile, 3)
        self.comb.getMaskedImage()[:] -= bias

        del bias
        del flatfile
        del combfile
        
    def tearDown(self):
        del self.flat
        del self.comb
        del self.bias

    def testFiberTraceFunctionFindingControl(self):
        """Test that we can create a FiberTraceFunctionFindingControl"""
        ftffc = drpStella.FiberTraceFunctionFindingControl()
        
        """Test that we can set the parameters of the FiberTraceFunctionFindingControl"""
        interpolation = "POLYNOMIAL"
        ftffc.fiberTraceFunctionControl.interpolation = interpolation
        self.assertEqual(ftffc.fiberTraceFunctionControl.interpolation, interpolation)

        order = 4
        ftffc.fiberTraceFunctionControl.order = order
        self.assertEqual(ftffc.fiberTraceFunctionControl.order, order)
        
        xLow = -5.
        ftffc.fiberTraceFunctionControl.xLow = xLow
        self.assertAlmostEqual(ftffc.fiberTraceFunctionControl.xLow, xLow)

        xHigh = 5.
        ftffc.fiberTraceFunctionControl.xHigh = xHigh
        self.assertAlmostEqual(ftffc.fiberTraceFunctionControl.xHigh, xHigh)
        
        apertureFWHM = 2.6
        ftffc.apertureFWHM = apertureFWHM
        self.assertAlmostEqual(ftffc.apertureFWHM, apertureFWHM, places=6)
        
        signalThreshold = 10.
        ftffc.signalThreshold = signalThreshold
        self.assertAlmostEqual(ftffc.signalThreshold, signalThreshold)
        
        nTermsGaussFit = 4
        ftffc.nTermsGaussFit = nTermsGaussFit
        self.assertEqual(ftffc.nTermsGaussFit, nTermsGaussFit)
        
        saturationLevel = 65550.
        ftffc.saturationLevel = saturationLevel
        self.assertAlmostEqual(ftffc.saturationLevel, saturationLevel)
        
        minLength = 20
        ftffc.minLength = minLength
        self.assertEqual(ftffc.minLength, minLength)
        
        maxLength = 4000
        ftffc.maxLength = maxLength
        self.assertEqual(ftffc.maxLength, maxLength)
        
        nLost = 20
        ftffc.nLost = nLost
        self.assertEqual(ftffc.nLost, nLost)
        
#        import pdb; pdb.set_trace()
    def testFiberTraceConstructors(self):

        """Test that we can construct a FiberTrace with the standard constructor"""
        fiberTrace = drpStella.FiberTraceF()
        self.assertEqual(fiberTrace.getHeight(), 0)
        self.assertEqual(fiberTrace.getWidth(), 0)
        self.assertFalse(fiberTrace.isXCentersCalculated())
        self.assertFalse(fiberTrace.isTraceSet())
        self.assertFalse(fiberTrace.isProfileSet())
        self.assertFalse(fiberTrace.isFiberTraceFunctionSet())
        self.assertFalse(fiberTrace.isFiberTraceProfileFittingControlSet())
        self.assertEqual(fiberTrace.getITrace(), 0)

        """Test that we can create a FiberTrace given width and height"""
        width = 5
        height = 100
        iTrace = 1
        fiberTrace = drpStella.FiberTraceF(width, height, iTrace)
        self.assertEqual(fiberTrace.getWidth(), width)
        self.assertEqual(fiberTrace.getImage().getWidth(), width)
        self.assertEqual(fiberTrace.getHeight(), height)
        self.assertEqual(fiberTrace.getImage().getHeight(), height)
        self.assertEqual(fiberTrace.getITrace(), iTrace)
        
        """Test that we can create a FiberTrace with a given dimension"""
        dimension = afwGeom.Extent2I(10,20)
        fiberTrace = drpStella.FiberTraceF(dimension, iTrace)
        self.assertEqual(fiberTrace.getWidth(), dimension[0])
        self.assertEqual(fiberTrace.getHeight(), dimension[1])
        
        """Test that we can create a FiberTrace given a MaskedImage"""
        """Flat"""
        fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), drpStella.FiberTraceFunctionFindingControl())
        fiberTrace = drpStella.FiberTraceF(self.flat.getMaskedImage(), fiberTraceSet.getFiberTrace(0).getFiberTraceFunction(), iTrace)
        self.assertEqual(self.flat.getHeight(), fiberTrace.getCCDHeight())
        self.assertEqual(self.flat.getWidth(), fiberTrace.getCCDWidth())
        
        """Test copy constructor"""
        fiberTraceCopy = drpStella.FiberTraceF(fiberTrace)
        self.assertEqual(fiberTraceCopy.getWidth(), fiberTrace.getWidth())
        self.assertEqual(fiberTraceCopy.getHeight(), fiberTrace.getHeight())
        self.assertEqual(fiberTraceCopy.getITrace(), fiberTrace.getITrace())
        self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], fiberTraceCopy.getTrace().getImage().getArray()[5,5])

    def testFiberTraceSetConstructors(self):
        """Test that we can create a FiberTraceSet from the standard constructor"""
        fiberTraceSet = drpStella.FiberTraceSetF()
        self.assertEqual(fiberTraceSet.size(), 0)
        
        nTraces = 3
        fiberTraceSet = drpStella.FiberTraceSetF(3)
        self.assertEqual(fiberTraceSet.size(), nTraces)
        for i in range(nTraces):
            self.assertEqual(fiberTraceSet.getFiberTrace(i).getITrace(), i)
            
        """Test that we can create a FiberTraceSet from another FiberTraceSet"""
        fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), drpStella.FiberTraceFunctionFindingControl())
        fiberTraceSetNew = drpStella.FiberTraceSetF(fiberTraceSet)
        self.assertEqual(fiberTraceSetNew.size(), fiberTraceSet.size())
        self.assertEqual(fiberTraceSet.getFiberTrace(1).getTrace().getImage().getArray()[5, 5], fiberTraceSetNew.getFiberTrace(1).getTrace().getImage().getArray()[5,5])
        for i in range(nTraces):
            self.assertEqual(fiberTraceSetNew.getFiberTrace(i).getITrace(), i)
        
#        """Test that we can create a FiberTraceSet from a vector of FiberTraces"""
#        fiberTraceSetNew = drpStella.FiberTraceSetF(fiberTraceSet.getTraces())
#        self.assertEqual(fiberTraceSetNew.size(), nTraces)
#        for i in range(nTraces):
#            self.assertEqual(fiberTraceSetNew.getFiberTrace(i).getITrace(), i)

    def testFiberTraceGetSetFunctions(self):
        """Test get methods"""

        iTrace = 1
        ftffc = drpStella.FiberTraceFunctionFindingControl()
        xHigh = 4.1
        ftffc.fiberTraceFunctionControl.xHigh = xHigh
        ftffc.signalThreshold = 110.
        fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), ftffc)
        
        """Test getFiberTrace"""
        fiberTrace = fiberTraceSet.getFiberTrace(iTrace)
        
        """Test getHeight()"""
        self.assertEqual(fiberTrace.getHeight(), fiberTrace.getFiberTraceFunction().yHigh - fiberTrace.getFiberTraceFunction().yLow + 1)
        
        """Test getWidth()"""
        self.assertGreaterEqual(fiberTrace.getWidth(), fiberTrace.getFiberTraceFunction().fiberTraceFunctionControl.xHigh - fiberTrace.getFiberTraceFunction().fiberTraceFunctionControl.xLow)
        self.assertLessEqual(fiberTrace.getWidth(), fiberTrace.getFiberTraceFunction().fiberTraceFunctionControl.xHigh - fiberTrace.getFiberTraceFunction().fiberTraceFunctionControl.xLow + 2)

        """Test getITrace()"""
        self.assertEqual(fiberTrace.getITrace(), iTrace)
        
        """Test getFiberTraceFunction"""
        self.assertAlmostEqual(fiberTrace.getFiberTraceFunction().fiberTraceFunctionControl.xHigh, xHigh, places=6)

        """Test getTrace"""
        self.assertEqual(fiberTrace.getTrace().getHeight(), fiberTrace.getHeight())
        flatPlusBias = self.flat.getMaskedImage()
        flatPlusBias += self.bias
        fiberTraceSetMIF = drpStella.findAndTraceAperturesF(flatPlusBias, ftffc)
        fiberTrace = fiberTraceSetMIF.getFiberTrace(iTrace)
        fiberTraceMIF = drpStella.FiberTraceF(flatPlusBias, fiberTrace.getFiberTraceFunction(), iTrace)
        arrayVal = fiberTraceMIF.getTrace().getImage().getArray()[5,5]
        print arrayVal

        print fiberTraceSetMIF.size()
        arrayMIFVal = fiberTraceSetMIF.getFiberTrace(iTrace).getTrace().getImage().getArray()[5,5]
        print arrayMIFVal
        self.assertEqual(arrayVal, arrayMIFVal)
        
        maskedImageWrongSize = afwImage.MaskedImageF(100,10)
        
        """Test setTrace"""
        val = 1000.
        fiberTrace.getTrace().getImage().getArray()[5,5] = val
        self.assertTrue(fiberTraceMIF.setTrace(fiberTrace.getTrace()))
        self.assertFalse(fiberTraceMIF.setTrace(maskedImageWrongSize))
        self.assertAlmostEqual(fiberTraceMIF.getTrace().getImage().getArray()[5,5], val)
        
        """Test setImage"""
        val = 1011.
        fiberTrace.getImage().getArray()[5,5] = val
        self.assertTrue(fiberTraceMIF.setImage(fiberTrace.getImage()))
        self.assertFalse(fiberTraceMIF.setImage(maskedImageWrongSize.getImage()))
        self.assertAlmostEqual(fiberTraceMIF.getImage().getArray()[5,5], val)
        
        """Test setVariance"""
        val = 1000.
        fiberTrace.getVariance().getArray()[5,5] = val
        self.assertTrue(fiberTraceMIF.setVariance(fiberTrace.getVariance()))
        self.assertFalse(fiberTraceMIF.setVariance(maskedImageWrongSize.getVariance()))
        self.assertAlmostEqual(fiberTraceMIF.getVariance().getArray()[5,5], val)
        
        """Test setMask"""
        print "fiberTrace.getMask().getArray()[5,5] = ", fiberTrace.getMask().getArray()[5,5]
        val = 1
        fiberTrace.getMask().getArray()[5,5] = val
        self.assertTrue(fiberTraceMIF.setMask(fiberTrace.getMask()))
        self.assertFalse(fiberTraceMIF.setMask(maskedImageWrongSize.getMask()))
        self.assertEqual(fiberTraceMIF.getMask().getArray()[5,5], val)
        
        """Test getProfile/setProfile"""
        val = 1111.
        fiberTrace.getProfile().getArray()[5,5] = val
        self.assertTrue(fiberTraceMIF.setProfile(fiberTrace.getProfile()))
        self.assertFalse(fiberTraceMIF.setProfile(maskedImageWrongSize.getImage()))
        self.assertEqual(fiberTraceMIF.getProfile().getArray()[5,5], val)
        
    def testFiberTraceExtractionMethods(self):
        ftffc = drpStella.FiberTraceFunctionFindingControl()
        ftffc.signalThreshold = 10.
        fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), ftffc)
        fiberTrace = fiberTraceSet.getFiberTrace(0)
        try:
            spectrum = fiberTrace.MkSlitFunc()
        except:
            e = sys.exc_info()[0]
            print e

    def testFiberTraceSetFunctions(self):
        if False:
            """Test that we can add a FiberTrace to the FiberTraceSet"""
            fts.addFiberTrace(f)
            print("fti added to fts")

            """Test that we can trace fibers"""
            msi = drpStella.MaskedSpectrographImageF(mif)
            print("msi created")

            msi.findAndTraceApertures(ftffc, 0, mif.getHeight(), 10)

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
