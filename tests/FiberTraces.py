#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python FiberTrace.py
or
   python
   >>> import FiberTrace; FiberTrace.run()
"""
import lsst.afw.image as afwImage
import lsst.afw.display.ds9 as ds9
import lsst.daf.persistence as dafPersist
import lsst.utils
import lsst.utils.tests as tests
import numpy as np
import os
import pfs.drp.stella as drpStella
import sys
import unittest

try:
    type(display)
except NameError:
    display = False

class FiberTraceTestCase(tests.TestCase):
    """A test case for measuring FiberTrace quantities"""

    def setUp(self):
        drpStellaDataDir = lsst.utils.getPackageDir("drp_stella_data")
        butler = dafPersist.Butler(os.path.join(drpStellaDataDir,"tests/data/PFS/"))
        dataId = dict(field="FLAT", visit=104, spectrograph=1, arm="r")
        self.flat = butler.get("postISRCCD", dataId, immediate=True)

        dataId = dict(field="ARC", visit=103, spectrograph=1, arm="r")
        self.arc = butler.get("postISRCCD", dataId, immediate=True)

        self.ftffc = drpStella.FiberTraceFunctionFindingControl()
        self.ftffc.fiberTraceFunctionControl.order = 5
        self.ftffc.fiberTraceFunctionControl.xLow = -5
        self.ftffc.fiberTraceFunctionControl.xHigh = 5

        # This particular flatfile has 12 FiberTraces scattered over the whole CCD
        # If in the future the test data change we need to change these numbers
        self.nFiberTraces = 11
        self.minLength = 3880
        self.maxLength = 3930

    def tearDown(self):
        del self.flat
        del self.arc
        del self.ftffc

    def testFiberTraceFunctionFindingControl(self):
        """Test that we can create a FiberTraceFunctionFindingControl"""
        self.ftffc.fiberTraceFunctionControl.order = 4

        """Test that we can set the parameters of the FiberTraceFunctionFindingControl"""
        interpolation = "POLYNOMIAL"
        self.ftffc.fiberTraceFunctionControl.interpolation = interpolation
        self.assertEqual(self.ftffc.fiberTraceFunctionControl.interpolation, interpolation)
        self.assertIsNot(interpolation, self.ftffc.fiberTraceFunctionControl.interpolation)
        oldInterpolation = self.ftffc.fiberTraceFunctionControl.interpolation
        interpolation = "POLYNOMAL"
        self.assertNotEqual(interpolation, oldInterpolation)
        self.ftffc.fiberTraceFunctionControl.interpolation = interpolation
        self.assertEqual(self.ftffc.fiberTraceFunctionControl.interpolation, interpolation)
        if interpolation is self.ftffc.fiberTraceFunctionControl.interpolation:
            print "interpolation IS self.ftffc.fiberTraceFunctionControl.interpolation"
        else:
            print "interpolation IS NOT self.ftffc.fiberTraceFunctionControl.interpolation"
        print " "

        order = 4
        self.ftffc.fiberTraceFunctionControl.order = order
        self.assertEqual(self.ftffc.fiberTraceFunctionControl.order, order)
        self.assertIs(order, self.ftffc.fiberTraceFunctionControl.order)
        if order is self.ftffc.fiberTraceFunctionControl.order:
            print "order IS self.ftffc.fiberTraceFunctionControl.order"
        else:
            print "order IS NOT self.ftffc.fiberTraceFunctionControl.order"
        print " "

        xLow = -5.
        self.ftffc.fiberTraceFunctionControl.xLow = xLow
        self.assertAlmostEqual(self.ftffc.fiberTraceFunctionControl.xLow, xLow)
        self.assertIsNot(xLow, self.ftffc.fiberTraceFunctionControl.xLow)
        if xLow is self.ftffc.fiberTraceFunctionControl.xLow:
            print "xLow IS self.ftffc.fiberTraceFunctionControl.xLow"
        else:
            print "xLow IS NOT self.ftffc.fiberTraceFunctionControl.xLow"
        print " "

        xHigh = 5.
        self.ftffc.fiberTraceFunctionControl.xHigh = xHigh
        self.assertAlmostEqual(self.ftffc.fiberTraceFunctionControl.xHigh, xHigh)
        self.assertIsNot(xHigh, self.ftffc.fiberTraceFunctionControl.xHigh)
        if xHigh is self.ftffc.fiberTraceFunctionControl.xHigh:
            print "xHigh IS self.ftffc.fiberTraceFunctionControl.xHigh"
        else:
            print "xHigh IS NOT self.ftffc.fiberTraceFunctionControl.xHigh"
        print " "

        apertureFWHM = 2.6
        self.ftffc.apertureFWHM = apertureFWHM
        self.assertAlmostEqual(self.ftffc.apertureFWHM, apertureFWHM, places=6)
        self.assertIsNot(apertureFWHM, self.ftffc.apertureFWHM)
        if apertureFWHM is self.ftffc.apertureFWHM:
            print "apertureFWHM IS self.ftffc.apertureFWHM"
        else:
            print "apertureFWHM IS NOT self.ftffc.apertureFWHM"
        print " "

        signalThreshold = 10.
        self.ftffc.signalThreshold = signalThreshold
        self.assertAlmostEqual(self.ftffc.signalThreshold, signalThreshold)
        self.assertIsNot(signalThreshold, self.ftffc.signalThreshold)
        if signalThreshold is self.ftffc.signalThreshold:
            print "signalThreshold IS self.ftffc.signalThreshold"
        else:
            print "signalThreshold IS NOT self.ftffc.signalThreshold"
        print " "

        nTermsGaussFit = 4
        self.ftffc.nTermsGaussFit = nTermsGaussFit
        self.assertEqual(self.ftffc.nTermsGaussFit, nTermsGaussFit)
        self.assertIs(nTermsGaussFit, self.ftffc.nTermsGaussFit)
        if nTermsGaussFit is self.ftffc.nTermsGaussFit:
            print "nTermsGaussFit IS self.ftffc.nTermsGaussFit"
        else:
            print "nTermsGaussFit IS NOT self.ftffc.nTermsGaussFit"
        print " "

        saturationLevel = 65550.
        self.ftffc.saturationLevel = saturationLevel
        self.assertAlmostEqual(self.ftffc.saturationLevel, saturationLevel)
        self.assertIsNot(saturationLevel, self.ftffc.saturationLevel)
        if saturationLevel is self.ftffc.saturationLevel:
            print "saturationLevel IS self.ftffc.saturationLevel"
        else:
            print "saturationLevel IS NOT self.ftffc.saturationLevel"
        print " "

        minLength = 20
        self.ftffc.minLength = minLength
        self.assertEqual(self.ftffc.minLength, minLength)
        self.assertIs(minLength, self.ftffc.minLength)
        if minLength is self.ftffc.minLength:
            print "minLength IS self.ftffc.minLength"
        else:
            print "minLength IS NOT self.ftffc.minLength"
        print " "

        maxLength = 4000
        self.ftffc.maxLength = maxLength
        self.assertEqual(self.ftffc.maxLength, maxLength)
        self.assertIsNot(maxLength, self.ftffc.maxLength)
        if maxLength is self.ftffc.maxLength:
            print "maxLength IS self.ftffc.maxLength"
        else:
            print "maxLength IS NOT self.ftffc.maxLength"
        print " "

        nLost = 20
        self.ftffc.nLost = nLost
        self.assertEqual(self.ftffc.nLost, nLost)
        self.assertIs(nLost, self.ftffc.nLost)
        if nLost is self.ftffc.nLost:
            print "nLost IS self.ftffc.nLost"
        else:
            print "nLost IS NOT self.ftffc.nLost"

    def testFiberTraceConstructors(self):
        if True:
            """Test that we can construct a FiberTrace with the standard constructor"""
            fiberTrace = drpStella.FiberTraceF()
            self.assertEqual(fiberTrace.getHeight(), 0)
            self.assertEqual(fiberTrace.getWidth(), 0)
            self.assertFalse(fiberTrace.isTraceSet())
            self.assertFalse(fiberTrace.isProfileSet())
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

            """Test that we can create a FiberTrace given a MaskedImage and a FiberTraceFunction"""
            """Flat"""
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            """Check that we found self.nFiberTraces FiberTraces"""
            self.assertEqual(fiberTraceSet.size(), self.nFiberTraces)
            """Check length of FiberTraces"""
            for i in range(fiberTraceSet.size()):
                self.assertLess(fiberTraceSet.getFiberTrace(i).getHeight(), self.maxLength)
                self.assertGreater(fiberTraceSet.getFiberTrace(i).getHeight(), self.minLength)
            fiberTrace = drpStella.FiberTraceF(self.flat.getMaskedImage(), fiberTraceSet.getFiberTrace(0).getFiberTraceFunction(), iTrace)
            self.assertEqual(fiberTraceSet.getFiberTrace(0).getXCenters()[5], fiberTrace.getXCenters()[5])

            """Test copy constructor - shallow copy"""
            fiberTraceCopy = drpStella.FiberTraceF(fiberTrace)
            self.assertEqual(fiberTraceCopy.getWidth(), fiberTrace.getWidth())
            self.assertEqual(fiberTraceCopy.getHeight(), fiberTrace.getHeight())
            self.assertEqual(fiberTraceCopy.getITrace(), fiberTrace.getITrace())
            self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], fiberTraceCopy.getTrace().getImage().getArray()[5,5])
            self.assertAlmostEqual(fiberTrace.getXCenters()[5], fiberTraceCopy.getXCenters()[5])
            val = 11.1
            fiberTrace.getTrace().getImage().getArray()[5,5] = val
            self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], val, places=6)
            self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], fiberTraceCopy.getTrace().getImage().getArray()[5,5])

            """Test copy constructor - deep copy"""
            fiberTraceCopy = drpStella.FiberTraceF(fiberTrace, True)
            self.assertEqual(fiberTraceCopy.getWidth(), fiberTrace.getWidth())
            self.assertEqual(fiberTraceCopy.getHeight(), fiberTrace.getHeight())
            self.assertEqual(fiberTraceCopy.getITrace(), fiberTrace.getITrace())
            self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], fiberTraceCopy.getTrace().getImage().getArray()[5,5])
            self.assertAlmostEqual(fiberTrace.getXCenters()[5], fiberTraceCopy.getXCenters()[5])
            val = 13.3
            fiberTrace.getTrace().getImage().getArray()[5,5] = val
            self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], val, places=6)
            self.assertNotAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], fiberTraceCopy.getTrace().getImage().getArray()[5,5], places=6)

    def testFiberTraceSetConstructors(self):
        if True:
            """Test that we can create a FiberTraceSet from the standard constructor"""
            fiberTraceSet = drpStella.FiberTraceSetF()
            self.assertEqual(fiberTraceSet.size(), 0)

            nTraces = 3
            fiberTraceSet = drpStella.FiberTraceSetF(3)
            self.assertEqual(fiberTraceSet.size(), nTraces)
            for i in range(nTraces):
                self.assertEqual(fiberTraceSet.getFiberTrace(i).getITrace(), i)

            """Test that we can create a FiberTraceSet from another FiberTraceSet"""
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
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
        if True:
            """Test get/set methods"""

            iTrace = 1
            strITrace = str(iTrace)
            xHigh = 4.1
            self.ftffc.fiberTraceFunctionControl.xHigh = xHigh
            self.ftffc.signalThreshold = 110.
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)

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
            flatPlusArc = self.flat.getMaskedImage()
            flatPlusArc += self.arc.getMaskedImage()
            fiberTraceSetMIF = drpStella.findAndTraceAperturesF(flatPlusArc, self.ftffc)
            fiberTrace = fiberTraceSetMIF.getFiberTrace(iTrace)
            fiberTraceMIF = drpStella.FiberTraceF(flatPlusArc, fiberTrace.getFiberTraceFunction(), iTrace)
            arrayVal = fiberTraceMIF.getTrace().getImage().getArray()[5,5]

            arrayMIFVal = fiberTraceSetMIF.getFiberTrace(iTrace).getTrace().getImage().getArray()[5,5]
            self.assertEqual(arrayVal, arrayMIFVal)

            height = 100
            width = 10
            maskedImageWrongSize = afwImage.MaskedImageF(width,height)
            maskedImageWrongSizeD = afwImage.MaskedImageD(width,height)

            """Test setTrace"""
            val = 1000.
            fiberTrace.getTrace().getImage().getArray()[5,5] = val
            self.assertTrue(fiberTraceMIF.setTrace(fiberTrace.getTrace()))
            try:
                fiberTraceMIF.setTrace(maskedImageWrongSize)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "FiberTrace"+strITrace+"::setTrace: ERROR: trace->getHeight(="+str(height)+") != _trace->getHeight(="+str(fiberTraceMIF.getHeight())+")"
                self.assertEqual(message[0],expected)
            self.assertAlmostEqual(fiberTraceMIF.getTrace().getImage().getArray()[5,5], val)
            fiberTraceMIF.getTrace().getImage().getArray()[5,5] = val+2
            self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], val+2)
            fiberTrace.getTrace().getImage().getArray()[5,5] = val
            self.assertAlmostEqual(fiberTraceMIF.getTrace().getImage().getArray()[5,5], val)

            """Test setImage"""
            val = 1011.
            fiberTrace.getImage().getArray()[5,5] = val
            self.assertTrue(fiberTraceMIF.setImage(fiberTrace.getImage()))
            try:
                fiberTraceMIF.setImage(maskedImageWrongSize.getImage())
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "FiberTrace.setImage: ERROR: image.getHeight(="+str(height)+") != _trace->getHeight(="+str(fiberTraceMIF.getHeight())+")"
                self.assertEqual(message[0],expected)
            self.assertAlmostEqual(fiberTraceMIF.getImage().getArray()[5,5], val)
            fiberTrace.getImage().getArray()[5,5] = val+2
            self.assertAlmostEqual(fiberTraceMIF.getImage().getArray()[5,5], val+2)

            """Test setVariance"""
            val = 1000.
            fiberTrace.getVariance().getArray()[5,5] = val
            self.assertTrue(fiberTraceMIF.setVariance(fiberTrace.getVariance()))
            try:
                fiberTraceMIF.setVariance(maskedImageWrongSize.getVariance())
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "FiberTrace.setVariance: ERROR: variance.getHeight(="+str(maskedImageWrongSize.getVariance().getHeight())+") != _trace->getHeight(="+str(fiberTraceMIF.getHeight())+")"
                self.assertEqual(message[0],expected)
            self.assertAlmostEqual(fiberTraceMIF.getVariance().getArray()[5,5], val)
            fiberTrace.getVariance().getArray()[5,5] = val+2
            self.assertAlmostEqual(fiberTraceMIF.getVariance().getArray()[5,5], val+2)

            """Test setMask"""
            val = 1
            fiberTrace.getMask().getArray()[5,5] = val
            self.assertTrue(fiberTraceMIF.setMask(fiberTrace.getMask()))
            try:
                fiberTraceMIF.setMask(maskedImageWrongSize.getMask())
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "FiberTrace.setMask: ERROR: mask.getHeight(="+str(maskedImageWrongSize.getMask().getHeight())+") != _trace->getHeight()(="+str(fiberTraceMIF.getMask().getHeight())+")"
                self.assertEqual(message[0],expected)

            self.assertEqual(fiberTraceMIF.getMask().getArray()[5,5], val)
            fiberTrace.getMask().getArray()[5,5] = val+2
            self.assertEqual(fiberTraceMIF.getMask().getArray()[5,5], val+2)

            """Test getProfile/setProfile"""
            val = 1111.
            fiberTrace.getProfile().getArray()[5,5] = val
            self.assertTrue(fiberTraceMIF.setProfile(fiberTrace.getProfile()))
            try:
                fiberTraceMIF.setProfile(maskedImageWrongSizeD.getImage())
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "FiberTrace.setProfile: ERROR: profile->getHeight(="+str(maskedImageWrongSizeD.getImage().getHeight())+") != _trace->getHeight(="+str(fiberTraceMIF.getProfile().getHeight())+")"
                self.assertEqual(message[0],expected)
            self.assertAlmostEqual(fiberTraceMIF.getProfile().getArray()[5,5], val)
            fiberTrace.getProfile().getArray()[5,5] = val+2
            self.assertAlmostEqual(fiberTraceMIF.getProfile().getArray()[5,5], fiberTrace.getProfile().getArray()[5,5])

            """Test getXCenters"""
            xCenters = fiberTrace.getXCenters()
            self.assertEqual(xCenters.shape[0], fiberTrace.getHeight())
            self.assertAlmostEqual(xCenters[5], fiberTrace.getXCenters()[5])

    def testFiberTraceCreateTrace(self):
        if True:
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            fiberTrace = fiberTraceSet.getFiberTrace(fiberTraceSet.size()-1)
            oldTrace = fiberTrace.getTrace().getImage().getArray().copy()
            self.assertTrue(fiberTrace.createTrace(self.flat.getMaskedImage()))
            trace = fiberTrace.getTrace().getImage().getArray()
            self.assertFalse(id(oldTrace) == id(trace))
            for i in range(fiberTrace.getHeight()):
                for j in range(fiberTrace.getWidth()):
                    self.assertAlmostEqual(oldTrace[i,j], trace[i,j])

    def testFiberTraceExtractionMethods(self):
        if True:
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            """Create flat profile from trace and extractSum"""
            for iTrace in range(0, fiberTraceSet.size()):
                fiberTrace = fiberTraceSet.getFiberTrace(iTrace)
                spectrum = fiberTrace.extractSum()
                profile = fiberTrace.getProfile()
                for row in range(0, fiberTrace.getHeight()):
                    profile.getArray()[row,:] = fiberTrace.getTrace().getImage().getArray()[row,:]
                    profile.getArray()[row,:] /= spectrum.getSpectrum()[row]
                self.assertTrue(fiberTrace.setProfile(profile))
                recImage = fiberTrace.getReconstructed2DSpectrum(spectrum)
                diff = fiberTrace.getTrace().getImage().getArray() - recImage.getArray()
                meanDiff = np.mean(diff)
                self.assertLess(np.absolute(meanDiff), 0.001)
                stdDevDiff = np.std(diff)
                self.assertLess(np.absolute(stdDevDiff), 0.001)

                spectrum = fiberTrace.extractFromProfile()
                recImage = fiberTrace.getReconstructed2DSpectrum(spectrum)
                diff = fiberTrace.getTrace().getImage().getArray() - recImage.getArray()
                meanDiff = np.mean(diff)
                self.assertLess(np.absolute(meanDiff), 0.001)
                stdDevDiff = np.std(diff)
                self.assertLess(np.absolute(stdDevDiff), 0.001)

            """Fit profile with MkSlitFunc"""
            ftpfc = drpStella.FiberTraceProfileFittingControl()
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            for iTrace in range(0, fiberTraceSet.size()):
                fiberTrace = fiberTraceSet.getFiberTrace(iTrace)
                try:
                    spectrum = fiberTrace.calcProfile()
                except:
                    e = sys.exc_info()[1]
                    message = str.split(e.message, "\n")
                    expected = "FiberTrace "+str(iTrace)+"::calcProfile: ERROR: _fiberTraceProfileFittingControl is not set"
                    self.assertEqual(message[0],expected)
                self.assertTrue(fiberTrace.setFiberTraceProfileFittingControl(ftpfc))
                self.assertTrue(fiberTrace.calcProfile())
                try:
                    spectrum = fiberTrace.extractFromProfile()
                except:
                    raise
                self.assertEqual(spectrum.getLength(), fiberTrace.getHeight())
                fiberTrace.getMask().getArray()[:,:] = 0
                oldestMask = fiberTrace.getMask().getArray().copy()
                spectrumFromProfile = fiberTrace.extractFromProfile()
                self.assertEqual(spectrum.getLength(), spectrumFromProfile.getLength())

                """Test createTrace"""
                oldHeight = fiberTrace.getHeight()
                oldWidth = fiberTrace.getWidth()
                oldProfile = fiberTrace.getProfile().getArray().copy()
                oldTrace = fiberTrace.getTrace().getImage().getArray().copy()
                oldMask = fiberTrace.getMask().getArray()

                self.assertTrue(fiberTrace.createTrace(self.flat.getMaskedImage()))
                trace = fiberTrace.getTrace().getImage().getArray()
                self.assertFalse(id(oldTrace) == id(trace))
                profile = fiberTrace.getProfile().getArray()
                self.assertFalse(id(oldProfile) == id(profile))
                mask = fiberTrace.getMask().getArray()

                self.assertEqual(oldHeight, fiberTrace.getHeight())
                self.assertEqual(oldWidth, fiberTrace.getWidth())
                self.assertEqual(oldProfile[5,5], fiberTrace.getProfile().getArray()[5,5])
                self.assertEqual(oldTrace[5,5], fiberTrace.getTrace().getImage().getArray()[5,5])

                fiberTrace.getMask().getArray()[:,:] = 0
                spectrum = fiberTrace.extractFromProfile()
                self.assertEqual(spectrum.getLength(), spectrumFromProfile.getLength())

                for i in range(spectrum.getLength()):#10, spectrum.getLength()-10):
                    for j in range(len(oldTrace[0,:])):
                        self.assertAlmostEqual(oldTrace[i,j], trace[i,j])
                        self.assertAlmostEqual(oldProfile[i,j], profile[i,j])
                        self.assertAlmostEqual(oldestMask[i,j], mask[i,j])
                        self.assertAlmostEqual(oldMask[i,j], mask[i,j])
                        self.assertGreaterEqual(profile[i,j], 0.)

                for i in range(spectrum.getLength()):
                    self.assertEqual(spectrum.getSpectrum()[i], spectrumFromProfile.getSpectrum()[i])

            self.assertTrue(fiberTrace.createTrace(self.arc.getMaskedImage()))
            self.assertEqual(oldHeight, fiberTrace.getHeight())
            self.assertEqual(oldWidth, fiberTrace.getWidth())
            self.assertEqual(oldProfile[5,5], fiberTrace.getProfile().getArray()[5,5])
            self.assertNotAlmostEqual(oldTrace[5,5], fiberTrace.getTrace().getImage().getArray()[5,5])
            spectrum = fiberTrace.extractFromProfile()
            self.assertEqual(spectrum.getLength(), spectrumFromProfile.getLength())
            self.assertNotAlmostEqual(spectrum.getSpectrum()[5], spectrumFromProfile.getSpectrum()[5])

    def testFiberTraceReconstruct(self):
        if True:
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            ftpfc = drpStella.FiberTraceProfileFittingControl()
            ftpfc.profileInterpolation = "PISKUNOV"
            for iTrace in range(0, fiberTraceSet.size()):
                fiberTrace = fiberTraceSet.getFiberTrace(iTrace)
                self.assertTrue(fiberTrace.setFiberTraceProfileFittingControl(ftpfc))
                bool = fiberTrace.calcProfile()
                self.assertTrue(bool)

                spectrum = fiberTrace.extractFromProfile()

                recImage = fiberTrace.getReconstructed2DSpectrum(spectrum)
                diff = fiberTrace.getTrace().getImage().getArray() - recImage.getArray()
                if display:
                    ds9.mtv(diff,title="reconstruction from MkSlitFunc",frame=fiberTraceSet.size()+iTrace)
                meanDiff = np.mean(diff)
                self.assertLess(np.absolute(meanDiff), 50.)
                stdDevDiff = np.std(diff)
                self.assertLess(np.absolute(stdDevDiff), 600.)

    def testFiberTraceOtherFunctions(self):
        if True:
            """Test FiberTrace.calcSwathBoundY(swathwidth)"""
            ftpffc = drpStella.FiberTraceProfileFittingControl()
            swathWidth = int(ftpffc.swathWidth)
            self.assertIsNot(swathWidth, ftpffc.swathWidth)
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            fiberTrace = drpStella.FiberTraceF(self.flat.getMaskedImage(), fiberTraceSet.getFiberTrace(0).getFiberTraceFunction(), 0)
            binBoundYOut = fiberTrace.calcSwathBoundY(swathWidth)
            for i in range(binBoundYOut.shape[0]):
                if i == 0:
                    self.assertEqual(binBoundYOut[i,0], 0)
                elif i == binBoundYOut.shape[0]-1:
                    self.assertEqual(binBoundYOut[i,1], fiberTrace.getHeight()-1)
                self.assertLess(binBoundYOut[i,0], binBoundYOut[i,1])
            for i in range(binBoundYOut.shape[0]-1):
                self.assertLess(binBoundYOut[i+1, 0], binBoundYOut[i,1])
            for i in range(binBoundYOut.shape[0]-2):
                self.assertEqual(binBoundYOut[i, 1]+1, binBoundYOut[i+2,0])

            width = 10
            height = 3901
            fiberTrace = drpStella.FiberTraceF(width, height)
            binBoundYOut = fiberTrace.calcSwathBoundY(75)
            self.assertEqual(binBoundYOut.shape[0], 102);
            self.assertEqual(binBoundYOut[binBoundYOut.shape[0]-1, 1], height-1)

    def testFiberTraceSetConstructor(self):
        if True:
            size = 0
            fts = drpStella.FiberTraceSetF(size)
            self.assertEqual(fts.size(), size)
            size = 2
            fts = drpStella.FiberTraceSetF(size)
            self.assertEqual(fts.size(), size)

            ftsa = drpStella.FiberTraceSetF(fts)
            self.assertEqual(fts.size(), ftsa.size())

    def testFiberTraceSetFunctions(self):
        if True:
            size = 0
            ftsEmpty = drpStella.FiberTraceSetF()

            """Test that we can trace fibers"""
            fts = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            self.assertGreater(fts.size(), 0)

            ft = drpStella.FiberTraceF()
            self.assertEqual(ft.getITrace(), 0)

            """Test that we can add a FiberTrace to an empty FiberTraceSet"""
            self.assertTrue(ftsEmpty.addFiberTrace(ft))
            self.assertEqual(ftsEmpty.size(), size+1)
            self.assertEqual(ftsEmpty.getFiberTrace(ftsEmpty.size()-1).getITrace(), 0)
            self.assertNotEqual(drpStella.getRawPointerFTF(ftsEmpty.getFiberTrace(0)), drpStella.getRawPointerFTF(ft.getPointer()))

            size = fts.size()
            self.assertTrue(fts.addFiberTrace(ft))
            self.assertEqual(fts.size(), size+1)
            self.assertEqual(fts.getFiberTrace(fts.size()-1).getITrace(), 0)
            self.assertNotEqual(drpStella.getRawPointerFTF(fts.getFiberTrace(size)), drpStella.getRawPointerFTF(ft))

            """Test that we can set a FiberTrace at a certain position"""
            ft = fts.getFiberTrace(2)
            comp = ft.getTrace().getImage().getArray()[5,5]
            pos = 0
            self.assertTrue(fts.setFiberTrace(pos, ft))
            self.assertEqual(fts.getFiberTrace(pos).getTrace().getImage().getArray()[5,5], comp)

            pos = fts.size()-1
            self.assertTrue(fts.setFiberTrace(pos, ft))
            self.assertEqual(fts.getFiberTrace(pos).getTrace().getImage().getArray()[5,5], comp)

            """Test that setting a FiberTrace just past the last position adds a FiberTrace"""
            size = fts.size()
            self.assertTrue(fts.setFiberTrace(fts.size(), ft))
            self.assertEqual(fts.size(), size+1)
            self.assertEqual(fts.getFiberTrace(fts.size()-1).getTrace().getImage().getArray()[5,5], comp)

            """Test that setting a FiberTrace outside legal position fails"""
            try:
                fts.setFiberTrace(fts.size()+1, ft)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "FiberTraceSet::setFiberTrace: ERROR: position for trace outside range!"
                self.assertEqual(message[0],expected)

            try:
                self.assertFalse(fts.setFiberTrace(-1, ft))
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "in method 'FiberTraceSetF_setFiberTrace', argument 2 of type 'size_t'"
                self.assertEqual(message[0],expected)

            """Test that we can erase a FiberTrace"""
            size = fts.size()

            self.assertTrue(fts.erase(fts.size()-1))
            self.assertEqual(fts.size(), size-1)

            self.assertTrue(fts.erase(0))
            self.assertEqual(fts.size(), size-2)
            self.assertEqual(fts.getFiberTrace(0).getITrace(), 1)

            for i in range(3):
                ft = drpStella.FiberTraceF(fts.getFiberTrace(0), True)
                self.assertTrue(fts.addFiberTrace(ft, fts.size()))
            size = fts.size()

            self.assertTrue(fts.erase(3,5))
            self.assertEqual(fts.size(), size-2)
            self.assertEqual(fts.getFiberTrace(3).getITrace(), 6)

            try:
                self.assertFalse(fts.erase(fts.size()))
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "FiberTraceSet::erase(iStart="+str(fts.size())+", iEnd=0): ERROR: iStart >= _traces->size()="+str(fts.size())
                self.assertEqual(message[0],expected)

            """Test that we can set the FiberTraceProfileFittingControl to all FiberTraces in FiberTraceSet"""
            ftpfc = drpStella.FiberTraceProfileFittingControl()
            swathWidth = 300
            ftpfc.swathWidth = swathWidth
            for i in range(fts.size()):
                self.assertFalse(fts.getFiberTrace(i).isFiberTraceProfileFittingControlSet())
            self.assertTrue(fts.setFiberTraceProfileFittingControl(ftpfc))
            for i in range(fts.size()):
                self.assertEqual(fts.getFiberTrace(i).getFiberTraceProfileFittingControl().swathWidth, swathWidth)
                self.assertTrue(fts.getFiberTrace(i).isFiberTraceProfileFittingControlSet())

            """Test that we can fit the spatial profiles of one FiberTrace in Set"""
            self.assertFalse(fts.getFiberTrace(3).isProfileSet())
            spectrum = fts.getFiberTrace(3).calcProfile();
            self.assertTrue(fts.getFiberTrace(3).isProfileSet())

            """Test that we can fit the spatial profiles for all FiberTraces"""
            for i in range(fts.size()):
                spectrum = fts.getFiberTrace(i).calcProfile();
                self.assertTrue(fts.getFiberTrace(i).isProfileSet())

            """Test that we can set all profiles for a new FiberTraceSet"""
            """Copy constructor - shallow copy"""
            val = 11.
            fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5] = val
            self.assertAlmostEqual(fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)
            ftsComb = drpStella.FiberTraceSetF(fts)
            self.assertAlmostEqual(ftsComb.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)
            self.assertTrue(ftsComb.createTraces(self.arc.getMaskedImage()))
            self.assertAlmostEqual(fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5], ftsComb.getFiberTrace(0).getTrace().getImage().getArray()[5,5])
            self.assertNotAlmostEqual(fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)

            """Copy constructor - deep copy"""
            fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5] = val
            self.assertAlmostEqual(fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)
            ftsComb = drpStella.FiberTraceSetF(fts, True)
            self.assertAlmostEqual(ftsComb.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)
            self.assertTrue(ftsComb.createTraces(self.arc.getMaskedImage()))
            self.assertNotAlmostEqual(fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5], ftsComb.getFiberTrace(0).getTrace().getImage().getArray()[5,5])
            self.assertAlmostEqual(fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)


            """Test that we can extract a FiberTrace from the spatial profile"""
            spectrum = ftsComb.extractTraceNumberFromProfile(3)
            self.assertEqual(spectrum.getLength(), ftsComb.getFiberTrace(3).getHeight())

            """Test that we can extract all FiberTraces from the spatial profile"""
            spectra = ftsComb.extractAllTracesFromProfile()
            self.assertEqual(spectra.size(), ftsComb.size())
            for i in range(spectra.size()):
                self.assertEqual(spectra.getSpectrum(i).getLength(), ftsComb.getFiberTrace(i).getHeight())

    def testAddFiberTraceToCcdImage(self):
        fts = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
        origMaskedImage = self.flat.getMaskedImage()
        imageShape = origMaskedImage.getImage().getArray().shape
        sumImage = afwImage.ImageF(np.zeros(shape=imageShape, dtype=np.float32))
        ftsOrig = fts
        for iFt in range(fts.size()):
            ft = fts.getFiberTrace(iFt)
            drpStella.addFiberTraceToCcdImage(ft, ft.getImage(), sumImage)
        self.assertTrue(fts.createTraces(afwImage.makeMaskedImage(sumImage)))
        for iFt in range(fts.size()):
            ftOrig = ftsOrig.getFiberTrace(iFt)
            ftNew = fts.getFiberTrace(iFt)
            np.testing.assert_array_equal(ftOrig.getImage().getArray(), ftNew.getImage().getArray())

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
