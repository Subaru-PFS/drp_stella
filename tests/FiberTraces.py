#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python FiberTrace.py
or
   python
   >>> import FiberTrace; FiberTrace.run()
"""
from builtins import str
from builtins import range
import os
import sys
import unittest

from astropy.io import fits as pyfits
import numpy as np

import lsst.afw.display as afwDisplay
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.daf.persistence as dafPersist
import lsst.log as log
import lsst.utils
import lsst.utils.tests as tests
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import readWavelengthFile

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

        self.ftpfc = drpStella.FiberTraceProfileFittingControl()

        # This particular flatfile has 11 FiberTraces scattered over the whole CCD
        # If in the future the test data change we need to change these numbers
        self.nFiberTraces = 11
        self.minLength = 3880
        self.maxLength = 3930
        self.maxMeanDiffSum = 0.00001
        self.maxStdDevDiffSum = 101.0
        self.maxMeanDiffOpt = 4.0
        self.maxStdDevDiffOpt = 115.0

        self.wLenFile = os.path.join(lsst.utils.getPackageDir('obs_pfs'),'pfs/RedFiberPixels.fits.gz')

        """Quiet down loggers which are too verbose"""
        log.setLevel("pfs.drp.stella.FiberTrace.assignTraceID", log.WARN)

    def tearDown(self):
        del self.flat
        del self.arc
        del self.ftffc

    def testFiberTraceFunctionFindingControl(self):
        # Test that we can create a FiberTraceFunctionFindingControl
        self.ftffc.fiberTraceFunctionControl.order = 4

        # Test that we can set the parameters of the FiberTraceFunctionFindingControl
        interpolation = "POLYNOMIAL"
        self.ftffc.fiberTraceFunctionControl.interpolation = interpolation
        self.assertEqual(self.ftffc.fiberTraceFunctionControl.interpolation, interpolation)
        oldInterpolation = self.ftffc.fiberTraceFunctionControl.interpolation
        interpolation = "POLYNOMAL"
        self.assertNotEqual(interpolation, oldInterpolation)
        self.ftffc.fiberTraceFunctionControl.interpolation = interpolation
        self.assertEqual(self.ftffc.fiberTraceFunctionControl.interpolation, interpolation)

        order = 4
        self.ftffc.fiberTraceFunctionControl.order = order
        self.assertEqual(self.ftffc.fiberTraceFunctionControl.order, order)

        xLow = -5.
        self.ftffc.fiberTraceFunctionControl.xLow = xLow
        self.assertAlmostEqual(self.ftffc.fiberTraceFunctionControl.xLow, xLow)

        xHigh = 5.
        self.ftffc.fiberTraceFunctionControl.xHigh = xHigh
        self.assertAlmostEqual(self.ftffc.fiberTraceFunctionControl.xHigh, xHigh)

        apertureFWHM = 2.6
        self.ftffc.apertureFWHM = apertureFWHM
        self.assertAlmostEqual(self.ftffc.apertureFWHM, apertureFWHM, places=6)

        signalThreshold = 10.
        self.ftffc.signalThreshold = signalThreshold
        self.assertAlmostEqual(self.ftffc.signalThreshold, signalThreshold)

        nTermsGaussFit = 4
        self.ftffc.nTermsGaussFit = nTermsGaussFit
        self.assertEqual(self.ftffc.nTermsGaussFit, nTermsGaussFit)

        saturationLevel = 65550.
        self.ftffc.saturationLevel = saturationLevel
        self.assertAlmostEqual(self.ftffc.saturationLevel, saturationLevel)

        minLength = 20
        self.ftffc.minLength = minLength
        self.assertEqual(self.ftffc.minLength, minLength)

        maxLength = 4000
        self.ftffc.maxLength = maxLength
        self.assertEqual(self.ftffc.maxLength, maxLength)

        nLost = 20
        self.ftffc.nLost = nLost
        self.assertEqual(self.ftffc.nLost, nLost)

    def testFiberTraceConstructors(self):
        # Test that we can create a FiberTrace given width and height
        width = 5
        height = 100
        fiberId = 1
        maskedImage = afwImage.MaskedImageF(width, height)
        fiberTrace = drpStella.FiberTrace(maskedImage, fiberId)
        self.assertEqual(fiberTrace.getTrace().getWidth(), width)
        self.assertEqual(fiberTrace.getTrace().getWidth(), width)
        self.assertEqual(fiberTrace.getTrace().getHeight(), height)
        self.assertEqual(fiberTrace.getTrace().getHeight(), height)
        self.assertEqual(fiberTrace.getFiberId(), fiberId)

        # Test that we can create a FiberTrace given a MaskedImage and a FiberTraceFunction
        # Flat
        fiberTraceSet = drpStella.findAndTraceApertures(self.flat.getMaskedImage(),
                                                        self.ftffc,
                                                        self.ftpfc)
        # Check that we found self.nFiberTraces FiberTraces
        self.assertEqual(fiberTraceSet.size(), self.nFiberTraces)
        # Check length of FiberTraces
        for i in range(fiberTraceSet.size()):
            self.assertLess(fiberTraceSet.getFiberTrace(i).getTrace().getHeight(), self.maxLength)
            self.assertGreater(fiberTraceSet.getFiberTrace(i).getTrace().getHeight(), self.minLength)
        fiberTrace = fiberTraceSet.getFiberTrace(0)

        # Test copy constructor - shallow copy
        fiberTraceCopy = drpStella.FiberTrace(fiberTrace)
        self.assertEqual(fiberTraceCopy.getTrace().getWidth(), fiberTrace.getTrace().getWidth())
        self.assertEqual(fiberTraceCopy.getTrace().getHeight(), fiberTrace.getTrace().getHeight())
        self.assertEqual(fiberTraceCopy.getFiberId(), fiberTrace.getFiberId())
        self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], fiberTraceCopy.getTrace().getImage().getArray()[5,5])
        self.assertAlmostEqual(fiberTrace.getXCenters()[5], fiberTraceCopy.getXCenters()[5])
        val = 11.1
        fiberTrace.getTrace().getImage().getArray()[5,5] = val
        self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], val, places=6)
        self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], fiberTraceCopy.getTrace().getImage().getArray()[5,5])

        # Test copy constructor - deep copy
        fiberTraceCopy = drpStella.FiberTrace(fiberTrace, True)
        self.assertEqual(fiberTraceCopy.getTrace().getWidth(), fiberTrace.getTrace().getWidth())
        self.assertEqual(fiberTraceCopy.getTrace().getHeight(), fiberTrace.getTrace().getHeight())
        self.assertEqual(fiberTraceCopy.getFiberId(), fiberTrace.getFiberId())
        self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], fiberTraceCopy.getTrace().getImage().getArray()[5,5])
        self.assertAlmostEqual(fiberTrace.getXCenters()[5], fiberTraceCopy.getXCenters()[5])
        val = 13.3
        fiberTrace.getTrace().getImage().getArray()[5,5] = val
        self.assertAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], val, places=6)
        self.assertNotAlmostEqual(fiberTrace.getTrace().getImage().getArray()[5,5], fiberTraceCopy.getTrace().getImage().getArray()[5,5], places=6)

    def testFiberTraceSetConstructors(self):
        # Test that we can create a FiberTraceSet from the standard constructor
        fiberTraceSet = drpStella.FiberTraceSet()
        self.assertEqual(fiberTraceSet.size(), 0)

        nTraces = 0
        fiberTraceSet = drpStella.FiberTraceSet()
        self.assertEqual(fiberTraceSet.size(), nTraces)

        # Test that we can create a FiberTraceSet from another FiberTraceSet
        fiberTraceSet = drpStella.findAndTraceApertures(self.flat.getMaskedImage(),
                                                        self.ftffc,
                                                        self.ftpfc)
        fiberTraceSetNew = drpStella.FiberTraceSet(fiberTraceSet)
        self.assertEqual(fiberTraceSetNew.size(), fiberTraceSet.size())

    def testFiberTraceGetSetFunctions(self):
        # Test get/set methods

        fiberId = 1
        strFiberId = str(fiberId)
        xHigh = 4.1
        self.ftffc.fiberTraceFunctionControl.xHigh = xHigh
        self.ftffc.signalThreshold = 110.
        fiberTraceSet = drpStella.findAndTraceApertures(self.flat.getMaskedImage(),
                                                        self.ftffc,
                                                        self.ftpfc)

        # Test getFiberTrace
        fiberTrace = fiberTraceSet.getFiberTrace(fiberId)

        # Test getFiberId()
        self.assertEqual(fiberTrace.getFiberId(), fiberId)

        # Test getTrace
        self.assertEqual(fiberTrace.getTrace().getHeight(), fiberTrace.getTrace().getHeight())
        flatPlusArc = self.flat.getMaskedImage()
        flatPlusArc += self.arc.getMaskedImage()
        fiberTraceSetMIF = drpStella.findAndTraceApertures(flatPlusArc,
                                                           self.ftffc,
                                                           self.ftpfc)
        fiberTrace = fiberTraceSetMIF.getFiberTrace(fiberId)

        height = 100
        width = 10
        maskedImageWrongSize = afwImage.MaskedImageF(width,height)

        # Test getXCenters
        xCenters = fiberTrace.getXCenters()
        self.assertEqual(xCenters.shape[0], fiberTrace.getTrace().getHeight())
        self.assertAlmostEqual(xCenters[5], fiberTrace.getXCenters()[5])

    def testFiberTraceExtractionMethods(self):
        fiberTraceSet = drpStella.findAndTraceApertures(self.flat.getMaskedImage(),
                                                        self.ftffc,
                                                        self.ftpfc)
        # test extractSum and extractFromProfile
        for fiberId in range(0, fiberTraceSet.size()):
            fiberTrace = fiberTraceSet.getFiberTrace(fiberId)

            arcMI = afwImage.MaskedImageF(self.arc.getMaskedImage(),
                                          fiberTrace.getTrace().getBBox())
            imArr = arcMI.getImage().getArray()
            varArr = arcMI.getVariance().getArray()

            mask = fiberTrace.getTrace().getMask()
            ftMask = 1 << mask.getMaskPlane("FIBERTRACE")
            maskBool = mask.getArray() & ftMask
            maskWhere = np.where(maskBool == ftMask)

            # extract sum
            spectrum = fiberTrace.extractSum(self.arc.getMaskedImage())
            recArr = fiberTrace.getReconstructed2DSpectrum(spectrum).getArray()

            diff = imArr - recArr
            if display:
                displayA = afwDisplay.Display(frame=1)
                displayA.mtv(afwImage.ImageF(diff),title="diff reconstruction from extractSum")

            diff = diff[maskWhere]
            meanDiff = np.mean(diff)
            self.assertLess(np.absolute(meanDiff), self.maxMeanDiffSum)
            stdDevDiff = np.std(diff)
            self.assertLess(np.absolute(stdDevDiff), self.maxStdDevDiffSum)

            # extract from profile
            spectrum = fiberTrace.extractFromProfile(self.arc.getMaskedImage())
            recArr = fiberTrace.getReconstructed2DSpectrum(spectrum).getArray()

            diff = imArr - recArr
            if display:
                displayB = afwDisplay.Display(frame=2)
                displayB.mtv(afwImage.ImageF(diff),title="diff reconstruction from extractFromProfile")
            diff = diff[maskWhere]
            meanDiff = np.mean(diff)
            self.assertLess(np.absolute(meanDiff), self.maxMeanDiffOpt)
            stdDevDiff = np.std(diff)
            self.assertLess(np.absolute(stdDevDiff), self.maxStdDevDiffOpt)

    def testFiberTraceSetConstructor(self):
        size = 0
        fts = drpStella.FiberTraceSet()
        self.assertEqual(fts.size(), size)

        ftsa = drpStella.FiberTraceSet(fts)
        self.assertEqual(fts.size(), ftsa.size())

    def testFiberTraceSetFunctions(self):
        size = 0
        ftsEmpty = drpStella.FiberTraceSet()

        # Test that we can trace fibers
        fts = drpStella.findAndTraceApertures(self.flat.getMaskedImage(),
                                              self.ftffc,
                                              self.ftpfc)
        self.assertGreater(fts.size(), 0)

        width = 10
        height = 100
        maskedImage = afwImage.MaskedImageF(width, height)
        ft = drpStella.FiberTrace(maskedImage)
        self.assertEqual(ft.getFiberId(), 0)

        # Test that we can add a FiberTrace to an empty FiberTraceSet
        ftsEmpty.addFiberTrace(ft)
        self.assertEqual(ftsEmpty.size(), size+1)
        self.assertEqual(ftsEmpty.getFiberTrace(ftsEmpty.size()-1).getFiberId(), 0)

        size = fts.size()
        fts.addFiberTrace(ft)
        self.assertEqual(fts.size(), size+1)
        self.assertEqual(fts.getFiberTrace(fts.size()-1).getFiberId(), 0)

        # Test that we can set a FiberTrace at a certain position
        ft = fts.getFiberTrace(2)
        comp = ft.getTrace().getImage().getArray()[5,5]
        pos = 0
        fts.setFiberTrace(pos, ft)
        self.assertEqual(fts.getFiberTrace(pos).getTrace().getImage().getArray()[5,5], comp)

        pos = fts.size()-1
        fts.setFiberTrace(pos, ft)
        self.assertEqual(fts.getFiberTrace(pos).getTrace().getImage().getArray()[5,5], comp)

        # Test that setting a FiberTrace just past the last position adds a FiberTrace
        size = fts.size()
        fts.setFiberTrace(fts.size(), ft)
        self.assertEqual(fts.size(), size+1)
        self.assertEqual(fts.getFiberTrace(fts.size()-1).getTrace().getImage().getArray()[5,5], comp)

        # Test that setting a FiberTrace outside legal position fails
        try:
            fts.setFiberTrace(fts.size()+1, ft)
        except:
            e = sys.exc_info()[1]
            message = str.split(str(e.message), "\n")
            expected = "FiberTraceSet::setFiberTrace: ERROR: position for trace outside range!"
            self.assertEqual(message[0],expected)

        try:
            fts.setFiberTrace(-1, ft)
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "setFiberTrace(): incompatible function arguments. " \
                       "The following argument types are supported:"
            self.assertEqual(message[0],expected)

        # Test that we can erase a FiberTrace
        size = fts.size()

        fts.erase(fts.size() - 1)
        self.assertEqual(fts.size(), size-1)

        fts.erase(0)
        self.assertEqual(fts.size(), size-2)
        self.assertEqual(fts.getFiberTrace(0).getFiberId(), 1)

        ft = drpStella.FiberTrace(fts.getFiberTrace(0), True)
        for i in range(4):
            fts.addFiberTrace(ft, fts.size())

        fts.erase(fts.size() - 1, fts.size())

        size = fts.size()

        fts.erase(3, 5)
        self.assertEqual(fts.size(), size-2)
        self.assertEqual(fts.getFiberTrace(3).getFiberId(), 6)

        try:
            fts.erase(fts.size())
        except:
            e = sys.exc_info()[1]
            message = str.split(str(e.message), "\n")
            expected = "FiberTraceSet::erase(iStart="+str(fts.size())+", iEnd=0): ERROR: iStart >= _traces->size()="+str(fts.size())
            self.assertEqual(message[0],expected)

        # Test that we can set all profiles for a new FiberTraceSet
        # Copy constructor - shallow copy
        val = 11.
        fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5] = val
        self.assertAlmostEqual(fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)
        ftsComb = drpStella.FiberTraceSet(fts)
        self.assertAlmostEqual(ftsComb.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)

        #Copy constructor - deep copy
        fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5] = val
        self.assertAlmostEqual(fts.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)
        ftsComb = drpStella.FiberTraceSet(fts, True)
        self.assertAlmostEqual(ftsComb.getFiberTrace(0).getTrace().getImage().getArray()[5,5], val)

        # Test that we can extract a FiberTrace from the spatial profile
        spectrum = ftsComb.extractTraceNumberFromProfile(self.arc.getMaskedImage(), 3)
        self.assertEqual(spectrum.getLength(), ftsComb.getFiberTrace(3).getTrace().getHeight())

        # Test that we can extract all FiberTraces from the spatial profile
        spectra = ftsComb.extractAllTracesFromProfile(self.arc.getMaskedImage())
        self.assertEqual(spectra.size(), ftsComb.size())
        for i in range(spectra.size()):
            self.assertEqual(spectra.getSpectrum(i).getLength(), ftsComb.getFiberTrace(i).getTrace().getHeight())

        fts = drpStella.findAndTraceApertures(self.flat.getMaskedImage(),
                                              self.ftffc,
                                              self.ftpfc)

    def testAssignTraceIDs(self):
        fts = drpStella.findAndTraceApertures(self.flat.getMaskedImage(),
                                              self.ftffc,
                                              self.ftpfc)

        # read wavelength file
        xCenters, wavelengths, traceIds = readWavelengthFile(self.wLenFile)

        traceNumbersCheck = [2,64,212,286,315,337,366,440,513,588,650]

        """ assign trace number to fiberTraceSet """
        fts.assignTraceIDs(traceIds, xCenters)
        for i in range( fts.size() ):
            fiberId = fts.getFiberTrace(i).getFiberId()
            self.assertEqual(fiberId, traceNumbersCheck[i])

    def testFiberTraceCenter(self):
        nRows = 100
        nCols = 20
        ccd = np.zeros(shape=(nRows,nCols), dtype=np.float32)
        row = np.zeros(shape=(nCols), dtype=np.float32)
        center = 10.0
        for i in range(nCols):
            row[i] = 10000. * np.exp(-(i-center)*(i-center) / 4.)
        for i in range(nRows):
            ccd[i,:] = row[:]
        fiberTraceFunctionControl = drpStella.FiberTraceFunctionControl()
        fiberTraceFunctionControl.interpolation = "POLYNOMIAL"
        fiberTraceFunctionControl.order = 0
        fiberTraceFunctionControl.xLow = -5.
        fiberTraceFunctionControl.xHigh = 5.
        fiberTraceFunctionControl.nPixCutLeft = 0
        fiberTraceFunctionControl.nPixCutRight = 0
        fiberTraceFunctionControl.nRows = nRows

        fiberTraceFunctionFindingControl = drpStella.FiberTraceFunctionFindingControl()
        fiberTraceFunctionFindingControl.fiberTraceFunctionControl = fiberTraceFunctionControl
        fiberTraceFunctionFindingControl.apertureFWHM = 4.
        fiberTraceFunctionFindingControl.signalThreshold = 1.
        fiberTraceFunctionFindingControl.nTermsGaussFit = 3
        fiberTraceFunctionFindingControl.saturationLevel = 65000.
        fiberTraceFunctionFindingControl.minLength = 10
        fiberTraceFunctionFindingControl.maxLength = nRows
        fiberTraceFunctionFindingControl.nLost = 10

        fiberTraceProfileFittingControl = drpStella.FiberTraceProfileFittingControl()

        fiberTraceSet = drpStella.findAndTraceApertures(
            afwImage.makeMaskedImage(afwImage.ImageF(ccd)),
            fiberTraceFunctionFindingControl,
            self.ftpfc)

        fiberTrace = fiberTraceSet.getFiberTrace(0)
        xCenters = fiberTrace.getXCenters()
        for i in range(xCenters.shape[0]):
            self.assertAlmostEqual(xCenters[i], center)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(FiberTraceTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Quiet down loggers which are too verbose"""
    for logger in ["afw.image.ExposureInfo",
                   "afw.image.Mask",
                   "CameraMapper",
                   "pfs.drp.stella.FiberTrace.calcProfile",
                   "pfs.drp.stella.FiberTrace.calcProfileSwath",
                   "pfs.drp.stella.FiberTrace.extractFromProfile",
                   "pfs.drp.stella.math.CurveFitting.LinFitBevingtonNdArray1D",
                   "pfs.drp.stella.math.CurveFitting.LinFitBevingtonNdArray2D",
                   "pfs.drp.stella.math.CurveFitting.PolyFit",
                   ]:
        log.Log.getLogger(logger).setLevel(log.WARN)

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
