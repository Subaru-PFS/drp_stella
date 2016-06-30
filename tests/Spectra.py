#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python Spectra.py
or
   python
   >>> import Spectra; Spectra.run()
"""

import os
import unittest
import sys
import numpy as np
import lsst.utils.tests as tests
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import pfs.drp.stella as drpStella
import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils
from astropy.io import fits as pyfits
import pfs.drp.stella.extractSpectraTask as esTask
import pfs.drp.stella.createFlatFiberTraceProfileTask as cfftpTask


try:
    type(display)
except NameError:
    display = False

class SpectraTestCase(tests.TestCase):
    """A test case for measuring Spectra quantities"""

    def setUp(self):
        drpStellaDataDir = os.environ.get('DRP_STELLA_DATA_DIR')
        flatfile = drpStellaDataDir+"/data/PFS/CALIB/FLAT/r/flat/pfsFlat-2016-01-12-0-2r.fits"
        arcfile = drpStellaDataDir+"/data/PFS/postISRCCD/2016-01-12/v0000004/PFFAr2.fits"
        self.flat = afwImage.ImageF(flatfile)
        self.flat = afwImage.makeExposure(afwImage.makeMaskedImage(self.flat))

        self.arc = afwImage.ImageF(arcfile)
        self.arc = afwImage.makeExposure(afwImage.makeMaskedImage(self.arc))
        
        self.ftffc = drpStella.FiberTraceFunctionFindingControl()
        self.ftffc.fiberTraceFunctionControl.order = 5
        self.ftffc.fiberTraceFunctionControl.xLow = -5
        self.ftffc.fiberTraceFunctionControl.xHigh = 5
        
        self.ftpfc = drpStella.FiberTraceProfileFittingControl()
        
        self.nFiberTraces = 11

        del flatfile
        del arcfile
        
    def tearDown(self):
        del self.flat
        del self.arc
        del self.ftffc
        del self.ftpfc
        del self.nFiberTraces

    def testSpectrumConstructors(self):
        if True:
            """Test that we can create a Spectrum with the standard constructor"""
            spec = drpStella.SpectrumF()
            self.assertEqual(spec.getLength(), 0)
            self.assertEqual(spec.getITrace(), 0)

            length = 10
            iTrace = 2
            spec = drpStella.SpectrumF(length, iTrace)
            self.assertEqual(spec.getLength(), length)
            self.assertEqual(spec.getITrace(), iTrace)

            """Test copy constructor"""
            specCopy = drpStella.SpectrumF(spec)
            self.assertEqual(specCopy.getLength(), length)
            self.assertEqual(specCopy.getITrace(), iTrace)
        
    def testSpectrumMethods(self):
        if True:
            """Test getSpectrum"""
            size = 100
            spec = drpStella.SpectrumF(size)
            vec = spec.getSpectrum()
            self.assertEqual(vec.shape[0], size)
            self.assertEqual(spec.getSpectrum().shape[0], size)

            """Test setSpectrum"""
            """Test that we can assign a spectrum of the correct length"""
            vecf = drpStella.indGenNdArrF(size)
            self.assertTrue(spec.setSpectrum(vecf))
            self.assertEqual(spec.getSpectrum()[3], vecf[3])

            """Test that we can't assign a spectrum of the wrong length"""
            vecf = drpStella.indGenNdArrF(size+1)
            try:
                spec.setSpectrum(vecf)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "pfs::drp::stella::Spectrum::setSpectrum: ERROR: spectrum->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
                self.assertEqual(message[0],expected)
            self.assertEqual(spec.getSpectrum().shape[0], size)

            """Test getVariance"""
            vec = spec.getVariance()
            self.assertEqual(vec.shape[0], size)

            """Test setVariance"""
            """Test that we can assign a variance vector of the correct length"""
            vecf = drpStella.indGenNdArrF(size)
            self.assertTrue(spec.setVariance(vecf))
            self.assertEqual(spec.getVariance()[3], vecf[3])

            """Test that we can't assign a variance vector of the wrong length"""
            vecf = drpStella.indGenNdArrF(size+1)
            try:
                self.assertFalse(spec.setVariance(vecf))
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "pfs::drp::stella::Spectrum::setVariance: ERROR: variance->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
                self.assertEqual(message[0],expected)
            self.assertEqual(spec.getVariance().shape[0], size)

            """Test getWavelength"""
            vec = spec.getWavelength()
            self.assertEqual(vec.shape[0], size)

            """Test setWavelength"""
            """Test that we can assign a wavelength vector of the correct length"""
            vecf = drpStella.indGenNdArrF(size)
            self.assertTrue(spec.setWavelength(vecf))
            self.assertEqual(spec.getWavelength()[3], vecf[3])

            """Test that we can't assign a wavelength vector of the wrong length"""
            vecf = drpStella.indGenNdArrF(size+1)
            try:
                self.assertFalse(spec.setWavelength(vecf))
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "pfsDRPStella::Spectrum::setWavelength: ERROR: wavelength->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
                self.assertEqual(message[0],expected)
            self.assertEqual(spec.getWavelength().shape[0], size)


            """Test getMask"""
            vec = spec.getMask()
            print "vec = ",vec
            print "dir(vec): ",dir(vec)
            self.assertEqual(vec.shape[0], size)

            """Test setMask"""
            """Test that we can assign a mask vector of the correct length"""
            vecf = drpStella.indGenNdArrUS(size)
            self.assertTrue(spec.setMask(vecf))
            self.assertEqual(spec.getMask()[3], vecf[3])

            """Test that we can't assign a mask vector of the wrong length"""
            vecus = drpStella.indGenNdArrUS(size+1)
            try:
                self.assertFalse(spec.setMask(vecus))
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "pfs::drp::stella::Spectrum::setMask: ERROR: mask->size()="+str(vecus.shape[0])+" != _length="+str(spec.getLength())
                self.assertEqual(message[0],expected)
            self.assertEqual(spec.getMask().shape[0], size)

            if True: 
                """Test setLength"""
                """If newLength < oldLength, vectors are supposed to be cut off, otherwise ZEROs are appended to the end of the vectors (last wavelength value for wavelength vector)"""
                """Test same size"""
                vecf = drpStella.indGenNdArrF(size)
                vecus = drpStella.indGenNdArrUS(size)
                self.assertTrue(spec.setLength(size))
                self.assertEqual(spec.getLength(), size)
                self.assertEqual(spec.getSpectrum()[size-1], vecf[size-1])
                self.assertEqual(spec.getSpectrum().shape[0], size)
                self.assertEqual(spec.getVariance().shape[0], size)
                self.assertEqual(spec.getVariance()[size-1], vecf[size-1])
                self.assertEqual(spec.getMask().shape[0], size)
                self.assertEqual(spec.getMask()[size-1], vecus[size-1])
                self.assertEqual(spec.getWavelength().shape[0], size)
                self.assertEqual(spec.getWavelength()[size-1], vecf[size-1])

            if True: 
                """Test longer size"""
                print 'trying to change length of Spectrum'
                self.assertTrue(spec.setLength(size+1))
                print 'spec = ',spec
            if True: 
                self.assertEqual(spec.getLength(), size+1)
                self.assertEqual(spec.getSpectrum()[size], 0)
                self.assertEqual(spec.getSpectrum().shape[0], size+1)
                self.assertEqual(spec.getVariance().shape[0], size+1)
                self.assertEqual(spec.getVariance()[size], 0)
                self.assertEqual(spec.getMask().shape[0], size+1)
                self.assertEqual(spec.getMask()[size], 0)
                self.assertEqual(spec.getWavelength().shape[0], size+1)
                self.assertAlmostEqual(spec.getWavelength()[size], 0.)

            if True: 
                """Test shorter size"""
                self.assertTrue(spec.setLength(size-1))
                self.assertEqual(spec.getLength(), size-1)
                self.assertEqual(spec.getSpectrum()[size-2], vecf[size-2])
                self.assertEqual(spec.getSpectrum().shape[0], size-1)
                self.assertEqual(spec.getVariance().shape[0], size-1)
                self.assertEqual(spec.getVariance()[size-2], vecf[size-2])
                self.assertEqual(spec.getMask().shape[0], size-1)
                self.assertEqual(spec.getMask()[size-2], vecus[size-2])
                self.assertEqual(spec.getWavelength().shape[0], size-1)
                self.assertEqual(spec.getWavelength()[size-2], vecf[size-2])

            """Test get/setITrace"""
            self.assertEqual(spec.getITrace(), 0)
            spec.setITrace(10)
            self.assertEqual(spec.getITrace(), 10)

            """Test isWaveLengthSet"""
            self.assertFalse(spec.isWavelengthSet())
            
    def testSpectrumSetConstructors(self):
        if True:
            """Test SpectrumSetConstructors"""
            """Test Standard Constructor"""
            specSet = drpStella.SpectrumSetF()
            self.assertEqual(specSet.size(), 0)

            size = 3
            specSet = drpStella.SpectrumSetF(size)
            self.assertEqual(specSet.size(), size)
            for i in range(size):
                self.assertEqual(specSet.getSpectrum(i).getSpectrum().shape[0], 0)

            length = 33
            specSet = drpStella.SpectrumSetF(size, length)
            self.assertEqual(specSet.size(), size)
            for i in range(size):
                self.assertEqual(specSet.getSpectrum(i).getSpectrum().shape[0], length)

            """Test copy constructor"""
            specSetCopy = drpStella.SpectrumSetF(specSet)
            self.assertEqual(specSetCopy.size(), specSet.size())
            for i in range(specSet.size()):
                self.assertEqual(specSetCopy.getSpectrum(i).getLength(), specSet.getSpectrum(i).getLength())
            if False:
                val = 3.3
                pos = 3
                vecf = list(drpStella.indGenF(length))
                print type(vecf)
                print vecf
                print dir(vecf)
                vecf[pos] = val
                pvecf = drpStella.SpecVectorF(vecf)
                self.assertTrue(specSet.getSpectrum(0).setSpectrum(pvecf))
                specSetCopy.getSpectrum(0).setSpectrum(pvecf)
                self.assertAlmostEqual(specSetCopy.getSpectrum(i).getSpectrum()[pos], val)

            """Test constructor from vector of spectra"""
            specSetV = drpStella.SpectrumSetF(specSet.getSpectra())
            self.assertEqual(specSet.size(), specSetV.size())
            for i in range(specSet.size()):
                self.assertEqual(specSetV.getSpectrum(i).getITrace(), i)
            
    def testExtractTask(self):
        if True:
            print "testing ExtractSpectraTask"
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            self.assertEqual(fiberTraceSet.size(), self.nFiberTraces)
            print 'found ',fiberTraceSet.size(),' traces'
            myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
            myProfileTask.run(fiberTraceSet)

            myExtractTask = esTask.ExtractSpectraTask()
            aperturesToExtract = [-1]
            spectrumSetFromProfile = myExtractTask.run(self.comb, fiberTraceSet, aperturesToExtract)
            self.assertEqual(spectrumSetFromProfile.size(), self.nFiberTraces)
            for i in range(spectrumSetFromProfile.size()):
                self.assertEqual(spectrumSetFromProfile.getSpectrum(i).getLength(), fiberTraceSet.getFiberTrace(i).getHeight())
            
    def testSpectrumSetAddSetErase(self):
        if True:
            size = 3
            length = 100
            specSet = drpStella.SpectrumSetF(size, length)
            spec = drpStella.SpectrumF(length)
            specNew = drpStella.SpectrumF(length+1)

            """Test that we cannot set a spectrum outside the limits 0 <= pos <= size"""
            try:
                specSet.setSpectrum(-1, specNew)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "in method 'SpectrumSetF_setSpectrum', argument 2 of type 'size_t'"
                self.assertEqual(message[0],expected)
            try:
                specSet.setSpectrum(size+1, specNew)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "SpectrumSet::setSpectrum(i="+str(size+1)+"): ERROR: i > _spectra.size()="+str(size)
                self.assertEqual(message[0],expected)

            """Test that we can set/add a spectrum"""
            self.assertTrue(specSet.setSpectrum(size-1, specNew))
            self.assertEqual(specSet.size(), size)
            self.assertTrue(specSet.setSpectrum(size, specNew))
            self.assertEqual(specSet.size(), size+1)

            specSet.addSpectrum(specNew)
            self.assertEqual(specSet.size(), size+2)

            """Test that we can't erase spectra outside the limits"""
            size = specSet.size()
            try:
                specSet.erase(size)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "SpectrumSet::erase(iStart="+str(size)+", iEnd=0): ERROR: iStart >= _spectra.size()="+str(size)
                self.assertEqual(message[0],expected)

            try:
                specSet.erase(size-1, size)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "SpectrumSet::erase(iStart="+str(size-1)+", iEnd="+str(size)+"): ERROR: iEnd >= _spectra.size()="+str(size)
                self.assertEqual(message[0],expected)

            try:
                specSet.erase(2, 1)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "SpectrumSet::erase(iStart=2, iEnd=1): ERROR: iStart > iEnd"
                self.assertEqual(message[0],expected)

            try:
                specSet.erase(-1)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "Wrong number or type of arguments for overloaded function 'SpectrumSetF_erase'."
                self.assertEqual(message[0],expected)

            """Test that we CAN erase spectra inside the limits"""
            self.assertTrue(specSet.erase(size-1))
            self.assertEqual(specSet.size(), size-1)

            self.assertTrue(specSet.erase(0, 1))
            self.assertEqual(specSet.size(), size-2)

            self.assertTrue(specSet.erase(0,2))
            self.assertEqual(specSet.size(), size-4)
        
    def testGetSpectra(self):
        if False:#FAILS because spectra is not recognized as a vector of Spectrum(s)
            """test getSpectra"""
            size = 3
            length = 100
            specSet = drpStella.SpectrumSetF(size,length)
            print 'specSet.size() = ',specSet.size()
            spectra = specSet.getSpectra()
            print 'spectra.size() = ',spectra.size()
            print 'type(spectra[0]) = ',type(spectra[0])
            print 'dir(spectra[0]) = ',dir(spectra[0])
            self.assertEqual(spectra[0].getSpectrum().shape[0], length)

    def testWavelengthCalibration(self):
        if True:
            print "testing wavelength calibration"
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            self.assertGreater(fiberTraceSet.size(), 0)
            print 'found ',fiberTraceSet.size(),' traces'
            myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
            myProfileTask.run(fiberTraceSet)

            myExtractTask = esTask.ExtractSpectraTask()
            aperturesToExtract = [-1]
            spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
            self.assertGreater(spectrumSetFromProfile.size(), 0)

            """ read line list """
            lineList = os.environ.get('OBS_PFS_DIR')+'/pfs/lineLists/CdHgKrNeXe_red.fits'
            hdulist = pyfits.open(lineList)
            tbdata = hdulist[1].data
            lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float32')
            lineListArr[:,0] = tbdata.field(0)
            lineListArr[:,1] = tbdata.field(1)

            """ read reference Spectrum """
            refSpec = os.environ.get('OBS_PFS_DIR')+'/pfs/arcSpectra/refSpec_CdHgKrNeXe_red.fits'
            hdulist = pyfits.open(refSpec)
            tbdata = hdulist[1].data
            refSpecArr = np.ndarray(shape=(len(tbdata)), dtype='float32')
            refSpecArr[:] = tbdata.field(0)

            refSpec = spectrumSetFromProfile.getSpectrum(int(spectrumSetFromProfile.size() / 2))
            ref = refSpec.getSpectrum()

        
            
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(SpectraTestCase)
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
