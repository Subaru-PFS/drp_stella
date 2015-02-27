#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python Spectra.py
or
   python
   >>> import Spectra; Spectra.run()
"""

import unittest
import sys
import numpy as np
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

class SpectraTestCase(tests.TestCase):
    """A test case for measuring Spectra quantities"""

    def setUp(self):
        latest = True
        if latest:
            flatfile = "sampledFlatx2-IR-0-23.fits"
            combfile = "sampledCombx2-IR-0-23.fits"
        else:
            flatfile = "/Users/azuri/spectra/pfs/2014-10-28/sampledFlatx2-IR-0-23-5-10-nonoise.fits"
            combfile = "/Users/azuri/spectra/pfs/2014-10-28/sampledCombx2-IR-0-23-5-10-nonoise.fits"
        self.flat = afwImage.ImageF(flatfile)
        self.flat = afwImage.makeExposure(afwImage.makeMaskedImage(self.flat))
        self.bias = afwImage.ImageF(flatfile, 3)
        self.flat.getMaskedImage()[:] -= self.bias

        self.comb = afwImage.ImageF(combfile)
        self.comb = afwImage.makeExposure(afwImage.makeMaskedImage(self.comb))
        bias = afwImage.ImageF(combfile, 3)
        self.comb.getMaskedImage()[:] -= bias
        
        self.ftffc = drpStella.FiberTraceFunctionFindingControl()
        self.ftffc.fiberTraceFunctionControl.order = 4
        
        self.ftpfc = drpStella.FiberTraceProfileFittingControl()

        del bias
        del flatfile
        del combfile
        del latest
        
    def tearDown(self):
        del self.flat
        del self.comb
        del self.bias
        del self.ftffc
        del self.ftpfc

    def testSpectrumConstructors(self):
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
        """Test getSpectrum"""
        size = 100
        spec = drpStella.SpectrumF(size)
        vec = spec.getSpectrum()
        self.assertEqual(vec.shape[0], size)
        
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
            expected = "pfsDRPStella::Spectrum::setSpectrum: ERROR: spectrum->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
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
            expected = "pfsDRPStella::Spectrum::setVariance: ERROR: variance->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
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
            expected = "pfsDRPStella::Spectrum::setMask: ERROR: mask->size()="+str(vecus.shape[0])+" != _length="+str(spec.getLength())
            self.assertEqual(message[0],expected)
        self.assertEqual(spec.getMask().shape[0], size)

        if False: 
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

            """Test longer size"""
            self.assertTrue(spec.setLength(size+1))
            self.assertEqual(spec.getLength(), size+1)
            self.assertEqual(spec.getSpectrum()[size], 0)
            self.assertEqual(spec.getSpectrum().shape[0], size+1)
            self.assertEqual(spec.getVariance().shape[0], size+1)
            self.assertEqual(spec.getVariance()[size], 0)
            self.assertEqual(spec.getMask().shape[0], size+1)
            self.assertEqual(spec.getMask()[size], 0)
            self.assertEqual(spec.getWavelength().shape[0], size+1)
            self.assertEqual(spec.getWavelength()[size], vecf[size-1])

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
            
    def testSpectrumSetAddSetErase(self):
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
            expected = "SpectrumSet::setSpectrum(i="+str(size+1)+"): ERROR: i > _spectra->size()="+str(size)
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
            expected = "SpectrumSet::erase(iStart="+str(size)+", iEnd=0): ERROR: iStart >= _spectra->size()="+str(size)
            self.assertEqual(message[0],expected)

        try:
            specSet.erase(size-1, size)
        except:
            e = sys.exc_info()[1]
            message = str.split(e.message, "\n")
            for i in range(len(message)):
                print "element",i,": <",message[i],">"
            expected = "SpectrumSet::erase(iStart="+str(size-1)+", iEnd="+str(size)+"): ERROR: iEnd >= _spectra->size()="+str(size)
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
        """test getSpectra"""
        size = 3
        length = 100
        specSet = drpStella.SpectrumSetF(size,length)
        self.assertEqual(specSet.getSpectra()[0].getSpectrum().shape[0], length)
            
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
