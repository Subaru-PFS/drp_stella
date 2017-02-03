#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python Spectra.py
or
   python
   >>> import Spectra; Spectra.run()
"""

from astropy.io import fits as pyfits
import lsst.afw.image as afwImage
import lsst.daf.persistence as dafPersist
import lsst.utils
import lsst.utils.tests as tests
import numpy as np
import os
import pfs.drp.stella as drpStella
import pfs.drp.stella.createFlatFiberTraceProfileTask as cfftpTask
import pfs.drp.stella.extractSpectraTask as esTask
import pfs.drp.stella.findAndTraceAperturesTask as fataTask
import pfs.drp.stella.math as drpStellaMath
import sys
import unittest

try:
    type(display)
except NameError:
    display = False

class SpectraTestCase(tests.TestCase):
    """A test case for measuring Spectra quantities"""

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

        self.dispCorControl = drpStella.DispCorControl()

        self.nFiberTraces = 11
        self.nRowsPrescan = 49

        # This value is a measured value which is otherwise poorly justified.
        # It serves as a regression test to make sure that changes in the code
        # didn't make it worse.
        self.maxRMS = 0.06

        self.lineList = os.path.join(lsst.utils.getPackageDir('obs_pfs'),'pfs/lineLists/CdHgKrNeXe_red.fits')
        self.refSpec = os.path.join(lsst.utils.getPackageDir('obs_pfs'),'pfs/arcSpectra/refSpec_CdHgKrNeXe_red.fits')
        self.wLenFile = os.path.join(lsst.utils.getPackageDir('obs_pfs'),'pfs/RedFiberPixels.fits.gz')

    def tearDown(self):
        del self.flat
        del self.arc
        del self.ftffc
        del self.nFiberTraces
        del self.lineList
        del self.refSpec
        del self.dispCorControl
        del self.wLenFile
        del self.maxRMS

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

        if True:
            """Test that we can't assign a spectrum of the wrong length"""
            vecf = drpStella.indGenNdArrF(size+1)
            try:
                spec.setSpectrum(vecf)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "pfs::drp::stella::Spectrum::setSpectrum: ERROR: spectrum->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
                self.assertEqual(message[0],expected)
            self.assertEqual(spec.getSpectrum().shape[0], size)

            """Test getVariance"""
            vec = spec.getVariance()
            self.assertEqual(vec.shape[0], size)

        if True:
            """Test setVariance"""
            """Test that we can assign a variance vector of the correct length"""
            vecf = drpStella.indGenNdArrF(size)
            self.assertTrue(spec.setVariance(vecf))
            self.assertEqual(spec.getVariance()[3], vecf[3])

        if True:
            """Test that we can't assign a variance vector of the wrong length"""
            vecf = drpStella.indGenNdArrF(size+1)
            try:
                self.assertFalse(spec.setVariance(vecf))
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
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
                expected = "pfsDRPStella::Spectrum::setWavelength: ERROR: wavelength->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
                self.assertEqual(message[0],expected)
            self.assertEqual(spec.getWavelength().shape[0], size)

        if True:
            """Test getMask"""
            vec = spec.getMask()
            self.assertEqual(vec.getWidth(), size)

            """Test setMask"""
            """Test that we can assign a mask vector of the correct length"""
            vecf = afwImage.MaskU(size, 1)
            self.assertTrue(spec.setMask(vecf))

            """Test that we can't assign a mask vector of the wrong length"""
            vecus = afwImage.MaskU(size+1, 1)
            try:
                self.assertFalse(spec.setMask(vecus))
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                for i in range(len(message)):
                    print "element",i,": <",message[i],">"
                expected = "pfs::drp::stella::Spectrum::setMask: ERROR: mask.getWidth()="+str(vecus.getWidth())+" != _length="+str(spec.getLength())
                self.assertEqual(message[0],expected)
            self.assertEqual(spec.getMask().getWidth(), size)

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
                self.assertEqual(spec.getMask().getWidth(), size)
                self.assertEqual(spec.getWavelength().shape[0], size)
                self.assertEqual(spec.getWavelength()[size-1], vecf[size-1])

            if True:
                """Test longer size"""
                self.assertTrue(spec.setLength(size+1))
            if True:
                self.assertEqual(spec.getLength(), size+1)
                self.assertEqual(spec.getSpectrum()[size], 0)
                self.assertEqual(spec.getSpectrum().shape[0], size+1)
                self.assertEqual(spec.getVariance().shape[0], size+1)
                self.assertEqual(spec.getVariance()[size], 0)
                self.assertEqual(spec.getMask().getWidth(), size+1)
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
                self.assertEqual(spec.getMask().getWidth(), size-1)
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
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            self.assertEqual(fiberTraceSet.size(), self.nFiberTraces)
            myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
            myProfileTask.run(fiberTraceSet)

            myExtractTask = esTask.ExtractSpectraTask()
            aperturesToExtract = [-1]
            spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
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
                expected = "Wrong number or type of arguments for overloaded function 'SpectrumSetF_setSpectrum'."
                self.assertEqual(message[0],expected)
            try:
                specSet.setSpectrum(size+1, specNew)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
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
                expected = "SpectrumSet::erase(iStart="+str(size)+", iEnd=0): ERROR: iStart >= _spectra->size()="+str(size)
                self.assertEqual(message[0],expected)

            try:
                specSet.erase(size-1, size)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "SpectrumSet::erase(iStart="+str(size-1)+", iEnd="+str(size)+"): ERROR: iEnd >= _spectra->size()="+str(size)
                self.assertEqual(message[0],expected)

            try:
                specSet.erase(2, 1)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
                expected = "SpectrumSet::erase(iStart=2, iEnd=1): ERROR: iStart > iEnd"
                self.assertEqual(message[0],expected)

            try:
                specSet.erase(-1)
            except:
                e = sys.exc_info()[1]
                message = str.split(e.message, "\n")
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
            spectra = specSet.getSpectra()
            self.assertEqual(spectra[0].getSpectrum().shape[0], length)

    def testProfile(self):
        """
        Test that there are no systematics in the profile
        We compare the original FiberTrace images to the reconstructed ones.
        The reconstructed images are calculated from the spatial profile images
        and the extracted spectra.
        We sort the ratio of both images (original / recreated) from all
        FiberTraces by the distance of the pixel to the FiberTrace's center,
        bin them in small bins (10 times the oversampling rate used for the
        profile calculation), and calculate the mean and standard deviation
        for each bin. The test passes if 1.0 is within 1 standard deviation
        of the mean value.
        """
        nBinsPerOverSampleStep = 10.
        myFindTask = fataTask.FindAndTraceAperturesTask()
        myFindTask.config.xLow = -5.
        myFindTask.config.xHigh = 5.
        fts = myFindTask.run(self.flat)

        myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
        myProfileTask.config.overSample = 10

        for interpolation in ['SPLINE3', 'PISKUNOV']:
            myProfileTask.config.profileInterpolation = interpolation
            myProfileTask.run(fts)

            distanceFromCenterOrigProfRec = drpStellaMath.getDistTraceProfRec(fts)
            dist, orig, rec = [distanceFromCenterOrigProfRec[:,0],
                               distanceFromCenterOrigProfRec[:,1],
                               distanceFromCenterOrigProfRec[:,3]]
            ratio = orig / np.where(rec < 0.1, 1., rec)

            binWidth = 1. / (myProfileTask.fiberTraceProfileFittingControl.overSample
                             * nBinsPerOverSampleStep)
            bins, mean, std = drpStellaMath.getMeanStdXBins(dist,
                                                            ratio,
                                                            binWidth)
            for iBin in range(bins.shape[0]):
                self.assertTrue(np.all(np.fabs(mean - 1.0) < std))

    def testWavelengthCalibrationWithRefSpec(self):
        if True:
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            self.assertGreater(fiberTraceSet.size(), 0)
            myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
            myProfileTask.run(fiberTraceSet)

            myExtractTask = esTask.ExtractSpectraTask()
            aperturesToExtract = [-1]
            spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
            self.assertEqual(spectrumSetFromProfile.size(), fiberTraceSet.size())

            self.dispCorControl.fittingFunction = "POLYNOMIAL"
            self.dispCorControl.order = 5
            self.dispCorControl.searchRadius = 2
            self.dispCorControl.fwhm = 2.6
            self.dispCorControl.radiusXCor = 35
            self.dispCorControl.lengthPieces = 500
            self.dispCorControl.nCalcs = 15
            self.dispCorControl.stretchMinLength = 450
            self.dispCorControl.stretchMaxLength = 550
            self.dispCorControl.nStretches = 100

            """ read line list """
            hdulist = pyfits.open(self.lineList)
            tbdata = hdulist[1].data
            lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float32')
            lineListArr[:,0] = tbdata.field(0)
            lineListArr[:,1] = tbdata.field(1)

            """ read reference Spectrum """
            hdulist = pyfits.open(self.refSpec)
            tbdata = hdulist[1].data
            refSpecArr = np.ndarray(shape=(len(tbdata)), dtype='float32')
            refSpecArr[:] = tbdata.field(0)

            for i in range(spectrumSetFromProfile.size()):
                spec = spectrumSetFromProfile.getSpectrum(i)
                specSpec = spec.getSpectrum()
                result = drpStella.stretchAndCrossCorrelateSpecFF(specSpec, refSpecArr, lineListArr, self.dispCorControl)
                spec.identifyF(result.lineList, self.dispCorControl, 8)

                """Check that wavelength solution is monotonic"""
                for j in range(spec.getLength()-1):
                    self.assertLess(spec.getWavelength()[j], spec.getWavelength()[j+1])

                """Check wavelength range"""
                self.assertGreater(spec.getWavelength()[0], 3800)
                self.assertLess(spec.getWavelength()[spec.getLength()-1], 9800)

                """Check RMS"""
                self.assertLess(spec.getDispRms(), self.maxRMS)

    def testWavelengthCalibrationWithoutRefSpec(self):
        if True:
            fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
            self.assertGreater(fiberTraceSet.size(), 0)
            myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
            myProfileTask.run(fiberTraceSet)

            myExtractTask = esTask.ExtractSpectraTask()
            aperturesToExtract = [-1]
            spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
            self.assertEqual(spectrumSetFromProfile.size(), fiberTraceSet.size())

            self.dispCorControl.fittingFunction = "POLYNOMIAL"
            self.dispCorControl.order = 5
            self.dispCorControl.searchRadius = 2
            self.dispCorControl.fwhm = 2.6

            """ read wavelength file """
            hdulist = pyfits.open(self.wLenFile)
            tbdata = hdulist[1].data
            traceIdsTemp = np.ndarray(shape=(len(tbdata)), dtype='int')
            xCenters = np.ndarray(shape=(len(tbdata)), dtype='float32')
            yCenters = np.ndarray(shape=(len(tbdata)), dtype='float32')
            wavelengths = np.ndarray(shape=(len(tbdata)), dtype='float32')
            traceIdsTemp[:] = tbdata[:]['fiberNum']
            traceIds = traceIdsTemp.astype('int32')
            wavelengths[:] = tbdata[:]['pixelWave']
            xCenters[:] = tbdata[:]['xc']
            yCenters[:] = tbdata[:]['yc']

            traceIdsUnique = np.unique(traceIds)

            """ assign trace number to fiberTraceSet """
            success = drpStella.assignITrace( fiberTraceSet, traceIds, xCenters, yCenters )
            iTraces = np.ndarray(shape=fiberTraceSet.size(), dtype='intp')
            for i in range( fiberTraceSet.size() ):
                iTraces[i] = fiberTraceSet.getFiberTrace(i).getITrace()

            self.assertTrue(success)

            """ read line list """
            hdulist = pyfits.open(self.lineList)
            tbdata = hdulist[1].data
            lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float32')
            lineListArr[:,0] = tbdata.field(0)
            lineListArr[:,1] = tbdata.field(1)

            for i in range(spectrumSetFromProfile.size()):
                spec = spectrumSetFromProfile.getSpectrum(i)
                spec.setITrace(iTraces[i])

                traceId = spec.getITrace()
                wLenTemp = np.ndarray( shape = traceIds.shape[0] / np.unique(traceIds).shape[0], dtype='float32' )
                k = 0
                l = -1
                for j in range(traceIds.shape[0]):
                    if traceIds[j] != l:
                        l = traceIds[j]
                    if traceIds[j] == traceIdsUnique[traceId]:
                        wLenTemp[k] = wavelengths[j]
                        k = k+1

                """cut off both ends of wavelengths where is no signal"""
                yCenter = fiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yCenter
                yLow = fiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yLow
                yHigh = fiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yHigh
                yMin = yCenter + yLow
                yMax = yCenter + yHigh
                wLen = wLenTemp[ yMin + self.nRowsPrescan : yMax + self.nRowsPrescan + 1]
                wLenArr = np.ndarray(shape=wLen.shape, dtype='float32')
                for j in range(wLen.shape[0]):
                    wLenArr[j] = wLen[j]
                wLenLines = lineListArr[:,0]
                wLenLinesArr = np.ndarray(shape=wLenLines.shape, dtype='float32')
                for j in range(wLenLines.shape[0]):
                    wLenLinesArr[j] = wLenLines[j]
                lineListPix = drpStella.createLineList(wLenArr, wLenLinesArr)
                spec.identifyF(lineListPix, self.dispCorControl, 8)

                """Check that wavelength solution is monotonic"""
                for j in range(spec.getLength()-1):
                    self.assertLess(spec.getWavelength()[j], spec.getWavelength()[j+1])

                """Check wavelength range"""
                self.assertGreater(spec.getWavelength()[0], 3800)
                self.assertLess(spec.getWavelength()[spec.getLength()-1], 9800)

    def testPolyFit(self):
        fiberTraceSet = drpStella.findAndTraceAperturesF(self.flat.getMaskedImage(), self.ftffc)
        self.assertEqual(fiberTraceSet.size(), self.nFiberTraces)
        myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
        myProfileTask.run(fiberTraceSet)

        myExtractTask = esTask.ExtractSpectraTask()
        aperturesToExtract = [-1]
        spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
        self.assertEqual(spectrumSetFromProfile.size(), self.nFiberTraces)

        spectrum = spectrumSetFromProfile.getSpectrum(0)

        """ read line list """
        hdulist = pyfits.open(self.lineList)
        tbdata = hdulist[1].data
        lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float32')
        lineListArr[:,0] = tbdata.field(0)
        lineListArr[:,1] = tbdata.field(1)

        """ read reference Spectrum """
        hdulist = pyfits.open(self.refSpec)
        tbdata = hdulist[1].data
        refSpecArr = np.ndarray(shape=(len(tbdata)), dtype='float32')
        refSpecArr[:] = tbdata.field(0)

        spec = spectrum.getSpectrum()
        result = drpStella.stretchAndCrossCorrelateSpecFF(spec, refSpecArr, lineListArr, self.dispCorControl)

        # we're not holding back any emission lines from the check to make
        # sure the line we will disturb is not one of the lines held back
        spectrum.identifyF(result.lineList, self.dispCorControl, 0)
        dispRMSOrig = spectrum.getDispRms()

        """Find an emission line"""
        distances = []
        for i in np.arange(1,lineListArr.shape[0]-1):
            distances.append(min(lineListArr[i][1]-lineListArr[i-1][1],
                                 lineListArr[i+1][1]-lineListArr[i][1]))
        linePos = 1 + max(xrange(len(distances)), key=distances.__getitem__)
        wavelengths = abs(spectrum.getWavelength() - lineListArr[linePos][0])
        linePos = min(xrange(len(wavelengths)), key=wavelengths.__getitem__)

        """include 'cosmic' next to line"""
        spectrum.getSpectrum()[linePos:linePos+4] += [10000.,20000.,30000., 20000.]

        spectrum.identifyF(result.lineList, self.dispCorControl, 0)# we're not holding back any emission lines
        dispRMSCosmic = spectrum.getDispRms()
        self.assertNotAlmostEqual(dispRMSOrig, dispRMSCosmic)
        mask = spectrum.getMask()
        maskArr = mask.getArray()
        maskVal = 1 << mask.getMaskPlane("REJECTED_LINES");
        print 'maskVal = ',maskVal
        self.assertEqual(maskArr[0,linePos-2],0)
        self.assertEqual(maskArr[0,linePos+4],0)
        for i in np.arange(linePos-1,linePos+4):
            self.assertEqual(maskArr[0,i], maskVal)

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
