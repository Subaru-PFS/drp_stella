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
import sys
import unittest

import numpy as np

import lsst.afw.image as afwImage
import lsst.daf.persistence as dafPersist
import lsst.log as log
import lsst.utils
import lsst.utils.tests as tests
import pfs.drp.stella as drpStella
import pfs.drp.stella.createFlatFiberTraceProfileTask as cfftpTask
import pfs.drp.stella.extractSpectraTask as esTask
import pfs.drp.stella.findAndTraceAperturesTask as fataTask
import pfs.drp.stella.math as drpStellaMath
from pfs.drp.stella.reduceArcRefSpecTask import ReduceArcRefSpecTask
from pfs.drp.stella.reduceArcTask import ReduceArcTask
from pfs.drp.stella.utils import readLineListFile, readReferenceSpectrum
from pfs.drp.stella.utils import readWavelengthFile

class SpectraTestCase(tests.TestCase):
    """A test case for measuring Spectra quantities"""

    def setUp(self):
        drpStellaDataDir = lsst.utils.getPackageDir("drp_stella_data")
        self.butler = dafPersist.Butler(os.path.join(drpStellaDataDir,"tests/data/PFS/"))
        self.dataIdFlat = dict(field="FLAT", visit=104, spectrograph=1, arm="r")
        self.flat = self.butler.get("postISRCCD", self.dataIdFlat, immediate=True)

        self.dataIdArc = dict(field="ARC", visit=103, spectrograph=1, arm="r")
        self.arc = self.butler.get("postISRCCD", self.dataIdArc, immediate=True)

        self.ftffc = drpStella.FiberTraceFunctionFindingControl()
        self.ftffc.fiberTraceFunctionControl.order = 5
        self.ftffc.fiberTraceFunctionControl.xLow = -5
        self.ftffc.fiberTraceFunctionControl.xHigh = 5

        self.dispCorControl = drpStella.DispCorControl()
        self.elements = 'Cd,Hg,Kr,Ne,Xe'

        self.nFiberTraces = 11
        self.nRowsPrescan = 49
        self.minPercentageOfLines = 66.7

        # These values are measured values which are otherwise poorly justified.
        # They serve as a regression test to make sure that changes in the code
        # didn't make it worse.
        self.maxRMS = 0.019
        self.maxRMSCheck = 0.18

        self.lineList = os.path.join(lsst.utils.getPackageDir('obs_pfs'),'pfs/lineLists/CdHgKrNeXe_red.fits')
        self.masterLineList = os.path.join(lsst.utils.getPackageDir('obs_pfs'),'pfs/lineLists/NeXeHgAr_1r.fits')
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
        del self.elements
        del self.wLenFile
        del self.masterLineList
        del self.maxRMS
        del self.maxRMSCheck

    def testSpectrumConstructors(self):
        # Test that we can create a Spectrum with the standard constructor
        spec = drpStella.Spectrum()
        self.assertEqual(spec.getLength(), 0)
        self.assertEqual(spec.getITrace(), 0)

        length = 10
        iTrace = 2
        spec = drpStella.Spectrum(length, iTrace)
        self.assertEqual(spec.getLength(), length)
        self.assertEqual(spec.getITrace(), iTrace)

        # Test copy constructor
        specCopy = drpStella.Spectrum(spec)
        self.assertEqual(specCopy.getLength(), length)
        self.assertEqual(specCopy.getITrace(), iTrace)

    def testSpectrumMethods(self):
        # Test getSpectrum
        size = 100
        spec = drpStella.Spectrum(size)
        vec = spec.getSpectrum()
        self.assertEqual(vec.shape[0], size)
        self.assertEqual(spec.getSpectrum().shape[0], size)

        # Test setSpectrum
        # Test that we can assign a spectrum of the correct length
        vecf = np.arange(size, dtype=np.float32)
        spec.setSpectrum(vecf)
        self.assertEqual(spec.getSpectrum()[3], vecf[3])

        # Test that we can't assign a spectrum of the wrong length
        vecf = np.arange(size + 1, dtype=np.float32)
        try:
            spec.setSpectrum(vecf)
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "pfs::drp::stella::Spectrum::setSpectrum: ERROR: spectrum->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
            self.assertEqual(message[0],expected)
        self.assertEqual(spec.getSpectrum().shape[0], size)

        # Test getVariance
        vec = spec.getVariance()
        self.assertEqual(vec.shape[0], size)

        # Test setVariance
        # Test that we can assign a variance vector of the correct length
        vecf = np.arange(size, dtype=np.float32)
        spec.setVariance(vecf)
        self.assertEqual(spec.getVariance()[3], vecf[3])

        # Test that we can't assign a variance vector of the wrong length
        vecf = np.arange(size + 1, dtype=np.float32)
        try:
            self.assertFalse(spec.setVariance(vecf))
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "pfs::drp::stella::Spectrum::setVariance: ERROR: variance->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
            self.assertEqual(message[0],expected)
        self.assertEqual(spec.getVariance().shape[0], size)

        # Test getWavelength
        vec = spec.getWavelength()
        self.assertEqual(vec.shape[0], size)

        # Test setWavelength
        # Test that we can assign a wavelength vector of the correct length
        vecf = np.arange(size, dtype=np.float32)
        spec.setWavelength(vecf)
        self.assertEqual(spec.getWavelength()[3], vecf[3])

        # Test that we can't assign a wavelength vector of the wrong length
        vecf = np.arange(size + 1, dtype=np.float32)
        try:
            self.assertFalse(spec.setWavelength(vecf))
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "pfsDRPStella::Spectrum::setWavelength: ERROR: wavelength->size()="+str(vecf.shape[0])+" != _length="+str(spec.getLength())
            self.assertEqual(message[0],expected)
        self.assertEqual(spec.getWavelength().shape[0], size)

        # Test getMask
        vec = spec.getMask()
        self.assertEqual(vec.getWidth(), size)

        # Test setMask
        # Test that we can assign a mask vector of the correct length
        vecf = afwImage.Mask(size, 1)
        spec.setMask(vecf)

        # Test that we can't assign a mask vector of the wrong length
        vecus = afwImage.Mask(size+1, 1)
        try:
            self.assertFalse(spec.setMask(vecus))
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "pfs::drp::stella::Spectrum::setMask: ERROR: mask.getWidth()="+str(vecus.getWidth())+" != _length="+str(spec.getLength())
            self.assertEqual(message[0],expected)
        self.assertEqual(spec.getMask().getWidth(), size)

        # Test setLength
        # If newLength < oldLength, vectors are supposed to be cut off,
        # otherwise ZEROs are appended to the end of the vectors (last
        # wavelength value for wavelength vector)
        # Test same size
        vecf = np.arange(size, dtype=np.float32)
        vecus = np.arange(size, dtype=np.uint32)
        spec.setLength(size)
        self.assertEqual(spec.getLength(), size)
        self.assertEqual(spec.getSpectrum()[size-1], vecf[size-1])
        self.assertEqual(spec.getSpectrum().shape[0], size)
        self.assertEqual(spec.getVariance().shape[0], size)
        self.assertEqual(spec.getVariance()[size-1], vecf[size-1])
        self.assertEqual(spec.getMask().getWidth(), size)
        self.assertEqual(spec.getWavelength().shape[0], size)
        self.assertEqual(spec.getWavelength()[size-1], vecf[size-1])

        # Test longer size
        spec.setLength(size+1)
        self.assertEqual(spec.getLength(), size+1)
        self.assertEqual(spec.getSpectrum()[size], 0)
        self.assertEqual(spec.getSpectrum().shape[0], size+1)
        self.assertEqual(spec.getVariance().shape[0], size+1)
        self.assertEqual(spec.getVariance()[size], 0)
        self.assertEqual(spec.getMask().getWidth(), size+1)
        self.assertEqual(spec.getWavelength().shape[0], size+1)
        self.assertAlmostEqual(spec.getWavelength()[size], 0.)

        # Test shorter size
        spec.setLength(size-1)
        self.assertEqual(spec.getLength(), size-1)
        self.assertEqual(spec.getSpectrum()[size-2], vecf[size-2])
        self.assertEqual(spec.getSpectrum().shape[0], size-1)
        self.assertEqual(spec.getVariance().shape[0], size-1)
        self.assertEqual(spec.getVariance()[size-2], vecf[size-2])
        self.assertEqual(spec.getMask().getWidth(), size-1)
        self.assertEqual(spec.getWavelength().shape[0], size-1)
        self.assertEqual(spec.getWavelength()[size-2], vecf[size-2])

        # Test get/setITrace
        self.assertEqual(spec.getITrace(), 0)
        spec.setITrace(10)
        self.assertEqual(spec.getITrace(), 10)

        # Test isWaveLengthSet
        self.assertFalse(spec.isWavelengthSet())

    def testSpectrumSetConstructors(self):
        # Test SpectrumSetConstructors
        # Test Standard Constructor
        specSet = drpStella.SpectrumSet()
        self.assertEqual(specSet.size(), 0)

        size = 3
        specSet = drpStella.SpectrumSet(size)
        self.assertEqual(specSet.size(), size)
        for i in range(size):
            self.assertEqual(specSet.getSpectrum(i).getSpectrum().shape[0], 0)

        length = 33
        specSet = drpStella.SpectrumSet(size, length)
        self.assertEqual(specSet.size(), size)
        for i in range(size):
            self.assertEqual(specSet.getSpectrum(i).getSpectrum().shape[0], length)

        # Test copy constructor
        specSetCopy = drpStella.SpectrumSet(specSet)
        self.assertEqual(specSetCopy.size(), specSet.size())
        for i in range(specSet.size()):
            self.assertEqual(specSetCopy.getSpectrum(i).getLength(), specSet.getSpectrum(i).getLength())

        # Test constructor from vector of spectra
        specSetV = drpStella.SpectrumSet(specSet.getSpectra())
        self.assertEqual(specSet.size(), specSetV.size())
        for i in range(specSet.size()):
            self.assertEqual(specSetV.getSpectrum(i).getITrace(), i)

    def testExtractTask(self):
        fiberTraceSet = drpStella.findAndTraceApertures(self.flat.getMaskedImage(), self.ftffc)

        # read wavelength file
        xCenters, wavelengths, traceIds = readWavelengthFile(self.wLenFile)

        # assign trace number to fiberTraceSet
        drpStella.assignITrace( fiberTraceSet, traceIds, xCenters )

        self.assertEqual(fiberTraceSet.size(), self.nFiberTraces)
        myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
        myProfileTask.run(fiberTraceSet)

        myExtractTask = esTask.ExtractSpectraTask()

        # test that we can extract all FiberTraces
        aperturesToExtract = [-1]
        spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
        self.assertEqual(spectrumSetFromProfile.size(), self.nFiberTraces)
        for i in range(spectrumSetFromProfile.size()):
            self.assertEqual(spectrumSetFromProfile.getSpectrum(i).getLength(), fiberTraceSet.getFiberTrace(i).getHeight())
            self.assertEqual(spectrumSetFromProfile.getSpectrum(i).getITrace(),
                             fiberTraceSet.getFiberTrace(i).getITrace())

        # test that we can extract individual FiberTraces
        for i in range(fiberTraceSet.size()):
            aperturesToExtract = [i]
            spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
            self.assertEqual(spectrumSetFromProfile.size(), 1)
            self.assertEqual(spectrumSetFromProfile.getSpectrum(0).getLength(), fiberTraceSet.getFiberTrace(i).getHeight())
            self.assertEqual(spectrumSetFromProfile.getSpectrum(0).getITrace(),
                             fiberTraceSet.getFiberTrace(i).getITrace())

        # test that we can extract 2 individual FiberTraces
        for i in range(fiberTraceSet.size()-1):
            aperturesToExtract = [i, i+1]
            spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
            self.assertEqual(spectrumSetFromProfile.size(), 2)
            self.assertEqual(spectrumSetFromProfile.getSpectrum(0).getLength(), fiberTraceSet.getFiberTrace(i).getHeight())
            self.assertEqual(spectrumSetFromProfile.getSpectrum(1).getLength(), fiberTraceSet.getFiberTrace(i+1).getHeight())
            self.assertEqual(spectrumSetFromProfile.getSpectrum(0).getITrace(),
                             fiberTraceSet.getFiberTrace(i).getITrace())
            self.assertEqual(spectrumSetFromProfile.getSpectrum(1).getITrace(),
                             fiberTraceSet.getFiberTrace(i+1).getITrace())

    def testSpectrumSetAddSetErase(self):
        size = 3
        length = 100
        specSet = drpStella.SpectrumSet(size, length)
        spec = drpStella.Spectrum(length)
        specNew = drpStella.Spectrum(length+1)

        # Test that we cannot set a spectrum outside the limits 0 <= pos <= size
        try:
            specSet.setSpectrum(-1, specNew)
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "setSpectrum(): incompatible function arguments." \
                       " The following argument types are supported:"
            self.assertEqual(message[0],expected)
        try:
            specSet.setSpectrum(size+1, specNew)
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "SpectrumSet::setSpectrum(i="+str(size+1)+"): ERROR: i > _spectra->size()="+str(size)
            self.assertEqual(message[0],expected)

        # Test that we can set/add a spectrum
        specSet.setSpectrum(size-1, specNew)
        self.assertEqual(specSet.size(), size)
        specSet.setSpectrum(size, specNew)
        self.assertEqual(specSet.size(), size+1)

        specSet.addSpectrum(specNew)
        self.assertEqual(specSet.size(), size+2)

        # Test that we can't erase spectra outside the limits
        size = specSet.size()
        try:
            specSet.erase(size)
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "SpectrumSet::erase(iStart="+str(size)+", iEnd=0): ERROR: iStart >= _spectra->size()="+str(size)
            self.assertEqual(message[0],expected)

        try:
            specSet.erase(2, 1)
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "SpectrumSet::erase(iStart=2, iEnd=1): ERROR: iStart > iEnd"
            self.assertEqual(message[0],expected)

        try:
            specSet.erase(-1)
        except Exception as e:
            message = str.split(str(e.message), "\n")
            expected = "erase(): incompatible function arguments. The following argument types are supported:"
            self.assertEqual(message[0],expected)

        # Test that we CAN erase spectra inside the limits
        self.assertTrue(specSet.erase(size-1, size))
        self.assertEqual(specSet.size(), size-1)
        self.assertTrue(specSet.erase(size-2))
        self.assertEqual(specSet.size(), size-2)

        self.assertTrue(specSet.erase(0, 1))
        self.assertEqual(specSet.size(), size-3)

        self.assertTrue(specSet.erase(0,2))
        self.assertEqual(specSet.size(), size-5)

    def testGetSpectra(self):
        """test getSpectra"""
        size = 3
        length = 100
        specSet = drpStella.SpectrumSet(size,length)
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

            # to avoid division by zero and effectively multiplying the
            # original values with a large number, we set the ratio to
            # unity where the reconstructed values are smaller than 0.1
            ratio = orig / np.where(rec < 0.1, 1., rec)

            binWidth = 1. / (myProfileTask.fiberTraceProfileFittingControl.overSample
                             * nBinsPerOverSampleStep)
            bins, mean, std = drpStellaMath.getMeanStdXBins(dist,
                                                            ratio,
                                                            binWidth)
            for iBin in range(bins.shape[0]):
                self.assertTrue(np.all(np.fabs(mean - 1.0) < std))

    def testWavelengthCalibrationWithRefSpec(self):
        myReduceArcTask = ReduceArcRefSpecTask()
        dataRefList = [ref for ref in self.butler.subset('postISRCCD',
                                                         'visit',
                                                         self.dataIdArc)]
        spectrumSetFromProfile = myReduceArcTask.run(
            dataRefList,
            self.butler,
            refSpec=self.refSpec,
            lineList=self.lineList
        )

        for i in range(spectrumSetFromProfile.size()):
            spec = spectrumSetFromProfile.getSpectrum(i)

            # Check that wavelength solution is monotonic
            wavelength = spec.getWavelength()
            for j in range(spec.getLength()-1):
                self.assertLess(wavelength[j], wavelength[j+1])

            # Check wavelength range
            self.assertGreater(wavelength[0], 380)
            self.assertLess(wavelength[spec.getLength()-1], 980)

            # Check RMS
            self.assertLess(spec.getDispRms(), self.maxRMS)
            self.assertLess(spec.getDispRmsCheck(), self.maxRMSCheck)

    def testWavelengthCalibrationWithoutRefSpec(self):
        myReduceArcTask = ReduceArcTask()
        myReduceArcTask.config.elements = self.elements
        dataRefList = [ref for ref in self.butler.subset("postISRCCD", 'visit', self.dataIdArc)]

        spectrumSetFromProfile = myReduceArcTask.run(dataRefList,
                                                     self.butler,
                                                     self.wLenFile)[0]

        for i in range(spectrumSetFromProfile.size()):
            spec = spectrumSetFromProfile.getSpectrum(i)

            # Check that wavelength solution is monotonic
            wavelength = spec.getWavelength()
            for j in range(spec.getLength()-1):
                self.assertLess(wavelength[j], wavelength[j+1])

            # Check wavelength range
            self.assertGreater(wavelength[0], 380)
            self.assertLess(wavelength[spec.getLength()-1], 980)

            # Check RMS
            self.assertLess(spec.getDispRms(), self.maxRMS)
            self.assertLess(spec.getDispRmsCheck(), self.maxRMSCheck)

    def testPolyFit(self):
        # This test is an integration test for <PolyFit> called by <identify>
        # We will disturb one line and then test that <PolyFit> properly
        # identified the line as outlier and rejected it from the fit

        fiberTraceSet = drpStella.findAndTraceApertures(self.flat.getMaskedImage(), self.ftffc)
        self.assertEqual(fiberTraceSet.size(), self.nFiberTraces)
        myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
        myProfileTask.run(fiberTraceSet)

        myExtractTask = esTask.ExtractSpectraTask()
        aperturesToExtract = [-1]
        spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
        self.assertEqual(spectrumSetFromProfile.size(), self.nFiberTraces)

        spectrum = spectrumSetFromProfile.getSpectrum(0)

        # read line list
        lineListArr = readLineListFile(self.lineList)

        # read reference Spectrum
        refSpecArr = readReferenceSpectrum(self.refSpec)

        spec = spectrum.getSpectrum()
        result = drpStella.stretchAndCrossCorrelateSpec(spec, refSpecArr, lineListArr, self.dispCorControl)

        # we're not holding back any emission lines from the check to make
        # sure the line we will disturb is not one of the lines held back
        self.dispCorControl.percentageOfLinesForCheck = 0
        spectrum.identify(drpStella.createLineListFromWLenPix(result.lineList),
                           self.dispCorControl)
        dispRMSOrig = spectrum.getDispRms()

        # Find an emission line which we can disturb to test that it is
        # identified as problematic and rejected by PolyFit
        distances = []
        for i in np.arange(1,lineListArr.shape[0]-1):
            distances.append(min(lineListArr[i][1]-lineListArr[i-1][1],
                                 lineListArr[i+1][1]-lineListArr[i][1]))
        linePos = 1 + max(xrange(len(distances)), key=distances.__getitem__)
        wavelengths = abs(spectrum.getWavelength() - lineListArr[linePos][0])
        linePos = min(xrange(len(wavelengths)), key=wavelengths.__getitem__)

        # include 'cosmic' next to line
        spectrum.getSpectrum()[linePos:linePos+4] += [10000.,20000.,30000., 20000.]

        #run identify again
        spectrum.identify(drpStella.createLineListFromWLenPix(result.lineList),
                           self.dispCorControl)# we're not holding back any emission lines
        dispRMSCosmic = spectrum.getDispRms()
        self.assertNotAlmostEqual(dispRMSOrig, dispRMSCosmic)
        mask = spectrum.getMask()
        maskArr = mask.getArray()
        maskVal = 1 << mask.getMaskPlane("REJECTED_LINES");
        self.assertEqual(maskArr[0,linePos-2],0)
        self.assertEqual(maskArr[0,linePos+4],0)
        for i in np.arange(linePos-1,linePos+4):
            self.assertEqual(maskArr[0,i], maskVal)

    def testMaxDistance(self):
        # The values we are comparing the RMS of the lines used for the wavelength
        # calibration and the RMS of the lines held back from the calibration
        # procedure to are solely measured values which are otherwise poorly
        # justified. They serve as a regression test to make sure that changes in the code
        # didn't make it worse.

        fiberTraceSet = drpStella.findAndTraceApertures(self.flat.getMaskedImage(), self.ftffc)
        self.assertGreater(fiberTraceSet.size(), 0)
        myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
        myProfileTask.run(fiberTraceSet)

        # read wavelength file
        xCenters, wavelengths, traceIds = readWavelengthFile(self.wLenFile)

        traceIdsUnique = np.unique(traceIds)
        nRows = traceIds.shape[0] / traceIdsUnique.shape[0]

        # assign trace number to fiberTraceSet
        drpStella.assignITrace( fiberTraceSet, traceIds, xCenters )
        iTraces = np.ndarray(shape=fiberTraceSet.size(), dtype='intp')
        for i in range( fiberTraceSet.size() ):
            iTraces[i] = fiberTraceSet.getFiberTrace(i).getITrace()

        myExtractTask = esTask.ExtractSpectraTask()
        aperturesToExtract = [0]
        spectrumSetFromProfile = myExtractTask.run(self.arc, fiberTraceSet, aperturesToExtract)
        self.assertEqual(spectrumSetFromProfile.size(), 1)

        self.dispCorControl.fittingFunction = "POLYNOMIAL"
        self.dispCorControl.order = 5
        self.dispCorControl.searchRadius = 2
        self.dispCorControl.fwhm = 2.6
        self.dispCorControl.minPercentageOfLines = self.minPercentageOfLines

        # read line list
        lineListArr = readLineListFile(self.lineList)

        for i in range(spectrumSetFromProfile.size()):
            spec = spectrumSetFromProfile.getSpectrum(i)
            traceId = spec.getITrace()

            wLenTemp = wavelengths[np.where(traceIds == traceId)]
            self.assertEqual(wLenTemp.shape[0], nRows)

            #cut off both ends of wavelengths where is no signal
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
            nLines = lineListPix.shape[0]
            minGoodLines = int(nLines * self.minPercentageOfLines / 100.0)

            maxLines = 0
            for maxDistance in np.arange(0.2,1.5,0.1):
                self.dispCorControl.maxDistance = maxDistance
                if maxDistance < 0.49:
                    try:
                        spec.identify(drpStella.createLineListFromWLenPix(lineListPix),
                                       self.dispCorControl)
                        self.assertTrue(False) # i.e. the previous line should raise an exception
                    except Exception as e:
                        message = str.split(str(e.message), "\n")
                        # the number minGoodLines is equal to
                        # nLines * self.minPercentageOfLines / 100, which is the
                        # minimum number of lines required for a successful
                        # wavelength calibration
                        expected = ("identify: ERROR: less than %d lines identified"
                                    % (minGoodLines))
                        self.assertEqual(message[0],expected)
                else:
                    spec.identify(drpStella.createLineListFromWLenPix(lineListPix),
                                   self.dispCorControl)
                    # make sure that the number of 'good' lines increases with
                    # a growing maxDistance
                    self.assertGreaterEqual(spec.getNGoodLines(),maxLines)
                    if spec.getNGoodLines() > maxLines:
                        maxLines = spec.getNGoodLines()

                    self.assertLess(spec.getDispRms(), self.maxRMS)
                    self.assertLess(spec.getDispRmsCheck(), self.maxRMSCheck)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(SpectraTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    #Quiet down loggers which are too verbose
    for logger in ["afw.image.ExposureInfo",
                   "CameraMapper",
                   "extractSpectra",
                   "pfs.drp.stella.math.assignITrace",
                   "pfs.drp.stella.math.findAndTraceApertures",
                   "pfs.drp.stella.FiberTrace.calcProfile",
                   "pfs.drp.stella.FiberTrace.calcProfileSwath",
                   "pfs.drp.stella.FiberTrace.createTrace",
                   "pfs.drp.stella.math.ccdToFiberTraceCoordinates",
                   "pfs.drp.stella.math.CurveFitting.LinFitBevingtonNdArray1D",
                   "pfs.drp.stella.math.CurveFitting.LinFitBevingtonNdArray2D",
                   "pfs::drp::stella::math::psfCoordinatesRelativeTo",
                   "pfs::drp::stella::PSF::extractPSFs",
                   "pfs::drp::stella::PSF::extractPSFFromCenterPosition",
                   "pfs::drp::stella::PSF::extractPSFFromCenterPositions",
                   "reduceArcTask",
                   "reduceArcRefSpecTask",
                   ]:
        log.setLevel(logger, log.WARN)

    for logger in ["extractSpectra",
                   "gaussFunc",
                   "pfs.drp.stella.Spectra.hIdentify",
                   "pfs.drp.stella.Spectra.identify",
                   "pfs.drp.stella.math.CurfFitting.PolyFit",
                   "writePfsArm"
                   ]:
        log.setLevel(logger, log.FATAL)

    #Run the tests
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
