#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""
This module describes utils for the STELLA pipeline.

@author Andreas Ritter, Princeton University
"""
from __future__ import absolute_import, division, print_function
from astropy.io import fits as pyfits
import lsst.afw.image as afwImage
import lsst.log as log
import numpy as np
from pfs.datamodel.pfsFiberTrace import PfsFiberTrace as pFT
import pfs.drp.stella as drpStella
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO

def makeFiberTraceSet(pfsFiberTrace, maskedImage=None):
    """
    take a pfsFiberTrace and return a FiberTraceSet
    @param pfsFiberTrace : pfsFiberTrace from which to reconstruct the FiberTraceSet
    @param maskedImage : if given, take the FiberTrace definitions and create a new
                         FiberTraceSet from it
    """
    if pfsFiberTrace.profiles is None or len(pfsFiberTrace.profiles) == 0:
        raise RuntimeError("There are no fiberTraces in the PfsFiberTrace object")

    fts = drpStella.FiberTraceSet()
    ftfc = drpStella.FiberTraceFunctionControl()
    ftf = drpStella.FiberTraceFunction()
    ftpfc = drpStella.FiberTraceProfileFittingControl()

    ftfc.interpolation = pfsFiberTrace.traceFunction
    ftfc.order = pfsFiberTrace.order
    ftfc.xLow = pfsFiberTrace.xLow
    ftfc.xHigh = pfsFiberTrace.xHigh
    ftfc.nPixCutLeft = pfsFiberTrace.nCutLeft
    ftfc.nPixCutRight = pfsFiberTrace.nCutRight
    ftfc.nRows = pfsFiberTrace.profiles[0].shape[0]

    ftpfc.profileInterpolation = pfsFiberTrace.interpol
    ftpfc.swathWidth = pfsFiberTrace.swathLength
    ftpfc.telluric = 'NONE'
    ftpfc.overSample = pfsFiberTrace.overSample
    ftpfc.maxIterSF = pfsFiberTrace.maxIterSF
    ftpfc.maxIterSky = 0
    ftpfc.maxIterSig = pfsFiberTrace.maxIterSig
    ftpfc.lambdaSF = pfsFiberTrace.lambdaSF
    ftpfc.lambdaSP = pfsFiberTrace.lambdaSP
    ftpfc.wingSmoothFactor = pfsFiberTrace.lambdaWing
    ftpfc.lowerSigma = pfsFiberTrace.lSigma
    ftpfc.upperSigma = pfsFiberTrace.uSigma

    for iFt in range(pfsFiberTrace.xCenter.shape[0]):
        ftf.fiberTraceFunctionControl = ftfc
        ftf.xCenter = pfsFiberTrace.xCenter[iFt]
        ftf.yCenter = pfsFiberTrace.yCenter[iFt]
        ftf.yLow = pfsFiberTrace.yLow[iFt]
        ftf.yHigh = pfsFiberTrace.yHigh[iFt]

        coeffs = np.ndarray(len(pfsFiberTrace.coeffs[iFt]), dtype=np.float64)
        for iCoeff in range(coeffs.shape[0]):
            coeffs[iCoeff] = pfsFiberTrace.coeffs[iFt][iCoeff]
        ftf.coefficients = coeffs

        ft = drpStella.FiberTrace()
        ft.setFiberTraceFunction(ftf)
        ft.setFiberTraceProfileFittingControl(ftpfc)

        ft.setITrace(pfsFiberTrace.fiberId[iFt]-1)

        profile = pfsFiberTrace.profiles[iFt]

        yMin = ftf.yCenter + ftf.yLow
        prof = afwImage.ImageF(profile.shape[1], ftf.yHigh - ftf.yLow + 1)
        prof.getArray()[:] = profile[yMin : yMin + prof.getHeight()].astype(np.float64)

        pixelData = afwImage.MaskedImageF(profile.shape[1], ftf.yHigh - ftf.yLow + 1)
        pixelData[:] = np.nan

        ft.setTrace(pixelData)
        ft.setProfile(prof)

        xCenters = drpStella.calculateXCenters(ftf)
        ft.setXCenters(xCenters)

        if maskedImage != None:
            ft.createTrace(maskedImage)

        if ft.getTrace().getHeight() != ft.getProfile().getHeight():
            raise RuntimeError("FiberTrace %d: trace and profile have different heights" % (ft.getITrace()))
        if ft.getTrace().getWidth() != ft.getProfile().getWidth():
            raise RuntimeError("FiberTrace %d: trace and profile have different widths" % (ft.getITrace()))

        fts.addFiberTrace(ft)
    return fts

def createPfsFiberTrace(dataId, fiberTraceSet, nRows):
    """
    take a fiberTraceSet and dataId and return a pfsFiberTrace object
    @param dataId : dataId for pfsFiberTrace to be returned
    @param fiberTraceSet : FiberTraceSet to convert to a pfsFiberTrace object
    @param nRows : number of CCD rows in the postISRCCD image
    """
    pfsFiberTrace = pFT(dataId['calibDate'], dataId['spectrograph'], dataId['arm'])

    ftf = fiberTraceSet.getFiberTrace(0).getFiberTraceFunction()
    ftfc = ftf.fiberTraceFunctionControl

    ftpfc = fiberTraceSet.getFiberTrace(0).getFiberTraceProfileFittingControl()

    pfsFiberTrace.fwhm = 0.
    pfsFiberTrace.threshold = 0.
    pfsFiberTrace.nTerms = 0
    pfsFiberTrace.saturationLevel = 0.
    pfsFiberTrace.minLength = 0
    pfsFiberTrace.maxLength = 0
    pfsFiberTrace.nLost = 0
    pfsFiberTrace.traceFunction = ftfc.interpolation
    pfsFiberTrace.order = ftfc.order
    pfsFiberTrace.xLow = ftfc.xLow
    pfsFiberTrace.xHigh = ftfc.xHigh
    pfsFiberTrace.nCutLeft = ftfc.nPixCutLeft
    pfsFiberTrace.nCutRight = ftfc.nPixCutRight

    pfsFiberTrace.interpol = ftpfc.profileInterpolation
    pfsFiberTrace.swathLength = ftpfc.swathWidth
    pfsFiberTrace.overSample = ftpfc.overSample
    pfsFiberTrace.maxIterSF = ftpfc.maxIterSF
    pfsFiberTrace.maxIterSig = ftpfc.maxIterSig
    pfsFiberTrace.lambdaSF = ftpfc.lambdaSF
    pfsFiberTrace.lambdaSP = ftpfc.lambdaSP
    pfsFiberTrace.lambdaWing = ftpfc.wingSmoothFactor
    pfsFiberTrace.lSigma = ftpfc.lowerSigma
    pfsFiberTrace.uSigma = ftpfc.upperSigma

    for iFt in range(fiberTraceSet.size()):
        ft = fiberTraceSet.getFiberTrace(iFt)
        pfsFiberTrace.fiberId.append(ft.getITrace()+1)
        ftf = ft.getFiberTraceFunction()
        pfsFiberTrace.xCenter.append(ftf.xCenter)
        pfsFiberTrace.yCenter.append(ftf.yCenter)
        pfsFiberTrace.yLow.append(ftf.yLow)
        pfsFiberTrace.yHigh.append(ftf.yHigh)
        pfsFiberTrace.coeffs.append(ftf.coefficients)
        prof = ft.getProfile()
        profOut = np.zeros(shape=[nRows,prof.getWidth()], dtype=np.float32)
        profOut[ftf.yCenter + ftf.yLow:ftf.yCenter + ftf.yHigh+1,:] = prof.getArray()[:,:]
        pfsFiberTrace.profiles.append(profOut)

    return pfsFiberTrace

def readWavelengthFile(wLenFile):
    """
    read wavelength file containing the predicted wavelengths per pixel
    @param wLenFile : name of file to read
    @return : dictionary{xCenters, yCenters, wavelengths, traceIds}
    """
    hdulist = pyfits.open(wLenFile)
    tbdata = hdulist[1].data
    traceIdsTemp = np.ndarray(shape=(len(tbdata)), dtype='int')
    xCenters = np.ndarray(shape=(len(tbdata)), dtype='float32')
    wavelengths = np.ndarray(shape=(len(tbdata)), dtype='float32')
    traceIdsTemp[:] = tbdata[:]['fiberNum']
    traceIds = traceIdsTemp.astype('int32')
    wavelengths[:] = tbdata[:]['pixelWave']
    xCenters[:] = tbdata[:]['xc']
    return [xCenters, wavelengths, traceIds]

def readLineListFile(lineList):
    """
    read line list
    @param lineList : name of file to read
    @return : ndarray of shape nLines x 2, [0: wavelength, 1: pixel]
    """
    hdulist = pyfits.open(lineList)
    tbdata = hdulist[1].data
    lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float32')
    lineListArr[:,0] = tbdata.field('wavelength')
    lineListArr[:,1] = tbdata.field('pixel')
    return lineListArr

def readReferenceSpectrum(refSpec):
    """
    read reference Spectrum
    @param refSpec : name of reference spectrum to read
    @return : flux ndarray of length nPixels
    """
    hdulist = pyfits.open(refSpec)
    tbdata = hdulist[1].data
    refSpecArr = np.ndarray(shape=(len(tbdata)), dtype='float32')
    refSpecArr[:] = tbdata.field('flux')
    return refSpecArr

def writePfsArm(butler, arcExposure, spectrumSet, dataId):
    """
    Do the I/O using a trampoline object PfsArmIO (to avoid adding butler-related details
    to the datamodel product)

    This is a bit messy as we need to include the pfsConfig file in the pfsArm file
    @param butler : butler to use
    @param arcExposure : Exposure object containing an Arc
    @param spectrumSet : SpectrumSet to convert to a pfsArm object
    @param dataId : Data ID for pfsArm object
    """
    md = arcExposure.getMetadata().toDict()
    key = "PFSCONFIGID"
    if key in md:
        pfsConfigId = md[key]
    else:
        log.log("writePfsArm",
                log.WARN,
                'No pfsConfigId is present in postISRCCD file for dataId %s' %
                str(dataId.items()))
        pfsConfigId = 0x0

    pfsConfig = butler.get("pfsConfig", pfsConfigId=pfsConfigId, dateObs=dataId["dateObs"])

    pfsArm = spectrumSetToPfsArm(pfsConfig, spectrumSet,
                                 dataId["visit"], dataId["spectrograph"], dataId["arm"])
    butler.put(PfsArmIO(pfsArm), 'pfsArm', dataId)

def addFiberTraceSetToMask(mask, fiberTraceSet, display=None, maskPlaneName="FIBERTRACE"):
    mask.addMaskPlane(maskPlaneName)

    ftMask = mask.getPlaneBitMask(maskPlaneName)
    for ft in fiberTraceSet.getTraces():
        drpStella.markFiberTraceInMask(ft, mask, ftMask)

    if display:
        display.setMaskPlaneColor(maskPlaneName, "GREEN")
