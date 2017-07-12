from astropy.io import fits as pyfits
import lsst.afw.image as afwImage
import lsst.log as log
import numpy as np
from pfs.datamodel.pfsFiberTrace import PfsFiberTrace as pFT
import pfs.drp.stella as drpStella
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO

def makeFiberTraceSet(pfsFiberTrace, maskedImage=None):
    if pfsFiberTrace.profiles is None or len(pfsFiberTrace.profiles) == 0:
        raise RuntimeError("There are no fiberTraces in the PfsFiberTrace object")

    fts = drpStella.FiberTraceSetF()
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

        ft = drpStella.FiberTraceF()
        if not ft.setFiberTraceFunction(ftf):
            raise RuntimeError("FiberTrace %d: Failed to set FiberTraceFunction" % iFt)
        if not ft.setFiberTraceProfileFittingControl(ftpfc):
            raise RuntimeError("FiberTrace %d: Failed to set FiberTraceProfileFittingControl" % iFt)

        ft.setITrace(pfsFiberTrace.fiberId[iFt]-1)

        profile = pfsFiberTrace.profiles[iFt]

        yMin = ftf.yCenter + ftf.yLow
        prof = afwImage.ImageD(profile.shape[1], ftf.yHigh - ftf.yLow + 1)
        prof.getArray()[:] = profile[yMin : yMin + prof.getHeight()].astype(np.float64)

        pixelData = afwImage.MaskedImageF(profile.shape[1], ftf.yHigh - ftf.yLow + 1)
        pixelData[:] = np.nan
        
        if not ft.setTrace(pixelData):
            raise RuntimeError("FiberTrace %d: Failed to set trace")
        if not ft.setProfile(prof):
            raise RuntimeError("FiberTrace %d: Failed to set profile")

        xCenters = drpStella.calculateXCenters(ftf)
        ft.setXCenters(xCenters)

        if maskedImage != None:
            if not ft.createTrace(maskedImage):
                raise RuntimeError("FiberTrace %d: Failed to create trace from maskedImage")

        if ft.getImage().getHeight() != ft.getProfile().getHeight():
            raise RuntimeError("FiberTrace %d: trace and profile have different sizes")
        if ft.getImage().getWidth() != ft.getProfile().getWidth():
            raise RuntimeError("FiberTrace %d: trace and profile have different sizes")

        fts.addFiberTrace(ft)
    return fts

def createPfsFiberTrace(dataId, fiberTraceSet, nRows):
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
    """read wavelength file"""
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
    """read line list"""
    hdulist = pyfits.open(lineList)
    tbdata = hdulist[1].data
    lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float32')
    lineListArr[:,0] = tbdata.field(0)
    lineListArr[:,1] = tbdata.field(1)
    return lineListArr

def readReferenceSpectrum(refSpec):
    """read reference Spectrum"""
    hdulist = pyfits.open(refSpec)
    tbdata = hdulist[1].data
    refSpecArr = np.ndarray(shape=(len(tbdata)), dtype='float32')
    refSpecArr[:] = tbdata.field(0)
    return refSpecArr

def writePfsArm(butler, arcExposure, spectrumSet, dataId):
    """
    Do the I/O using a trampoline object PfsArmIO (to avoid adding butler-related details
    to the datamodel product)

    This is a bit messy as we need to include the pfsConfig file in the pfsArm file
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
