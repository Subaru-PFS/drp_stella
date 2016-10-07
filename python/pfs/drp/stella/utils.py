#from pfs.datamodel.pfsFiberTrace import PfsFiberTrace
import pfs.drp.stella as drpStella
import numpy as np
import lsst.afw.image as afwImage

def makeFiberTraceSet(pfsFiberTrace, maskedImage=None):
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
    print 'ftfc.nRows set to ',ftfc.nRows

    ftpfc.profileInterpolation = pfsFiberTrace.interpol
#    ftpfc.ccdReadOutNoise = pfsFiberTrace.
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
        print 'pfsFiberTrace.coeffs[iFt] = ',pfsFiberTrace.coeffs[iFt]
        coeffs = np.ndarray(len(pfsFiberTrace.coeffs[iFt]), dtype=np.float64)
        for iCoeff in range(coeffs.shape[0]):
            coeffs[iCoeff] = pfsFiberTrace.coeffs[iFt][iCoeff]
        print 'coeffs = ',coeffs.shape,': ',coeffs
        ftf.coefficients = coeffs
        ft = drpStella.FiberTraceF()
        if not ft.setFiberTraceFunction(ftf):
            raise RuntimeError("FiberTrace %d: Failed to set FiberTraceFunction" % iFt)
        if not ft.setFiberTraceProfileFittingControl(ftpfc):
            raise RuntimeError("FiberTrace %d: Failed to set FiberTraceProfileFittingControl" % iFt)
        ft.setITrace(pfsFiberTrace.fiberId[iFt]-1)
        profile = pfsFiberTrace.profiles[iFt]
        trace = np.ndarray(shape=(ftf.yHigh - ftf.yLow + 1, profile.shape[1]), dtype=np.float32)
        if not ft.setTrace(afwImage.makeMaskedImage(afwImage.ImageF(trace))):
            raise RuntimeError("FiberTrace %d: Failed to set trace")
        prof = np.ndarray(shape=(ftf.yHigh - ftf.yLow + 1, profile.shape[1]), dtype=np.float64)
        for iRow in range(prof.shape[0]):
            prof[iRow,:] = profile[ftf.yCenter + ftf.yLow + iRow,:]
        if not ft.setProfile(afwImage.ImageD(prof)):
            raise RuntimeError("FiberTrace %d: Failed to set profile")
        xCenters = drpStella.calculateXCenters(ftf)
        ft.setXCenters(xCenters)
        if maskedImage != None:
            if not ft.createTrace(maskedImage):
                raise RuntimeError("FiberTrace %d: Failed to create trace from maskedImage")
        fts.addFiberTrace(ft)
    return fts
