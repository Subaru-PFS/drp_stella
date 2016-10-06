from pfs.datamodel.pfsFiberTrace import PfsFiberTrace
import pfs.drp.stella as drpStella

def makeFiberTraceSet(pfsFiberTrace, maskedImage=None):
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
        ftf.coefficients = pfsFiberTrace.coeffs[iFt]
        ft = drpStella.FiberTrace()
        if not ft.setFiberTraceFunction(ftf):
            raise RuntimeError("FiberTrace %d: Failed to set FiberTraceFunction" % iFt)
        if not ft.setFiberTraceProfileFittingControl(ftpfc):
            raise RuntimeError("FiberTrace %d: Failed to set FiberTraceProfileFittingControl" % iFt)
        if not ft.setITrace(pfsFiberTrace.fiberId[iFt]-1):
            raise RuntimeError("FiberTrace %d: Failed to set FiberId" % iFt)
        if maskedImage != None:
            if not ft.createTrace(maskedImage):
                raise RuntimeError("FiberTrace %d: Failed to create trace from maskedImage")
        profile = pfsFiberTrace.profiles[iFt]
        if not ft.setProfile(profile):
            raise RuntimeError("FiberTrace %d: Failed to set profile")
        fts.addFiberTrace(ft)
    return fts