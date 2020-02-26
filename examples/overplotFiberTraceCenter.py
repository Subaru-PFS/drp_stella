#!/Users/azuri/anaconda/envs/lsst-v12_1/bin/python
from builtins import range
import os
import lsst.log as log
import lsst.utils
import lsst.daf.persistence as dafPersist
import pfs.drp.stella as drpStella

import lsst.afw.display as afwDisplay

# Silence verbose loggers
for logger in ["afw.ExposureFormatter",
               "afw.image.ExposureInfo",
               "afw.image.Mask",
               "CameraMapper",
               "daf.persistence.LogicalLocation",
               "daf.persistence.butler",
               "pfs.drp.stella.FiberTrace.calcProfile",
               "pfs.drp.stella.FiberTrace.calcProfileSwath",
               "pfs.drp.stella.FiberTrace.extractFromProfile",
               "pfs.drp.stella.math.CurveFitting.LinFitBevingtonNdArray1D",
               "pfs.drp.stella.math.CurveFitting.LinFitBevingtonNdArray2D",
               "pfs.drp.stella.math.CurveFittingPolyFit",
               ]:
    log.Log.getLogger(logger).setLevel(log.WARN)

display = afwDisplay.Display(1)

dataDir = os.path.join(lsst.utils.getPackageDir("drp_stella_data"), 'tests/data/PFS')
butler = dafPersist.Butler(dataDir)

# dataId for Flat
dataId = dict(visit=104, spectrograph=1, arm='r')

# get postISRCCD Flat exposure from butler
exposure = butler.get('postISRCCD', dataId)

# Trace apertures
fiberTraceFunctionFindingControl = drpStella.FiberTraceFunctionFindingControl()
fts = drpStella.findAndTraceApertures(exposure.getMaskedImage(),
                                      fiberTraceFunctionFindingControl)

# Mark FiberTrace in mask
maskPlane = "FIBERTRACE"
exposure.getMaskedImage().getMask().addMaskPlane(maskPlane)
display.setMaskPlaneColor(maskPlane, "GREEN")
ftMask = 1 << exposure.getMaskedImage().getMask().getMaskPlane(maskPlane)
for ft in fts.getTraces():
    drpStella.markFiberTraceInMask(ft, exposure.getMaskedImage().getMask(), ftMask)

# display image
display.setMaskTransparency(50)
display.mtv(exposure, title="parent")

# Mark FiberTrace centers in image
with display.Buffering():
    for ft in fts.getTraces():
        xCenters = ft.getXCenters()
        ftFunction = ft.getFiberTraceFunction()
        yLow = ftFunction.yCenter + ftFunction.yLow
        yHigh = ftFunction.yCenter + ftFunction.yHigh

        for y in range(yLow, yHigh):
            pointA = [xCenters[y-yLow], y]
            pointB = [xCenters[y-yLow+1], y+1]
            points = [pointA, pointB]
            display.line(points, ctype='red')
