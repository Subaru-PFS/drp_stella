from astropy.io import fits as pyfits
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.log as log
import numpy as np
import pfs.drp.stella as drpStella
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO

def makeFiberTraceSet(pfsFiberTrace):
    if pfsFiberTrace.traces is None or len(pfsFiberTrace.traces) == 0:
        raise RuntimeError("There are no fiberTraces in the PfsFiberTrace object")

    fts = drpStella.FiberTraceSet()

    for iFt in range(len(pfsFiberTrace.traces)):
        ft = drpStella.FiberTrace(pfsFiberTrace.traces[iFt],
                                  pfsFiberTrace.fiberId[iFt] - 1)

        fts.addFiberTrace(ft)
    return fts

def readWavelengthFile(wLenFile):
    """read wavelength file and return 1-D arrays of length nFibre*nwavelength

    These arrays are used by evaluating e.g. wavelengths[np.where(traceId == fid)]
    """
    hdulist = pyfits.open(wLenFile)
    tbdata = hdulist[1].data
    traceIds = tbdata[:]['fiberNum'].astype('int32')
    wavelengths = tbdata[:]['pixelWave'].astype('float32')
    xCenters = tbdata[:]['xc'].astype('float32')

    traceIdSet = np.unique(traceIds)
    assert len(wavelengths) == len(traceIds[traceIds == traceIdSet[0]])*len(traceIdSet) # could check all

    return [xCenters, wavelengths, traceIds]

def readLineListFile(lineList):
    """read line list"""
    hdulist = pyfits.open(lineList)
    tbdata = hdulist[1].data
    lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float32')
    lineListArr[:,0] = tbdata.field(0)  # wavelength
    lineListArr[:,1] = tbdata.field(1)  # ??
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

def addFiberTraceSetToMask(mask, fiberTraceSet):
    for ft in fiberTraceSet.getTraces():
        traceMask = ft.getTrace().mask
        if False:                       # requires w_2017_32 or later
            mask[traceMask.getBBox(), afwImage.PARENT] |= mask
        else:
            mask.Factory(mask, traceMask.getBBox(), afwImage.PARENT)[:] |= traceMask
