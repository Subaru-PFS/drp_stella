import os
import re
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.log as log
import numpy as np
import pfs.drp.stella as drpStella
import pfs.drp.stella.detectorMap as detectorMap
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO

def makeFiberTraceSet(pfsFiberTrace):
    if pfsFiberTrace.traces is None or len(pfsFiberTrace.traces) == 0:
        raise RuntimeError("There are no fiberTraces in the PfsFiberTrace object")

    fts = drpStella.FiberTraceSet()

    for iFt in range(len(pfsFiberTrace.traces)):
        ft = drpStella.FiberTrace(pfsFiberTrace.traces[iFt], pfsFiberTrace.fiberId[iFt])

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

def makeDetectorMap(butler, dataId, wLenFile, nKnot=25):
    """Return a DetectorMap from the specified file

    N.b. we need to get this from the butler in the longer run
    """
    
    xCenters, wavelengths, fiberIds = readWavelengthFile(wLenFile)
    #
    # N.b. Andreas calls these "traceIds" but I'm assuming that they are actually be 1-indexed fiberIds
    #
    nFiber = len(set(fiberIds))
    fiberIds = fiberIds.reshape(nFiber, len(fiberIds)//nFiber)
    xCenters = xCenters.reshape(fiberIds.shape)
    wavelengths = wavelengths.reshape(fiberIds.shape)
    fiberIds = fiberIds.reshape(fiberIds.shape)[:, 0].copy()
    #
    # The RedFiberPixels.fits.gz file has an extra 48 pixels at the bottom of each row
    # (corresponding to the pixels that taper to the serials and which we trip off in the ISR)
    #
    nRowsPrescan = 48
    xCenters = xCenters[:, nRowsPrescan:]
    wavelengths = wavelengths[:, nRowsPrescan:]

    missing = (wavelengths == 0)
    xCenters[missing] = np.nan
    wavelengths[missing] = np.nan

    detector = butler.get('raw_detector', dataId)
    bbox = detector.getBBox()

    return detectorMap.DetectorMap(bbox, fiberIds, xCenters, wavelengths, nKnot=nKnot)

def readLineListFile(lineList, lamps=["Ar", "Cd", "Hg", "Ne", "Xe"], minIntensity=0):
    """Read line list

    Return:
       list of drp::ReferenceLine

    This file is basically CdHgKrNeXe_use
    """
    try:
        hdulist = pyfits.open(lineList)
    except IOError:
        hdulist = None

    if hdulist:
        tbdata = hdulist[1].data
        
        wavelength = tbdata.field(1)
        element = tbdata.field(2)
        intensity = tbdata.field(3)         # Comment (intensity + notes)
        tmp = np.empty(len(intensity))
        for i in range(len(intensity)):
            tmp[i] = np.float(intensity[i].split()[0])
        intensity = tmp; del tmp
    else:                               # must be a text file;  wavelength intensity element
        with open(lineList) as fd:
            element = []
            wavelength = []
            intensity = []
            for line in fd:
                line = re.sub(r"\s*#.*$", "", line).rstrip() # strip comments

                if not line:
                    continue
                fields = line.split()
                lam, I, elem = fields[:3]

                element.append(elem)
                wavelength.append(float(lam))
                intensity.append(float(I))
    #
    # Pack into a list of ReferenceLines
    #
    referenceLines = []
    for elem, lam, I in zip(element, wavelength, intensity):
        if lamps:
            keep = False
            for lamp in lamps:
                if elem.startswith(lamp):
                    keep = True
                    break

            if not keep:
                continue

        if minIntensity > 0:
            if I < minIntensity:
                continue

        referenceLines.append(drpStella.ReferenceLine(elem, wavelength=lam, guessedIntensity=I))

    return referenceLines

def plotReferenceLines(referenceLines, what, ls=':', alpha=1):
    """Plot a set of reference lines using axvline"""
    for rl in referenceLines:
        color = 'black'
        if not (rl.status & rl.Status.FIT):
            color = 'red'
        elif (rl.status & rl.Status.MISIDENTIFIED):
            color = 'blue'

        plt.axvline(getattr(rl, what), ls=ls, color=color, alpha=alpha)
    
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
