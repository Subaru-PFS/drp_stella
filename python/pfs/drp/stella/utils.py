from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object
import os
import re
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
import lsst.daf.base as dafBase
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.log as log
import numpy as np
import pfs.drp.stella as drpStella
import pfs.drp.stella.detectorMap as detectorMap
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO, PfsConfigIO
import pfs.drp.stella.detectorMap

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

def writeDetectorMap(detectorMap, outFile, metadata=None):
    """write a DetectorMap to the specified file
    @param detectorMap: The DetectorMap
    @param outFile:  Name of desired FITS file
    """
    #
    # Unpack detectorMap into python objects
    #
    detMapIo = pfs.drp.stella.detectorMap.DetectorMapIO(detectorMap)
    bbox = detMapIo.getBBox()
    fiberIds = np.array(detMapIo.getFiberIds(), dtype=np.int32)
    slitOffsets = detectorMap.getSlitOffsets()

    nKnot = len(detMapIo.getXCenter(fiberIds[0])[0])
    splineDataArr = np.empty((len(fiberIds), 4, nKnot))
    for i, fiberId in enumerate(fiberIds):
        splineDataArr[i][0], splineDataArr[i][1] = detMapIo.getXCenter(fiberId)
        splineDataArr[i][2], splineDataArr[i][3] = detMapIo.getWavelength(fiberId)

    del detMapIo

    throughputs = np.empty_like(slitOffsets[0])
    for i, fiberId in enumerate(fiberIds):
        throughputs[i] = detectorMap.getThroughput(fiberId)    
    #
    # OK, we've unpacked the DetectorMap; time to write the contents to disk
    #
    hdus = pyfits.HDUList()

    hdr = pyfits.Header()
    hdr["MINX"] = bbox.getMinX()
    hdr["MINY"] = bbox.getMinY()
    hdr["MAXX"] = bbox.getMaxX()
    hdr["MAXY"] = bbox.getMaxY()
    if metadata:
        for k in metadata.names():
            hdr[k] = metadata.get(k)
            
    hdus.append(pyfits.PrimaryHDU(header=hdr))

    hdu = pyfits.ImageHDU(fiberIds)
    hdu.name = "FIBERID"
    hdu.header["INHERIT"] = True
    hdus.append(hdu)

    hdu = pyfits.ImageHDU(slitOffsets)
    hdu.name = "SLITOFF"
    hdu.header["INHERIT"] = True
    hdus.append(hdu)

    hdu = pyfits.ImageHDU(splineDataArr)
    hdu.name = "SPLINE"
    hdu.header["INHERIT"] = True
    hdus.append(hdu)

    hdu = pyfits.ImageHDU(throughputs)
    hdu.name = "THROUGHPUT"
    hdu.header["INHERIT"] = True
    hdus.append(hdu)

    # clobber=True in writeto prints a message, so use open instead
    with open(outFile, "w") as fd:
        hdus.writeto(fd)
    
def readDetectorMap(inFile):
    """Return a DetectorMap from the specified file
    @param inFile:  Name of FITS file to read
    """
    with pyfits.open(inFile) as fd:
        pdu = fd[0]
        minX = pdu.header['MINX']
        minY = pdu.header['MINY']
        maxX = pdu.header['MAXX']
        maxY = pdu.header['MAXY']

        bbox = afwGeom.BoxI(afwGeom.PointI(minX, minY), afwGeom.PointI(maxX, maxY))

        hdu = fd["FIBERID"]
        fiberIds = hdu.data
        fiberIds = fiberIds.astype(np.int32)   # why is this astype needed? BITPIX=32, no BZERO/BSCALE

        hdu = fd["SLITOFF"]
        slitOffsets = hdu.data.astype(np.float32)

        hdu = fd["SPLINE"]
        splineDataArr = hdu.data.astype(np.float32)

        try:
            hdu = fd["THROUGHPUT"]
            throughputs = hdu.data.astype(np.float32)
        except KeyError:
            throughputs = np.ones_like(slitOffsets[0])

    nKnot = len(splineDataArr[0][0])

    detMapIo = pfs.drp.stella.detectorMap.DetectorMapIO(bbox, fiberIds, nKnot)
    detMapIo.setSlitOffsets(slitOffsets)

    for i in range(len(fiberIds)):
        fiberId = fiberIds[i]
        xcKnot, xc, wlKnot, wl = splineDataArr[i]

        detMapIo.setXCenter(fiberId,    xcKnot, xc)
        detMapIo.setWavelength(fiberId, wlKnot, wl)

    detMap = detMapIo.getDetectorMap()

    for i, fiberId in enumerate(fiberIds):
        detMap.setThroughput(fiberId, throughputs[i])

    return detMap

def makeDetectorMapIO(detectorMap, visitInfo):
    """Make a DetectorMapIO with needed metadata

    \param detectorMap: The DetectorMap
    \param visitInfo:  VisitInfo with e.g. the date
    """
    dt = visitInfo.getDate()
    dateObs = dt.toPython(dt.UTC).strftime("%Y-%m-%d")
    
    metadata = dafBase.PropertyList()
    metadata.set("OBSTYPE", "DETECTORMAP")
    metadata.set("HIERARCH calibDate", dateObs) # avoid warnings on long keywords
    return DetectorMapIO(detectorMap, metadata)

def _makeDetectorMap(butler, dataId, wLenFile, nKnot=25):
    """Return a DetectorMap from the specified file

    OBSELETE:  Use makeDetectorMap.  Keep until we've decided
    how to manage/curate the pfsDetectorMap file
    """
    
    xCenters, wavelengths, fiberIds = readWavelengthFile(wLenFile)
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

    detMap = detectorMap.DetectorMap(bbox, fiberIds, xCenters, wavelengths, nKnot=nKnot)

    slitOffsets = np.zeros((3, len(fiberIds)), dtype='float32')
    DX     = slitOffsets[detMap.FIBER_DX]
    DY     = slitOffsets[detMap.FIBER_DY]
    DFOCUS = slitOffsets[detMap.FIBER_DFOCUS]

    # Update the slit offsets
    for fid, dx, dy in [                # (fiberId, dx, dy)
            (  5,   8,   6),
            ( 67,   3,   4),
            (194,   2,   2),
            (257,   1,   2),
            (315,  30,   2),
            (337, -40,   2),
            (393,  -3,   4),
            (455,  -3,   6),
            (582,  -7,   9),
            (644,  -7,  10),
            ]:
        idx = detMap.getFiberIdx(fid)
        DX[idx] = dx
        DY[idx] = dy

    detMap.setSlitOffsets(slitOffsets)

    return detMap

def readLineListFile(lineList, lamps=["Ar", "Cd", "Hg", "Ne", "Xe"], minIntensity=0):
    """Read line list

    File consists of lines of
       lambda Intensity Species Flag
    where lambda is the wavelength in vacuum nm

    Flag:  bitwise OR of:
      0   Good
      1   Not visible
      2   Blend; don't use
      4   Unknown; check

    Return:
       list of drp::ReferenceLine
    """
    #
    # Pack into a list of ReferenceLines
    #
    referenceLines = []

    with open(lineList) as fd:
        for line in fd:
            line = re.sub(r"\s*#.*$", "", line).rstrip() # strip comments

            if not line:
                continue
            fields = line.split()
            try:
                lam, I, species, flag = fields
            except Exception as e:
                print("%s: %s" % (e, fields))
                raise

            flag = int(flag)
            if flag != 0:
                continue

            try:
                I = float(I)
            except ValueError:
                I = np.nan

            if lamps:
                keep = False
                for lamp in lamps:
                    if species.startswith(lamp):
                        keep = True
                        break

                if not keep:
                    continue

            if minIntensity > 0:
                if not np.isfinite(I) or I < minIntensity:
                    continue

            referenceLines.append(drpStella.ReferenceLine(species, wavelength=float(lam), guessedIntensity=I))

    if len(referenceLines) == 0:
        raise RuntimeError("You have not selected any lines from %s" % lineList)

    return referenceLines

def plotReferenceLines(referenceLines, what, ls=':', alpha=1, color=None, label=None, labelStatus=True,
                       labelLines=False, wavelength=None, spectrum=None):
    """Plot a set of reference lines using axvline

    \param referenceLines   List of ReferenceLine
    \param what   which field in ReferenceLine to plot
    \param ls Linestyle (default: ':')
    \param alpha Transparency (default: 1)
    \param color Colour (default: None => let matplotlib choose)
    \param label Label for lines (default: None => use "what" or "what status")
    \param labelStatus Include status in labels (default: True)
    \param labelLines Label lines with their ion (default: False)
    \param wavelength Wavelengths array for underlying plot (default: None)
    \param spectrum   Intensity array for underlying plot (default: None)

    If labelLines is True the lines will be labelled at the top of the plot; if you provide the spectrum
    the labels will appear near the peaks of the lines
    """
    if label == '':
        label = None

    def maybeSetLabel(status):
        if labelLines:
            if labelStatus:
                lab = "%s %s" % (what, status) # n.b. label is in caller's scope
            else:
                lab = what

            lab = None if lab in labels else lab
            labels[lab] = True

            return lab
        else:
            return label

    labels = {}                         # labels that we've used if labelLines is True
    for rl in referenceLines:
        if not (rl.status & rl.Status.FIT):
            color = 'black'
            label = maybeSetLabel("Bad fit")
        elif (rl.status & rl.Status.RESERVED):
            color = 'blue'
            label = maybeSetLabel("Reserved")
        elif (rl.status & rl.Status.SATURATED):
            color = 'magenta'
            label = maybeSetLabel("Saturated")
        elif (rl.status & rl.Status.CR):
            color = 'cyan'
            label = maybeSetLabel("Cosmic ray")
        elif (rl.status & rl.Status.MISIDENTIFIED):
            color = 'brown'
            label = maybeSetLabel("Misidentified")
        elif (rl.status & rl.Status.CLIPPED):
            color = 'red'
            label = maybeSetLabel("Clipped")
        else:
            color = 'green'
            label = maybeSetLabel("Fit")

        x = getattr(rl, what)
        plt.axvline(x, ls=ls, color=color, alpha=alpha, label=label)
        label = None

        if labelLines:
            if spectrum is None:
                y = 0.95*plt.ylim()[1]
            else:
                if wavelength is None:
                    ix = x
                else:
                    ix = np.searchsorted(wavelength, x)

                    if ix == 0 or ix == len(wavelength):
                        continue

                i0 = max(0, int(ix) - 2)
                i1 = min(len(spectrum), int(ix) + 2 + 1)
                y = 1.05*spectrum[i0:i1].max()

            plt.text(x, y, rl.description, ha='center')
    
def writePfsArm(butler, arcExposure, spectrumSet, dataId):
    """
    Do the I/O using a trampoline object PfsArmIO (to avoid adding butler-related details
    to the datamodel product)

    This is a bit messy as we need to include the pfsConfig file in the pfsArm file
    """
    md = arcExposure.getMetadata()
    key = "PFSCONFIGID"
    if md.exists(key):
        pfsConfigId = md.get(key)
    else:
        log.log("writePfsArm",
                log.WARN,
                'No pfsConfigId is present in postISRCCD file for dataId %s' %
                str(dataId.items()))
        pfsConfigId = 0x0

    pfsConfig = butler.get("pfsConfig", pfsConfigId=pfsConfigId, dateObs=dataId["dateObs"])

    if False:                           # calls bypass_pfsConfig, repeating warnings from the get()
        createPfsConfig = not butler.datasetExists("pfsConfig",
                                                   pfsConfigId=pfsConfigId, dateObs=dataId["dateObs"])
    else:
        fileName = butler.get("pfsConfig_filename", pfsConfigId=pfsConfigId, dateObs=dataId["dateObs"])[0]
        createPfsConfig = not os.path.exists(fileName)

    if createPfsConfig:
        assert pfsConfig.fiberId is None
        fiberId = []
        for spec in spectrumSet:
            fiberId.append(spec.getFiberId())
        pfsConfig.fiberId = np.array(fiberId)

    pfsArm = spectrumSetToPfsArm(pfsConfig, spectrumSet,
                                 dataId["visit"], dataId["spectrograph"], dataId["arm"])
    butler.put(PfsArmIO(pfsArm), 'pfsArm', dataId)

    if createPfsConfig:
        butler.put(PfsConfigIO(pfsConfig), 'pfsConfig', dataId, pfsConfigId=0x0)

def addFiberTraceSetToMask(mask, fiberTraceSet):
    for ft in fiberTraceSet.getTraces():
        traceMask = ft.getTrace().mask
        if False:                       # requires w_2017_32 or later
            mask[traceMask.getBBox(), afwImage.PARENT] |= mask
        else:
            mask.Factory(mask, traceMask.getBBox(), afwImage.PARENT)[:] |= traceMask


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class DetectorMapIO(object):
    """A class to perform butler-based I/O for DetectorMap
    """

    def __init__(self, detectorMap, metadata=None):
        self._detectorMap = detectorMap
        self._metadata = metadata

    @staticmethod
    def readFits(pathName, hdu=None, flags=None):
        return readDetectorMap(pathName)

    def writeFits(self, pathName, flags=None):
        writeDetectorMap(self._detectorMap, pathName, self._metadata)
