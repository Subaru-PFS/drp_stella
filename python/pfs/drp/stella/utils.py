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
import collections
import os.path
import re

from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

import lsst.afw.image as afwImage
import lsst.log as log
import lsst.utils
from pfs.datamodel.pfsFiberTrace import PfsFiberTrace as pFT
import pfs.drp.stella as drpStella
from pfs.drp.stella import NistLine, NistLineMeas
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO
from pfs.drp.stella.math import gauss, gaussFit, makeArtificialSpectrum

distStrength = collections.namedtuple('distStrength', ['distance', 'strengthRatio'])

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

def createLineListForLamps(elements, lineListSuffix, removeLines):
    """
    Read NIST file for each element in elements and return a list
    of 'NistLine's sorted by laboratoryWavelength
    @param elements : array of elements, e.g. [Hg,Xe]
    @param lineListSuffix : should be either 'vac' or 'air'
    @param removeLines : array of elements and ions to remove from the line list, e.g. [HgI,XeII]
    @return array of all NistLines sorted by laboratoryWavelength
    """
    logger = log.Log.getLogger("createLineListForLamps")
    logger.debug('createLineListForLamps started')
    logger.trace('elements = %s' % elements)
    lines = []
    dir = lsst.utils.getPackageDir("obs_pfs")
    subDir = 'pfs/lineLists'
    iLine = 1
    for el in elements.split(','):
        logger.debug('reading line list for element %s' % el)
        inFile = os.path.join(dir,subDir,"%s_nist_%s.txt"
                                         % (el, lineListSuffix))
        logger.debug('reading line list <%s>' % inFile)
        f = open(inFile, "r")
        for line in f:
            if len(line) == 0 or line == "\n" or line[0] == "#":
                continue
            # example lines:
            #Ar I |  738.39800 | 10000 | 8.47e+06 | C | 1 | 2 | | T1242n |  L2634  |
            #Hg I |  467.0831  |       |          |   | 1 | 1 | |        | L11984  |
            match = re.match("(\w{2}) (I*) *\| *(\d*\.\d*) *\| *(\d*\.*\d*)(\S*) *\|.*\|.*\|.*\|.*\|.*\|.*\| *(\S*)  \|$", line)
            if not match:
                continue
            element, ion, lam, strength, flag, sources = match.groups()
            if strength == "" or strength == "0" or strength == "00":
                strength = "1"
            if ('|' in flag) or ('|' in lam):
                logger.trace('line = <%s>' % (line))
                logger.trace('element = %s, ion = %s, lam = %s, strength = %s, flag = %s, sources = %s'
                             % (element, ion, lam, strength, flag, sources))
                logger.warn("'|' found in flag(=%s) or in lam(=%s)" % (flag, lam))
            if element not in removeLines.split(','):
                line = NistLine()
                line.element=element
                line.flags=flag
                line.id=iLine
                line.ion=ion
                line.sources=sources
                fac = 1.0
                if sources == 'L11760':
                    fac = 1700.
                line.predictedStrength=float(strength) / fac
                line.laboratoryWavelength=float(lam)
                lines.append(line)
            iLine += 1
        f.close()

    lines = sorted(lines, key=lambda x: x.laboratoryWavelength)
    return lines

def writeLineList(lines, linePos, fileName, force=False):
    """
    Here we write a line list
    @param lines : list of NistLinesMeas to write to fileName
    @param linePos : position of lines in pixels
    @param fileName : output file name without path (path = obs_pfs/pfs/lineLists)
    @param force : overwrite existing file?
    """
    dir = lsst.utils.getPackageDir("obs_pfs")
    subDir = 'pfs/lineLists'
    fileWithPath = os.path.join(dir,subDir,fileName)

    # Check if file exists. If it does and force is not set, raise an error
    if os.path.isfile(fileWithPath):
        if not force:
            raise RuntimeError(
                "writeLineList: File '%s' already exists. Use force to overwrite"
                % (fileWithPath)
            )

    # create data columns
    col1 = pyfits.Column(name='wavelength',
                         format='E',
                         array=[line.nistLine.laboratoryWavelength for line in lines])
    col2 = pyfits.Column(name='pixel',
                         format='E',
                         array=linePos)
    col3 = pyfits.Column(name='strength',
                         format='E',
                         array=[line.nistLine.predictedStrength for line in lines])
    col4 = pyfits.Column(name='element',
                         format='8A',
                         array=[line.nistLine.element for line in lines])

    # write fits file
    cols = pyfits.ColDefs([col1, col2, col3, col4])
    tbhdu = pyfits.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fileWithPath, clobber=force)

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

def getLinesInWavelengthRange(lines, lambdaMin, lambdaMax):
    """
    Remove NIST lines outside the wavelength range (lambdaMin, lambdaMax) and
    return a list of 'NistLine's sorted by laboratoryWavelength
    @param lines : array of NistLines, output from createLineLampList
    @param lambdaMin : minimum wavelength of the FiberTrace spectrum
    @param lambdaMax : maximum wavelength of the FiberTrace spectrum
    @return : array of NistLines in (lambdaMin, lambdaMax), sorted by laboratoryWavelength
    """
    assert(lambdaMin < lambdaMax)
    logger = log.Log.getLogger("getLinesInWavelengthRange")
    linesUnsorted = []
    for line in lines:
        if line.laboratoryWavelength <= lambdaMin or line.laboratoryWavelength >= lambdaMax:
            logger.debug('removing line %d as it is outside the wavelength range'
                              % (line.id,))
            continue
        linesUnsorted.append(line)
    linesOut = sorted(linesUnsorted, key=lambda k: k.laboratoryWavelength)
    return linesOut

def measureLinesInPixelSpace(lines, lambdaPix, fluxPix, fwhm, minStrength=None):
    """
    Fit a Gaussian in pixel space to all lines which are at least minStrength high
    @param lines : array of NistLines, must have laboratoryWavelength set,
                   normally the output from createLineListForLamps
    @param lambdaPix : Predicted wavelengths for each pixel
    @param fluxPix : extracted Spectrum
    @param fwhm : approximate FWHM of the Gaussians
    @param minStrength : minimum line.predictedStrength for the line to be measured
    @return : array of NistLineMeas, where nistLineMeas.pixelPosPredicted,
              nistLineMeas.gaussCoeffsPixel, and nistLineMeas.eGaussCoeffsPixel
              are set
    """
    logger = log.Log.getLogger("measureLinesInPixelSpace")
    logger.trace('len(lines) = %d' % (len(lines)))
    logger.trace('lambdaPix = %s' % (np.array_str(np.array(lambdaPix))))
    logger.trace('fluxPix = %s' % (np.array_str(np.array(fluxPix))))
    if minStrength == None:
        minStrength = 0.0
    measuredLines = drpStella.getNistLineMeasVec()
    xWidth = int(1.5 * fwhm)

    # cut off zero value pixels from the predicted wavelengths
    lambdaPixGTZero = np.where(np.array(lambdaPix) > 0.)
    logger.trace('lambdaPixGTZero = %s' % (np.array_str(np.array(lambdaPixGTZero))))
    lambdaPixGTZeroStart = lambdaPixGTZero[0][1]
    lambdaPixGTZeroEnd = lambdaPixGTZero[0][len(lambdaPixGTZero[0])-2]
    lambdaPixGood = lambdaPix[lambdaPixGTZeroStart:lambdaPixGTZeroEnd]
    logger.trace('lambdaPixGood = %s' % (np.array_str(np.array(lambdaPixGood))))
    pix = np.arange(lambdaPixGTZeroStart,lambdaPixGTZeroEnd)

    for iLine in range(len(lines)):
        lineMeas = NistLineMeas()
        lineMeas.nistLine = lines[iLine]
        lineMeas.pixelPosPredicted = np.interp(lineMeas.nistLine.laboratoryWavelength,
                                               lambdaPixGood,
                                               pix)
        logger.trace('lineMeas[%d].pixelPosPredicted = %f'
                     % (iLine, lineMeas.pixelPosPredicted))
        logger.trace('lines[%d].predictedStrength(=%f)'
                     % (iLine, lines[iLine].predictedStrength))
        if lines[iLine].predictedStrength > minStrength:
            logger.trace('lines[%d].predictedStrength(=%f) > minStrength(=%f)'
                         % (iLine, lines[iLine].predictedStrength, minStrength))
            # fit Gaussian
            x = np.arange(int(lineMeas.pixelPosPredicted - xWidth),
                          int(lineMeas.pixelPosPredicted + xWidth + 1))
            logger.trace('lineMeas[%d]: x = %s' % (iLine, np.array_str(x)))
            if x[0] < 0 or x[len(x)-1] >= fluxPix.shape[0]:
                continue
            y = fluxPix[x]
            logger.trace('lineMeas[%d]: y = %s' % (iLine, np.array_str(y)))

            guess = drpStella.GaussCoeffs()
            guess.strength, guess.mu, guess.sigma = [np.max(y)-np.min(y),
                                                     int(lineMeas.pixelPosPredicted),
                                                     fwhm/2.355]
            logger.trace('lineMeas[%d]: guess: strength = %f, mu = %f, sigma = %f' %
                          (iLine, guess.strength, guess.mu, guess.sigma))

            gaussFitResult = gaussFit(x, y, guess)
            logger.trace('gaussFitResult = %d: %s'
                         % (len(gaussFitResult),
                            np.array_str(np.array(gaussFitResult))))
            if (gaussFitResult[1] > x[0]) and (gaussFitResult[1] < x[len(x)-1]):
                lineMeas.gaussCoeffsPixel.strength = gaussFitResult[0]
                lineMeas.gaussCoeffsPixel.mu = gaussFitResult[1]
                lineMeas.gaussCoeffsPixel.sigma = gaussFitResult[2]
                logger.trace('lineMeas[%d]: lineMeas.gaussCoeffsPixel: strength = %f, mu = %f, sigma = %f'
                             % (iLine,
                                lineMeas.gaussCoeffsPixel.strength,
                                lineMeas.gaussCoeffsPixel.mu,
                                lineMeas.gaussCoeffsPixel.sigma))
                lineMeas.eGaussCoeffsPixel.strength = gaussFitResult[3]
                lineMeas.eGaussCoeffsPixel.mu = gaussFitResult[4]
                lineMeas.eGaussCoeffsPixel.sigma = gaussFitResult[5]
                logger.trace('lineMeas[%d].eGaussCoeffsPixel = [strength=%f, mu=%f, sigma=%f]'
                            % (iLine,
                               lineMeas.eGaussCoeffsPixel.strength,
                               lineMeas.eGaussCoeffsPixel.mu,
                               lineMeas.eGaussCoeffsPixel.sigma))
            else:
                logger.debug('GaussFit line %d failed because gaussFitResult[1](=%f) <= x[0](=%f) or >= x[len(x)-1](=%f)'
                             % (iLine, gaussFitResult[1], x[0],
                                x[len(x)-1]))
        measuredLines.append(lineMeas)

    return measuredLines

def removeBadLines(lines, fluxPix, plot, fwhm, minDistance, maxDistance, minStrength, minRatio):
    """
    Identify and remove lines which are not suitable for the calibration.
    Blends are identified as follows:
    * For each line search for lines with a fitted position close to its own
      position (within minDistance * fwhm pixels)
    * If a possible blend is identified, compare the predicted lines strengths.
    * If one line is at least minRatio times stronger than the other lines
      in the blend, keep the strong line, otherwise discard all lines in the
      blend
    If the plot parameter is True then this function will
    plot diagnostic a plot which shows the spectrum (fluxPix), Gauss fits
    of the good lines, and vertical lines with the NIST strength for all
    lines from the NIST database
    @param lines : array of NistLineMeas, output from measureLinesInPixelSpace
    @param fluxPix : extracted Spectrum
    @param plot : plot diagnostic plot if True
    @param fwhm : FWHM of emission lines
    @param minDistance : minimum distance in pixels for 2 lines to be not blended
    @param maxDistance : maximum distance in pixels between predicted and measured position
    @param minStrength : minimum height of emission lines
    @param minRatio : minimum flux ratio for the stronger line in a blend to be kept
    @return : array of NistLineMeas, remaining lines useful for the calibration]
    """
    logger = log.Log.getLogger("removeBadLines")
    logger.trace('len(lines) = %d' % (len(lines)))
    goodLines = drpStella.getNistLineMeasVec()
    xWidth = int(2. * fwhm)

    if plot:
        plt.plot(fluxPix, 'g-', label='measured spectrum')
        plt.xlabel('pixel')
        plt.ylabel('flux [ADU]')
        plt.title('spectrum and fitted good lines')
        plt.xlim(500.,1500.)

    # find and remove bad lines
    for iLine in range(len(lines)):
        line = lines[iLine]

        if np.abs(line.gaussCoeffsPixel.mu - line.pixelPosPredicted) > maxDistance:
            logger.debug('Removing iLine = %d: line.pixelPosPredicted = %f: line.gaussCoeffsPixel.mu = %f: difference in position > 1.0'
                         % (iLine,line.pixelPosPredicted,line.gaussCoeffsPixel.mu))
            line.flags = ''.join([line.flags, 'p'])
            continue
        if line.gaussCoeffsPixel.strength < minStrength:
            logger.debug('Removing iLine = %d: line.pixelPosPredicted = %f: line = %s: line strength too low'
                           % (iLine,line.pixelPosPredicted,line,))
            line.flags = ''.join([line.flags, 's'])
            continue
        if line.gaussCoeffsPixel.sigma < 0.:
            logger.debug('Removing iLine = %d: line.pixelPosPredicted = %f: line = %s: negative sigma in Gaussfit'
                           % (iLine,line.pixelPosPredicted,line,))
            line.flags = ''.join([line.flags, 'm'])
            continue
        goodLines.append(line)

    sigmas = [line.gaussCoeffsPixel.sigma for line in goodLines]
    meanSigma = np.mean(sigmas)
    stddevSigma = np.std(sigmas)
    goodLinesOut = drpStella.getNistLineMeasVec()
    for j in range(len(goodLines)):
        strengthA = goodLines[j].nistLine.predictedStrength
        distances = []
        for i in range(len(lines)):
            if goodLines[j].nistLine.laboratoryWavelength != lines[i].nistLine.laboratoryWavelength:
                strengthB = lines[i].nistLine.predictedStrength
                ratio = strengthA / strengthB
                distances.append(distStrength(
                    distance=abs(lines[i].gaussCoeffsPixel.mu - goodLines[j].gaussCoeffsPixel.mu),
                    strengthRatio=ratio)
                )
        passed = True
        for i in np.arange(len(distances)-1,-1,-1):
            if (distances[i].distance < minDistance
                and (distances[i].strengthRatio < minRatio)):
                logger.debug("rejecting line %d because it didn't pass the distance / strength test"
                             % (j,))
                passed = False
        if not passed:
            goodLines[j].flags = ''.join([goodLines[j].flags, 'b'])
        if goodLines[j].nistLine.flags != "":
            logger.debug("rejecting line %d because it is flagged as problematic"
                         % (j,))
            goodLines[j].flags = ''.join([goodLines[j].flags, 'n'])
            passed = False
        if np.abs(goodLines[j].gaussCoeffsPixel.sigma - meanSigma) > 3. * stddevSigma:
            logger.debug("rejecting line %d because its fitted sigma value is out of bounds"
                         % (j,))
            goodLines[j].flags = ''.join([goodLines[j].flags, 'a'])
            passed = False
        if not passed:
            continue

        if plot:
            x = np.arange(int(goodLines[j].pixelPosPredicted - xWidth),
                          int(goodLines[j].pixelPosPredicted + xWidth + 1))
            gaussY = gauss(x, goodLines[j].gaussCoeffsPixel)
            plt.plot([goodLines[j].pixelPosPredicted,
                      goodLines[j].pixelPosPredicted],
                     [0.0, goodLines[j].nistLine.predictedStrength / 10.])
            plt.plot(x, gaussY)
            plt.text(goodLines[j].pixelPosPredicted,
                     np.max(gaussY),
                     '%s%s %f' % (goodLines[j].nistLine.element,
                                  goodLines[j].nistLine.ion,
                                  goodLines[j].nistLine.laboratoryWavelength))

        goodLines[j].flags = ''.join([goodLines[j].flags, 'g'])
        goodLinesOut.append(goodLines[j])
    if plot:
        plt.show()
    logger.debug('type(goodLinesOut) = %s, len(goodLinesOut) = %d'
                 % (type(goodLinesOut), len(goodLinesOut)))
    logger.info('found %d good lines for the wavelength calibration'
                % (len(goodLinesOut,)))
    return goodLinesOut

def createLineListForFiberTrace(lambdaPix,
                                fluxPix,
                                lines,
                                plot=False,
                                fwhm=1.0,
                                xCorRadius=5.0,
                                minDistance=1.0,
                                maxDistance=1.0,
                                minStrength=10.0,
                                minRatio=100.0):
    """
    Create the line list for a given FiberTrace
    If the plot is True then this function will
    plot a diagnostic plot one which shows the measured and the
    artificial spectrum with the good lines labeled
    @param lambdaPix : predicted wavelengths for each pixel in fluxPix
    @param fluxPix : Arc spectrum, same length as lambdaPix
    @param lines : NistLines for all elements created from the NIST output
    @param fwhm : FWHM of emission lines in pixels
    @param xCorRadius : cross correlation radius between artificial and measured spectrum
    @param minDistance : minimum distance in pixels for 2 lines to be not blended
    @param maxDistance : maximum distance in pixels between predicted and measured position
    @param minStrength : minimum height of emission lines
    @param minRatio : minimum flux ratio for the stronger line in a blend to be kept
    @return dictionary {"lineList": array of good NistLineMeas,
                        "measuredLines": array of NistLineMeas,
                        "offset" : measured offset in pixels between artificial
                        and measured spectrum
                       }
    """
    logger = log.Log.getLogger("createLineListForFiberTrace")
    logger.debug('len(lines) = %d' % (len(lines)))
    logger.trace('lambdaPix = %d: %s' % (len(lambdaPix),
                                         np.array_str(lambdaPix)))
    logger.trace('fluxPix = %s' % np.array_str(fluxPix))

    xWidth = int(2. * fwhm)
    logger.trace('xWidth = %f' % xWidth)
    lambdaPixGTZeroInd = np.where(lambdaPix > 0.)
    logger.trace('len(lambdaPixGTZeroInd[0])-xWidth = %d'
        % (len(lambdaPixGTZeroInd[0])-xWidth))
    lambdaMin = np.min(lambdaPix[lambdaPixGTZeroInd[0][xWidth:len(
        lambdaPixGTZeroInd[0])-xWidth]])
    lambdaMax = np.max(lambdaPix)
    logger.debug('%f < lambda < %f' % (lambdaMin, lambdaMax))

    # Remove lines outside the wavelength range
    linesInWavelengthRange = getLinesInWavelengthRange(lines = lines,
                                                       lambdaMin = lambdaMin,
                                                       lambdaMax = lambdaMax)
    logger.debug('len(linesInWavelengthRange) = %d' % (len(linesInWavelengthRange)))

    # Create an artificial spectrum
    for iLine in range(len(linesInWavelengthRange)):
        logger.trace('linesInWavelengthRange[%d]: laboratoryWavelength = %f, predictedStrength = %f' % (
                     iLine,
                     linesInWavelengthRange[iLine].laboratoryWavelength,
                     linesInWavelengthRange[iLine].predictedStrength,
                     ))
    calculatedFlux = makeArtificialSpectrum(lambdaPix = lambdaPix,
                                            lines = linesInWavelengthRange,
                                            lambdaMin = lambdaMin,
                                            lambdaMax = lambdaMax,
                                            fwhm = fwhm)
    logger.debug('max(calculatedFlux) = %f' % (np.max(calculatedFlux)))
    #Add continuum
    sortedFlux = np.sort(fluxPix[0:int(len(fluxPix)/4)])
    continuum = sortedFlux[int(len(fluxPix)/5)]
    logger.trace('continuum = %f' % (continuum))
    calculatedFlux[:] += continuum

    # cross-correlate the artificial spectrum with the spectrum from the
    # FiberTrace to find the offset in pixels and apply the offset to the
    # predicted wavelengths
    lambdaPixShifted, calculatedFlux, offset = findPixelOffsetFunction(fluxPix,
                                                                       calculatedFlux,
                                                                       lambdaPix,
                                                                       xCorRadius)

    # get the wavelength range
    lambdaPixGTZeroInd = np.where(lambdaPixShifted > 0.)
    logger.trace('len(lambdaPixGTZeroInd[0])-xWidth = %d'
        % (len(lambdaPixGTZeroInd[0])-xWidth))
    lambdaMin = np.min(lambdaPixShifted[lambdaPixGTZeroInd[0][xWidth:len(
        lambdaPixGTZeroInd[0])-xWidth]])
    lambdaMax = np.max(lambdaPixShifted)
    logger.debug('%f < lambda < %f' % (lambdaMin, lambdaMax))

    # measure lines
    logger.trace('starting measureLinesInPixelSpace')
    measuredLines = measureLinesInPixelSpace(lines = linesInWavelengthRange,
                                             lambdaPix = lambdaPixShifted,
                                             fluxPix = fluxPix,
                                             fwhm = fwhm)
    logger.debug('type(measuredLines) = %s, len(measuredLines) = %d'
                 % (type(measuredLines), len(measuredLines)))

    # Remove bad lines
    goodLines = removeBadLines(lines = measuredLines,
                               fluxPix = fluxPix,
                               plot = plot,
                               fwhm = fwhm,
                               minDistance = minDistance,
                               maxDistance = maxDistance,
                               minStrength = minStrength,
                               minRatio = minRatio)

    # Mark line as good by setting the 'g' flag.
    # If the line gets rejected or held back in Spectra::identify it will get
    # another flag.
    for line in goodLines:
        line.flags = line.flags.join('g')
    logger.debug('type(goodLines) = %s, len(goodLines) = %d'
                 % (type(goodLines), len(goodLines)))

    # create diagnostic plot
    if plot:
        plt.plot(fluxPix,'r-', label = 'measured spectrum')
        plt.plot(calculatedFlux,'g-', label = 'artificial spectrum')
        plt.xlabel('pixel')
        plt.ylabel('flux [ADU]')
        plt.title("measured and artificial spectrum + lines' names")
        xMin = 0.
        xMax = len(fluxPix)
        plt.xlim(xMin,xMax)
        yMin = 0.0
        yMax = np.max([np.max(fluxPix[xMin:xMax]), np.max(calculatedFlux[xMin:xMax])])
        plt.ylim(yMin,yMax)
        logger.trace('yMin = %f, yMax = %f' % (yMin, yMax))
        for iLine in range(len(goodLines)):
            plt.text(goodLines[iLine].pixelPosPredicted,
                     calculatedFlux[goodLines[iLine].pixelPosPredicted],
                     '%s%s %f' % (goodLines[iLine].nistLine.element,
                                  goodLines[iLine].nistLine.ion,
                                  goodLines[iLine].nistLine.laboratoryWavelength
                                 ),
                     color='g')
        plt.legend()
        plt.show()
    output = {"lineList" : goodLines,
              "measuredLines" : measuredLines,
              "offset" : offset
             }
    return output

def measureLinesInWavelengthSpace(lines, flux, wavelength, sigma=1.0, plot=False):
    """
    Measure the wavelength of all lines in 'lines' from the spectrum flux vs.
    wavelength. Save the results in line.gaussCoeffsLambda and line.eGaussCoeffsLambda
    @param lines : list of NistLineMeas to measure the position in wavelength space
    @param flux : spectrum flux
    @param wavelength : spectrum wavelength, same length as flux
    @param sigma : Gauss sigma
    @param plot : Plot spectrum and fitted lines?
    """
    logger = log.Log.getLogger("measureLinesInWavelengthSpace")
    logger.trace('len(lines) = %d' % (len(lines)))
    logger.trace('flux = %d: %s' % (len(flux), np.array_str(np.array(flux))))
    wavelengthArr = np.array(wavelength)
    logger.trace('wavelength = %d: %s' % (len(wavelength),
                                          np.array_str(wavelengthArr)))
    for line in lines:
        guess = drpStella.GaussCoeffs()
        guess.mu, guess.sigma, guess.strength=[line.nistLine.laboratoryWavelength,
                                               sigma,
                                               line.nistLine.predictedStrength]
        logger.trace('guess = [mu=%f, strength=%f, sigma=%f]'
                     % (guess.mu, guess.strength, guess.sigma))
        indices = np.where(wavelengthArr >= (guess.mu - (3. * guess.sigma)))[0]
        logger.trace('indices = %s' % (np.array_str(indices)))
        indices = indices[np.where(wavelengthArr[indices] <= guess.mu + (3. * guess.sigma))[0]]
        logger.trace('indices = %s' % (np.array_str(indices)))
        x = wavelengthArr[indices[0]:indices[len(indices)-1]+1]
        y = flux[indices]
        logger.trace('x = %s' % (np.array_str(x)))
        logger.trace('y = %s' % (np.array_str(y)))
        gaussFitResult = gaussFit(x,y,guess)
        logger.trace('gaussFitResult = %s'
                     % (np.array_str(np.array(gaussFitResult))))
        gaussCoeffs = drpStella.GaussCoeffs()
        gaussCoeffs.strength, gaussCoeffs.mu, gaussCoeffs.sigma = [gaussFitResult[0],
                                                                   gaussFitResult[1],
                                                                   gaussFitResult[2]]
        eGaussCoeffs = drpStella.GaussCoeffs()
        eGaussCoeffs.strength, eGaussCoeffs.mu, eGaussCoeffs.sigma = [gaussFitResult[3],
                                                                      gaussFitResult[4],
                                                                      gaussFitResult[5]]
        logger.trace('gaussCoeffs = [mu=%f, strength=%f, sigma=%f]'
                     % (gaussCoeffs.mu, gaussCoeffs.strength, gaussCoeffs.sigma))
        logger.trace('eGaussCoeffs = [mu=%f, strength=%f, sigma=%f]'
                     % (eGaussCoeffs.mu, eGaussCoeffs.strength, eGaussCoeffs.sigma))
        line.gaussCoeffsLambda = gaussCoeffs
        line.eGaussCoeffsLambda = eGaussCoeffs
        logger.trace('line.gaussCoeffsLambda = [mu=%f, strength=%f, sigma=%f]'
                     % (line.gaussCoeffsLambda.mu,
                        line.gaussCoeffsLambda.strength,
                        line.gaussCoeffsLambda.sigma))
        logger.trace('line.eGaussCoeffsLambda = [mu=%f, strength=%f, sigma=%f]'
                     % (line.eGaussCoeffsLambda.mu,
                        line.eGaussCoeffsLambda.strength,
                        line.eGaussCoeffsLambda.sigma))
    if plot:
        plt.plot(wavelength, flux,'r-', label='spectrum')
        plt.title("measured spectrum + lines")
        plt.xlabel('wavelength [nm]')
        plt.ylabel('flux [ADU]')
        for line in lines:
            x = wavelength[np.where(wavelength > line.nistLine.laboratoryWavelength - (3. * sigma))]
            x = x[np.where(x < line.nistLine.laboratoryWavelength + (3. * sigma))]
            y = gauss(x,line.gaussCoeffsLambda)
            plt.plot(x,y,'b-')
            plt.text(line.nistLine.laboratoryWavelength,
                     y[int(len(y)/2)],
                     '%s%s %f' % (line.nistLine.element,
                                  line.nistLine.ion,
                                  line.nistLine.laboratoryWavelength
                                 ),
                     color='g')
        plt.legend()
        plt.show()

def plotSpectrumWithLines(fiberIdx,
                          lineList,
                          pfsArm,
                          fwhm=1.0,
                          lambdaMin=0.0,
                          lambdaMax=0.0,
                          yMin=0.0,
                          yMax=0.0,
                          showPlot=True):
    """Plot an Arc spectrum with the identified emission lines

    @param fiberIdx : FiberTrace index to plot
    @param lineList : line list, NistLineMeasPtrVector
    @param pfsArm : Arc spectrum to plot
    @param fwhm : Full Width at Half Maximum of emission lines in pixels
    @param lambdaMin : minimum wavelength to plot
    @param lambdaMax : maximum wavelength to plot
    @param yMin : minimum flux to plot
    @param yMax : maximum flux to plot
    @param showPlot : if True then show plot on screen
    """
    logger = log.Log.getLogger("plotSpectrumWithLines")
    logger.trace('lambdaMin = %f, lambdaMax = %f' % (lambdaMin, lambdaMax))
    xlabel = "Wavelength [nm]"

    # plot measured spectrum
    lam = pfsArm.lam[fiberIdx][~np.isnan(pfsArm.lam[fiberIdx])]
    flux = pfsArm.flux[fiberIdx][~np.isnan(pfsArm.lam[fiberIdx])]
    plt.title("measured spectrum with lines")
    plt.plot(lam, flux, label='measured spectrum', color='k')
    plt.xlabel('wavelength [nm]')
    plt.ylabel('flux [ADU]')

    # silence a verbose logger
    tmpLogger = log.Log.getLogger("makeArtificialSpectrum")
    tmpLogger.setLevel(log.WARN)

    # calculate artificial spectrum
    nistLines = [line.nistLine for line in lineList]
    logger.debug('len(nistLines) = %d' % (len(nistLines)))
    artificialSpectrum = makeArtificialSpectrum(lambdaPix = lam,
                                                lines = nistLines,
                                                lambdaMin = lambdaMin,
                                                lambdaMax = lambdaMax,
                                                fwhm = fwhm)

    # calculate xRange for the plot
    xMin = lambdaMin
    if xMin == 0.0:
        xMin = np.min(lam)
    xMax = lambdaMax
    if xMax == 0.0:
        xMax = np.max(lam)
    logger.trace('xMin = %f, xMax = %f' % (xMin, xMax))

    xRangeMin=np.where(lam >= xMin)
    xRangeMax=np.where(lam[xRangeMin] <= xMax)
    xRange = xRangeMin[0][xRangeMax]
    if yMax == 0.0:
        yMax = np.max([flux[xRange],artificialSpectrum[xRange]])

    # plot artificial spectrum
    plt.plot(lam, artificialSpectrum, label='artificial spectrum', color='blue')

    # add verticle lines and line text to plot
    for line in lineList:
        if ((line.nistLine.laboratoryWavelength >= xMin)
                and (line.nistLine.laboratoryWavelength <= xMax)):
            color='red'
            if 'g' in line.flags:
                color='green'
            plt.plot([line.nistLine.laboratoryWavelength,
                      line.nistLine.laboratoryWavelength],
                     [0.0, line.nistLine.predictedStrength],
                     color=color)
            if line.nistLine.predictedStrength < yMax:
                plt.text(line.nistLine.laboratoryWavelength,
                         line.nistLine.predictedStrength,
                         '%s%s %f' % (line.nistLine.element,
                                      line.nistLine.ion,
                                      line.nistLine.laboratoryWavelength),
                         color=color)

    plt.xlabel(xlabel)
    plt.legend()

    plt.xlim(xMin,xMax)
    plt.ylim(yMin,yMax)

    if showPlot:
        plt.show()

def plotWavelengthResiduals(lineList, traceId=0, rms=0.0):
    """
    create a plot showing the residuals between the laboratory wavelength and the
    measured wavelength
    @param lineList : Line list (NistLineMeasPtrVector) for which to plot the residuals.
                      Only lines with the flag 'g' are plotted, no lines with the flag
                      'i' are plotted as they could not be fitted by a Gaussian in pixel
                      space
    """
    logger = log.Log.getLogger("plotWavelengthResiduals")
    logger.debug('len(lineList) = %d ' % (len(lineList)))

    lines = drpStella.getNistLineMeasVec()
    for line in lineList:
        lines.append(line)
        logger.trace('line.flags = %s' % (line.flags))
    logger.debug('%d lines in lines' % (len(lines)))

    allFittedLines = drpStella.getLinesWithFlags(lines, 'g', 'i')
    fittedLines = drpStella.getLinesWithFlags(lines, 'g', 'hif')
    heldBackLines = drpStella.getLinesWithFlags(lines, 'h')
    rejectedLines = drpStella.getLinesWithFlags(lines, 'f')
    rejectedLinesByGaussFit = drpStella.getLinesWithFlags(lines, 'i')

    logger.trace('all fitted lines: %d' % (len(allFittedLines)))
    logger.trace('%d good fitted lines, not rejected' % (len(fittedLines)))
    logger.trace('%d lines held back' % (len(heldBackLines)))
    logger.trace('%d lines rejected by PolyFit' % (len(rejectedLines)))
    logger.trace('%d lines rejected by GaussFit' % (len(rejectedLinesByGaussFit)))

    lamFitted = np.array([line.nistLine.laboratoryWavelength for line in fittedLines])
    measFitted = np.array([line.wavelengthFromPixelPosAndPoly for line in fittedLines])
    dLamFitted = lamFitted - measFitted

    lamFittedAll = np.array([line.nistLine.laboratoryWavelength for line in allFittedLines])
    measFittedAll = np.array([line.wavelengthFromPixelPosAndPoly for line in allFittedLines])
    dLamFittedAll = lamFittedAll - measFittedAll

    lamHeldBack = np.array([line.nistLine.laboratoryWavelength for line in heldBackLines])
    measHeldBack = np.array([line.wavelengthFromPixelPosAndPoly for line in heldBackLines])
    dLamHeldBack = lamHeldBack - measHeldBack

    lamRejected = np.array([line.nistLine.laboratoryWavelength for line in rejectedLines])
    measRejected = np.array([line.wavelengthFromPixelPosAndPoly for line in rejectedLines])
    dLamRejected = lamRejected - measRejected

    # the random data
    logger.trace('lamFitted = %s' % (np.array_str(np.array(lamFitted))))
    logger.trace('lamHeldBack = %s' % (np.array_str(np.array(lamHeldBack))))
    logger.trace('lamRejected = %s' % (np.array_str(np.array(lamRejected))))
    logger.trace('measFitted = %s' % (np.array_str(np.array(measFitted))))
    logger.trace('measHeldBack = %s' % (np.array_str(np.array(measHeldBack))))
    logger.trace('measRejected = %s' % (np.array_str(np.array(measRejected))))
    logger.trace('dLamFitted = %s' % (np.array_str(np.array(dLamFitted))))
    logger.trace('dLamHeldBack = %s' % (np.array_str(np.array(dLamHeldBack))))
    logger.trace('dLamRejected = %s' % (np.array_str(np.array(dLamRejected))))

    x = lamFittedAll
    logger.trace('len(x) = %d' % (len(x)))
    logger.trace('x = %s' % (np.array_str(np.array(x))))

    y = dLamFittedAll
    logger.trace('len(y) = %d' % (len(y)))
    logger.trace('y = %s' % (np.array_str(np.array(y))))
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.85
    left_h = 0.81

    rect_scatter = [left, bottom, width, height]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, color='green')
    axScatter.set_xlabel('wavelength [nm]')
    axScatter.set_ylabel('residual [nm]')
    title = ''
    if traceId > 0:
        title += 'FiberTraceId = %d' % (traceId)
    if rms > 0.0:
        title += ' RMS = %f' % (rms)
    axScatter.set_title(title)

    # now determine nice limits by hand:
    ybinwidth = (np.max(y) - np.min(y)) / 100.

    axScatter.set_xlim((0.99*np.min(x), 1.01*np.max(x)))
    axScatter.set_ylim((1.05*np.min(y), 1.05*np.max(y)))

    ybins = np.arange(np.min(y), np.max(y) + ybinwidth, ybinwidth)

    axHisty.set_ylim(axScatter.get_ylim())
    axHisty.hist(y, bins=ybins, orientation='horizontal', color='blue')

    x = lamFitted
    y = dLamFitted
    axHisty.hist(y, bins=ybins, orientation='horizontal', color='green')

    x = lamHeldBack
    y = dLamHeldBack
    axScatter.scatter(x, y, color='blue')

    x = lamRejected
    y = dLamRejected
    logger.trace('x = lamRejected = %s' % (np.array_str(np.array(x))))
    logger.trace('y = dLamRejected = %s' % (np.array_str(np.array(y))))
    axHisty.hist(y, bins=ybins, orientation='horizontal', color='red')
    axScatter.scatter(x, y, color='red')

    plt.show()

def readLineList(fileName):
    """
    Read the line list containing all good NistLines for all elements
    @param fileName : name of file to read
    @return : NistLineVec
    """
    hdulist = pyfits.open(fileName)
    tbdata = hdulist[1].data
    lines = drpStella.getNistLineVec()
    for i in range(len(tbdata)):
        line = NistLine()
        line.element=tbdata[i]['element']
        line.flags=tbdata[i]['flags']
        line.ion=tbdata[i]['ion']
        line.sources=tbdata[i]['sources']
        line.predictedStrength=float(tbdata[i]['strength'])
        line.laboratoryWavelength=float(tbdata[i]['wavelength'])
        lines.append(line)
    return lines


def getLineList(fileName, elements):
    """
    create the line list for all elements from fileName
    @param fileName : name of file containing all lines, created by constructLineListTask
    @param elements : tuple with all elements for which the line list is to be created
    @return : NistLineVec
    """
    allLines = readLineList(fileName)
    linesOut = drpStella.getNistLineVec()
    for line in allLines:
        for elem in getElements(elements):
            if elem in line.element:
                foundInFlags = False
                for el in getElements(elements):
                    if el in line.flags:
                        foundInFlags = True
                if not foundInFlags:
                    linesOut.append(line)
    return linesOut

def getElements(elementString):
    """
    extract elements from element string
    @param elementString : string containing the elements, separated by ',' or not
    @return tuple of elements in element string
    """
    logger = log.Log.getLogger("getElements")
    elements = []
    logger.trace('getElements: elementString = <%s>' % (elementString))
    if ',' in elementString:
        for el in elementString.split(','):
            elements.append(el)
    else:
        for i in np.arange(0,len(elementString),2):
            elem = elementString[i*2:(i*2)+2]
            logger.trace('getElements: elem = %s' % (elem))
            elements.append(elem)
    return elements

def getElementsString(elements):
    """
    construct element string without kommas from element string with kommas
    @param elements : string containing the elements separated by ','
    @param return : string containing the elements without ','
    """
    elementsString = ''
    for el in elements.split(','):
        elementsString += el
    return elementsString

def findPixelOffset(measuredSpectrum,
                    artificialSpectrum,
                    lambdaPix,
                    xCorRadius):
    # cross-correlate the artificial spectrum with the spectrum from the
    # FiberTrace to find the offset in pixels and apply the offset to the
    # predicted wavelengths
    logger = log.Log.getLogger("findPixelOffset")
    try:
        res = drpStella.crossCorrelate(measuredSpectrum,
                                       artificialSpectrum,
                                       xCorRadius,
                                       xCorRadius)
    except Exception, e:
        raise RuntimeError("crossCorrelate failed: %s", e)
    logger.debug('res.pixShift = %f' % (res.pixShift))

    x = range(measuredSpectrum.shape[0])
    u = [el - res.pixShift for el in x]
    artificialSpectrum = np.interp(u,x,artificialSpectrum).astype(np.float32)

    lambdaPixShifted = np.interp(u,x,lambdaPix).astype(np.float32)
    if False:
        plt.title('measured and artificial spectrum1')
        plt.plot(lambdaPixShifted, measuredSpectrum1,'g-', label = 'measured spectrum')
        plt.plot(lambdaPixShifted, artificialSpectrum1, 'b-', label = 'artificial spectrum')
        plt.xlabel('wavelength [nm]')
        plt.ylabel('flux [ADU]')
        plt.legend()
        plt.show()
    return [lambdaPixShifted, artificialSpectrum, res.pixShift]

def findPixelOffsetFunction(measuredSpectrum,
                            artificialSpectrum,
                            lambdaPix,
                            xCorRadius,
                            plot=False):
    # cross-correlate the artificial spectrum with the spectrum from the
    # FiberTrace to find the offset in pixels and apply the offset to the
    # predicted wavelengths
    logger = log.Log.getLogger("findPixelOffsetFunction")
    try:
        resLow = drpStella.crossCorrelate(
            measuredSpectrum[0:int(measuredSpectrum.shape[0]/4)],
            artificialSpectrum[0:int(measuredSpectrum.shape[0]/4)],
            xCorRadius,
            xCorRadius
        )
        resHigh = drpStella.crossCorrelate(
            measuredSpectrum[int(measuredSpectrum.shape[0]*3/4):measuredSpectrum.shape[0]],
            artificialSpectrum[int(measuredSpectrum.shape[0]*3/4):measuredSpectrum.shape[0]],
            xCorRadius,
            xCorRadius
        )
    except Exception, e:
        raise RuntimeError("crossCorrelate failed: %s", e)
    logger.debug('resLow.pixShift = %f' % (resLow.pixShift))
    logger.debug('resHigh.pixShift = %f' % (resHigh.pixShift))

    dy = resHigh.pixShift - resLow.pixShift
    dx = measuredSpectrum.shape[0] * 0.75
    logger.debug('measuredSpectrum.shape[0] = %d' % (measuredSpectrum.shape[0]))
    logger.debug('dx = %f, dy = %f' % (dx, dy))
    m = dy/dx
    logger.debug('m = %f' % (m))
    n = ((resLow.pixShift * 7.0) - resHigh.pixShift) / 6.0
    logger.debug('n = %f' % (n))
    y_l = (m * measuredSpectrum.shape[0] / 8.0) + n
    logger.debug('y_l = %f' % (y_l))
    y_h = (m * measuredSpectrum.shape[0] * 7.0 / 8.0) + n
    logger.debug('y_h = %f' % (y_h))

    pixShift = np.array([m * el + n for el in range(measuredSpectrum.shape[0])])
    logger.debug('pixShift = %s' % (np.array_str(pixShift)))

    x = range(measuredSpectrum.shape[0])
    u = x - pixShift#[el - res.pixShift for el in x]
    artificialSpectrum = np.interp(u,x,artificialSpectrum).astype(np.float32)

    lambdaPixShifted = np.interp(u,x,lambdaPix).astype(np.float32)
    if plot:
        plt.title("measured and artificial spectrum")
        plt.plot(lambdaPixShifted, measuredSpectrum,'g-', label='measuredSpectrum2')
        plt.plot(lambdaPixShifted, artificialSpectrum, 'b-', label='artificialspectrum2')
        plt.xlabel('wavelength [nm]')
        plt.ylabel('flux [ADU]')
        plt.legend()
        plt.show()
    return [lambdaPixShifted, artificialSpectrum, pixShift]
