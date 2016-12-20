#/Users/azuri/stella-git/drp_stella/bin.src/reduceArc.py '/Users/azuri/spectra/pfs/PFS' --id visit=4 --wLenFile '/Users/azuri/stella-git/obs_pfs/pfs/RedFiberPixels.fits.gz' --lineList '/Users/azuri/stella-git/obs_pfs/pfs/lineLists/CdHgKrNeXe_red.fits' --loglevel 'info' --calib '/Users/azuri/spectra/pfs/PFS/CALIB/' --output '/Users/azuri/spectra/pfs/PFS'
import os
import sys

import pfs.drp.stella.extractSpectraTask as esTask
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import makeFiberTraceSet
import lsst.log
from lsst.utils import getPackageDir
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, TaskRunner, ArgumentParser, CmdLineTask
import numpy as np
from astropy.io import fits as pyfits
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO
import traceback

class ReduceArcConfig(Config):
    """Configuration for reducing arc images"""
    function = Field( doc = "Function for fitting the dispersion", dtype=str, default="POLYNOMIAL" );
    order = Field( doc = "Fitting function order", dtype=int, default = 5 );
    searchRadius = Field( doc = "Radius in pixels relative to line list to search for emission line peak", dtype = int, default = 2 );
    fwhm = Field( doc = "FWHM of emission lines", dtype=float, default = 2.6 );
    nRowsPrescan = Field( doc = "Number of prescan rows in raw CCD image", dtype=int, default = 49 );
    radiusXCor = Field( doc = "Radius in pixels in which to cross correlate a spectrum relative to the reference spectrum", dtype=int, default=10);
    lengthPieces = Field( doc = "Length of pieces of spectrum to match to reference spectrum by stretching and shifting", dtype=int, default=2000);
    nCalcs = Field( doc = "Number of iterations > spectrumLength / lengthPieces, e.g. spectrum length is 3800 pixels, <lengthPieces> = 500, <nCalcs> = 15: run 1: pixels 0-499, run 2: 249-749,...", dtype=int, default=5);
    stretchMinLength = Field( doc = "Minimum length to stretched pieces to (< lengthPieces)", dtype=int, default=2000);
    stretchMaxLength = Field( doc = "Maximum length to stretched pieces to (> lengthPieces)", dtype=int, default=2000);
    nStretches = Field( doc = "Number of stretches between <stretchMinLength> and <stretchMaxLength>", dtype=int, default=1);
    wavelengthFile = Field( doc = "reference pixel-wavelength file including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/RedFiberPixels.fits.gz"));
    lineList = Field( doc = "reference line list including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/lineLists/CdHgKrNeXe_red.fits"));
    maxDistance = Field( doc = "Reject emission lines which center is more than this value away from the predicted position", dtype=float, default = 2.5 );
    percentageOfLinesForCheck = Field( doc = "Hold back this percentage of lines in the line list for check", dtype=int, default=10)

class ReduceArcTaskRunner(TaskRunner):
    """Get parsed values into the ReduceArcTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        print 'ReduceArcTask.getTargetList: kwargs = ',kwargs
        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, wLenFile=parsedCmd.wLenFile, lineList=parsedCmd.lineList)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            self.logger.info('ReduceArcTask.__call__: args = %s' % args)
            result = task.run(**args)
        else:
            try:
                result = task.run(**args)
            except Exception, e:
                task.log.fatal("Failed: %s" % e)
                traceback.print_exc(file=sys.stderr)

        if self.doReturnResults:
            return Struct(
                args = args,
                metadata = task.metadata,
                result = result,
            )

class ReduceArcTask(CmdLineTask):
    """Task to reduce Arc images"""
    ConfigClass = ReduceArcConfig
    RunnerClass = ReduceArcTaskRunner
    _DefaultName = "reduceArcTask"

    def __init__(self, *args, **kwargs):
        super(ReduceArcTask, self).__init__(*args, **kwargs)
        self.logger = lsst.log.Log.getLogger("pfs.drp.stella.ReduceArcTask")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="postISRCCD",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
        parser.add_argument("--wLenFile", help='directory and name of pixel vs. wavelength file')
        parser.add_argument("--lineList", help='directory and name of line list')
        return parser# ReduceArcArgumentParser(name=cls._DefaultName, *args, **kwargs)

    # lambdaPix and fluxPix have the shape (fiberTrace.yHigh - fiberTrace.yLow + 1)
    def createLineList(self, lambdaPix, fluxPix, lambdaLines, strengthLines, dispCorControl, nRows):

        calculatedFlux = np.ndarray(shape=lambdaPix.shape[0], dtype=np.float32)
        calculatedFlux[:] = np.median(fluxPix)

        lineList = np.ndarray(shape=(lambdaLines.shape[0],2),dtype=np.float32)
        lineList[:,0] = lambdaLines

        linePos = []
        for k in range(len(lambdaLines)):
            linePos.append(lambdaLines[k])
            minDist = 1000
            for yFt in range(len(lambdaPix)):
                dist = abs(lambdaLines[k] - lambdaPix[yFt])
                if dist < minDist:
                    minDist = dist
                    linePos[len(linePos)-1] = yFt
            self.logger.debug('linePos for line %d is at %f' % (len(linePos)-1, linePos[len(linePos)-1]))
            xWidth = int(2. * self.config.fwhm)
            x = np.linspace(-1 * xWidth, xWidth, (2 * xWidth) + 1)
            self.logger.debug('x = %s, strengthLines[%d] = %f' % (np.array_str(x), k, strengthLines[k]))
            fac = np.exp(-np.power(nRows - linePos[len(linePos) - 1], 2.) / (2 * np.power(1000., 2.)))
            self.logger.debug('fac = %f' % fac)
            gaussian = fac * strengthLines[k] * np.exp(-np.power(x, 2.) / (2 * np.power(self.config.fwhm/2.34, 2.)))
            self.logger.debug('%s' % np.array_str(gaussian))
            calculatedFlux[linePos[len(linePos) - 1] - xWidth:linePos[len(linePos) - 1] + xWidth + 1] += gaussian
            if np.isnan(np.min(calculatedFlux)):
                raise RuntimeError("calculatedFlux contains NaNs")

        lineList[:,1] = linePos
        return drpStella.stretchAndCrossCorrelateSpecFF(fluxPix,calculatedFlux,lineList,dispCorControl).lineList

    def run(self, expRefList, butler, wLenFile=None, lineList=None, immediate=True):
        if wLenFile == None:
            wLenFile = self.config.wavelengthFile
        if lineList == None:
            lineList = self.config.lineList
        self.logger.debug('len(expRefList) = %d' % len(expRefList))
        self.logger.debug('wLenFile = %s' % wLenFile)
        self.logger.debug('lineList = %s' % lineList)

        for arcRef in expRefList:
            self.logger.debug('arcRef.dataId = %s' % arcRef.dataId)
            self.logger.debug('arcRef = %s' % arcRef)

            try:
                fiberTrace = arcRef.get('fiberTrace', immediate=True)
                flatFiberTraceSet = makeFiberTraceSet(fiberTrace)
            except Exception, e:
                raise RuntimeError("Unable to load fiberTrace for %s from %s: %s" %
                                   (arcRef.dataId, arcRef.get('fiberTrace_filename')[0], e))

            arcExp = arcRef.get("postISRCCD", immediate=True)
            self.logger.debug('arcExp = %s' % arcExp)
            self.logger.debug('type(arcExp) = %s' % type(arcExp))

            """ optimally extract arc spectra """
            self.logger.debug('extracting arc spectra')

            """ read wavelength file """
            hdulist = pyfits.open(wLenFile)
            tbdata = hdulist[1].data
            traceIdsTemp = np.ndarray(shape=(len(tbdata)), dtype='int')
            traceIdsTemp[:] = tbdata[:]['fiberNum']
            traceIds = traceIdsTemp.astype('int32')
            lambdaPix = np.ndarray(shape=(len(tbdata)), dtype='float32')
            lambdaPix[:] = tbdata[:]['pixelWave']
            xCenters = np.ndarray(shape=(len(tbdata)), dtype='float32')
            xCenters[:] = tbdata[:]['xc']

            traceIdsUnique = np.unique(traceIds)

            """ assign trace number to flatFiberTraceSet """
            drpStella.assignITrace( flatFiberTraceSet, traceIds, xCenters )
            iTraces = np.ndarray(shape=flatFiberTraceSet.size(), dtype='intp')
            for i in range( flatFiberTraceSet.size() ):
                iTraces[i] = flatFiberTraceSet.getFiberTrace(i).getITrace()
            self.logger.debug('iTraces = %s' % np.array_str(iTraces))

            myExtractTask = esTask.ExtractSpectraTask()
            aperturesToExtract = [-1]
            spectrumSetFromProfile = myExtractTask.run(arcExp, flatFiberTraceSet, aperturesToExtract)

            for i in range(spectrumSetFromProfile.size()):
                # THIS DOESN'T WORK, again a problem with changing getSpectrum()
                spectrumSetFromProfile.getSpectrum(i).setITrace(flatFiberTraceSet.getFiberTrace(i).getITrace())
                self.logger.debug("spectrumSetFromProfile.getSpectrum(i).getITrace() = %d" % spectrumSetFromProfile.getSpectrum(i).getITrace())
                self.logger.debug("flatFiberTraceSet.getFiberTrace(i).getITrace() = %d" % flatFiberTraceSet.getFiberTrace(i).getITrace())

            """ read line list """
            hdulist = pyfits.open(lineList)
            tbdata = hdulist[1].data
            lambdaLines = np.ndarray(shape=(len(tbdata)), dtype='float32')
            lambdaLines[:] = tbdata.field(0)
            strengthLines = np.ndarray(shape=(len(tbdata)), dtype='float32')
            strengthLines[:] = tbdata.field(2)

            dispCorControl = drpStella.DispCorControl()
            dispCorControl.fittingFunction = self.config.function
            dispCorControl.order = self.config.order
            dispCorControl.searchRadius = self.config.searchRadius
            dispCorControl.fwhm = self.config.fwhm
            dispCorControl.radiusXCor = self.config.radiusXCor
            dispCorControl.lengthPieces = self.config.lengthPieces
            dispCorControl.nCalcs = self.config.nCalcs
            dispCorControl.stretchMinLength = self.config.stretchMinLength
            dispCorControl.stretchMaxLength = self.config.stretchMaxLength
            dispCorControl.nStretches = self.config.nStretches
            dispCorControl.verticalPrescanHeight = self.config.nRowsPrescan
            dispCorControl.maxDistance = self.config.maxDistance
            self.logger.debug('dispCorControl.fittingFunction = %s' % dispCorControl.fittingFunction)
            self.logger.debug('dispCorControl.order = %d' % dispCorControl.order)
            self.logger.debug('dispCorControl.searchRadius = %d' % dispCorControl.searchRadius)
            self.logger.debug('dispCorControl.fwhm = %g' % dispCorControl.fwhm)
            self.logger.debug('dispCorControl.maxDistance = %g' % dispCorControl.maxDistance)

            for i in range(spectrumSetFromProfile.size()):
                spec = spectrumSetFromProfile.getSpectrum(i)
                spec.setITrace(iTraces[i])
                self.logger.debug('flatFiberTraceSet.getFiberTrace(%d).getITrace() = %d, spec.getITrace() = %d' %(i,flatFiberTraceSet.getFiberTrace(i).getITrace(), spec.getITrace()))
                fluxPix = spec.getSpectrum()
                self.logger.debug('fluxPix.shape = %d' % fluxPix.shape)
                self.logger.debug('type(fluxPix) = %s: <%s>' % (type(fluxPix),type(fluxPix[0])))
                self.logger.debug('type(spec) = %s: <%s>: <%s>' % (type(spec),type(spec.getSpectrum()),type(spec.getSpectrum()[0])))

                nRows = traceIds.shape[0] / traceIdsUnique.shape[0]
                traceId = spec.getITrace()
                self.logger.info('traceId = %d' % traceId)

                """cut off both ends of wavelengths where is no signal"""
                yMin = (flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yCenter +
                        flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yLow)
                yMax = (flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yCenter +
                        flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yHigh)
                self.logger.debug('fiberTrace %d: yMin = %d' % (i, yMin))
                self.logger.debug('fiberTrace %d: yMax = %d' % (i, yMax))

                lambdaPixFiberTrace = lambdaPix[drpStella.firstIndexWithValueGEFrom(traceIds,
                                                                                    traceIdsUnique[traceId]) + self.config.nRowsPrescan+yMin:
                                                drpStella.firstIndexWithValueGEFrom(traceIds,
                                                                                    traceIdsUnique[traceId]) + self.config.nRowsPrescan+yMax+1]
                if len(lambdaPixFiberTrace) != len(fluxPix):
                    raise RuntimeError("reduceArcTask.py: ERROR: len(lambdaPixFiberTrace)(=%d) != len(fluxPix)(=%d)" % (len(lambdaPixFiberTrace), len(fluxPix)))
                if len(lambdaLines) != len(strengthLines):
                    raise RuntimeError("reduceArcTask.py: ERROR: len(lambdaLines)(=%d) != len(strengthLines)(=%d)" % (len(lambdaLines), len(strengthLines)))
                lineListWLenPix = self.createLineList(lambdaPixFiberTrace,
                                                      spec.getSpectrum(),
                                                      lambdaLines,
                                                      strengthLines,
                                                      dispCorControl,
                                                      nRows)
                try:
                    spec.identifyF(lineListWLenPix,
                                   dispCorControl,
                                   int(lineListWLenPix.shape[0] * self.config.percentageOfLinesForCheck / 100))
                except:
                    e = sys.exc_info()[1]
                    message = str.split(e.message, "\n")
                    for k in range(len(message)):
                        print "element",k,": <",message[k],">"
                self.logger.debug("FiberTrace %d: spec.getDispCoeffs() = %s" % (i,np.array_str(spec.getDispCoeffs())))
                self.logger.info("FiberTrace %d: spec.getDispRms() = %f" % (i, spec.getDispRms()))
                self.logger.info("FiberTrace %d: spec.getDispRmsCheck() = %f" % (i, spec.getDispRmsCheck()))

            #
            # Do the I/O using a trampoline object PfsArmIO (to avoid adding butler-related details
            # to the datamodel product)
            #
            # This is a bit messy as we need to include the pfsConfig file in the pfsArm file
            #
            dataId = arcRef.dataId

            md = arcExp.getMetadata().toDict()
            key = "PFSCONFIGID"
            if key in md:
                pfsConfigId = md[key]
            else:
                self.logger.info('No pfsConfigId is present in postISRCCD file for dataId %s' %
                              str(dataId.items()))
                pfsConfigId = 0x0

            pfsConfig = butler.get("pfsConfig", pfsConfigId=pfsConfigId, dateObs=dataId["dateObs"])

            pfsArm = spectrumSetToPfsArm(pfsConfig, spectrumSetFromProfile,
                                         dataId["visit"], dataId["spectrograph"], dataId["arm"])
            butler.put(PfsArmIO(pfsArm), 'pfsArm', dataId)
        return spectrumSetFromProfile
