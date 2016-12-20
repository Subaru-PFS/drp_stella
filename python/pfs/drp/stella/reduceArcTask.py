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
    wavelengthFile = Field( doc = "reference pixel-wavelength file including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/RedFiberPixels.fits.gz"));
    lineList = Field( doc = "reference line list including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/lineLists/CdHgKrNeXe_red.fits"));
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
            xCenters = np.ndarray(shape=(len(tbdata)), dtype='float32')
            yCenters = np.ndarray(shape=(len(tbdata)), dtype='float32')
            wavelengths = np.ndarray(shape=(len(tbdata)), dtype='float32')
            traceIdsTemp[:] = tbdata[:]['fiberNum']
            traceIds = traceIdsTemp.astype('int32')
            wavelengths[:] = tbdata[:]['pixelWave']
            xCenters[:] = tbdata[:]['xc']
            yCenters[:] = tbdata[:]['yc']

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
            lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float32')
            lineListArr[:,0] = tbdata.field(0)
            lineListArr[:,1] = tbdata.field(1)

            dispCorControl = drpStella.DispCorControl()
            dispCorControl.fittingFunction = self.config.function
            dispCorControl.order = self.config.order
            dispCorControl.searchRadius = self.config.searchRadius
            dispCorControl.fwhm = self.config.fwhm
            self.logger.debug('dispCorControl.fittingFunction = %s' % dispCorControl.fittingFunction)
            self.logger.debug('dispCorControl.order = %d' % dispCorControl.order)
            self.logger.debug('dispCorControl.searchRadius = %d' % dispCorControl.searchRadius)
            self.logger.debug('dispCorControl.fwhm = %g' % dispCorControl.fwhm)
            self.logger.debug('dispCorControl.maxDistance = %g' % dispCorControl.maxDistance)

            for i in range(spectrumSetFromProfile.size()):
                spec = spectrumSetFromProfile.getSpectrum(i)
                spec.setITrace(iTraces[i])
                specSpec = spec.getSpectrum()
                self.logger.debug('flatFiberTraceSet.getFiberTrace(%d).getITrace() = %d, spec.getITrace() = %d' %(i,flatFiberTraceSet.getFiberTrace(i).getITrace(), spec.getITrace()))
                self.logger.debug('type(spec) = %s: <%s>: <%s>' % (type(spec),type(spec.getSpectrum()),type(spec.getSpectrum()[0])))

                traceId = spec.getITrace()
    #            print 'traceId = ',traceId
                wLenTemp = np.ndarray( shape = traceIds.shape[0] / np.unique(traceIds).shape[0], dtype='float32' )
                k = 0
                l = -1
                for j in range(traceIds.shape[0]):
                    if traceIds[j] != l:
                        l = traceIds[j]
                    if traceIds[j] == traceIdsUnique[traceId]:
                        wLenTemp[k] = wavelengths[j]
                        k = k+1

                """cut off both ends of wavelengths where is no signal"""
                xCenter = flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().xCenter
                yCenter = flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yCenter
                yLow = flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yLow
                yHigh = flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yHigh
                yMin = yCenter + yLow
                yMax = yCenter + yHigh
                wLen = wLenTemp[ yMin + self.config.nRowsPrescan : yMax + self.config.nRowsPrescan + 1]
                wLenArr = np.ndarray(shape=wLen.shape, dtype='float32')
                for j in range(wLen.shape[0]):
                    wLenArr[j] = wLen[j]
                wLenLines = lineListArr[:,0]
                wLenLinesArr = np.ndarray(shape=wLenLines.shape, dtype='float32')
                for j in range(wLenLines.shape[0]):
                    wLenLinesArr[j] = wLenLines[j]
                lineListPix = drpStella.createLineList(wLenArr, wLenLinesArr)
                self.logger.debug('fiberTrace %d: yMin = %d' % (i, yMin))
                self.logger.debug('fiberTrace %d: yMax = %d' % (i, yMax))
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
