#/Users/azuri/stella-git/drp_stella/bin.src/reduceArc.py '/Users/azuri/spectra/pfs/PFS' --id visit=4 --wLenFile '/Users/azuri/stella-git/obs_pfs/pfs/RedFiberPixels.fits.gz' --lineList '/Users/azuri/stella-git/obs_pfs/pfs/lineLists/CdHgKrNeXe_red.fits' --loglevel 'info' --calib '/Users/azuri/spectra/pfs/PFS/CALIB/' --output '/Users/azuri/spectra/pfs/PFS'
import os
import sys

import pfs.drp.stella.extractSpectraTask as esTask
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import makeFiberTraceSet
from lsst.utils import getPackageDir
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, TaskRunner, ArgumentParser, CmdLineTask
import numpy as np
from astropy.io import fits as pyfits
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO

class ReduceArcConfig(Config):
    """Configuration for reducing arc images"""
    function = Field( doc = "Function for fitting the dispersion", dtype=str, default="POLYNOMIAL" );
    order = Field( doc = "Fitting function order", dtype=int, default = 5 );
    searchRadius = Field( doc = "Radius in pixels relative to line list to search for emission line peak", dtype = int, default = 2 );
    fwhm = Field( doc = "FWHM of emission lines", dtype=float, default = 2.6 );
    nRowsPrescan = Field( doc = "Number of prescan rows in raw CCD image", dtype=int, default = 49 );
    wavelengthFile = Field( doc = "reference pixel-wavelength file including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/RedFiberPixels.fits.gz"));
    lineList = Field( doc = "reference line list including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/lineLists/CdHgKrNeXe_red.fits"));

class ReduceArcTaskRunner(TaskRunner):
    """Get parsed values into the ReduceArcTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        print 'ReduceArcTask.getTargetList: kwargs = ',kwargs
        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, wLenFile=parsedCmd.wLenFile, lineList=parsedCmd.lineList)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            self.log.info('ReduceArcTask.__call__: args = %s' % args)
            result = task.run(**args)
        else:
            try:
                result = task.run(**args)
            except Exception, e:
                task.log.fatal("Failed: %s" % e)
#                traceback.print_exc(file=sys.stderr)

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
        self.log.info('expRefList = %s' % expRefList)
        self.log.info('len(expRefList) = %d' % len(expRefList))
        self.log.info('wLenFile = %s' % wLenFile)
        self.log.info('lineList = %s' % lineList)

        for arcRef in expRefList:
            self.log.info('arcRef.dataId = %s' % arcRef.dataId)
            self.log.info('arcRef = %s' % arcRef)
            self.log.info('type(arcRef) = %s' % type(arcRef))

            try:
                fiberTrace = arcRef.get('fiberTrace', immediate=True)
            except Exception, e:
                raise RuntimeError("Unable to retrieve fiberTrace for %s: %s" % (arcRef.dataId, e))

            arcExp = arcRef.get("postISRCCD", immediate=True)
            self.log.info('arcExp = %s' % arcExp)
            self.log.info('type(arcExp) = %s' % type(arcExp))
            flatFiberTraceSet = makeFiberTraceSet(fiberTrace)

            """ optimally extract arc spectra """
            print 'extracting arc spectra'

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
            success = drpStella.assignITrace( flatFiberTraceSet, traceIds, xCenters, yCenters )
            iTraces = np.ndarray(shape=flatFiberTraceSet.size(), dtype='intp')
            for i in range( flatFiberTraceSet.size() ):
                iTraces[i] = flatFiberTraceSet.getFiberTrace(i).getITrace()

            if success == False:
                print 'assignITrace FAILED'

            myExtractTask = esTask.ExtractSpectraTask()
            aperturesToExtract = [-1]
            spectrumSetFromProfile = myExtractTask.run(arcExp, flatFiberTraceSet, aperturesToExtract)

            for i in range(spectrumSetFromProfile.size()):
                # THIS DOESN'T WORK, again a problem with changing getSpectrum()
                spectrumSetFromProfile.getSpectrum(i).setITrace(flatFiberTraceSet.getFiberTrace(i).getITrace())

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
            self.log.info('dispCorControl.fittingFunction = %s' % dispCorControl.fittingFunction)
            self.log.info('dispCorControl.order = %d' % dispCorControl.order)
            self.log.info('dispCorControl.searchRadius = %d' % dispCorControl.searchRadius)
            self.log.info('dispCorControl.fwhm = %g' % dispCorControl.fwhm)

            for i in range(spectrumSetFromProfile.size()):
                spec = spectrumSetFromProfile.getSpectrum(i)
                spec.setITrace(iTraces[i])
                self.log.info('flatFiberTraceSet.getFiberTrace(%d).getITrace() = %d, spec.getITrace() = %d' %(i,flatFiberTraceSet.getFiberTrace(i).getITrace(), spec.getITrace()))
                specSpec = spec.getSpectrum()
                self.log.info('yCenters.shape = %d' % yCenters.shape)
                self.log.info('specSpec.shape = %d' % specSpec.shape)
                self.log.info('lineListArr.shape = [%d,%d]' % (lineListArr.shape[0], lineListArr.shape[1]))
                self.log.info('type(specSpec) = %s: <%s>' % (type(specSpec),type(specSpec[0])))
                self.log.info('type(lineListArr) = %s: <%s>' % (type(lineListArr),type(lineListArr[0][0])))
                self.log.info('type(spec) = %s: <%s>: <%s>' % (type(spec),type(spec.getSpectrum()),type(spec.getSpectrum()[0])))

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
                self.log.info('fiberTrace %d: xCenter = %d' % (i, xCenter))
                self.log.info('fiberTrace %d: yCenter = %d' % (i, yCenter))
                self.log.info('fiberTrace %d: yLow = %d' % (i, yLow))
                self.log.info('fiberTrace %d: yHigh = %d' % (i, yHigh))
                self.log.info('fiberTrace %d: yMin = %d' % (i, yMin))
                self.log.info('fiberTrace %d: yMax = %d' % (i, yMax))
                wLen = wLenTemp[ yMin + self.config.nRowsPrescan : yMax + self.config.nRowsPrescan + 1]
                wLenArr = np.ndarray(shape=wLen.shape, dtype='float32')
                for j in range(wLen.shape[0]):
                    wLenArr[j] = wLen[j]
                wLenLines = lineListArr[:,0]
                wLenLinesArr = np.ndarray(shape=wLenLines.shape, dtype='float32')
                for j in range(wLenLines.shape[0]):
                    wLenLinesArr[j] = wLenLines[j]
                lineListPix = drpStella.createLineList(wLenArr, wLenLinesArr)
                try:
                    spec.identifyF(lineListPix, dispCorControl, 8)
                except:
                    e = sys.exc_info()[1]
                    message = str.split(e.message, "\n")
                    for k in range(len(message)):
                        print "element",k,": <",message[k],">"
                print "FiberTrace ",i,": spec.getDispCoeffs() = ",spec.getDispCoeffs()
                print "FiberTrace ",i,": spec.getDispRms() = ",spec.getDispRms()
                if spectrumSetFromProfile.setSpectrum(i, spec ):
                    print 'setSpectrum for spectrumSetFromProfile[',i,'] done'
                else:
                    print 'setSpectrum for spectrumSetFromProfile[',i,'] failed'

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
                self.log.info('No pfsConfigId is present in postISRCCD file for dataId %s' %
                              str(dataId.items()))
                pfsConfigId = 0x0
                                                                                                 
            pfsConfig = butler.get("pfsConfig", pfsConfigId=pfsConfigId, dateObs=dataId["dateObs"])

            pfsArm = spectrumSetToPfsArm(pfsConfig, spectrumSetFromProfile,
                                         dataId["visit"], dataId["spectrograph"], dataId["arm"])
            butler.put(PfsArmIO(pfsArm), 'pfsArm', dataId)

        return spectrumSetFromProfile
