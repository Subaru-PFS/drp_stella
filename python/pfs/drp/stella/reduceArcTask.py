import os
import sys

from astropy.io import fits as pyfits
import numpy as np

import lsstDebug
import lsst.log as log
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, TaskRunner, ArgumentParser, CmdLineTask
from lsst.utils import getPackageDir
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO
from pfs.drp.stella.utils import makeFiberTraceSet, readWavelengthFile
from pfs.drp.stella.utils import readLineListFile, writePfsArm, addFiberTraceSetToMask

class ReduceArcConfig(Config):
    """Configuration for reducing arc images"""
    function = Field( doc = "Function for fitting the dispersion", dtype=str, default="POLYNOMIAL" );
    order = Field( doc = "Fitting function order", dtype=int, default = 5 );
    searchRadius = Field( doc = "Radius in pixels relative to line list to search for emission line peak", dtype = int, default = 2 );
    fwhm = Field( doc = "FWHM of emission lines", dtype=float, default = 2.6 );
    nRowsPrescan = Field( doc = "Number of prescan rows in raw CCD image", dtype=int, default = 49 );
    wavelengthFile = Field( doc = "reference pixel-wavelength file including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/RedFiberPixels.fits.gz"));
    lineList = Field( doc = "reference line list including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/lineLists/CdHgKrNeXe_red.fits"));
    maxDistance = Field( doc = "Reject emission lines which center is more than this value away from the predicted position", dtype=float, default = 2.5 );

class ReduceArcTaskRunner(TaskRunner):
    """Get parsed values into the ReduceArcTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, wLenFile=parsedCmd.wLenFile, lineList=parsedCmd.lineList)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            self.log.debug('ReduceArcTask.__call__: args = %s' % args)
            result = task.run(**args)
        else:
            try:
                result = task.run(**args)
            except Exception, e:
                task.log.warn("Failed: %s" % e)

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

        self.debugInfo = lsstDebug.Info(__name__)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="raw",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
        parser.add_argument("--wLenFile", help='directory and name of pixel vs. wavelength file')
        parser.add_argument("--lineList", help='directory and name of line list')
        return parser

    def run(self, expRefList, butler, wLenFile=None, lineList=None, immediate=True):
        if wLenFile == None:
            wLenFile = self.config.wavelengthFile
        if lineList == None:
            lineList = self.config.lineList
        self.log.debug('expRefList = %s' % expRefList)
        self.log.debug('len(expRefList) = %d' % len(expRefList))
        self.log.debug('wLenFile = %s' % wLenFile)
        self.log.debug('lineList = %s' % lineList)

        # read wavelength file
        xCenters, wavelengths, traceIds = readWavelengthFile(wLenFile)

        # read line list
        lineListArr = readLineListFile(lineList)

        for arcRef in expRefList:
            self.log.debug('arcRef.dataId = %s' % arcRef.dataId)
            self.log.debug('arcRef = %s' % arcRef)
            self.log.debug('type(arcRef) = %s' % type(arcRef))

            # read pfsFiberTrace and then construct FiberTraceSet
            try:
                fiberTrace = arcRef.get('fibertrace')
            except Exception, e:
                raise RuntimeError("Unable to load fiberTrace for %s: %s" % (arcRef.dataId, e))

            flatFiberTraceSet = makeFiberTraceSet(fiberTrace)
            self.log.debug('fiberTrace calibration file contains %d fibers' % flatFiberTraceSet.size())

            arcExp = None
            for dataType in ["calexp", "postISRCCD"]:
                if arcRef.datasetExists(dataType):
                    arcExp = arcRef.get(dataType)
                    break

            if arcExp is None:
                raise RuntimeError("Unable to load postISRCCD or calexp image for %s" % (arcRef.dataId))

            if self.debugInfo.display and self.debugInfo.arc_frame >= 0:
                display = afwDisplay.Display(self.debugInfo.arc_frame)

                addFiberTraceSetToMask(inExposure.getMaskedImage().getMask(),
                                       inFiberTraceSetWithProfiles.getTraces(), display)
                
                display.setMaskTransparency(50)
                display.mtv(arcExp, "Arcs")

            # optimally extract arc spectra
            self.log.info('extracting arc spectra from %s', arcRef.dataId)

            # assign trace number to flatFiberTraceSet
            drpStella.assignITrace( flatFiberTraceSet, traceIds, xCenters )
            for i in range( flatFiberTraceSet.size() ):
                self.log.debug('iTraces[%d] = %d' % (i, flatFiberTraceSet.getFiberTrace(i).getITrace()))

            extractSpectraTask = ExtractSpectraTask()
            spectrumSetFromProfile = extractSpectraTask.run(arcExp, flatFiberTraceSet)

            dispCorControl = drpStella.DispCorControl()
            dispCorControl.fittingFunction = self.config.function
            dispCorControl.order = self.config.order
            dispCorControl.searchRadius = self.config.searchRadius
            dispCorControl.fwhm = self.config.fwhm
            dispCorControl.maxDistance = self.config.maxDistance
            self.log.debug('dispCorControl.fittingFunction = %s' % dispCorControl.fittingFunction)
            self.log.debug('dispCorControl.order = %d' % dispCorControl.order)
            self.log.debug('dispCorControl.searchRadius = %d' % dispCorControl.searchRadius)
            self.log.debug('dispCorControl.fwhm = %g' % dispCorControl.fwhm)
            self.log.debug('dispCorControl.maxDistance = %g' % dispCorControl.maxDistance)

            for i in range(spectrumSetFromProfile.size()):
                spec = spectrumSetFromProfile.getSpectrum(i)
                specSpec = spec.getSpectrum()
                self.log.debug('specSpec.shape = %d' % specSpec.shape)
                self.log.debug('lineListArr.shape = [%d,%d]' % (lineListArr.shape[0], lineListArr.shape[1]))
                self.log.debug('type(specSpec) = %s: <%s>' % (type(specSpec),type(specSpec[0])))
                self.log.debug('type(lineListArr) = %s: <%s>' % (type(lineListArr),type(lineListArr[0][0])))
                self.log.debug('type(spec) = %s: <%s>: <%s>' % (type(spec),type(spec.getSpectrum()),type(spec.getSpectrum()[0])))

                traceId = spec.getITrace()
                wLenTemp = wavelengths[np.where(traceIds == traceId)]
                assert wLenTemp.shape[0] == traceIds.shape[0] / np.unique(traceIds).shape[0]

                # cut off both ends of wavelengths where is no signal
                xCenter = flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().xCenter
                yCenter = flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yCenter
                yLow = flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yLow
                yHigh = flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yHigh
                yMin = yCenter + yLow
                yMax = yCenter + yHigh
                self.log.debug('fiberTrace %d: xCenter = %d' % (i, xCenter))
                self.log.debug('fiberTrace %d: yCenter = %d' % (i, yCenter))
                self.log.debug('fiberTrace %d: yLow = %d' % (i, yLow))
                self.log.debug('fiberTrace %d: yHigh = %d' % (i, yHigh))
                self.log.debug('fiberTrace %d: yMin = %d' % (i, yMin))
                self.log.debug('fiberTrace %d: yMax = %d' % (i, yMax))
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
                    lineListPix = drpStella.createLineList(wLenArr, wLenLinesArr)
                except Exception as e:
                    self.log.warn("Error occured during createLineList: %s" % (e.message))

                # Idendify emission lines and fit dispersion
                try:
                    spec.identify(lineListPix, dispCorControl, 8)
                except Exception as e:
                    self.log.warn("Error occured during identify: %s" % (e.message))

                self.log.trace("FiberTrace %d: spec.getDispCoeffs() = %s"
                               % (i, np.array_str(spec.getDispCoeffs())))
                self.log.info("FiberTrace %d: spec.getDispRms() = %f"
                              % (i, spec.getDispRms()))

                spectrumSetFromProfile.setSpectrum(i, spec)

            writePfsArm(butler, arcExp, spectrumSetFromProfile, arcRef.dataId)

        return spectrumSetFromProfile
