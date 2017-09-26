import os

import numpy as np

import lsstDebug
import lsst.log as log
from lsst.pex.config import Config, Field
from lsst.pipe.base import TaskRunner, ArgumentParser, CmdLineTask
from lsst.utils import getPackageDir
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask
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
        return [(parsedCmd.id.refList,
                 dict(butler=parsedCmd.butler, wLenFile=parsedCmd.wLenFile, lineList=parsedCmd.lineList))]

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
        del xCenters

        # read line list
        lineListArr = readLineListFile(lineList)
        wLenLinesArr = np.array(lineListArr[:, 0])

        for arcRef in expRefList:
            self.log.debug('arcRef.dataId = %s' % arcRef.dataId)
            self.log.debug('arcRef = %s' % arcRef)
            self.log.debug('type(arcRef) = %s' % type(arcRef))

            # read pfsFiberTrace and then construct FiberTraceSet
            try:
                self.log.debug('fiberTrace file name = %s' % (arcRef.get('fibertrace_filename')))
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

                addFiberTraceSetToMask(arcExp.maskedImage.mask, flatFiberTraceSet)
                
                display.setMaskTransparency(50)
                display.mtv(arcExp, "Arcs")

            # optimally extract arc spectra
            self.log.info('extracting arc spectra from %s', arcRef.dataId)

            extractSpectraTask = ExtractSpectraTask()
            spectrumSet = extractSpectraTask.run(arcExp, flatFiberTraceSet).spectrumSet

            # Fit the wavelength solution

            dispCorControl = drpStella.DispCorControl()
            dispCorControl.fittingFunction = self.config.function
            dispCorControl.order = self.config.order
            dispCorControl.searchRadius = self.config.searchRadius
            dispCorControl.fwhm = self.config.fwhm
            dispCorControl.maxDistance = self.config.maxDistance

            if self.debugInfo.display and self.debugInfo.residuals_frame >= 0:
                display = afwDisplay.Display(self.debugInfo.residuals_frame)
                residuals = arcExp.maskedImage.clone()
            else:
                residuals = None

            for i in range(spectrumSet.size()):
                spec = spectrumSet.getSpectrum(i)

                traceId = spec.getITrace()
                wLenTemp = wavelengths[np.where(traceIds == traceId)]
                wLenTemp = wLenTemp[self.config.nRowsPrescan:] # this should be fixed in the wavelengths file

                # cut off both ends of wavelengths where is no signal
                bbox = flatFiberTraceSet.getFiberTrace(i).getTrace().getBBox()
                wLenArr = np.array(wLenTemp[bbox.getMinY() : bbox.getMaxY() + 1])

                lineListPix = drpStella.createLineList(wLenArr, wLenLinesArr)

                # Identify emission lines and fit dispersion
                spec.identify(lineListPix, dispCorControl, 8)

                self.log.info("FiberTrace %d: spec.getDispRms() = %f" % (i, spec.getDispRms()))

                spectrumSet.setSpectrum(i, spec)

                if residuals is not None:
                    ft = flatFiberTraceSet.getFiberTrace(i)
                    reconIm = ft.getReconstructed2DSpectrum(spec)
                    residuals[reconIm.getBBox()] -= reconIm

            writePfsArm(butler, arcExp, spectrumSet, arcRef.dataId)

        if residuals is not None:
            display.mtv(residuals, title='Residuals')
            
        return spectrumSet
    #
    # Disable writing metadata (doesn't work with lists of dataRefs anyway)
    #
    def _getMetadataName(self):
        return None
