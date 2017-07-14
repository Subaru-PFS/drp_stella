import os

import numpy as np

import lsstDebug
import lsst.log as log
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, TaskRunner, ArgumentParser, CmdLineTask
from lsst.utils import getPackageDir
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask
from pfs.drp.stella.math import makeArtificialSpectrum
from pfs.drp.stella.utils import createLineListForLamps, findPixelOffsetFunction
from pfs.drp.stella.utils import getLineList, makeFiberTraceSet
from pfs.drp.stella.utils import measureLinesInPixelSpace, readWavelengthFile
from pfs.drp.stella.utils import writePfsArm

lineListFileName = os.path.join(
    getPackageDir("obs_pfs"),
    "pfs/lineLists/NeXeHgAr_%d%s.fits") # % (spectrograph, arm)

class ReduceArcConfig(Config):
    """Configuration for reducing arc images"""
    elements = Field(
        doc = "Element(s) to create line list for, separated by ',' (e.g. Hg,Ar)",
        dtype = str,
        default = "Hg,Ar"
    )
    function = Field(
        doc = "Function for fitting the dispersion",
        dtype = str,
        default = "POLYNOMIAL"
    )
    fwhm = Field(
        doc = "FWHM of emission lines in pixels",
        dtype = float,
        default = 2.6
    )
    lineListSuffix = Field(
        doc = "Suffix of line list to read (vac or air)",
        dtype = str,
        default = 'vac'
    )
    maxDistance = Field(
        doc = "Reject emission lines which center is more than this value in pixels away from the predicted position",
        dtype = float,
        default = 3.2
    )
    minDistance = Field(
        doc="Minimum distance between lines for creation of line list in FWHM (see below)",
        dtype = float,
        default = 2.1
    )
    minDistanceLines = Field(
        doc="Minimum distance between 2 lines to be identified",
        dtype = float,
        default = 1.5
    )
    minErr = Field(
        doc="Minimum measure error for PolyFit",
        dtype = float,
        default = 0.01
    )
    minPercentageOfLines = Field(
        doc = "Minimum percentage of lines to be identified for <identify> to pass",
        dtype = float,
        default = 66.6
    )
    nIterReject = Field(
        doc = "Number of sigma rejection iterations",
        dtype = int,
        default = 3
    )
    nRowsPrescan = Field(
        doc = "Number of prescan rows in raw CCD image",
        dtype = int,
        default = 48
    )
    order = Field(
        doc = "Fitting function order",
        dtype = int,
        default = 6
    )
    percentageOfLinesForCheck = Field(
        doc = "Hold back this percentage of lines in the line list for check",
        dtype = int,
        default = 10
    )
    plot = Field(
        doc = "FiberTraceId to plot (set to -1 for none)",
        dtype = int,
        default = -1
    )
    removeLines = Field(
        doc = "Remove lines from line list (e.g. HgII,NeII)",
        dtype = str,
        default = 'HgII,NeII'
    )
    searchRadius = Field(
        doc = "Radius in pixels relative to line list to search for emission line peak",
        dtype = int,
        default = 1
    )
    sigmaReject = Field(
        doc = "Sigma rejection threshold for polynomial fitting",
        dtype = float,
        default = 2.5,
    )
    wavelengthFile = Field(
        doc = "Reference pixel-wavelength file including path",
        dtype = str,
        default = os.path.join(getPackageDir("obs_pfs"), "pfs/RedFiberPixels.fits.gz")
    )
    xCorRadius = Field(
        doc = "Radius in pixels for cross correlating spectrum and line list",
        dtype = int,
        default = 15,
    )

class ReduceArcTaskRunner(TaskRunner):
    """Get parsed values into the ReduceArcTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(parsedCmd.id.refList, dict(butler=parsedCmd.butler, wLenFile=parsedCmd.wLenFile))]

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

    def run(self,
            expRefList,
            butler,
            wLenFile=None,
            immediate=True,
           ):
        """
        @param expRefList : reference list of Arc exposures
        @param butler : butler to use
        @param wLenFile : simulator output with the predicted wavelengths for
                          each pixel
        @param immediate : let butler read file immediately or only when needed?
        """
        # Silence verbose loggers
        for logger in ["afw.ExposureFormatter",
                       "afw.image.ExposureInfo",
                       "afw.image.Mask",
                       "afw.ImageFormatter",
                       "CameraMapper",
                       "createLineListForFiberTrace",
                       "createLineListForLamps",
                       "daf.persistence.butler",
                       "daf.persistence.LogicalLocation",
                       "extractSpectra",
                       "gaussFit",
                       "getLinesInWavelengthRange",
                       "makeArtificialSpectrum",
                       "measureLines",
                       "measureLinesInPixelSpace",
                       "pfs::drp::stella::FiberTrace::createTrace",
                       "pfs::drp::stella::math::assignITrace",
                       "pfs::drp::stella::math::CurfFitting::PolyFit",
                       "pfs::drp::stella::math::findITrace",
                       "removeBadLines",
                      ]:
            log.setLevel(logger, log.WARN)
        for logger in ["gaussFunc",
                       "pfs::drp::stella::Spectra::identify",
                      ]:
            log.setLevel(logger, log.FATAL)

        if wLenFile == None:
            wLenFile = self.config.wavelengthFile
        self.log.debug('len(expRefList) = %d' % len(expRefList))
        self.log.debug('wLenFile = %s' % wLenFile)

        # read wavelength file
        xCenters, lambdaPix, traceIds = readWavelengthFile(wLenFile)

        # create DispCorControl
        dispCorControl = drpStella.DispCorControl()
        dispCorControl.fittingFunction = self.config.function
        dispCorControl.fwhm = self.config.fwhm
        dispCorControl.maxDistance = self.config.maxDistance
        dispCorControl.minDistanceLines = self.config.minDistanceLines
        dispCorControl.minErr = self.config.minErr
        dispCorControl.minPercentageOfLines = self.config.minPercentageOfLines
        dispCorControl.nIterReject = self.config.nIterReject
        dispCorControl.order = self.config.order
        dispCorControl.percentageOfLinesForCheck = self.config.percentageOfLinesForCheck
        dispCorControl.searchRadius = self.config.searchRadius
        dispCorControl.sigmaReject = self.config.sigmaReject
        dispCorControl.verticalPrescanHeight = self.config.nRowsPrescan

        self.log.trace('dispCorControl.fittingFunction = %s' % dispCorControl.fittingFunction)
        self.log.trace('dispCorControl.order = %d' % dispCorControl.order)
        self.log.trace('dispCorControl.searchRadius = %d' % dispCorControl.searchRadius)
        self.log.trace('dispCorControl.fwhm = %g' % dispCorControl.fwhm)
        self.log.trace('dispCorControl.maxDistance = %g' % dispCorControl.maxDistance)

        # create the line list from the master line list
        lines = createLineListForLamps(self.config.elements,
                                       self.config.lineListSuffix,
                                       self.config.removeLines)
        self.log.info('raw line list contains %d lines' % (len(lines)))

        measuredLinesPerArc = []
        offsetPerArc = []
        for arcRef in expRefList:
            self.log.debug('arcRef.dataId = %s' % arcRef.dataId)

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

                addFiberTraceSetToMask(arcExp.maskedImage.mask, flatFiberTraceSet, display)

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

            measuredLinesPerFiberTrace = []
            offsetPerFiberTrace = []
            for i in range(spectrumSetFromProfile.size()):
                spec = spectrumSetFromProfile.getSpectrum(i)
                self.log.trace('i = %d: spec.getITrace() = %d' % (i, spec.getITrace()))
                fluxPix = spec.getSpectrum()
                self.log.trace('fluxPix.shape = %d' % fluxPix.shape)
                self.log.trace('type(fluxPix) = %s: <%s>' % (type(fluxPix),type(fluxPix[0])))
                self.log.trace('type(spec) = %s: <%s>: <%s>'
                    % (type(spec),type(spec.getSpectrum()),type(spec.getSpectrum()[0])))

                traceId = spec.getITrace()
                self.log.debug('traceId = %d' % traceId)

                # cut off both ends of wavelengths where is no signal
                yMin = (flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yCenter +
                        flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yLow)
                yMax = (flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yCenter +
                        flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().yHigh)
                self.log.debug('fiberTrace %d: yMin = %d' % (i, yMin))
                self.log.debug('fiberTrace %d: yMax = %d' % (i, yMax))

                startIndex = drpStella.firstIndexWithValueGEFrom(traceIds,traceId)
                self.log.trace('startIndex = %d' % startIndex)
                self.log.trace('copying lambdaPix[%d:%d]'
                    % (startIndex + self.config.nRowsPrescan + yMin,
                       startIndex + self.config.nRowsPrescan + yMax + 1))
                lambdaPixFiberTrace = lambdaPix[startIndex + self.config.nRowsPrescan + yMin:
                                                startIndex + self.config.nRowsPrescan + yMax + 1]
                if len(lambdaPixFiberTrace) != len(fluxPix):
                    raise RuntimeError(
                        "reduceArcTask.py: ERROR: len(lambdaPixFiberTrace)(=%d) != len(fluxPix)(=%d)"
                        % (len(lambdaPixFiberTrace), len(fluxPix)))

                lineList = getLineList(lineListFileName % (arcRef.dataId['spectrograph'],
                                                           arcRef.dataId['arm']),
                                       self.config.elements)
                self.log.debug('type(lineList) = %s, len(lineList) = %d'
                               % (type(lineList), len(lineList)))

                artificialSpectrum = makeArtificialSpectrum(lambdaPixFiberTrace, lineList)
                assert(len(fluxPix) == len(lambdaPixFiberTrace))
                lambdaPixShifted, artificialSpectrum, offset = findPixelOffsetFunction(fluxPix,
                                                                                       artificialSpectrum,
                                                                                       lambdaPixFiberTrace,
                                                                                       self.config.xCorRadius)
                self.log.trace('offset = %s' % (np.array_str(offset)))
                measuredLines = measureLinesInPixelSpace(lines = lineList,
                                                         lambdaPix = lambdaPixShifted,
                                                         fluxPix = fluxPix,
                                                         fwhm = self.config.fwhm)
                for line in measuredLines:
                    line.flags += 'g'

                self.log.debug('new line list created')
                self.log.info('len(lineList) = %d' % (len(lineList)))

                # use line list to calibrate the Arc spectrum
                try:
                    spec.identifyF(measuredLines, dispCorControl)
                except Exception, e:
                    raise RuntimeError(
                        "reduceArcTask.py: %dth FiberTrace: traceId = %d: ERROR: %s"
                        % (i,traceId,e.message))
                spectrumSetFromProfile.setSpectrum(i, spec)

                measuredLinesPerFiberTrace.append(measuredLines)
                offsetPerFiberTrace.append(offset)

                self.log.trace("FiberTrace %d: spec.getWavelength() = %s"
                    % (i, np.array_str(spec.getWavelength())))
                self.log.trace("FiberTrace %d: spec.getDispCoeffs() = %s"
                    % (i,np.array_str(spec.getDispCoeffs())))
                self.log.info("FiberTrace %d (ID=%d): spec.getDispRms() = %f"
                    % (i, spec.getITrace(), spec.getDispRms()))
                self.log.info("FiberTrace %d (ID=%d): spec.getDispRmsCheck() = %f"
                    % (i, spec.getITrace(), spec.getDispRmsCheck()))
                spectrumSetFromProfile.setSpectrum(i, spec)

            measuredLinesPerArc.append(measuredLinesPerFiberTrace)
            offsetPerArc.append(offsetPerFiberTrace)
            writePfsArm(butler, arcExp, spectrumSetFromProfile, arcRef.dataId)
        return [spectrumSetFromProfile,
                measuredLinesPerArc,
                offsetPerArc,
                flatFiberTraceSet]

    #
    # Disable writing metadata (doesn't work with lists of dataRefs anyway)
    #
    def _getMetadataName(self):
        return None
