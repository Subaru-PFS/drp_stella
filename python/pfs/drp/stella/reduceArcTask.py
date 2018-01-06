import os
import numpy as np
if False:                               # will be imported if needed (matplotlib import can be slow)
    import matplotlib.pyplot as plt
import lsstDebug
import lsst.pex.config as pexConfig
from lsst.utils import getPackageDir
from lsst.pipe.base import TaskRunner, ArgumentParser, CmdLineTask
import lsst.afw.display as afwDisplay
from .calibrateWavelengthsTask import CalibrateWavelengthsTask
from .extractSpectraTask import ExtractSpectraTask
from .utils import makeFiberTraceSet, DetectorMapIO, makeDetectorMapIO
from .utils import readLineListFile, writePfsArm, addFiberTraceSetToMask
from lsst.obs.pfs.utils import getLampElements

class ReduceArcConfig(pexConfig.Config):
    """Configuration for reducing arc images"""
    extractSpectra = pexConfig.ConfigurableField(
        target=ExtractSpectraTask,
        doc="""Task to extract spectra using the fibre traces""",
    )
    calibrateWavelengths = pexConfig.ConfigurableField(
        target=CalibrateWavelengthsTask,
        doc="""Calibrate a SpectrumSet's wavelengths""",
    )
    lineList=pexConfig.Field(doc="reference line list including path",
                             dtype=str, default=os.path.join(getPackageDir("obs_pfs"),
                                                             "pfs/lineLists/ArCdHgKrNeXe.txt"));
    minArcLineIntensity=pexConfig.Field(doc="Minimum 'NIST' intensity to use emission lines",
                                        dtype=float, default=100);
    fiberDy=pexConfig.Field(doc="Offset to add to all FIBER_DY values (used when bootstrapping)",
                            dtype=float, default=0);
    randomSeed=pexConfig.Field(doc="Seed to pass to np.random.seed()", dtype=int, default=0)

class ReduceArcTaskRunner(TaskRunner):
    """Get parsed values into the ReduceArcTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(parsedCmd.id.refList, dict(butler=parsedCmd.butler))]

class ReduceArcTask(CmdLineTask):
    """Task to reduce Arc images"""
    ConfigClass = ReduceArcConfig
    RunnerClass = ReduceArcTaskRunner
    _DefaultName = "reduceArcTask"

    def __init__(self, *args, **kwargs):
        super(ReduceArcTask, self).__init__(*args, **kwargs)

        self.makeSubtask("calibrateWavelengths")
        self.makeSubtask("extractSpectra")

        self.debugInfo = lsstDebug.Info(__name__)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="raw",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
        return parser

    def run(self, expRefList, butler, immediate=True):
        self.log.debug('expRefList = %s' % expRefList)
        self.log.debug('len(expRefList) = %d' % len(expRefList))

        if self.config.randomSeed != 0:
            np.random.seed(self.config.randomSeed)

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

            detectorMap = butler.get('detectormap', arcRef.dataId)

            if self.config.fiberDy != 0.0:
                slitOffsets = detectorMap.getSlitOffsets()
                slitOffsets[detectorMap.FIBER_DY] += self.config.fiberDy
                detectorMap.setSlitOffsets(slitOffsets)

            flatFiberTraceSet = makeFiberTraceSet(fiberTrace)
            self.log.debug('fiberTrace calibration file contains %d fibers' % flatFiberTraceSet.getNtrace())

            arcExp = None
            for dataType in ["calexp", "postISRCCD"]:
                if arcRef.datasetExists(dataType):
                    arcExp = arcRef.get(dataType)
                    break

            if arcExp is None:
                raise RuntimeError("Unable to load postISRCCD or calexp image for %s" % (arcRef.dataId))

            # read line list
            lamps = getLampElements(arcExp.getMetadata())
            self.log.info("Arc lamp elements are: %s" % " ".join(lamps))
            arcLines = readLineListFile(self.config.lineList, lamps,
                                        minIntensity=self.config.minArcLineIntensity)
            arcLineWavelengths = np.array([line.wavelength for line in arcLines], dtype='float32')

            if self.debugInfo.display and self.debugInfo.arc_frame >= 0:
                display = afwDisplay.Display(self.debugInfo.arc_frame)

                display.setMaskPlaneColor("FIBERTRACE", afwDisplay.YELLOW)
                addFiberTraceSetToMask(arcExp.maskedImage.mask, flatFiberTraceSet)
                
                display.setMaskTransparency(50)
                display.mtv(arcExp, "Arcs %(visit)d %(arm)s%(spectrograph)d" % (arcRef.dataId))

            # optimally extract arc spectra
            self.log.info('extracting arc spectra from %(visit)d %(arm)s%(spectrograph)d' % arcRef.dataId)

            spectrumSet = self.extractSpectra.run(arcExp, flatFiberTraceSet, detectorMap).spectrumSet

            self.log.info('calibrating wavelengths for %(visit)d %(arm)s%(spectrograph)d' % arcRef.dataId)
            self.calibrateWavelengths.run(detectorMap, spectrumSet, arcLines)

            writePfsArm(butler, arcExp, spectrumSet, arcRef.dataId)
            #
            # Now the updated DetectorMap.  We could derive this task from CalibTask, except
            # that that depends on BatchPoolTask and that'd be a pain as it assumes multiprocessing
            #
            detectorMapIO = makeDetectorMapIO(detectorMap, arcExp.getInfo().getVisitInfo())
            arcRef.put(detectorMapIO, 'detectormap', visit0=arcRef.dataId['visit'])
            #
            # Done; time for debugging plots
            #
            if self.debugInfo.display and self.debugInfo.arc_frame >= 0 and self.debugInfo.showArcLines:
                for spec in spectrumSet:
                    fiberId = spec.getFiberId()

                    x = detectorMap.findPoint(fiberId, arcLines[0].wavelength)[0]
                    y = 0.5*arcExp.getHeight()
                    display.dot(str(fiberId), x, y + 10*(fiberId%2), ctype='blue')

                    for rl in arcLines:
                        x, y = detectorMap.findPoint(fiberId, rl.wavelength)
                        display.dot('o', x, y, ctype='blue')

            if self.debugInfo.display and self.debugInfo.residuals_frame >= 0:
                display = afwDisplay.Display(self.debugInfo.residuals_frame)
                residuals = arcExp.maskedImage.clone()

                for ft, spec in zip(flatFiberTraceSet, spectrumSet):
                    reconIm = ft.getReconstructed2DSpectrum(spec)
                    reconIm *= detectorMap.getThroughput(spec.getFiberId())
                    residuals[reconIm.getBBox()] -= reconIm

                display.mtv(residuals, title="Residuals %(visit)d %(arm)s%(spectrograph)d" % (arcRef.dataId))
    #
    # Disable writing metadata (doesn't work with lists of dataRefs anyway)
    #
    def _getMetadataName(self):
        return None
