import os

import numpy as np
if False:                               # will be imported if needed (matplotlib import can be slow)
    import matplotlib.pyplot as plt
import lsstDebug
import lsst.log as log
import lsst.pex.config as pexConfig
from lsst.pipe.base import TaskRunner, ArgumentParser, CmdLineTask
from lsst.utils import getPackageDir
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
from pfs.drp.stella.extractSpectraTask import ExtractSpectraTask
from pfs.drp.stella.utils import makeFiberTraceSet, makeDetectorMap, getLampElements
from pfs.drp.stella.utils import readLineListFile, writePfsArm, addFiberTraceSetToMask

@pexConfig.wrap(drpStella.DispCorControl) # should wrap IdentifyLinesTaskConfig when it's written
class ReduceArcConfig(pexConfig.Config):
    """Configuration for reducing arc images"""
    extractSpectra = pexConfig.ConfigurableField(
        target=ExtractSpectraTask,
        doc="""Task to extract spectra using the fibre traces""",
    )

    fittingFunction=pexConfig.Field(doc="Function for fitting the dispersion", dtype=str, default="POLYNOMIAL");
    order=pexConfig.Field(doc="Fitting function order", dtype=int, default=5);
    searchRadius=pexConfig.Field(doc="Radius in pixels relative to line list to search for emission line peak",
                                 dtype=int, default=2);
    fwhm=pexConfig.Field(doc="FWHM of emission lines", dtype=float, default=2.6);
    wavelengthFile=pexConfig.Field( doc="reference pixel-wavelength file including path",
                                    dtype=str, default=os.path.join(getPackageDir("obs_pfs"),
                                                                    "pfs/RedFiberPixels.fits.gz"));
    lineList=pexConfig.Field(doc="reference line list including path",
                             dtype=str, default=os.path.join(getPackageDir("obs_pfs"),
                                                             "pfs/lineLists/CdHgKrNeXe_red.fits"));
    maxDistance=pexConfig.Field(doc="Reject arc lines with center more than maxDistance from predicted position",
                                dtype=float, default=2.5);
    minArcLineIntensity=pexConfig.Field(doc="Minimum 'NIST' intensity to use emission lines",
                                        dtype=float, default=0);
    nLinesKeptBack=pexConfig.Field(doc="Number of lines to withhold from line fitting to estimate errors",
                                   dtype=int, default=4);

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

        self.makeSubtask("extractSpectra")

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

            detMap = makeDetectorMap(butler, arcRef.dataId, wLenFile)

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
            lamps = getLampElements(arcExp)
            arcLines = readLineListFile(lineList, lamps, minIntensity=self.config.minArcLineIntensity)
            arcLineWavelengths = np.array(arcLines[:, 0])

            if self.debugInfo.display and self.debugInfo.arc_frame >= 0:
                display = afwDisplay.Display(self.debugInfo.arc_frame)

                addFiberTraceSetToMask(arcExp.maskedImage.mask, flatFiberTraceSet)
                
                display.setMaskTransparency(50)
                display.mtv(arcExp, "Arcs")

            # optimally extract arc spectra
            self.log.info('extracting arc spectra from %s', arcRef.dataId)

            spectrumSet = self.extractSpectra.run(arcExp, flatFiberTraceSet).spectrumSet

            # Fit the wavelength solution
            dispCorControl = self.config.makeControl()

            if self.debugInfo.display and self.debugInfo.residuals_frame >= 0:
                display = afwDisplay.Display(self.debugInfo.residuals_frame)
                residuals = arcExp.maskedImage.clone()
            else:
                residuals = None

            for i in range(spectrumSet.getNtrace()):
                spec = spectrumSet.getSpectrum(i)

                traceId = spec.getITrace()
                fiberWavelengths = detMap.getWavelength(traceId)

                assert len(fiberWavelengths) == arcExp.getHeight() # this is the fundamental assumption
                lineListPix = drpStella.createLineList(fiberWavelengths, arcLineWavelengths)

                # Identify emission lines and fit dispersion
                try:
                    spec.identify(lineListPix, dispCorControl, 8)
                    self.log.info("FiberTrace %d: spec.getDispRms() = %f" % (traceId, spec.getDispRms()))
                except Exception as e:
                    print(e)

                if residuals is not None:
                    ft = flatFiberTraceSet.getFiberTrace(i)
                    reconIm = ft.getReconstructed2DSpectrum(spec)
                    residuals[reconIm.getBBox()] -= reconIm

                if self.debugInfo.display and self.debugInfo.showFibers is not None:
                    import matplotlib.pyplot as plt

                    fiberId = spec.getITrace()

                    if self.debugInfo.showFibers and fiberId not in self.debugInfo.showFibers:
                        continue

                    refLines = spec.getReferenceLines()
                    plt.plot(spec.wavelength, spec.spectrum)
                    plt.xlabel("Wavelength (vacuum nm)")
                    plt.title("FiberId %d" % fiberId)
                    for rl in refLines:
                        if rl.status & rl.Status.FIT:
                            plt.axvline(rl.wavelength, ls='-', color='black', alpha=0.5)
                        else:
                            plt.axvline(rl.wavelength, ls='-', color='red', alpha=0.25)
                    plt.show()

            writePfsArm(butler, arcExp, spectrumSet, arcRef.dataId)

        if residuals is not None:
            display.mtv(residuals, title='Residuals')
            
        return spectrumSet
    #
    # Disable writing metadata (doesn't work with lists of dataRefs anyway)
    #
    def _getMetadataName(self):
        return None
