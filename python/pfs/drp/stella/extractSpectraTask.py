#!/usr/bin/env python
import lsst.pex.config as pexConfig
from lsst.pipe.base import Task
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
import pfs.drp.stella.utils as dsUtils

class ExtractSpectraConfig(pexConfig.Config):
    saturationLevel = pexConfig.Field(
          doc = "CCD saturation level",
          dtype = float,
          default = 65000.,
          check = lambda x : x > 0.)

class ExtractSpectraTask(Task):
    ConfigClass = ExtractSpectraConfig
    _DefaultName = "ExtractSpectraTask"

    def __init__(self, *args, **kwargs):
        super(ExtractSpectraTask, self).__init__(*args, **kwargs)

        import lsstDebug
        self.debugInfo = lsstDebug.Info(__name__)

    def extractSpectra(self, inExposure, inFiberTraceSetWithProfiles, inTraceNumbers):

        traceNumbers = inTraceNumbers
        if inTraceNumbers[0] == -1:
            traceNumbers = range(inFiberTraceSetWithProfiles.size())
        self.log.debug("inTraceNumbers = %s" % inTraceNumbers)
        self.log.debug("traceNumbers = %s" % traceNumbers)

        spectrumSet = drpStella.SpectrumSetF()

        if inExposure != None:
            inMaskedImage = inExposure.getMaskedImage()

            if self.debugInfo.display:
                  display = afwDisplay.Display(frame=self.debugInfo.input_frame)

                  dsUtils.addFiberTraceSetToMask(inExposure.getMaskedImage().getMask(),
                                                 inFiberTraceSetWithProfiles.getTraces(), display)

                  display.mtv(inExposure, "input")

        # Store pixel values from inMaskedImage in inFiberTraceSetWithProfile's FiberTraces
        # and proceed to extract the spectra

        for i in range(len(traceNumbers)):
            fiberTrace = inFiberTraceSetWithProfiles.getFiberTrace(traceNumbers[i])
            if not fiberTrace.isProfileSet():
                raise Exception("profile not set")

            # Set pixels in FiberTrace from inMaskedImage
            if inExposure != None:
                fiberTrace.createTrace(inMaskedImage)
            #
            # There is no guarantee that createTrace generates images ("trace"s) whose dimensions
            # match the profiles.  This is a bug (PIPE2D-219); for now we'll trim the traces
            #
            trace = fiberTrace.getTrace()
            width = fiberTrace.getProfile().getWidth()
            if trace.getWidth() != width:
                fiberTrace.setTrace(trace[:width, :])

            # Extract spectrum from profile
            try:
                spectrum = fiberTrace.extractFromProfile()
            except Exception as e:
                self.log.warn("Extraction of fibre %d failed: %s" % (fiberTrace.getITrace(), e))
                continue

            spectrumSet.addSpectrum(spectrum)

        return spectrumSet

    def run(self, inExposure, inFiberTraceSetWithProfiles, inTraceNumbers=[-1]):
        """Create traces from inExposure and extract spectra from profiles in inFiberTraceSetWithProfiles

        This method is the top-level for running the automatic 1D extraction of the fiber traces on the Exposure
        of the object spectra as a stand-alone BatchPoolTask.

        This method returns a SpectrumSet
        """

        spectrumSet = self.extractSpectra(inExposure, inFiberTraceSetWithProfiles, inTraceNumbers)
        return spectrumSet
