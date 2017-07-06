#!/usr/bin/env python
import lsst.pex.config as pexConfig
from lsst.pipe.base import Task
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella

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

        # Create FiberTraces for inExposure and store them in inFiberTraceSetWithProfile
        if inExposure != None:
            inMaskedImage = inExposure.getMaskedImage()

            if self.debugInfo.display:
                  from pfs.drp.stella import markFiberTraceInMask
                  disp = afwDisplay.Display(frame=self.debugInfo.input_frame)

                  maskPlane = "FIBERTRACE"
                  mask = inExposure.getMaskedImage().getMask()
                  mask.addMaskPlane(maskPlane)
                  disp.setMaskPlaneColor(maskPlane, "GREEN")

                  ftMask = mask.getPlaneBitMask(maskPlane)
                  for ft in inFiberTraceSetWithProfiles.getTraces():
                        markFiberTraceInMask(ft, mask, ftMask)
                  
                  disp.mtv(inExposure, "input")

        # Create traces and extract spectrum
        for i in range(len(traceNumbers)):
            trace = inFiberTraceSetWithProfiles.getFiberTrace(traceNumbers[i])
            if not trace.isProfileSet():
                raise Exception("profile not set")

            # Create trace from inMaskedImage
            if inExposure != None:
                trace.createTrace(inMaskedImage)

            # Extract spectrum from profile
            spectrum = trace.extractFromProfile()
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
