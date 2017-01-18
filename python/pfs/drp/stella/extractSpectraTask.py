#!/usr/bin/env python
import lsst.pex.config as pexConfig
from lsst.pipe.base import Task
import pfs.drp.stella as drpStella

class ExtractSpectraConfig(pexConfig.Config):
      saturationLevel = pexConfig.Field(
          doc = "CCD saturation level",
          dtype = float,
          default = 65000.,
          check = lambda x : x > 0.)

class ExtractSpectraTask(Task):
    ConfigClass = ExtractSpectraConfig
    _DefaultName = "extractSpectra"

    def __init__(self, *args, **kwargs):
        super(ExtractSpectraTask, self).__init__(*args, **kwargs)

    def extractSpectra(self, inExposure, inFiberTraceSetWithProfiles, inTraceNumbers):

        if inTraceNumbers[0] == -1:
            traceNumbers = range(inFiberTraceSetWithProfiles.size())
        else:
            traceNumbers = inTraceNumbers
        self.log.info("inTraceNumbers = %s" % inTraceNumbers)
        self.log.info("traceNumbers = %s" % traceNumbers)

        spectrumSet = drpStella.SpectrumSetF()

        """Create FiberTraces for inExposure and store them in inFiberTraceSetWithProfile"""
        if inExposure != None:
            inMaskedImage = inExposure.getMaskedImage()

        """Create traces and extract spectrum"""
        for i in traceNumbers:
            if i < 0:
                raise Exception("i < 0")
            elif i >= inFiberTraceSetWithProfiles.size():
                raise Exception("i >= inFiberTraceSetWithProfiles.size()")

            trace = inFiberTraceSetWithProfiles.getFiberTrace(i)
            if trace.isProfileSet() == False:
                raise Exception("profile not set")

            """Create trace from inMaskedImage"""
            if inExposure != None:
                trace.setITrace(i)
                trace.createTrace(inMaskedImage)

            """Extract spectrum from profile"""
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
