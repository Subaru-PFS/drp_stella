#!/usr/bin/env python
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
import pfs.drp.stella.utils as dsUtils

class ExtractSpectraConfig(pexConfig.Config):
    saturationLevel = pexConfig.Field(
          doc = "CCD saturation level",
          dtype = float,
          default = 65000.,
          check = lambda x : x > 0.)

class ExtractSpectraTask(pipeBase.Task):
    ConfigClass = ExtractSpectraConfig
    _DefaultName = "ExtractSpectraTask"

    def __init__(self, *args, **kwargs):
        super(ExtractSpectraTask, self).__init__(*args, **kwargs)

        import lsstDebug
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, exposure, fiberTraceSet, traceNumbers=None):
        """Create traces from exposure and extract spectra from profiles in fiberTraceSet

        This method is the top-level for running the automatic 1D extraction of the fiber traces on the Exposure
        of the object spectra as a stand-alone BatchPoolTask.

        @return pipe_base Struct containing these fields:
         - spectrumSet: set of extracted spectra
        """

        if traceNumbers is None:
            traceNumbers = range(fiberTraceSet.size())
        self.log.debug("traceNumbers = %s" % traceNumbers)

        spectrumSet = drpStella.SpectrumSet()

        if exposure != None:
            if self.debugInfo.display:
                  display = afwDisplay.Display(frame=self.debugInfo.input_frame)

                  dsUtils.addFiberTraceSetToMask(exposure.mask, fiberTraceSet)

                  display.mtv(exposure, "input")

        # extract the spectra

        for i in range(len(traceNumbers)):
            fiberTrace = fiberTraceSet.getFiberTrace(traceNumbers[i])

            # Extract spectrum from profile
            try:
                spectrum = fiberTrace.extractFromProfile(exposure.getMaskedImage())
            except Exception as e:
                self.log.warn("Extraction of fibre %d failed: %s" % (fiberTrace.getITrace(), e))
                continue
            bbox = fiberTrace.getTrace().getBBox()
            spectrum.setMinY(bbox.getMinY())
            spectrum.setMaxY(bbox.getMaxY())

            spectrumSet.addSpectrum(spectrum)

        return pipeBase.Struct(
            spectrumSet=spectrumSet,
        )
