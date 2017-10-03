#!/usr/bin/env python
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
import pfs.drp.stella.utils as dsUtils

class ExtractSpectraConfig(pexConfig.Config):
    extractionAlgorithm = pexConfig.ChoiceField(
        dtype=str,
        doc="Algorithm used to extract spectra",
        allowed={
            "OPTIMAL": "classical \"optimal\" extraction",
            "BOXCAR": "simple sum of pixels within the trace",
        },
        default="OPTIMAL",
        optional=False,
    )

class ExtractSpectraTask(pipeBase.Task):
    ConfigClass = ExtractSpectraConfig
    _DefaultName = "ExtractSpectraTask"

    def __init__(self, *args, **kwargs):
        super(ExtractSpectraTask, self).__init__(*args, **kwargs)

        import lsstDebug
        self.debugInfo = lsstDebug.Info(__name__)

        self._useProfile = self.config.extractionAlgorithm == "OPTIMAL"

    def run(self, exposure, fiberTraceSet, traceNumbers=None):
        """Create traces from exposure and extract spectra from profiles in fiberTraceSet

        This method is the top-level for running the automatic 1D extraction of the fiber traces on the Exposure
        of the object spectra as a stand-alone BatchPoolTask.

        @return pipe_base Struct containing these fields:
         - spectrumSet: set of extracted spectra
        """

        if traceNumbers is None:
            traceNumbers = range(fiberTraceSet.getNtrace())
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
                spectrum = fiberTrace.extractSpectrum(exposure.getMaskedImage(), self._useProfile)
            except Exception as e:
                self.log.warn("Extraction of fibre %d failed: %s" % (fiberTrace.getITrace(), e))
                continue

            spectrumSet.addSpectrum(spectrum)

        return pipeBase.Struct(
            spectrumSet=spectrumSet,
        )
