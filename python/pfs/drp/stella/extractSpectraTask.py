import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella
import pfs.drp.stella.utils as dsUtils

from .identifyLines import IdentifyLinesTask


class ExtractSpectraConfig(pexConfig.Config):
    useOptimal = pexConfig.Field(dtype=bool, default=True,
                                 doc="Use optimal extraction? "
                                     "Otherwise, use a simple sum of pixels within the trace.")
    identifyLines = pexConfig.ConfigurableField(target=IdentifyLinesTask, doc="Identify reference lines")


class ExtractSpectraTask(pipeBase.Task):
    ConfigClass = ExtractSpectraConfig
    _DefaultName = "ExtractSpectraTask"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("identifyLines")
        import lsstDebug
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, maskedImage, fiberTraceSet, detectorMap=None, lines=None):
        """Extract spectra from the image

        We extract the spectra using the profiles in the provided
        fiber traces.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to extract spectra; modified if
            ``doSubtractContinuum`` is set.
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Fiber traces to extract.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Map of expected detector coordinates to fiber, wavelength.
            If provided, they will be used to normalise the spectrum
            and provide a rough wavelength calibration.
        lines : `list` of `pfs.drp.stella.ReferenceLine`, optional
            Reference lines.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """

        if self.debugInfo.display:
            display = afwDisplay.Display(frame=self.debugInfo.input_frame)
            dsUtils.addFiberTraceSetToMask(maskedImage.mask, fiberTraceSet)
            display.mtv(maskedImage, "input")

        spectra = self.extractAllSpectra(maskedImage, fiberTraceSet, detectorMap)
        if lines:
            self.identifyLines.run(spectra, detectorMap, lines)
        return spectra

    def extractAllSpectra(self, maskedImage, fiberTraceSet, detectorMap=None):
        """Extract all spectra in the fiberTraceSet

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to extract spectra; modified if
            ``doSubtractContinuum`` is set.
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Fiber traces to extract.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Map of expected detector coordinates to fiber, wavelength.
            If provided, they will be used to normalise the spectrum
            and provide a rough wavelength calibration.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        spectra = drpStella.SpectrumSet()
        for fiberTrace in fiberTraceSet:
            spectrum = self.extractSpectrum(maskedImage, fiberTrace, detectorMap)
            if spectrum is not None:
                spectra.addSpectrum(spectrum)
        return spectra

    def extractSpectrum(self, maskedImage, fiberTrace, detectorMap=None):
        """Extract a single spectrum from the image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to extract spectra.
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Fiber traces to extract.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Map of expected detector coordinates to fiber, wavelength.
            If provided, they will be used to normalise the spectrum
            and provide a rough wavelength calibration.

        Returns
        -------
        spectrum : `pfs.drp.stella.SpectrumSet`
            Extracted spectra, or `None` if the extraction failed.
        """
        fiberId = fiberTrace.getFiberId()
        try:
            spectrum = fiberTrace.extractSpectrum(maskedImage, self.config.useOptimal)
        except Exception as exc:
            self.log.warn("Extraction of fibre %d failed: %s", fiberId, exc)
            return None
        if detectorMap is not None:
            spectrum.spectrum /= detectorMap.getThroughput(fiberId)
            spectrum.setWavelength(detectorMap.getWavelength(fiberId))
        return spectrum
