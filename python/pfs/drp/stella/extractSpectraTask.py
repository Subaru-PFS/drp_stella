import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella


class ExtractSpectraConfig(pexConfig.Config):
    useOptimal = pexConfig.Field(dtype=bool, default=True,
                                 doc="Use optimal extraction? "
                                     "Otherwise, use a simple sum of pixels within the trace.")
    fiberId = pexConfig.ListField(dtype=int, default=[], doc="If non-empty, only extract these fiberIds")


class ExtractSpectraTask(pipeBase.Task):
    ConfigClass = ExtractSpectraConfig
    _DefaultName = "extractSpectra"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import lsstDebug
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, maskedImage, fiberTraceSet, detectorMap=None):
        """Extract spectra from the image

        We extract the spectra using the profiles in the provided
        fiber traces.

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
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        if self.debugInfo.display:
            display = afwDisplay.Display(frame=self.debugInfo.input_frame)
            fiberTraceSet.applyToMask(maskedImage.mask)
            display.mtv(maskedImage, "input")
        if self.config.fiberId:
            # Extract only the fiberTraces we care about
            num = sum(1 for ft in fiberTraceSet if ft.fiberId in self.config.fiberId)
            newTraces = drpStella.FiberTraceSet(num)
            for ft in fiberTraceSet:
                if ft.fiberId in self.config.fiberId:
                    newTraces.add(ft)
            fiberTraceSet = newTraces
        spectra = self.extractAllSpectra(maskedImage, fiberTraceSet, detectorMap)
        return pipeBase.Struct(spectra=spectra)

    def extractAllSpectra(self, maskedImage, fiberTraceSet, detectorMap=None):
        """Extract all spectra in the fiberTraceSet

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
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        spectra = fiberTraceSet.extractSpectra(maskedImage)
        for spectrum in spectra:
            spectrum.setWavelength(detectorMap.getWavelength(spectrum.fiberId))
        return spectra

    def extractSpectrum(self, maskedImage, fiberTrace, detectorMap=None):
        """Extract a single spectrum from the image

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which to extract spectra.
        fiberTrace : `pfs.drp.stella.FiberTrace`
            Fiber traces to extract.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Map of expected detector coordinates to fiber, wavelength.
            If provided, they will be used to normalise the spectrum
            and provide a rough wavelength calibration.

        Returns
        -------
        spectrum : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        fiberId = fiberTrace.getFiberId()
        spectrum = fiberTrace.extractSpectrum(maskedImage, self.config.useOptimal)
        if detectorMap is not None:
            spectrum.setWavelength(detectorMap.getWavelength(fiberId))
        return spectrum
