from typing import Optional
import numpy as np

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella

from lsst.afw.image import MaskedImage
from .DetectorMap import DetectorMap
from .FiberTrace import FiberTrace
from .FiberTraceSet import FiberTraceSet
from .Spectrum import Spectrum
from .SpectrumSet import SpectrumSet


class ExtractSpectraConfig(pexConfig.Config):
    fiberId = pexConfig.ListField(dtype=int, default=[], doc="If non-empty, only extract these fiberIds")
    mask = pexConfig.ListField(dtype=str, default=["NO_DATA", "BAD", "SAT", "CR", "BAD_FLAT"],
                               doc="Mask pixels to ignore in extracting spectra")
    minFracMask = pexConfig.Field(dtype=float, default=0.0,
                                  doc="Minimum fractional contribution of pixel for mask to be accumulated")


class ExtractSpectraTask(pipeBase.Task):
    ConfigClass = ExtractSpectraConfig
    _DefaultName = "extractSpectra"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import lsstDebug
        self.debugInfo = lsstDebug.Info(__name__)

    def run(
        self,
        maskedImage: MaskedImage,
        fiberTraceSet: FiberTraceSet,
        detectorMap: Optional[DetectorMap] = None,
        fiberId: Optional[np.ndarray] = None,
    ) -> pipeBase.Struct:
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
        fiberId : `numpy.ndarray` of `int`
            Fiber identifiers to include in output.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        if len(fiberTraceSet) == 0:
            raise RuntimeError("No fiber traces to extract")
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
        if fiberId is not None:
            spectra = self.includeSpectra(spectra, fiberId, detectorMap)
        return pipeBase.Struct(spectra=spectra)

    def extractAllSpectra(
        self,
        maskedImage: MaskedImage,
        fiberTraceSet: FiberTraceSet,
        detectorMap: Optional[DetectorMap] = None,
    ) -> SpectrumSet:
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
        badBitMask = maskedImage.mask.getPlaneBitMask(self.config.mask)
        spectra = fiberTraceSet.extractSpectra(maskedImage, badBitMask, self.config.minFracMask)
        if detectorMap is not None:
            for spectrum in spectra:
                spectrum.setWavelength(detectorMap.getWavelength(spectrum.fiberId))
        return spectra

    def extractSpectrum(
        self, maskedImage: MaskedImage, fiberTrace: FiberTrace, detectorMap: Optional[DetectorMap] = None
    ) -> Spectrum:
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
        spectrum = fiberTrace.extractSpectrum(maskedImage)
        if detectorMap is not None:
            spectrum.setWavelength(detectorMap.getWavelength(fiberId))
        return spectrum

    def includeSpectra(
        self, spectra: SpectrumSet, fiberId: np.ndarray, detectorMap: Optional[DetectorMap] = None
    ) -> SpectrumSet:
        """Include in the output spectra for the provided fiberIds

        If we haven't extracted spectra for a particular fiberId, it's added as
        ``NaN``.

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        fiberId : `numpy.ndarray` of `int`
            Fiber identifiers to include in output.
        detectorMap : `pfs.drp.stella.DetectorMap`, optional
            Map of expected detector coordinates to fiber, wavelength.

        Returns
        -------
        new : `pfs.drp.stella.SpectrumSet`
            Spectra for each of the provided fiberIds.
        """
        specFibers = spectra.getAllFiberIds()
        if set(specFibers) == set(fiberId):
            return spectra
        length = spectra.getLength()
        new = drpStella.SpectrumSet(len(fiberId), length)
        fiberToIndex = {ff: ii for ii, ff in enumerate(specFibers)}
        for ii, ff in enumerate(fiberId):
            if ff in fiberToIndex:
                target = spectra[fiberToIndex[ff]]
            else:
                target = drpStella.Spectrum(length, ff)
                target.flux[:] = np.nan
                target.mask.array[:] = target.mask.getPlaneBitMask("NO_DATA")
                target.covariance[:] = np.nan
                target.background[:] = np.nan
                wavelength = np.nan
                if detectorMap is not None and ff in detectorMap:
                    wavelength = detectorMap.getWavelength(ff)
                target.wavelength[:] = wavelength
            new[ii] = target
        return new
