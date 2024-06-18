from typing import Optional

import numpy as np
import scipy.linalg

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella

from lsst.afw.image import MaskedImage
from .datamodel import PfsFiberNorms
from .DetectorMap import DetectorMap
from .FiberTrace import FiberTrace
from .FiberTraceSet import FiberTraceSet
from .Spectrum import Spectrum
from .SpectrumSet import SpectrumSet


class ExtractSpectraConfig(pexConfig.Config):
    fiberId = pexConfig.ListField(dtype=int, default=[], doc="If non-empty, only extract these fiberIds")
    mask = pexConfig.ListField(dtype=str, default=["NO_DATA", "BAD", "SAT", "CR", "BAD_FLAT"],
                               doc="Mask pixels to ignore in extracting spectra")
    minFracMask = pexConfig.Field(dtype=float, default=0.25,
                                  doc="Minimum fractional contribution of pixel for mask to be accumulated")
    doCrosstalk = pexConfig.Field(dtype=bool, default=False, doc="Correct for optical crosstalk?")
    crosstalk = pexConfig.ListField(
        dtype=float,
        default=[4.41695363e-03, 1.26907573e-03, 6.37238677e-04, 3.99808286e-04],
        doc=("Optical crosstalk coefficients, in increasing distance from the fiber of interest. "
             "These are the fraction of a given fiber's flux that appears in the neighboring fibers."),
    )


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
        isBoxcar: bool = False,
        fiberNorms : Optional[PfsFiberNorms] = None,
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
        isBoxcar : `bool`
            fiberTraceSet consists of boxcar apertures (peak 1, not row-normalised to 1)
        fiberNorms : `PfsFiberNorms`, optional
            Fiber normalizations.

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
            fiberId = self.config.fiberId

        if fiberId is not None and \
           len(set(fiberId) ^ set(fiberTraceSet.fiberId)) != 0:  # i.e. we don't want all the traces
            # Extract only the fiberTraces we care about
            num = sum(1 for ft in fiberTraceSet if ft.fiberId in fiberId)
            newTraces = drpStella.FiberTraceSet(num)
            for ft in fiberTraceSet:
                if ft.fiberId in fiberId:
                    newTraces.add(ft)
            fiberTraceSet = newTraces

        if fiberId is not None and len(fiberTraceSet) != len(fiberId):
            if len(fiberTraceSet) == 0:
                raise RuntimeError("None of the desired fiberIds are in the fiberTraceSet")
            else:
                missing = sorted(set(fiberId) ^ set(fiberTraceSet.fiberId))
                self.log.warning(f"fiberIds {missing} are not in the fiberTraceSet")

        spectra = self.extractAllSpectra(maskedImage, fiberTraceSet, detectorMap, isBoxcar, fiberNorms)

        if fiberId is not None:
            spectra = self.includeSpectra(spectra, fiberId, detectorMap)

        if self.config.doCrosstalk:
            self.crosstalkCorrection(spectra)

        return pipeBase.Struct(spectra=spectra)

    def extractAllSpectra(
        self,
        maskedImage: MaskedImage,
        fiberTraceSet: FiberTraceSet,
        detectorMap: Optional[DetectorMap] = None,
        isBoxcar: bool = False,
        fiberNorms : Optional[PfsFiberNorms] = None,
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
        isBoxcar : `bool`
            fiberTraceSet consists of boxcar apertures (peak 1, not row-normalised to 1)
        fiberNorms : `PfsFiberNorms`, optional
            Fiber normalizations.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        badBitMask = maskedImage.mask.getPlaneBitMask(self.config.mask)
        if isBoxcar:
            spectra = SpectrumSet(maskedImage.getHeight())
            spectra.reserve(len(fiberTraceSet))
            for ft in fiberTraceSet:
                spectrum = ft.extractAperture(maskedImage, badBitMask)
                spectra.add(spectrum)
        else:
            spectra = fiberTraceSet.extractSpectra(maskedImage, badBitMask, self.config.minFracMask)

        if detectorMap is not None:
            for spectrum in spectra:
                spectrum.setWavelength(detectorMap.getWavelength(spectrum.fiberId))

        if fiberNorms is not None:
            for spectrum in spectra:
                if spectrum.fiberId in fiberNorms:
                    spectrum.norm *= fiberNorms.calculate(spectrum.fiberId)

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

    def crosstalkCorrection(self, spectra: SpectrumSet):
        """Perform optical crosstalk correction

        Given a set of measured coefficients (representing the fraction of a
        fiber's flux that appears in its neighbours; assumed constant for all
        rows), we solve the matrix equation and apply.

        Parameters
        ----------
        spectra : `SpectrumSet`
            Spectra to correct; modified in-place.
        """
        # Set up the coefficients array
        halfSize = len(self.config.crosstalk)
        fullSize = 2*halfSize + 1
        coeff = np.zeros(fullSize)
        coeff[:halfSize] = np.array(self.config.crosstalk)[::-1]
        coeff[halfSize] = 1.0
        coeff[halfSize + 1:] = self.config.crosstalk

        # Calculate the crosstalk crosstalkMatrix
        # We generate the crosstalkMatrix that we'd get with full fiber sampling, and
        # then sub-sample it to contain only the fibers for which we have
        # spectra.
        fiberId = spectra.getAllFiberIds()
        minFiberId = fiberId.min()
        maxFiberId = fiberId.max()
        numFullFibers = maxFiberId - minFiberId + fullSize + 1
        fullFiberId = np.arange(
            minFiberId - halfSize,
            maxFiberId + halfSize + 2,  # +1 for center, +1 for 1-based fiberId
            dtype=int,
        )
        fullCrosstalkMatrix = np.zeros((numFullFibers, numFullFibers), dtype=float)

        for ii in range(halfSize, numFullFibers - halfSize):
            fullCrosstalkMatrix[ii, ii - halfSize:ii + halfSize + 1] = coeff

        haveFiberId = np.isin(fullFiberId, fiberId)
        crosstalkMatrix = fullCrosstalkMatrix[haveFiberId].T[haveFiberId].T

        icrosstalkMatrix = scipy.linalg.inv(crosstalkMatrix)

        flux = spectra.getAllFluxes()
        bad = (spectra.getAllMasks() & spectra[0].mask.getPlaneBitMask(self.config.mask)) != 0

        # Set bad pixels to zero.
        # This saves infecting all other pixels, but we could do better by
        # interpolating masked fluxes in the spectral dimension. But the
        # correction is small, and hopefully we won't be doing it this way
        # for long.
        if True:                   # masking bad pixels may not be a good idea.  RHL
            flux[bad] = 0.0
        flux[~np.isfinite(flux)] = 0.0

        # Correct in place row-by-row
        for ii in range(spectra.getLength()):
            flux[:, ii] = icrosstalkMatrix@flux[:, ii]

        for spectrum, corr in zip(spectra, flux):
            spectrum.flux[:] = corr
