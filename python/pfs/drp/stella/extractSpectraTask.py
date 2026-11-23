from typing import Optional
import warnings

import numpy as np
import scipy.linalg
import scipy.ndimage

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella

from lsst.afw.image import ImageF, MaskedImage
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
    minFracImage = pexConfig.Field(
        dtype=float,
        default=0.4,
        doc="Minimum total fractional contribution for measurement to be considered reliable",
    )
    doCrosstalk = pexConfig.Field(dtype=bool, default=False, doc="Correct for optical crosstalk?")
    crosstalk = pexConfig.ListField(
        dtype=float,
        default=[4.41695363e-03, 1.26907573e-03, 6.37238677e-04, 3.99808286e-04],
        doc=("Optical crosstalk coefficients, in increasing distance from the fiber of interest. "
             "These are the fraction of a given fiber's flux that appears in the neighboring fibers."),
    )
    doRejectOutliers = pexConfig.Field(
        dtype=bool,
        default=False,
        doc=("Reject outliers (e.g., cosmic rays) by clipping pixels that are inconsistent with the model "
             "constructed from the extracted spectra and the fiber profiles, and re-extracting?"),
    )
    rejectThreshold = pexConfig.Field(
        dtype=float,
        default=8.0,
        doc=("Rejection threshold (positive residual/sigma) for outliers. This is deliberately high: false "
             "rejections (e.g., at bright lines) are worse than letting faint cosmic rays through."),
    )
    rejectIterations = pexConfig.Field(
        dtype=int, default=3, doc="Maximum number of rejection iterations (each followed by re-extraction)"
    )
    rejectGrow = pexConfig.Field(
        dtype=int,
        default=1,
        doc=("Number of pixels by which to grow rejected pixels along the row (i.e., across the fibers). "
             "We don't grow in the spectral direction, as that is expensive for data binned in that "
             "direction."),
    )
    rejectGrowThreshold = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc=("Grow each rejected region to include the connected pixels (in both directions, within the "
             "traces) whose positive residual exceeds this many standard deviations, so that all of a "
             "detected outlier is rejected. Disabled if <= 0."),
    )
    rejectFiberFraction = pexConfig.Field(
        dtype=float,
        default=0.1,
        doc=("If the rejected pixels in a row carry more than this fraction of a fiber's (normalised) "
             "profile, reject all of that fiber's trace in that row. The core of an outlier that resembles "
             "the profile is absorbed into the flux, so it can't be identified from the residuals. No "
             "further rejections are made within the trace of such a fiber in that row, as the model there "
             "is no longer reliable. Disabled if <= 0."),
    )
    rejectSystematicFraction = pexConfig.Field(
        dtype=float,
        default=0.05,
        doc=("Fraction of the model to add in quadrature to the noise when rejecting outliers, to allow "
             "for errors in the fiber profiles (including the non-separability of the PSF). The model used "
             "is its local maximum along the row (see rejectSystematicWidth)."),
    )
    rejectSystematicWidth = pexConfig.Field(
        dtype=int,
        default=3,
        doc=("Half-width (pixels along the row) of the maximum filter applied to the model for the "
             "systematic error term, so that the wings of bright features get the systematic error of the "
             "peak"),
    )
    rejectMaskPlane = pexConfig.Field(dtype=str, default="CR", doc="Mask plane for rejected pixels")
    rejectMinProfile = pexConfig.Field(
        dtype=float,
        default=0.02,
        doc=("Minimum value of the (row-normalised) fiber profiles for a pixel to be considered for "
             "rejection. Pixels with small profile values contribute little to the fluxes, and those in "
             "the gaps between fibers may contain light that isn't in the model."),
    )
    rejectReadNoiseWidth = pexConfig.Field(
        dtype=int, default=31, doc="Width (columns) of median filter for smoothing the read noise estimate"
    )
    rejectBackgroundMaxProfile = pexConfig.Field(
        dtype=float,
        default=0.005,
        doc=("Maximum value of the (row-normalised) fiber profiles for a pixel to be used in estimating the "
             "background between the fibers, which is used as a floor for the model when rejecting outliers"),
    )
    rejectBackgroundBinSize = pexConfig.Field(
        dtype=int,
        default=64,
        doc="Size (columns) of the bins in which the background between the fibers is estimated",
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
        rejectOutliers: Optional[bool] = None,
    ) -> pipeBase.Struct:
        """Extract spectra from the image

        We extract the spectra using the profiles in the provided
        fiber traces.

        If outlier rejection is enabled, pixels that are inconsistent with
        the model constructed from the extracted spectra and fiber profiles
        are flagged in the image's mask (with ``config.rejectMaskPlane``) and
        the spectra are re-extracted without them.

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
        rejectOutliers : `bool`, optional
            Reject outliers? If ``None``, use ``config.doRejectOutliers``.
            Rejection is not performed for boxcar extraction.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        if rejectOutliers is None:
            rejectOutliers = self.config.doRejectOutliers
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

        spectra = self.extractAllSpectra(maskedImage, fiberTraceSet, detectorMap, isBoxcar, rejectOutliers)

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
        rejectOutliers: bool = False,
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
        rejectOutliers : `bool`
            Reject outliers (not performed for boxcar extraction)? Rejected
            pixels are flagged in the ``maskedImage``'s mask.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        """
        badBitMask = maskedImage.mask.getPlaneBitMask(self.config.mask)
        if rejectOutliers and not isBoxcar:
            maskedImage.mask.addMaskPlane(self.config.rejectMaskPlane)
            badBitMask |= maskedImage.mask.getPlaneBitMask(self.config.rejectMaskPlane)
        if isBoxcar:
            spectra = SpectrumSet(maskedImage.getHeight())
            spectra.reserve(len(fiberTraceSet))
            for ft in fiberTraceSet:
                spectrum = ft.extractAperture(maskedImage, badBitMask)
                spectra.add(spectrum)
        else:
            spectra = fiberTraceSet.extractSpectra(
                maskedImage, badBitMask, self.config.minFracMask, self.config.minFracImage
            )
            if rejectOutliers:
                total = 0
                iteration = -1
                for iteration in range(self.config.rejectIterations):
                    numRejected = self.rejectOutliers(maskedImage, fiberTraceSet, spectra, badBitMask)
                    self.log.debug("Iteration %d: rejected %d pixels", iteration, numRejected)
                    if numRejected == 0:
                        break
                    total += numRejected
                    spectra = fiberTraceSet.extractSpectra(
                        maskedImage, badBitMask, self.config.minFracMask, self.config.minFracImage
                    )
                self.log.info(
                    "Rejected %d outlying pixels (masked as %s) in %d iterations",
                    total, self.config.rejectMaskPlane, iteration + 1,
                )

        if detectorMap is not None:
            for spectrum in spectra:
                spectrum.setWavelength(detectorMap.getWavelength(spectrum.fiberId))

        return spectra

    def makeModelImage(self, maskedImage: MaskedImage, fiberTraceSet: FiberTraceSet, spectra: SpectrumSet):
        """Construct the model image from the extracted spectra

        This is the model that the extraction fit: the extraction normalises
        the profile in each row by its sum (the spectrum's ``norm``), so the
        model for each fiber is ``flux/norm*profile``.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which the spectra were extracted.
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Fiber traces used for the extraction.
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra (not yet normalised in any other way).

        Returns
        -------
        model : `numpy.ndarray` of `float`
            Model image.
        profile : `numpy.ndarray` of `float`
            Sum of the normalised profiles of all the fibers (i.e., the model
            for unit flux in every fiber): the fractional contribution of
            each pixel to the fibers' fluxes.
        """
        bbox = maskedImage.getBBox()
        model = ImageF(bbox)
        model.set(0.0)
        profile = ImageF(bbox)
        profile.set(0.0)
        spectraByFiberId = {spectrum.fiberId: spectrum for spectrum in spectra}
        for ft in fiberTraceSet:
            spectrum = spectraByFiberId.get(ft.fiberId)
            if spectrum is None:
                continue
            norm = spectrum.norm
            good = (norm != 0) & np.isfinite(norm) & np.isfinite(spectrum.flux)
            with np.errstate(invalid="ignore", divide="ignore"):
                inverseNorm = np.where(good, 1.0/norm, 0.0).astype(np.float32)
            ft.constructImage(model, np.where(good, spectrum.flux, 0.0).astype(np.float32)*inverseNorm)
            ft.constructImage(profile, inverseNorm)
        return model.array, profile.array

    def rejectOutliers(
        self, maskedImage: MaskedImage, fiberTraceSet: FiberTraceSet, spectra: SpectrumSet, badBitMask: int
    ) -> int:
        """Flag pixels that are inconsistent with the model of the spectra

        We construct the model image from the extracted spectra and the fiber
        profiles, and flag pixels within the fiber traces whose positive
        residual exceeds ``config.rejectThreshold`` standard deviations. Only
        pixels that contribute significantly to the fluxes (normalised profile
        exceeding ``config.rejectMinProfile``) are considered: outliers
        elsewhere don't matter much for the extraction, and the gaps between
        fibers can contain light that isn't in the model (e.g., unmodelled
        scattered light), which would produce false positives. The
        noise is computed from the *model* rather than the data (so that an
        outlier doesn't inflate its own noise): the Poisson contribution of
        the data in the variance plane is replaced by that of the model, and
        a fraction (``config.rejectSystematicFraction``) of the model is added
        in quadrature to allow for errors in the fiber profiles.

        Because each row is fit independently, real spectral structure (e.g.,
        sky lines) is absorbed into that row's fitted fluxes; only flux whose
        shape across the fibers doesn't match the profiles is flagged. The
        corollary is that an outlier that resembles the profile (e.g., a
        cosmic ray covering the central couple of pixels of a trace) is
        absorbed into the flux and not rejected: because the systematic error
        scales with the local peak of the model, an outlier is only rejected
        if its residual exceeds roughly ``rejectThreshold*rejectSystematicFraction``
        of that peak, whatever its amplitude. We choose to let such outliers
        through rather than risk rejecting real features.

        The noise is ``readNoise^2 + model + (f*peak)^2``, where the read
        noise is estimated from the image (see ``estimateReadNoise2``) and
        ``peak`` is the maximum of the model within
        ``config.rejectSystematicWidth`` pixels along the row: the PSF isn't
        separable into the fiber profile and the LSF, so near bright features
        there are residuals in the profile wings that scale with the peak.

        The rejection is performed before the scattered light is subtracted
        (as the scattered light is estimated from the spectra), so there is
        smooth light in the image that isn't in the model. Within the bright
        part of a trace it is absorbed into the fiber's flux, but in the wings
        of a faint (or empty) fiber next to a gap it isn't, and the
        systematic error (which scales with the fiber's own small peak) doesn't
        cover it. We therefore use as the model the maximum of the model and
        the background measured between the fibers (see
        ``estimateBackground``).

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which the spectra were extracted. The mask is modified.
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Fiber traces used for the extraction.
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        badBitMask : `int`
            Bitmask for pixels to ignore.

        Returns
        -------
        numRejected : `int`
            Number of newly rejected pixels.
        """
        model, profile = self.makeModelImage(maskedImage, fiberTraceSet, spectra)
        inTrace = profile > self.config.rejectMinProfile
        image = maskedImage.image.array
        variance = maskedImage.variance.array
        mask = maskedImage.mask.array
        rejectBitMask = maskedImage.mask.getPlaneBitMask(self.config.rejectMaskPlane)
        good = ((mask & badBitMask) == 0) & np.isfinite(image) & np.isfinite(variance) & (variance > 0)

        readNoise2 = self.estimateReadNoise2(image, variance, model, good)
        background = self.estimateBackground(image, profile, good)
        model = np.maximum(model, background)  # light between the fibers that isn't in the model
        with np.errstate(invalid="ignore"):
            noise2 = readNoise2[np.newaxis, :] + np.maximum(model, 0.0)  # noise of the model, not the data
            # The PSF isn't separable into the fiber profile and the LSF, so near bright lines (or in bright
            # fibers) there are residuals in the profile wings that scale with the peak flux, not the local
            # model value. Scale the systematic error by the local maximum of the model along the row.
            systematic = scipy.ndimage.maximum_filter1d(
                np.abs(model), size=2*self.config.rejectSystematicWidth + 1, axis=1, mode="nearest"
            )
            noise2 += (self.config.rejectSystematicFraction*systematic)**2
            chi = (image - model)/np.sqrt(noise2)
            candidate = inTrace & good & np.isfinite(chi)
            if self.config.rejectFiberFraction > 0:
                # Don't reject in the traces of fibers that have already been rejected in that row: the model
                # there is unreliable
                rejected = (mask & rejectBitMask) != 0
                _, protect = self.rejectFibers(maskedImage, fiberTraceSet, spectra, rejected)
                candidate &= ~protect
            outlier = candidate & (chi > self.config.rejectThreshold)
            if self.config.rejectGrowThreshold > 0 and np.any(outlier):
                # Grow to the connected pixels with a significant excess (including those already rejected,
                # so that a region can be connected through them)
                alreadyRejected = (mask & rejectBitMask) != 0
                excess = (candidate & (chi > self.config.rejectGrowThreshold)) | outlier | alreadyRejected
                labels, _ = scipy.ndimage.label(excess, structure=np.ones((3, 3), dtype=bool))
                seeds = np.unique(labels[outlier])
                outlier |= np.isin(labels, seeds[seeds > 0]) & inTrace & ~alreadyRejected
        if self.config.rejectGrow > 0:
            structure = np.ones((1, 2*self.config.rejectGrow + 1), dtype=bool)
            outlier = scipy.ndimage.binary_dilation(outlier, structure=structure) & inTrace
        if self.config.rejectFiberFraction > 0 and np.any(outlier):
            outlier |= self.rejectFibers(maskedImage, fiberTraceSet, spectra, outlier)[0]

        new = outlier & ((mask & rejectBitMask) == 0)
        mask[new] |= rejectBitMask
        return int(np.count_nonzero(new))

    def rejectFibers(
        self,
        maskedImage: MaskedImage,
        fiberTraceSet: FiberTraceSet,
        spectra: SpectrumSet,
        outlier: np.ndarray,
    ) -> np.ndarray:
        """Extend rejections to the whole of a fiber's trace in a row

        The core of an outlier that resembles the fiber profile is absorbed
        into the fitted flux, so it can't be identified from the residuals.
        If the rejected pixels in a row carry more than
        ``config.rejectFiberFraction`` of a fiber's (normalised) profile, we
        reject all of that fiber's trace (where its normalised profile exceeds
        ``config.rejectMinProfile``) in that row.

        We also return the footprint of the traces of such fibers in those
        rows. The flux of such a fiber is then fit from the remaining pixels
        in its wings, which (before the scattered light is subtracted) contain
        light that isn't in the model, so the model in its footprint
        (including the neighbouring fibers) is unreliable; making further
        rejections there leads to a cascade of rejections in the neighbours.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which the spectra were extracted.
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Fiber traces used for the extraction.
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        outlier : `numpy.ndarray` of `bool`
            Rejected pixels.

        Returns
        -------
        reject : `numpy.ndarray` of `bool`
            Additional pixels to reject.
        footprint : `numpy.ndarray` of `bool`
            Footprint of the traces of the fibers rejected in each row.
        """
        reject = np.zeros_like(outlier)
        footprint = np.zeros_like(outlier)
        x0, y0 = maskedImage.getXY0()
        spectraByFiberId = {spectrum.fiberId: spectrum for spectrum in spectra}
        for ft in fiberTraceSet:
            spectrum = spectraByFiberId.get(ft.fiberId)
            if spectrum is None:
                continue
            box = ft.trace.getBBox()
            rows = slice(box.getMinY() - y0, box.getMaxY() + 1 - y0)
            columns = slice(box.getMinX() - x0, box.getMaxX() + 1 - x0)
            norm = spectrum.norm[rows]
            with np.errstate(invalid="ignore", divide="ignore"):
                inverseNorm = np.where((norm != 0) & np.isfinite(norm), 1.0/norm, 0.0)
            inTrace = (ft.trace.mask.array & ft.trace.mask.getPlaneBitMask("FIBERTRACE")) != 0
            profile = np.where(inTrace, ft.trace.image.array, 0.0)*inverseNorm[:, np.newaxis]
            fraction = np.sum(profile*outlier[rows, columns], axis=1)
            select = fraction > self.config.rejectFiberFraction
            if np.any(select):
                reject[rows, columns][select] |= profile[select] > self.config.rejectMinProfile
                footprint[rows, columns][select] |= profile[select] > 0
        return reject, footprint

    def estimateReadNoise2(
        self, image: np.ndarray, variance: np.ndarray, model: np.ndarray, good: np.ndarray
    ) -> np.ndarray:
        """Estimate the read noise variance as a function of column

        For a variance plane constructed from the data (as in the ISR), the
        variance minus the image is the read noise variance. We take the
        median over pixels in each column for which the model is small (so
        that the variance is dominated by the read noise, and its estimate is
        robust whether or not the variance was computed from the noisy data),
        and smooth along the row so that the result follows the amplifiers.

        Parameters
        ----------
        image, variance, model : `numpy.ndarray` of `float`
            Image, variance and model arrays.
        good : `numpy.ndarray` of `bool`
            Pixels that may be used.

        Returns
        -------
        readNoise2 : `numpy.ndarray` of `float`
            Read noise variance for each column.
        """
        excess = variance - image
        lowFlux = good & (np.abs(model) < np.maximum(excess, 0.0))  # Poisson noise < read noise
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            readNoise2 = np.nanmedian(np.where(lowFlux, excess, np.nan), axis=0)
        haveValue = np.isfinite(readNoise2)
        if not np.any(haveValue):
            return np.zeros(image.shape[1], dtype=float)
        columns = np.arange(image.shape[1])
        readNoise2 = np.interp(columns, columns[haveValue], readNoise2[haveValue])
        readNoise2 = scipy.ndimage.median_filter(
            readNoise2, size=self.config.rejectReadNoiseWidth, mode="nearest"
        )
        return np.maximum(readNoise2, 0.0)

    def estimateBackground(self, image: np.ndarray, profile: np.ndarray, good: np.ndarray) -> np.ndarray:
        """Estimate the light between the fibers

        For each row, we take the median of the pixels between the fibers
        (where the sum of the normalised fiber profiles is less than
        ``config.rejectBackgroundMaxProfile``) in bins of
        ``config.rejectBackgroundBinSize`` columns, and interpolate linearly
        between the bins.

        Parameters
        ----------
        image : `numpy.ndarray` of `float`
            Image array.
        profile : `numpy.ndarray` of `float`
            Sum of the row-normalised fiber profiles.
        good : `numpy.ndarray` of `bool`
            Pixels that may be used.

        Returns
        -------
        background : `numpy.ndarray` of `float`
            Background image; ``-inf`` in rows with no measurement.
        """
        height, width = image.shape
        binSize = self.config.rejectBackgroundBinSize
        numBins = (width + binSize - 1)//binSize
        between = np.full((height, numBins*binSize), np.nan, dtype=float)
        use = good & (profile < self.config.rejectBackgroundMaxProfile)
        between[:, :width] = np.where(use, image, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bins
            binned = np.nanmedian(between.reshape(height, numBins, binSize), axis=2)
        centers = np.minimum((np.arange(numBins) + 0.5)*binSize, width - 1)
        columns = np.arange(width)
        background = np.full((height, width), -np.inf, dtype=float)
        for row in range(height):
            haveValue = np.isfinite(binned[row])
            if np.any(haveValue):
                background[row] = np.interp(columns, centers[haveValue], binned[row][haveValue])
        return background

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
                target.variance[:] = np.nan
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
