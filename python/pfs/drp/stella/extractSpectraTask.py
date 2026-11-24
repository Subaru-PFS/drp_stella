from typing import Optional, Tuple
import warnings

import numpy as np
import scipy.linalg
import scipy.ndimage

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.display as afwDisplay
import pfs.drp.stella as drpStella

from lsst.afw.image import ImageF, MaskedImage
from lsst.geom import Point2I
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
        doc=("Number of pixels by which to dilate rejected pixels along the row (i.e., across the fibers), "
             "unconditionally (but only into pixels that may be rejected), after the growth by "
             "rejectGrowThreshold. There is no unconditional dilation in the spectral direction, as that "
             "is expensive for data binned in that direction."),
    )
    rejectGrowThreshold = pexConfig.Field(
        dtype=float,
        default=3.0,
        doc=("Grow each rejected region to include the connected pixels (within the traces) whose positive "
             "residual exceeds this many standard deviations, so that all of a detected outlier is "
             "rejected. Disabled if <= 0. Growth into an adjacent row uses rejectGrowAcrossRowsThreshold."),
    )
    rejectGrowAcrossRowsThreshold = pexConfig.Field(
        dtype=float,
        default=5.0,
        doc=("Threshold (positive residual/sigma) for growing a rejected region into an adjacent row; "
             "within that row it then grows with rejectGrowThreshold. This is higher than "
             "rejectGrowThreshold because for data binned in the spectral direction an adjacent row may be "
             "many pixels away: a cosmic ray that crosses into it leaves a significant excess there, and "
             "growing into noise rejects good data (and too much of a fiber's profile, losing the row)."),
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
    rejectMaskPlane = pexConfig.Field(
        dtype=str,
        default="CR",
        doc=("Mask plane for rejected pixels. This is shared with cosmic-ray masks made elsewhere (which "
             "are respected, and through which a rejected region may grow)."),
    )
    rejectOutlierMaskPlane = pexConfig.Field(
        dtype=str,
        default="OUTLIER",
        doc=("Additional mask plane for pixels rejected as outliers in their own right (as opposed to "
             "pixels rejected because the rest of their fiber's trace was rejected; see "
             "rejectFiberFraction, or pixels masked elsewhere, e.g., by the ISR). Only these count "
             "towards the fibers' rejected fractions: counting the rest of a rejected fiber's trace "
             "would propagate the rejection across the fibers, and the masks from elsewhere are trusted "
             "(the fibers' fluxes are fit from the remaining pixels, as usual)."),
    )
    rejectRuinedFiberMaskPlane = pexConfig.Field(
        dtype=str,
        default="RUINED_FIBER",
        doc=("Additional mask plane for the pixels containing the light of a fiber in a row in which our "
             "rejection left too little of its profile to measure its flux (see "
             "rejectRuinedProfileThreshold). They are masked (with rejectMaskPlane too) and the row "
             "re-extracted without the fiber, as otherwise its flux is fit from the faint edges of its "
             "profile, which it shares with its neighbours, biasing them. These pixels don't count as "
             "masked when deciding whether a neighbour is also ruined."),
    )
    rejectRuinedProfileThreshold = pexConfig.Field(
        dtype=float,
        default=0.02,
        doc=("When excluding a ruined fiber from the fit (see rejectRuinedFiberMaskPlane), mask the pixels "
             "in which its normalised profile (which sums to unity over the row) exceeds this value, so "
             "that little of its light is left for its neighbours to absorb. Lower values leave less of "
             "its light but mask more of the neighbours' profiles (e.g., 0.005 makes many neighbours SUSPECT "
             "in b4, whose profiles have broad wings)."),
    )
    rejectBadFitThreshold = pexConfig.Field(
        dtype=float,
        default=10.0,
        doc=("When rejecting bad fits (see ExtractSpectraTask.run's rejectBadFits), the chi^2 per degree of "
             "freedom of a fiber's fit in a row (over the pixels in which its normalised profile exceeds "
             "rejectMinProfile, with both positive and negative residuals) above which the fit is bad. A "
             "cosmic ray on the core of a fiber is largely absorbed into the flux, leaving residuals that "
             "are too small to reject (the systematic error scales with the raised model) but a fit that "
             "is poor (low in the wings). The positive pixels of a bad fit (with at least half of the "
             "largest residual) are rejected. As those are the core of the profile, the rest of the "
             "fiber's trace in that row is then usually rejected too (rejectFiberFraction) and the "
             "spectral pixel flagged NO_DATA. Not used for the preliminary extraction."),
    )
    preliminaryRejectIterations = pexConfig.Field(
        dtype=int,
        default=1,
        doc=("Maximum number of rejection iterations in a preliminary extraction (one whose spectra are used "
             "to model the scattered light, before it is subtracted). Only the bright outliers matter for "
             "the scattered light, and the model is poorer before the scattered light is subtracted, so "
             "we reject less there."),
    )
    preliminaryRejectThreshold = pexConfig.Field(
        dtype=float,
        default=8.0,
        doc="Rejection threshold (positive residual/sigma) for outliers in a preliminary extraction",
    )
    preliminaryRejectFiberFraction = pexConfig.Field(
        dtype=float,
        default=0.0,
        doc=("Value of rejectFiberFraction in a preliminary extraction. A fiber rejected in a row is "
             "refit from its wings, which (before the scattered light is subtracted) contain light that "
             "isn't in the model. Disabled if <= 0."),
    )
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
        preliminary: bool = False,
        rejectBadFits: bool = False,
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
        preliminary : `bool`, optional
            Is this a preliminary extraction (e.g., one whose spectra are used
            to model the scattered light, which hasn't yet been subtracted)?
            If so, the rejection uses the ``preliminaryReject*`` parameters.
        rejectBadFits : `bool`, optional
            Also reject the positive pixels of fibers whose fits are bad in a
            row (``config.rejectBadFitThreshold``)? This assumes that the
            fiber profiles describe the light well, which isn't true near
            bright lines in unbinned data (the PSF isn't separable into the
            fiber profile and the LSF), but is for data binned in the spectral
            direction. Not used for the preliminary extraction.

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

        spectra = self.extractAllSpectra(
            maskedImage, fiberTraceSet, detectorMap, isBoxcar, rejectOutliers, preliminary, rejectBadFits
        )

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
        preliminary: bool = False,
        rejectBadFits: bool = False,
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
        preliminary : `bool`
            Is this a preliminary extraction? If so, the rejection uses the
            ``preliminaryReject*`` parameters.
        rejectBadFits : `bool`
            Also reject the positive pixels of bad fits (not for a preliminary
            extraction)? See ``run``.

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
                self.checkGeometry(maskedImage, fiberTraceSet)
                badFitThreshold = 0.0
                if preliminary:
                    iterations = self.config.preliminaryRejectIterations
                    threshold = self.config.preliminaryRejectThreshold
                    fiberFraction = self.config.preliminaryRejectFiberFraction
                else:
                    iterations = self.config.rejectIterations
                    threshold = self.config.rejectThreshold
                    fiberFraction = self.config.rejectFiberFraction
                    if rejectBadFits:
                        badFitThreshold = self.config.rejectBadFitThreshold
                total = 0
                numIterations = 0  # iterations that rejected something
                numRejected = 0
                for iteration in range(iterations):
                    numRejected = self.rejectOutliers(
                        maskedImage, fiberTraceSet, spectra, badBitMask, threshold, fiberFraction,
                        badFitThreshold,
                    )
                    self.log.debug("Iteration %d: rejected %d pixels", iteration, numRejected)
                    if numRejected == 0:
                        break
                    total += numRejected
                    numIterations += 1
                    spectra = fiberTraceSet.extractSpectra(
                        maskedImage, badBitMask, self.config.minFracMask, self.config.minFracImage
                    )
                self.log.info(
                    "Rejected %d outlying pixels (masked as %s) in %d iterations%s%s",
                    total, self.config.rejectMaskPlane, numIterations,
                    " (preliminary extraction)" if preliminary else "",
                    "; stopped at the iteration limit" if numRejected > 0 else "",
                )
                spectra, numNoData = self.flagRejectedSpectra(maskedImage, fiberTraceSet, spectra, badBitMask)
                if numNoData > 0:
                    self.log.info(
                        "Flagged %d spectral pixels as NO_DATA after rejecting outliers, and re-extracted "
                        "their rows without them", numNoData
                    )

        if detectorMap is not None:
            for spectrum in spectra:
                spectrum.setWavelength(detectorMap.getWavelength(spectrum.fiberId))

        return spectra

    @staticmethod
    def checkGeometry(maskedImage: MaskedImage, fiberTraceSet: FiberTraceSet):
        """Check the geometry assumed by the outlier rejection

        The rejection indexes the spectra (which are indexed by row relative
        to the image origin) and the image with the traces' bounding boxes,
        and ``FiberTrace.constructImage`` indexes the spectra by absolute row,
        so the image must have zero origin and contain the traces.

        Raises
        ------
        RuntimeError
            If the image origin is not zero, or a trace isn't contained in the
            image.
        """
        if maskedImage.getXY0() != Point2I(0, 0):
            raise RuntimeError(f"Outlier rejection requires zero image origin; xy0={maskedImage.getXY0()}")
        bbox = maskedImage.getBBox()
        for ft in fiberTraceSet:
            if not bbox.contains(ft.trace.getBBox()):
                raise RuntimeError(
                    f"Trace for fiberId={ft.fiberId} ({ft.trace.getBBox()}) is not contained in the image "
                    f"({bbox})"
                )

    def flagRejectedSpectra(
        self, maskedImage: MaskedImage, fiberTraceSet: FiberTraceSet, spectra: SpectrumSet, badBitMask: int
    ) -> Tuple[SpectrumSet, int]:
        """Flag as NO_DATA the spectral pixels that our rejection has ruined

        If too little of a fiber's profile remains unmasked in a row (less
        than ``config.minFracImage``), the extraction fits its flux from the
        faint edges of the profile and flags it ``SUSPECT``; such fluxes can
        be wildly wrong, and as the fiber shares those pixels with its
        neighbours, they bias the neighbours' fluxes too. Where our rejection
        contributed to that (there are pixels we rejected as outliers within
        the fiber's footprint in that row), we:

        - mask the pixels containing the fiber's light in that row
          (``config.rejectRuinedProfileThreshold``), with
          ``config.rejectMaskPlane`` and ``config.rejectRuinedFiberMaskPlane``;
        - re-extract, excluding the fiber from the fit in that row; and
        - flag the spectral pixel as ``NO_DATA`` (as the extraction does when
          the whole of the trace is masked), with zero flux and variance,
          retaining the masks that caused it (e.g., ``CR``).

        Fibers that are mostly masked for other reasons are left as they are,
        and pixels masked only because they contain a ruined fiber's light
        don't count towards ruining a neighbour.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which the spectra were extracted. The mask is modified.
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Fiber traces used for the extraction (unmodified on return).
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        badBitMask : `int`
            Bitmask for pixels to ignore; must include ``config.rejectMaskPlane``.

        Returns
        -------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra: re-extracted if any fiber was ruined, and with
            the ruined spectral pixels flagged.
        numFlagged : `int`
            Number of spectral pixels flagged.
        """
        mask = maskedImage.mask.array
        if self.config.rejectOutlierMaskPlane not in maskedImage.mask.getMaskPlaneDict():
            return spectra, 0
        outlierBitMask = maskedImage.mask.getPlaneBitMask(self.config.rejectOutlierMaskPlane)
        rejectBitMask = maskedImage.mask.getPlaneBitMask(self.config.rejectMaskPlane)
        maskedImage.mask.addMaskPlane(self.config.rejectRuinedFiberMaskPlane)
        ruinedBitMask = maskedImage.mask.getPlaneBitMask(self.config.rejectRuinedFiberMaskPlane)
        image = maskedImage.image.array
        variance = maskedImage.variance.array
        x0, y0 = maskedImage.getXY0()
        spectraByFiberId = {spectrum.fiberId: spectrum for spectrum in spectra}
        ruined = []  # (fiberTrace, rows of the trace, rows of the spectrum)
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
            traceMask = mask[rows, columns]  # a view
            # Pixels the C++ extraction doesn't use, whatever their mask
            invalid = ~(np.isfinite(image[rows, columns]) & np.isfinite(variance[rows, columns])
                        & (variance[rows, columns] > 0))
            # Pixels masked only because they contain the light of a ruined fiber (ours or a neighbour's)
            # count as unmasked: they don't ruin a fiber by themselves (no spreading from fiber to fiber),
            # and a fiber we've already ruined remains ruined by its outliers.
            onlyRuined = ((traceMask & ruinedBitMask) != 0) & ((traceMask & outlierBitMask) == 0)
            onlyRuined &= (traceMask & (badBitMask & ~rejectBitMask)) == 0
            bad = (((traceMask & badBitMask) != 0) & ~onlyRuined) | invalid
            unmasked = np.sum(profile*~bad, axis=1)
            # Our rejection must be what ruined the row: it contains our outliers, and without the rejection
            # (the rejectMaskPlane, which external cosmic-ray masks share) enough of the profile is left.
            # Rows ruined by other masks (e.g., saturation) are left as the extraction flagged them.
            other = ((traceMask & (badBitMask & ~rejectBitMask)) != 0) | invalid
            unmaskedOther = np.sum(profile*~other, axis=1)
            ours = np.any((profile > 0) & ((traceMask & outlierBitMask) != 0), axis=1)
            ours &= unmaskedOther >= self.config.minFracImage
            select = ours & (unmasked < self.config.minFracImage) & (norm != 0)
            if not np.any(select):
                continue
            light = np.zeros_like(inTrace)
            light[select] = profile[select] > self.config.rejectRuinedProfileThreshold
            # Only mark pixels that aren't already masked: those that are must keep counting as masked, or
            # the fiber would no longer appear ruined in a later pass and would be fit again.
            light &= (traceMask & badBitMask) == 0
            traceMask[light] |= rejectBitMask | ruinedBitMask
            traceRows = np.where(select)[0]
            ruined.append((ft, traceRows, traceRows + box.getMinY() - y0))

        if not ruined:
            return spectra, 0

        # Exclude the ruined fibers from the fit in those rows: they have no FIBERTRACE pixels there
        saved = []  # (fiberTrace, rows of the trace, original mask of those rows)
        try:
            for ft, traceRows, _ in ruined:
                traceMaskArray = ft.trace.mask.array
                saved.append((ft, traceRows, traceMaskArray[traceRows].copy()))
                traceMaskArray[traceRows] &= ~ft.trace.mask.getPlaneBitMask("FIBERTRACE")
            newSpectra = fiberTraceSet.extractSpectra(
                maskedImage, badBitMask, self.config.minFracMask, self.config.minFracImage
            )
        finally:
            for ft, traceRows, original in reversed(saved):
                ft.trace.mask.array[traceRows] = original

        ruinedRows = {ft.fiberId: index for ft, _, index in ruined}
        numFlagged = 0
        for spectrum in newSpectra:
            index = ruinedRows.get(spectrum.fiberId)
            if index is None:
                continue
            old = spectraByFiberId[spectrum.fiberId]
            spectrum.flux[index] = 0.0
            spectrum.variance[index] = 0.0
            spectrum.norm[index] = old.norm[index]
            spectrum.mask.array[0, index] = old.mask.array[0, index] | old.mask.getPlaneBitMask("NO_DATA")
            numFlagged += len(index)
        return newSpectra, numFlagged

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
        self,
        maskedImage: MaskedImage,
        fiberTraceSet: FiberTraceSet,
        spectra: SpectrumSet,
        badBitMask: int,
        threshold: Optional[float] = None,
        fiberFraction: Optional[float] = None,
        badFitThreshold: float = 0.0,
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

        In a preliminary extraction, the rejection is performed before the
        scattered light is subtracted (as the scattered light is estimated
        from the spectra), so there is smooth light in the image that isn't
        in the model. Within the bright part of a trace it is absorbed into
        the fiber's flux, but in the wings of a faint (or empty) fiber next to
        a gap it isn't, and the systematic error (which scales with the
        fiber's own small peak) doesn't cover it. We therefore use as the
        model the maximum of the model and the background measured between
        the fibers (see ``estimateBackground``); after the scattered light
        has been subtracted, this background should be small.

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
        threshold : `float`, optional
            Rejection threshold; defaults to ``config.rejectThreshold``.
        fiberFraction : `float`, optional
            Fraction of a fiber's profile that, if rejected in a row, causes
            the whole of its trace in that row to be rejected; defaults to
            ``config.rejectFiberFraction``. Disabled if <= 0.
        badFitThreshold : `float`, optional
            Chi^2 per degree of freedom above which a fiber's fit in a row is
            bad, and its positive pixels are rejected (see ``findBadFits``).
            Disabled if <= 0.

        Returns
        -------
        numRejected : `int`
            Number of newly rejected pixels.
        """
        if threshold is None:
            threshold = self.config.rejectThreshold
        if fiberFraction is None:
            fiberFraction = self.config.rejectFiberFraction
        model, profile = self.makeModelImage(maskedImage, fiberTraceSet, spectra)
        inTrace = profile > self.config.rejectMinProfile
        image = maskedImage.image.array
        variance = maskedImage.variance.array
        mask = maskedImage.mask.array
        rejectBitMask = maskedImage.mask.getPlaneBitMask(self.config.rejectMaskPlane)
        maskedImage.mask.addMaskPlane(self.config.rejectOutlierMaskPlane)
        outlierBitMask = maskedImage.mask.getPlaneBitMask(self.config.rejectOutlierMaskPlane)
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
            # Pixels we previously rejected as outliers in their own right (e.g., in a preliminary extraction
            # made without rejecting whole fibers)
            previousSeeds = (mask & outlierBitMask) != 0
            if fiberFraction > 0:
                # Don't reject anything else in the traces of fibers whose previously rejected pixels exceed
                # the fraction (we reject the rest of those traces below): the model there is unreliable
                _, protect = self.rejectFibers(
                    maskedImage, fiberTraceSet, spectra, previousSeeds, fiberFraction
                )
                candidate &= ~protect
            outlier = candidate & (chi > threshold)
            if badFitThreshold > 0:
                outlier |= self.findBadFits(
                    maskedImage, fiberTraceSet, spectra, chi, candidate, badFitThreshold
                )
            if self.config.rejectGrowThreshold > 0 and np.any(outlier):
                # Grow to the connected pixels with a significant excess (including those already rejected,
                # so that a region can be connected through them; but not the traces of ruined fibers, which
                # are masked whatever their contents). Already-rejected pixels include cosmic rays masked
                # elsewhere and the rest of rejected fibers' traces, so a region may travel through them,
                # but only the candidates are newly rejected.
                alreadyRejected = (mask & rejectBitMask) != 0
                ruined = np.zeros_like(alreadyRejected)
                if self.config.rejectRuinedFiberMaskPlane in maskedImage.mask.getMaskPlaneDict():
                    ruinedBitMask = maskedImage.mask.getPlaneBitMask(self.config.rejectRuinedFiberMaskPlane)
                    ruined = (mask & ruinedBitMask) != 0
                connect = outlier | (alreadyRejected & ~ruined)
                alongRow = (candidate & (chi > self.config.rejectGrowThreshold)) | connect
                acrossRows = (candidate & (chi > self.config.rejectGrowAcrossRowsThreshold)) | connect
                outlier |= self.growOutliers(outlier, alongRow, acrossRows) & inTrace & ~alreadyRejected
        if self.config.rejectGrow > 0:
            structure = np.ones((1, 2*self.config.rejectGrow + 1), dtype=bool)
            outlier |= scipy.ndimage.binary_dilation(outlier, structure=structure) & candidate
        fiberReject = np.zeros_like(outlier)
        if fiberFraction > 0:
            # The fractions include our previous rejections
            seeds = outlier | previousSeeds
            fiberReject = self.rejectFibers(maskedImage, fiberTraceSet, spectra, seeds, fiberFraction)[0]
            fiberReject &= ~seeds

        unmasked = (mask & rejectBitMask) == 0
        mask[outlier & unmasked] |= rejectBitMask | outlierBitMask
        mask[fiberReject & unmasked] |= rejectBitMask
        return int(np.count_nonzero((outlier | fiberReject) & unmasked))

    def findBadFits(
        self,
        maskedImage: MaskedImage,
        fiberTraceSet: FiberTraceSet,
        spectra: SpectrumSet,
        chi: np.ndarray,
        candidate: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """Find the positive pixels of fibers that are badly fit

        A cosmic ray on the core of a fiber is largely absorbed into the
        fiber's flux, raising the model (and hence the systematic error), so
        that its residuals are too small to reject; but the fit is poor, as
        the model is too high in the wings. For each fiber and row, we compute
        the chi^2 per degree of freedom over the candidate pixels in which the
        fiber's normalised profile exceeds ``config.rejectMinProfile``.

        A fit can also be poor because the profile is wrong (in every row of
        that fiber, e.g., profiles contaminated by a neighbouring broken cobra)
        or because of a feature that's in every fiber in that row (e.g., a
        bright sky line). A cosmic ray affects few fibers and rows, so a fit
        is bad if its chi^2 exceeds the threshold multiplied by the larger of
        unity, the fiber's median chi^2 over its rows, and the row's median
        chi^2 over the fibers. For a bad fit we return the pixels with
        positive residuals at least half of the largest. These are usually the
        core of the profile, so the rest of the fiber's trace in that row is
        then rejected too (``config.rejectFiberFraction``) and the spectral
        pixel is flagged ``NO_DATA`` (see ``flagRejectedSpectra``).

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            Image from which the spectra were extracted.
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Fiber traces used for the extraction.
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra.
        chi : `numpy.ndarray` of `float`
            Residuals divided by the noise.
        candidate : `numpy.ndarray` of `bool`
            Pixels that may be rejected (and are used to measure the fit).
        threshold : `float`
            Chi^2 per degree of freedom (relative to the fiber's and row's
            typical values) above which a fit is bad.

        Returns
        -------
        reject : `numpy.ndarray` of `bool`
            Pixels to reject.
        """
        reject = np.zeros_like(candidate)
        height = maskedImage.getHeight()
        x0, y0 = maskedImage.getXY0()
        spectraByFiberId = {spectrum.fiberId: spectrum for spectrum in spectra}
        chi2 = np.full((len(fiberTraceSet), height), np.nan)  # chi^2 per degree of freedom, by fiber and row
        traces = []  # (index, rows, columns, use, traceChi)
        for ii, ft in enumerate(fiberTraceSet):
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
            use = (profile > self.config.rejectMinProfile) & candidate[rows, columns]
            traceChi = np.where(use, chi[rows, columns], 0.0)
            num = use.sum(axis=1)
            chi2[ii, rows] = np.where(num >= 3, np.sum(traceChi**2, axis=1)/np.maximum(num - 1, 1), np.nan)
            traces.append((ii, rows, columns, use, traceChi))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN slices
            fiberMedian = np.nanmedian(chi2, axis=1)
            rowMedian = np.nanmedian(chi2, axis=0)
        baseline = np.maximum(np.nan_to_num(fiberMedian, nan=1.0)[:, np.newaxis],
                              np.nan_to_num(rowMedian, nan=1.0)[np.newaxis, :])
        baseline = np.maximum(baseline, 1.0)
        with np.errstate(invalid="ignore"):
            bad = chi2 > threshold*baseline  # False where chi2 is NaN

        for ii, rows, columns, use, traceChi in traces:
            traceBad = bad[ii, rows]
            if not np.any(traceBad):
                continue
            positive = np.where(use & (traceChi > 0), traceChi, 0.0)
            peak = positive.max(axis=1)
            select = traceBad[:, np.newaxis] & (positive > 0) & (positive >= 0.5*peak[:, np.newaxis])
            reject[rows, columns] |= select
        return reject

    @staticmethod
    def growOutliers(seeds: np.ndarray, alongRow: np.ndarray, acrossRows: np.ndarray) -> np.ndarray:
        """Grow outliers to the connected pixels with a significant excess

        A region grows along a row through the pixels in ``alongRow``, and into
        an adjacent row (including diagonally) only through the pixels in
        ``acrossRows`` (a subset of ``alongRow`` with a higher threshold),
        from which it continues to grow along that row.

        Parameters
        ----------
        seeds : `np.ndarray` of `bool`
            Pixels from which to grow.
        alongRow : `np.ndarray` of `bool`
            Pixels into which a region may grow within a row.
        acrossRows : `np.ndarray` of `bool`
            Pixels into which a region may grow from an adjacent row.

        Returns
        -------
        grown : `np.ndarray` of `bool`
            The grown regions (including the seeds).
        """
        alongRowStructure = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool)
        labels, _ = scipy.ndimage.label(alongRow | seeds, structure=alongRowStructure)
        grown = seeds.copy()
        while True:
            segments = np.unique(labels[grown])
            grown = np.isin(labels, segments[segments > 0]) | grown
            new = scipy.ndimage.binary_dilation(grown, structure=np.ones((3, 3), dtype=bool)) & acrossRows
            new &= ~grown
            if not np.any(new):
                return grown
            grown |= new

    def rejectFibers(
        self,
        maskedImage: MaskedImage,
        fiberTraceSet: FiberTraceSet,
        spectra: SpectrumSet,
        outlier: np.ndarray,
        fiberFraction: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extend rejections to the whole of a fiber's trace in a row

        The core of an outlier that resembles the fiber profile is absorbed
        into the fitted flux, so it can't be identified from the residuals.
        If the rejected pixels in a row carry more than ``fiberFraction``
        of a fiber's (normalised) profile, we reject all of that fiber's trace
        (where its normalised profile exceeds ``config.rejectMinProfile``) in
        that row. The rejected pixels provided should be the outliers
        themselves, not pixels rejected because the rest of their fiber was,
        or the rejection would propagate across the fibers.

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
        fiberFraction : `float`, optional
            Fraction of a fiber's profile that must be rejected in a row;
            defaults to ``config.rejectFiberFraction``.

        Returns
        -------
        reject : `numpy.ndarray` of `bool`
            Additional pixels to reject.
        footprint : `numpy.ndarray` of `bool`
            Footprint of the traces of the fibers rejected in each row.
        """
        if fiberFraction is None:
            fiberFraction = self.config.rejectFiberFraction
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
            select = fraction > fiberFraction
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
