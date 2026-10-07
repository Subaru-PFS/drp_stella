from typing import Iterable, List

import numpy as np

import lsst.afw.geom as afwGeom
import lsst.geom as geom
import lsst.meas.algorithms as measAlg
import lsst.meas.extensions.piff.piffPsfDeterminer  # noqa: F401  (registers "piff" in psfDeterminerRegistry)
from lsst.afw.image import ExposureF
from lsst.afw.table import SourceCatalog
from lsst.meas.algorithms import MakePsfCandidatesTask, ReserveSourcesTask
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from pfs.drp.stella.gen3 import readDatasetRefs

from ..adjustDetectorMap import AdjustDetectorMapTask
from ..centroidLines import CentroidLinesTask
from ..datamodel import PfsConfig
from ..DetectorMapContinued import DetectorMap
from ..readLineList import ReadLineListTask
from ..referenceLine import ReferenceLineStatus

__all__ = ("FitImagePsfTask", "selectGoodSources", "maskNeighboringFibers")


def selectGoodSources(catalog: SourceCatalog, badLineStatus: ReferenceLineStatus) -> np.ndarray:
    """Select sources suitable for use as PSF candidates

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        Catalog of line measurements, as returned by
        `CentroidLinesTask.measureCatalog`.
    badLineStatus : `pfs.drp.stella.ReferenceLineStatus`
        Combined status bits that disqualify a line from PSF fitting.

    Returns
    -------
    good : `numpy.ndarray` of `bool`
        True for rows that are suitable for use as PSF candidates.
    """
    status = catalog["status"].astype(np.int64)
    return (
        ~catalog["ignore"]
        & ((status & int(badLineStatus)) == 0)
        & ~catalog["centroid_flag"]
        & ~catalog["flux_flag"]
    )


def maskNeighboringFibers(
    candidates: List,
    catalog: SourceCatalog,
    detectorMap: DetectorMap,
    referenceLines,
    exposureHeight: int,
    halfWidth: float,
    halfHeight: float,
    threshold: float,
    log=None,
) -> None:
    """Mask out neighboring fibers' arc lines that leak into a stamp

    Fibers that are supposed to be unilluminated (or blocked) are not
    always perfectly dark, and a faint, leaked arc line can land inside a
    neighboring fiber's PSF-candidate stamp (fiber pitch is typically only
    a few times the stamp size). For each candidate, we check every
    *other* fiber whose trace passes through the stamp, at every
    reference line's expected position for that fiber; if there's a real
    peak there (not just noise), we mask a small box around it. If there's
    no significant peak, that fiber is taken to be genuinely dark there,
    and nothing is masked -- so this only ever removes pixels that show
    actual leaked signal, not whole trace columns.

    This relies on the candidate's stamp already being cached at
    ``config.makePsfCandidates.kernelSize`` (matching
    ``config.psfDeterminer.stampSize``, set in `FitImagePsfConfig.setDefaults`)
    so that the mask edits made here on ``cand.getMaskedImage()`` are the
    same pixels the determiner later reads, rather than a fresh re-extraction
    from the original (unmasked) exposure.

    Parameters
    ----------
    candidates : `list` of `lsst.meas.algorithms.PsfCandidate`
        Candidates to mask, modified in-place.
    catalog : `lsst.afw.table.SourceCatalog`
        Catalog aligned 1:1 with ``candidates``, providing each one's
        ``fiberId``.
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping of fiberId,wavelength to x,y.
    referenceLines : `pfs.drp.stella.ReferenceLineSet`
        Reference lines for this visit, used to predict where a
        neighboring fiber's arc lines would fall.
    exposureHeight : `int`
        Height of the exposure the candidates were drawn from, for
        clipping row lookups.
    halfWidth : `float`
        Half-width (pixels, cross-dispersion) of the local box masked
        around a neighboring fiber's arc line.
    halfHeight : `float`
        Half-height (pixels, dispersion direction) of the local box.
    threshold : `float`
        Signal-to-noise threshold for a neighboring fiber's arc line to be
        considered a real, leaked peak.
    log : optional
        Logger (with a ``debug`` method) to report how many stamps/positions
        were affected. If not provided, nothing is logged.
    """
    if not candidates or len(referenceLines) == 0:
        return
    fiberId = detectorMap.fiberId
    wavelength = referenceLines.wavelength
    # x-position of every fiber, at every row: computed once per exposure, reused for every candidate.
    xCenter = np.array([detectorMap.getXCenter(int(ff)) for ff in fiberId])

    numMasked = 0
    numChecked = 0
    for cand, targetFiberId in zip(candidates, catalog["fiberId"]):
        stamp = cand.getMaskedImage()
        bbox = stamp.getBBox()
        x0, y0 = bbox.getMinX(), bbox.getMinY()
        width, height = bbox.getWidth(), bbox.getHeight()
        xc, yc = cand.getXCenter(), cand.getYCenter()

        row0 = int(np.clip(round(yc), 0, exposureHeight - 1))
        searchHalfWidth = 0.5 * width + 5.0  # margin for trace slope across the stamp
        nearby = np.flatnonzero(
            (np.abs(xCenter[:, row0] - xc) < searchHalfWidth) & (fiberId != targetFiberId)
        )
        if len(nearby) == 0:
            continue

        imageArray = stamp.image.array
        varianceArray = stamp.variance.array
        maskArray = stamp.mask.array
        badBit = stamp.mask.getPlaneBitMask("BAD")
        stampWasMasked = False

        for ff in nearby:
            points = detectorMap.findPoint(int(fiberId[ff]), wavelength)
            px, py = points[:, 0], points[:, 1]
            inStamp = (
                np.isfinite(px)
                & np.isfinite(py)
                & (px >= x0 + halfWidth)
                & (px < x0 + width - halfWidth)
                & (py >= y0 + halfHeight)
                & (py < y0 + height - halfHeight)
            )
            for lineX, lineY in zip(px[inStamp], py[inStamp]):
                numChecked += 1
                col, row = int(round(lineX - x0)), int(round(lineY - y0))
                colLo, colHi = col - int(np.ceil(halfWidth)), col + int(np.ceil(halfWidth)) + 1
                rowLo, rowHi = row - int(np.ceil(halfHeight)), row + int(np.ceil(halfHeight)) + 1
                colLo, colHi = max(0, colLo), min(width, colHi)
                rowLo, rowHi = max(0, rowLo), min(height, rowHi)
                window = imageArray[rowLo:rowHi, colLo:colHi]
                varWindow = varianceArray[rowLo:rowHi, colLo:colHi]
                with np.errstate(invalid="ignore", divide="ignore"):
                    snr = window / np.sqrt(varWindow)
                peak = np.nanmax(snr) if snr.size else np.nan
                if not np.isfinite(peak) or peak < threshold:
                    continue  # no significant peak here: this fiber is genuinely dark, nothing to mask
                maskArray[rowLo:rowHi, colLo:colHi] |= badBit
                stampWasMasked = True
        if stampWasMasked:
            numMasked += 1
    if log is not None:
        log.debug(
            "Masked a neighboring-fiber line in %d/%d candidate stamps (%d neighbor positions checked)",
            numMasked,
            len(candidates),
            numChecked,
        )


class FitImagePsfConnections(PipelineTaskConnections, dimensions=("instrument", "arm", "spectrograph")):
    """Connections for FitImagePsfTask"""

    exposures = InputConnection(
        name="postISRCCD",
        doc="Input ISR-corrected exposures, one per visit, with different fibers lit",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    pfsConfigs = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration, one per visit",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
        multiple=True,
    )
    detectorMap = PrerequisiteConnection(
        name="detectorMap_calib",
        doc="Mapping from fiberId,wavelength to x,y: measured from real data, possibly from a "
        "different visit (so may not precisely apply to any of these exposures; see "
        "adjustedDetectorMaps).",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    psf = OutputConnection(
        name="imagePsf",
        doc="PSF fit from isolated fiber images, treating the data as ordinary imaging data",
        storageClass="Psf",
        dimensions=("instrument", "arm", "spectrograph"),
    )
    adjustedDetectorMaps = OutputConnection(
        name="detectorMap_imagePsf",
        doc="detectorMap, adjusted to each individual visit (slit/optics can drift between the "
        "detectorMap_calib's own visit and these); for reuse by e.g. FitImagePsfQaTask.",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )


class FitImagePsfConfig(PipelineTaskConfig, pipelineConnections=FitImagePsfConnections):
    """Configuration for FitImagePsfTask"""

    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read line lists")
    centroidLines = ConfigurableField(target=CentroidLinesTask, doc="Find and measure isolated line images")
    makePsfCandidates = ConfigurableField(target=MakePsfCandidatesTask, doc="Make PSF candidates")
    reserve = ConfigurableField(target=ReserveSourcesTask, doc="Reserve some sources for validation")
    adjustDetectorMap = ConfigurableField(
        target=AdjustDetectorMapTask,
        doc="Adjust the (possibly different-visit) calib detectorMap to each individual exposure",
    )
    computeExclusionRadius = Field(
        dtype=bool,
        default=True,
        doc="Automatically set readLineList.exclusionRadius from makePsfCandidates.kernelSize and the "
        "detectorMap's own dispersion (nm/pixel), so a line only gets used if no other reference line "
        "falls within one stamp height of it. Overrides any readLineList.exclusionRadius configured "
        "directly; set False to use the configured value as-is.",
    )
    psfDeterminer = measAlg.psfDeterminerRegistry.makeField("PSF determination algorithm", default="piff")
    badLineStatus = Field(
        dtype=str,
        default="NOT_VISIBLE,BLEND,SUSPECT,REJECTED",
        doc="Comma-separated ReferenceLineStatus names that disqualify a line from PSF fitting",
    )
    attachDummyWcs = Field(
        dtype=bool,
        default=True,
        doc="Attach a synthetic, trivial pixel-scale Wcs to each exposure? PFS exposures have no real "
        "Wcs, but some psfDeterminers (e.g. piff) unconditionally query the exposure's pixel scale.",
    )
    dummyPixelScale = Field(
        dtype=float,
        default=1.0,
        doc="Pixel scale (arcsec/pixel) of the synthetic Wcs attached when attachDummyWcs is True",
    )
    neighborMaskHalfWidth = Field(
        dtype=float,
        default=2.0,
        doc="Half-width (pixels, cross-dispersion) of the local box masked around a neighboring fiber's "
        "arc line, to guard against light leaking through fibers that are supposed to be "
        "unilluminated/blocked. Should be less than half the fiber pitch.",
    )
    neighborMaskHalfHeight = Field(
        dtype=float,
        default=5.0,
        doc="Half-height (pixels, dispersion direction) of the local box masked around a neighboring "
        "fiber's arc line.",
    )
    neighborPeakThreshold = Field(
        dtype=float,
        default=5.0,
        doc="Signal-to-noise threshold for a neighboring fiber's arc line to be considered a real, "
        "leaked peak (and so be masked). If a neighboring fiber shows no significant peak at a "
        "reference line's position, it is taken to be genuinely dark there, and nothing is masked.",
    )

    def setDefaults(self):
        super().setDefaults()
        # samplingSize=0.7 (see below) needs modelSize/samplingSize=25/0.7=35 pixels of stamp to
        # validate; keep kernelSize in lock-step with it (see below).
        self.makePsfCandidates.kernelSize = 35
        # The psfDeterminer must request candidate stamps at exactly this size, or it will silently
        # re-extract a fresh, unmasked copy from the original exposure instead of reusing the one we
        # mask in maskNeighboringFibers() (PsfCandidate only caches/reuses a stamp at one fixed size).
        for name in self.psfDeterminer.registry:
            determinerConfig = self.psfDeterminer[name]
            determinerConfig.stampSize = self.makePsfCandidates.kernelSize
            # Default of 300 silently down-samples away most of our candidates; our informed
            # selection means we usually have far more good, isolated lines available than that.
            determinerConfig.maxCandidates = 1000
            # Several neighboring-fiber bands are typically masked within one stamp (see
            # neighborMaskHalfWidth), so the usual 0.5 minimum would reject most/all candidates.
            if hasattr(determinerConfig, "minimumUnmaskedFraction"):
                determinerConfig.minimumUnmaskedFraction = 0.3
            # Our fiber PSF core is undersampled (FWHM only ~3 pixels); the stack's generic default
            # of no oversampling (1.0) isn't fine enough to resolve it well.
            if hasattr(determinerConfig, "samplingSize"):
                determinerConfig.samplingSize = 0.7


class FitImagePsfTask(PipelineTask):
    """Fit an LSST-style imaging PSF from a series of sparse-fiber exposures

    Some arc exposures expose only a subset of fibers (e.g. every 4th fiber).
    At that spacing, each lit fiber's arc lines are isolated point sources, so
    the image can be treated like ordinary astronomical imaging data, and the
    LSST stack's own PSF-determination machinery
    (`lsst.meas.algorithms`/`lsst.meas.extensions.piff`) can be used directly,
    rather than reimplementing PSF fitting ourselves.

    No single exposure covers the whole detector (only the lit fibers are
    useful), so we combine the isolated-source "stamps" from a series of
    exposures -- each with a different subset of fibers lit -- into a single
    list of PSF candidates before handing them to the (single-exposure) LSST
    stack PSF determiner.

    Sources are found using the existing `CentroidLinesTask` machinery: known
    arc lines are looked up via the ``detectorMap`` at the position of each
    lit fiber (per ``pfsConfig``), so this is an "informed" search rather than
    blind source detection.
    """

    ConfigClass = FitImagePsfConfig
    _DefaultName = "fitImagePsf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")
        self.makeSubtask("centroidLines")
        self.makeSubtask("makePsfCandidates")
        self.makeSubtask("adjustDetectorMap")
        self.makeSubtask("psfDeterminer")  # no schema: we're not adding any calib_psf_* flags via this path
        # PsfCandidate/the psfDeterminer expect slot_Shape and slot_PsfFlux to be set (CentroidLinesTask
        # only sets up slot_Centroid, since ArcLineSet consumers access fields by name instead).
        aliases = self.centroidLines.schema.getAliasMap()
        aliases.set("slot_Shape", self.centroidLines.shapeName)
        aliases.set("slot_PsfFlux", self.centroidLines.photometryName)
        # The reserve subtask adds a flag field to the schema, so it must be constructed after
        # centroidLines (whose schema it modifies) and before any catalogs are built from that schema.
        self.makeSubtask(
            "reserve",
            columnName="calib_psf",
            schema=self.centroidLines.schema,
            doc="Reserved from PSF determination for validation",
        )
        self.badLineStatus = ReferenceLineStatus.fromNames(
            *(name.strip() for name in self.config.badLineStatus.split(","))
        )

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        arm = inputRefs.exposures[0].dataId.arm.name
        spectrograph = inputRefs.exposures[0].dataId.spectrograph.num
        data = readDatasetRefs(butler, inputRefs, "exposures", "pfsConfigs")
        detectorMap = butler.get(inputRefs.detectorMap)
        outputs = self.run(data.exposures, data.pfsConfigs, detectorMap, arm=arm, spectrograph=spectrograph)
        butler.put(outputs.psf, outputRefs.psf)
        for ref, adjusted in zip(outputRefs.adjustedDetectorMaps, outputs.adjustedDetectorMaps):
            butler.put(adjusted, ref)
        return outputs

    def run(
        self,
        exposures: Iterable[ExposureF],
        pfsConfigs: Iterable[PfsConfig],
        detectorMap: DetectorMap,
        arm: str = "",
        spectrograph: int = 0,
    ) -> Struct:
        """Fit a PSF from a series of sparse-fiber exposures

        Parameters
        ----------
        exposures : iterable of `lsst.afw.image.ExposureF`
            ISR-corrected exposures, one per visit, each with a different
            subset of fibers lit. All must be of the same detector (arm,
            spectrograph).
        pfsConfigs : iterable of `pfs.datamodel.PfsConfig`
            Top-end fiber configuration for each exposure, aligned 1:1 with
            ``exposures``.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y: the starting point for a
            per-visit adjustment (see `adjustDetectorMapForVisit`), since it
            may come from a different visit than any of these exposures.
        arm : `str`
            Spectrograph arm in use (``b``, ``r``, ``n``, ``m``); required in
            order to adjust the detectorMap for each visit.
        spectrograph : `int`, optional
            Spectrograph module number, for logging only.

        Returns
        -------
        psf : `lsst.afw.detection.Psf`
            Fitted PSF, valid across the combined spatial coverage of all the
            input exposures.
        cellSet : `lsst.afw.math.SpatialCellSet`
            PSF candidates used/rejected by the determiner, for diagnostics.
        usedCatalog : `lsst.afw.table.SourceCatalog`
            Combined catalog of line measurements that were available to the
            determiner (both used and reserved for validation).
        allCandidates : `list` of `lsst.meas.algorithms.PsfCandidate`
            PSF candidates for every row of ``usedCatalog`` (same order),
            including those reserved for validation (and so excluded from
            ``cellSet`, which only holds candidates seen by the determiner).
        adjustedDetectorMaps : `list` of `pfs.drp.stella.DetectorMap`
            The per-visit adjusted detectorMap for each input exposure (same
            order, one per element of ``exposures``).
        """
        if self.config.computeExclusionRadius:
            midFiber = detectorMap.fiberId[len(detectorMap.fiberId) // 2]
            dispersion = detectorMap.getDispersionAtCenter(midFiber)  # nm/pixel
            exclusionRadius = self.config.makePsfCandidates.kernelSize * dispersion
            self.log.info(
                "Setting readLineList.exclusionRadius = %.3f nm (dispersion=%.4f nm/pixel, kernelSize=%d)",
                exclusionRadius,
                dispersion,
                self.config.makePsfCandidates.kernelSize,
            )
            self.readLineList.config.exclusionRadius = exclusionRadius

        allCandidates = []
        goodStarCat = SourceCatalog(self.centroidLines.schema)
        referenceExposure = None
        numExposures = 0
        nextId = 1  # Each per-visit catalog gets its own ids starting from 1; renumber to be unique overall
        adjustedDetectorMaps = []

        for exposure, pfsConfig in zip(exposures, pfsConfigs):
            visitDetectorMap = self.adjustDetectorMapForVisit(exposure, pfsConfig, detectorMap, arm)
            adjustedDetectorMaps.append(visitDetectorMap)

            refLines = self.readLineList.run(visitDetectorMap, exposure.getMetadata())
            if len(refLines) == 0:
                self.log.warn("No reference lines for visit %d; skipping", exposure.visitInfo.id)
                continue
            catalog = self.centroidLines.measureCatalog(
                exposure, refLines, visitDetectorMap, pfsConfig, seed=exposure.visitInfo.id
            )
            good = selectGoodSources(catalog, self.badLineStatus)
            selected = catalog[good].copy(deep=True)
            self.log.info(
                "Visit %d: %d/%d line measurements selected as PSF candidates",
                exposure.visitInfo.id,
                len(selected),
                len(catalog),
            )
            if len(selected) == 0:
                continue
            for record in selected:
                record.setId(nextId)
                nextId += 1

            if self.config.attachDummyWcs:
                exposure.setWcs(self.makeDummyWcs(exposure))
            if referenceExposure is None:
                referenceExposure = exposure

            candidates = self.makePsfCandidates.run(selected, exposure)
            maskNeighboringFibers(
                candidates.psfCandidates,
                candidates.goodStarCat,
                visitDetectorMap,
                refLines,
                exposure.getBBox().getHeight(),
                self.config.neighborMaskHalfWidth,
                self.config.neighborMaskHalfHeight,
                self.config.neighborPeakThreshold,
                log=self.log,
            )
            allCandidates.extend(candidates.psfCandidates)
            goodStarCat.extend(candidates.goodStarCat, deep=True)
            numExposures += 1

        if referenceExposure is None or not allCandidates:
            raise RuntimeError("No usable PSF candidates found in any input exposure")

        reserveResult = self.reserve.run(goodStarCat, expId=referenceExposure.visitInfo.id)
        psfCandidateList: List = [cand for cand, use in zip(allCandidates, reserveResult.use) if use]
        self.log.info(
            "Sending %d/%d candidates (from %d exposures) to PSF determiner",
            len(psfCandidateList),
            len(allCandidates),
            numExposures,
        )

        psf, cellSet = self.psfDeterminer.determinePsf(referenceExposure, psfCandidateList, self.metadata)
        size = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()
        if not np.isfinite(size):
            raise RuntimeError(f"Fitted PSF ({arm}{spectrograph}) has a non-finite size")
        self.log.info("Fitted PSF (%s%s) size: %f pixels", arm, spectrograph, size)

        return Struct(
            psf=psf,
            cellSet=cellSet,
            usedCatalog=goodStarCat,
            allCandidates=allCandidates,
            adjustedDetectorMaps=adjustedDetectorMaps,
        )

    def adjustDetectorMapForVisit(
        self, exposure: ExposureF, pfsConfig: PfsConfig, detectorMap: DetectorMap, arm: str
    ) -> DetectorMap:
        """Adjust the calib detectorMap to this specific visit

        The calib detectorMap may come from a different visit than
        ``exposure``, and the slit/optics can drift between visits, so we
        measure lines against it and fit a low-order correction specific to
        this exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure to adjust the detectorMap for.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end fiber configuration for this exposure.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Starting-point (calib) detectorMap.
        arm : `str`
            Spectrograph arm in use (``b``, ``r``, ``n``, ``m``).

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            detectorMap adjusted to this exposure.
        """
        # Use the full (unfiltered) line list for this bootstrap pass, to maximise the training set
        # for the geometric fit; temporarily override our own (possibly nonzero) exclusion radius.
        exclusionRadius = self.readLineList.config.exclusionRadius
        self.readLineList.config.exclusionRadius = 0.0
        try:
            refLines = self.readLineList.run(detectorMap, exposure.getMetadata())
            arcLines = self.centroidLines.run(
                exposure, refLines, detectorMap, pfsConfig, seed=exposure.visitInfo.id
            )
        finally:
            self.readLineList.config.exclusionRadius = exclusionRadius

        adjusted = self.adjustDetectorMap.run(
            detectorMap, arcLines, arm, exposure.visitInfo, seed=exposure.visitInfo.id
        )
        self.log.info(
            "Visit %d: adjusted detectorMap xRms=%.4f yRms=%.4f pixels (%d/%d lines used)",
            exposure.visitInfo.id,
            adjusted.xRms,
            adjusted.yRms,
            adjusted.selection.sum(),
            len(adjusted.selection),
        )
        return adjusted.detectorMap

    def selectGoodSources(self, catalog: SourceCatalog) -> np.ndarray:
        """Select sources suitable for use as PSF candidates

        See the module-level `selectGoodSources` function.
        """
        return selectGoodSources(catalog, self.badLineStatus)

    def maskNeighboringFibers(
        self,
        candidates: List,
        catalog: SourceCatalog,
        detectorMap: DetectorMap,
        referenceLines,
        exposureHeight: int,
    ) -> None:
        """Mask out neighboring fibers' arc lines that leak into a stamp

        See the module-level `maskNeighboringFibers` function.
        """
        maskNeighboringFibers(
            candidates,
            catalog,
            detectorMap,
            referenceLines,
            exposureHeight,
            self.config.neighborMaskHalfWidth,
            self.config.neighborMaskHalfHeight,
            self.config.neighborPeakThreshold,
            log=self.log,
        )

    def makeDummyWcs(self, exposure: ExposureF):
        """Construct a trivial, synthetic pixel-scale Wcs

        PFS exposures have no real (celestial) Wcs, but some ``psfDeterminer``
        implementations (e.g. ``piff``) unconditionally query the exposure's
        pixel scale. We attach a synthetic tangent-plane Wcs with exactly
        ``config.dummyPixelScale`` arcsec/pixel (no rotation), purely so that
        query succeeds; its sky position is arbitrary and not meaningful.

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure for which to construct a Wcs.

        Returns
        -------
        wcs : `lsst.afw.geom.SkyWcs`
            Synthetic Wcs.
        """
        center = geom.Point2D(exposure.getBBox().getCenter())
        crval = geom.SpherePoint(0.0 * geom.degrees, 0.0 * geom.degrees)
        cdMatrix = afwGeom.makeCdMatrix(scale=self.config.dummyPixelScale * geom.arcseconds)
        return afwGeom.makeSkyWcs(center, crval, cdMatrix)
