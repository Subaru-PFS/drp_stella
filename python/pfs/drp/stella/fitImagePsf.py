from typing import Iterable, List

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import lsst.afw.geom as afwGeom
import lsst.geom as geom
import lsst.meas.algorithms as measAlg
import lsst.meas.extensions.piff.piffPsfDeterminer  # noqa: F401  (registers "piff" in psfDeterminerRegistry)
from lsst.afw.image import ExposureF, makeExposure
from lsst.afw.table import SourceCatalog
from lsst.meas.algorithms import MakePsfCandidatesTask, ReserveSourcesTask
from lsst.pex.config import ConfigurableField, Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection

from pfs.drp.stella.gen3 import readDatasetRefs

from .adjustDetectorMap import AdjustDetectorMapTask
from .centroidLines import CentroidLinesTask
from .datamodel import PfsConfig
from .DetectorMapContinued import DetectorMap
from .readLineList import ReadLineListTask
from .referenceLine import ReferenceLineStatus

__all__ = ("FitImagePsfTask", "FitImagePsfQaTask", "selectGoodSources", "maskNeighboringFibers")


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
        isCalibration=True,
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
        doc="Automatically use (for this Task's own calls to readLineList.run only, without modifying "
        "readLineList.config, which may be frozen under pipetask) the greater of the configured "
        "readLineList.exclusionRadius and a value computed from makePsfCandidates.kernelSize and the "
        "detectorMap's own dispersion (nm/pixel), so a line only gets used as a PSF candidate if no "
        "other reference line falls within one stamp height of it. Set False to use "
        "readLineList.exclusionRadius as configured, with no automatic minimum.",
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
        default=3.0,
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
        default=3.0,
        doc="Signal-to-noise threshold for a neighboring fiber's arc line to be considered a real, "
        "leaked peak (and so be masked). If a neighboring fiber shows no significant peak at a "
        "reference line's position, it is taken to be genuinely dark there, and nothing is masked.",
    )

    def setDefaults(self):
        super().setDefaults()
        # samplingSize=0.7 (see below) needs modelSize/samplingSize pixels of stamp to
        # validate; keep kernelSize in lock-step with it (see below).
        self.makePsfCandidates.kernelSize = 17
        # The psfDeterminer must request candidate stamps at exactly this size, or it will silently
        # re-extract a fresh, unmasked copy from the original exposure instead of reusing the one we
        # mask in maskNeighboringFibers() (PsfCandidate only caches/reuses a stamp at one fixed size).
        determinerConfig = self.psfDeterminer["piff"]
        determinerConfig.modelSize = 11
        determinerConfig.stampSize = self.makePsfCandidates.kernelSize
        # Default of 300 silently down-samples away most of our candidates; our informed
        # selection means we usually have far more good, isolated lines available than that.
        determinerConfig.maxCandidates = 1000
        # Several neighboring-fiber bands are typically masked within one stamp (see
        # neighborMaskHalfWidth), so the usual 0.5 minimum would reject most/all candidates.
        determinerConfig.minimumUnmaskedFraction = 0.3
        # Our fiber PSF core is undersampled (FWHM only ~3 pixels); the stack's generic default
        # of no oversampling (1.0) isn't fine enough to resolve it well.
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
        exclusionRadius = None  # None means: leave readLineList.config.exclusionRadius as configured
        if self.config.computeExclusionRadius:
            midFiber = detectorMap.fiberId[len(detectorMap.fiberId) // 2]
            dispersion = detectorMap.getDispersionAtCenter(midFiber)  # nm/pixel
            minExclusionRadius = self.config.makePsfCandidates.kernelSize * dispersion
            exclusionRadius = max(self.readLineList.config.exclusionRadius, minExclusionRadius)
            self.log.info(
                "Using readLineList.exclusionRadius = %.3f nm (dispersion=%.4f nm/pixel, kernelSize=%d)",
                exclusionRadius,
                dispersion,
                self.config.makePsfCandidates.kernelSize,
            )

        allCandidates = []
        goodStarCat = SourceCatalog(self.centroidLines.schema)
        referenceExposure = None
        numExposures = 0
        nextId = 1  # Each per-visit catalog gets its own ids starting from 1; renumber to be unique overall
        adjustedDetectorMaps = []

        for exposure, pfsConfig in zip(exposures, pfsConfigs):
            visitDetectorMap = self.adjustDetectorMapForVisit(exposure, pfsConfig, detectorMap, arm)
            adjustedDetectorMaps.append(visitDetectorMap)

            refLines = self.readLineList.run(
                visitDetectorMap, exposure.getMetadata(), exclusionRadius=exclusionRadius
            )
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
        # for the geometric fit; override our own (possibly nonzero) exclusion radius for this call only.
        refLines = self.readLineList.run(detectorMap, exposure.getMetadata(), exclusionRadius=0.0)
        arcLines = self.centroidLines.run(
            exposure, refLines, detectorMap, pfsConfig, seed=exposure.visitInfo.id
        )

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


class FitImagePsfQaConnections(PipelineTaskConnections, dimensions=("instrument", "arm", "spectrograph")):
    """Connections for FitImagePsfQaTask"""

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
    detectorMaps = InputConnection(
        name="detectorMap_imagePsf",
        doc="detectorMap adjusted to each individual visit, as produced by FitImagePsfTask",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    psf = InputConnection(
        name="imagePsf",
        doc="Fitted PSF to assess",
        storageClass="Psf",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    qualityPlot = OutputConnection(
        name="imagePsfQualityPlot",
        doc="PSF fit quality QA figure (chi2 distributions, stacked residual, sample stamps)",
        storageClass="Plot",
        dimensions=("instrument", "arm", "spectrograph"),
    )
    spatialPlot = OutputConnection(
        name="imagePsfSpatialPlot",
        doc="PSF spatial variation QA figure (image mosaic, FWHM map, ellipticity map)",
        storageClass="Plot",
        dimensions=("instrument", "arm", "spectrograph"),
    )
    residuals = OutputConnection(
        name="imagePsfResidual",
        doc="Exposure with every line used to fit the PSF subtracted, as an end-to-end sanity check",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )


class FitImagePsfQaConfig(PipelineTaskConfig, pipelineConnections=FitImagePsfQaConnections):
    """Configuration for FitImagePsfQaTask"""

    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read line lists")
    centroidLines = ConfigurableField(target=CentroidLinesTask, doc="Find and measure isolated line images")
    makePsfCandidates = ConfigurableField(target=MakePsfCandidatesTask, doc="Make PSF candidates")
    reserve = ConfigurableField(
        target=ReserveSourcesTask, doc="Reserve some sources for an independent used-vs-held-out check"
    )
    badLineStatus = Field(
        dtype=str,
        default="NOT_VISIBLE,BLEND,SUSPECT,REJECTED",
        doc="Comma-separated ReferenceLineStatus names that disqualify a line from use in these QA checks",
    )
    neighborMaskHalfWidth = Field(
        dtype=float, default=2.0, doc="Half-width (pixels, cross-dispersion) of the neighbor-fiber mask box"
    )
    neighborMaskHalfHeight = Field(
        dtype=float, default=5.0, doc="Half-height (pixels, dispersion direction) of the neighbor-fiber mask"
    )
    neighborPeakThreshold = Field(
        dtype=float, default=5.0, doc="Signal-to-noise threshold for masking a neighboring fiber's line"
    )
    computeExclusionRadius = Field(
        dtype=bool,
        default=True,
        doc="Automatically use (for this Task's own calls to readLineList.run only, without modifying "
        "readLineList.config, which may be frozen under pipetask) the greater of the configured "
        "readLineList.exclusionRadius and a value computed from makePsfCandidates.kernelSize and the "
        "detectorMap's dispersion, matching FitImagePsfTask's own behaviour. Set False to use "
        "readLineList.exclusionRadius as configured, with no automatic minimum.",
    )
    goodPixelMaskPlanes = ListField(
        dtype=str,
        default=["BAD", "CR", "INTRP", "SAT", "SUSPECT", "NO_DATA"],
        doc="Mask planes excluded from the residual/chi2 calculations (matches Piff's zeroWeightMaskBits)",
    )
    maxChi2Sample = Field(dtype=int, default=1500, doc="Maximum number of candidates to compute chi2 for")
    numSampleStamps = Field(dtype=int, default=36, doc="Number of individual residual stamps to display")
    gridNx = Field(dtype=int, default=7, doc="Number of grid points (x) for the spatial-variation figure")
    gridNy = Field(dtype=int, default=8, doc="Number of grid points (y) for the spatial-variation figure")
    gridMargin = Field(dtype=float, default=60.0, doc="Margin (pixels) to keep grid points from the edge")
    rngSeed = Field(dtype=int, default=0, doc="Seed for sampling/reservation random numbers")

    def setDefaults(self):
        super().setDefaults()
        # Should match the kernelSize the psf being assessed was actually fit with (FitImagePsfConfig's
        # own default); if that's been customized in the fitting pipeline, customize this to match.
        self.makePsfCandidates.kernelSize = 35


class FitImagePsfQaTask(PipelineTask):
    """Quality-assurance diagnostics for a PSF fitted by FitImagePsfTask

    Kept as a separate, downstream task (rather than folded into
    `FitImagePsfTask` itself) since it's purely diagnostic: it re-measures
    the same isolated arc lines used for fitting (now against the
    already visit-adjusted detectorMaps that task produced) and compares them
    to the already-fitted PSF, rather than redoing the (expensive) fit.
    """

    ConfigClass = FitImagePsfQaConfig
    _DefaultName = "fitImagePsfQa"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")
        self.makeSubtask("centroidLines")
        self.makeSubtask("makePsfCandidates")
        aliases = self.centroidLines.schema.getAliasMap()
        aliases.set("slot_Shape", self.centroidLines.shapeName)
        aliases.set("slot_PsfFlux", self.centroidLines.photometryName)
        self.makeSubtask(
            "reserve",
            columnName="calib_psf",
            schema=self.centroidLines.schema,
            doc="Reserved for an independent used-vs-held-out QA check",
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
        data = readDatasetRefs(butler, inputRefs, "exposures", "pfsConfigs", "detectorMaps")
        psf = butler.get(inputRefs.psf)
        outputs = self.run(data.exposures, data.pfsConfigs, data.detectorMaps, psf)
        butler.put(outputs.qualityPlot, outputRefs.qualityPlot)
        butler.put(outputs.spatialPlot, outputRefs.spatialPlot)
        for ref, residual in zip(outputRefs.residuals, outputs.residuals):
            butler.put(residual, ref)
        return outputs

    def run(
        self,
        exposures: Iterable[ExposureF],
        pfsConfigs: Iterable[PfsConfig],
        detectorMaps: Iterable[DetectorMap],
        psf,
    ) -> Struct:
        """Build QA products for a fitted PSF

        Parameters
        ----------
        exposures : iterable of `lsst.afw.image.ExposureF`
            The same ISR-corrected exposures used to fit ``psf``.
        pfsConfigs : iterable of `pfs.datamodel.PfsConfig`
            Top-end fiber configuration for each exposure, aligned 1:1 with
            ``exposures``.
        detectorMaps : iterable of `pfs.drp.stella.DetectorMap`
            Per-visit adjusted detectorMap for each exposure (aligned 1:1),
            as produced by `FitImagePsfTask`.
        psf : `lsst.afw.detection.Psf`
            Fitted PSF to assess.

        Returns
        -------
        qualityPlot, spatialPlot : `matplotlib.figure.Figure`
            QA figures.
        residuals : `list` of `lsst.afw.image.ExposureF`
            Per-visit exposures with every used line subtracted.
        """
        exposures = list(exposures)
        pfsConfigs = list(pfsConfigs)
        detectorMaps = list(detectorMaps)
        rng = np.random.RandomState(self.config.rngSeed)

        exclusionRadius = None  # None means: leave readLineList.config.exclusionRadius as configured
        if self.config.computeExclusionRadius:
            detMap0 = detectorMaps[0]
            midFiber = detMap0.fiberId[len(detMap0.fiberId) // 2]
            dispersion = detMap0.getDispersionAtCenter(midFiber)
            minExclusionRadius = self.config.makePsfCandidates.kernelSize * dispersion
            exclusionRadius = max(self.readLineList.config.exclusionRadius, minExclusionRadius)
            self.log.info("Using readLineList.exclusionRadius = %.3f nm", exclusionRadius)

        allCandidates: List = []
        goodStarCat = SourceCatalog(self.centroidLines.schema)
        nextId = 1
        residuals = []

        for exposure, pfsConfig, detectorMap in zip(exposures, pfsConfigs, detectorMaps):
            refLines = self.readLineList.run(
                detectorMap, exposure.getMetadata(), exclusionRadius=exclusionRadius
            )
            visitCandidates: List = []
            if len(refLines) > 0:
                catalog = self.centroidLines.measureCatalog(
                    exposure, refLines, detectorMap, pfsConfig, seed=exposure.visitInfo.id
                )
                good = selectGoodSources(catalog, self.badLineStatus)
                selected = catalog[good].copy(deep=True)
                for record in selected:
                    record.setId(nextId)
                    nextId += 1
                if len(selected) > 0:
                    candidates = self.makePsfCandidates.run(selected, exposure)
                    maskNeighboringFibers(
                        candidates.psfCandidates,
                        candidates.goodStarCat,
                        detectorMap,
                        refLines,
                        exposure.getBBox().getHeight(),
                        self.config.neighborMaskHalfWidth,
                        self.config.neighborMaskHalfHeight,
                        self.config.neighborPeakThreshold,
                        log=self.log,
                    )
                    visitCandidates = candidates.psfCandidates
                    allCandidates.extend(visitCandidates)
                    goodStarCat.extend(candidates.goodStarCat, deep=True)
            else:
                self.log.warn("No reference lines for visit %d", exposure.visitInfo.id)

            residuals.append(self.buildResidualExposure(exposure, psf, visitCandidates))

        if not allCandidates:
            raise RuntimeError("No usable candidates found for PSF QA")

        reserved = self.reserve.run(goodStarCat, expId=self.config.rngSeed).reserved

        diagnostics = self.computeDiagnostics(psf, allCandidates, goodStarCat, reserved, rng)
        qualityPlot = self.makeQualityPlot(diagnostics)
        spatialPlot = self.makeSpatialPlot(psf, exposures[0].getBBox(), rng)

        return Struct(qualityPlot=qualityPlot, spatialPlot=spatialPlot, residuals=residuals)

    def buildResidualExposure(self, exposure: ExposureF, psf, candidates: List) -> ExposureF:
        """Subtract every candidate line from a copy of the exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.ExposureF`
            Exposure to subtract from (not modified).
        psf : `lsst.afw.detection.Psf`
            Fitted PSF to subtract.
        candidates : `list` of `lsst.meas.algorithms.PsfCandidate`
            Candidates (lines) to subtract, from this exposure.

        Returns
        -------
        residual : `lsst.afw.image.ExposureF`
            Exposure with every candidate line subtracted.
        """
        residual = exposure.maskedImage.clone()
        detectedBit = residual.mask.getPlaneBitMask("DETECTED")
        imageBBox = residual.getBBox()

        numSubtracted = 0
        numSkipped = 0
        for cand in candidates:
            xc, yc = cand.getXCenter(), cand.getYCenter()
            if not (np.isfinite(xc) and np.isfinite(yc)):
                numSkipped += 1
                continue
            try:
                stampBBox = psf.computeImage(geom.Point2D(xc, yc)).getBBox()
                stampBBox.clip(imageBBox)
                if stampBBox.isEmpty():
                    numSkipped += 1
                    continue
                residual.mask[stampBBox].array[:] |= detectedBit
                measAlg.subtractPsf(psf, residual, xc, yc)
                numSubtracted += 1
            except Exception as exc:
                self.log.debug("candidate at (%s,%s) could not be subtracted: %s", xc, yc, exc)
                numSkipped += 1
        self.log.info(
            "Visit %d: subtracted %d candidates, skipped %d", exposure.visitInfo.id, numSubtracted, numSkipped
        )

        residualExposure = makeExposure(residual)
        residualExposure.setDetector(exposure.getDetector())
        residualExposure.getInfo().setVisitInfo(exposure.getInfo().getVisitInfo())
        return residualExposure

    def computeDiagnostics(self, psf, candidates: List, catalog: SourceCatalog, reserved, rng) -> Struct:
        """Compute per-candidate chi2 and stacked/sample residual stamps

        Does the model subtraction manually (not via ``measAlg.subtractPsf``)
        so that: (1) the comparison is restricted to exactly the footprint
        used for fitting (reusing each candidate's own cached, already
        neighbor-masked stamp) rather than the determiner's internal "natural"
        draw size (which, for an oversampled model, can be considerably
        larger and pulls in unrelated, far-field neighbor-fiber
        contamination); and (2) it stays honest about which pixels were
        actually masked during fitting.

        Parameters
        ----------
        psf : `lsst.afw.detection.Psf`
            Fitted PSF.
        candidates : `list` of `lsst.meas.algorithms.PsfCandidate`
            All candidates (aligned 1:1 with ``catalog``).
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog aligned 1:1 with ``candidates``.
        reserved : `numpy.ndarray` of `bool`
            True for candidates held out as an independent check.
        rng : `numpy.random.RandomState`
            Random number generator for subsampling.

        Returns
        -------
        diagnostics : `lsst.pipe.base.Struct`
            ``candX``, ``candY``, ``candChi2``, ``candFlux``, ``reserved``
            (arrays, one per candidate), ``stampSize``, ``stackedResidual``
            (2D array), ``sampleStamps``, ``sampleReserved`` (lists).
        """
        numCand = len(candidates)
        stampSize = self.config.makePsfCandidates.kernelSize
        goodPixelMaskBits = list(self.config.goodPixelMaskPlanes)

        usedIdx = np.flatnonzero(~reserved)
        reservedIdx = np.flatnonzero(reserved)
        sampleUsed = rng.choice(usedIdx, size=min(self.config.maxChi2Sample, len(usedIdx)), replace=False)
        chiIdx = np.concatenate([sampleUsed, reservedIdx])
        rng.shuffle(chiIdx)

        candX = np.full(numCand, np.nan)
        candY = np.full(numCand, np.nan)
        candChi2 = np.full(numCand, np.nan)
        candFlux = np.array(catalog["flux_instFlux"], dtype=float)

        stackSum = np.zeros((stampSize, stampSize))
        stackCount = np.zeros((stampSize, stampSize))
        sampleStamps = []
        sampleReserved = []
        sampleSlots = set(
            rng.choice(chiIdx, size=min(self.config.numSampleStamps, len(chiIdx)), replace=False).tolist()
        )

        for i in chiIdx:
            cand = candidates[i]
            xc, yc = cand.getXCenter(), cand.getYCenter()
            candX[i], candY[i] = xc, yc
            try:
                stamp = cand.getMaskedImage()  # cached at kernelSize, already neighbor-masked
                dataBBox = stamp.getBBox()
                modelImage = psf.computeImage(geom.Point2D(xc, yc))
                modelSub = modelImage[dataBBox]
            except Exception as exc:
                self.log.debug("candidate %d: could not compare to model: %s", i, exc)
                continue

            badBit = stamp.mask.getPlaneBitMask(goodPixelMaskBits)
            good = (
                ((stamp.mask.array & badBit) == 0)
                & (stamp.variance.array > 0)
                & np.isfinite(stamp.variance.array)
                & np.isfinite(stamp.image.array)
            )
            if good.sum() < 10:
                continue

            d = stamp.image.array[good].astype(float)
            m = modelSub.array[good].astype(float)
            v = stamp.variance.array[good].astype(float)
            sumMM = np.sum(m * m / v)
            if sumMM <= 0:
                continue
            amp = np.sum(m * d / v) / sumMM
            resid = d - amp * m
            candChi2[i] = np.sum(resid**2 / v) / max(good.sum() - 1, 1)

            with np.errstate(invalid="ignore", divide="ignore"):
                chiImage = (stamp.image.array - amp * modelSub.array) / np.sqrt(stamp.variance.array)
            chiImage[~good] = np.nan

            if not reserved[i]:
                valid = np.isfinite(chiImage)
                stackSum[valid] += chiImage[valid]
                stackCount[valid] += 1
            if i in sampleSlots:
                sampleStamps.append(chiImage.copy())
                sampleReserved.append(bool(reserved[i]))

        for i, cand in enumerate(candidates):
            if np.isnan(candX[i]):
                candX[i] = cand.getXCenter()
                candY[i] = cand.getYCenter()

        with np.errstate(invalid="ignore", divide="ignore"):
            stackedResidual = stackSum / np.maximum(stackCount, 1)
        stackedResidual[stackCount == 0] = np.nan

        return Struct(
            candX=candX,
            candY=candY,
            candChi2=candChi2,
            candFlux=candFlux,
            reserved=reserved,
            stampSize=stampSize,
            stackedResidual=stackedResidual,
            sampleStamps=np.array(sampleStamps) if sampleStamps else np.zeros((0, stampSize, stampSize)),
            sampleReserved=np.array(sampleReserved, dtype=bool),
        )

    def makeQualityPlot(self, diag: Struct) -> Figure:
        """Build the fit-quality QA figure

        Parameters
        ----------
        diag : `lsst.pipe.base.Struct`
            As returned by `computeDiagnostics`.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
        """
        candChi2, candReserved = diag.candChi2, diag.reserved
        candX, candY, candFlux = diag.candX, diag.candY, diag.candFlux
        stackedResidual = diag.stackedResidual
        sampleStamps, sampleReserved = diag.sampleStamps, diag.sampleReserved

        fig = Figure(figsize=(16, 9), layout="constrained")
        fig.suptitle("FitImagePsfTask QA: fit quality", fontweight="bold")
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])

        ax = fig.add_subplot(gs[0, 0])
        valid = np.isfinite(candChi2)
        used, resv = valid & ~candReserved, valid & candReserved
        if valid.sum() > 0:
            bins = np.linspace(0, np.nanpercentile(candChi2[valid], 99), 60)
            ax.hist(candChi2[used], bins=bins, alpha=0.6, density=True, label=f"used (n={used.sum()})")
            ax.hist(candChi2[resv], bins=bins, alpha=0.6, density=True, label=f"reserved (n={resv.sum()})")
            ax.axvline(1.0, color="k", linestyle=":", linewidth=1, label="chi2=1")
            ax.legend(fontsize="small")
        ax.set_xlabel("reduced chi2 (per candidate)")
        ax.set_ylabel("density")
        ax.set_title("Residual chi2: used vs. held-out")

        ax = fig.add_subplot(gs[0, 1])
        cx, cy = np.nanmedian(candX), np.nanmedian(candY)
        radius = np.hypot(candX - cx, candY - cy)
        ax.scatter(radius[used], candChi2[used], s=4, alpha=0.3, rasterized=True, label="used")
        ax.scatter(radius[resv], candChi2[resv], s=6, alpha=0.6, rasterized=True, label="reserved")
        if valid.sum() > 0:
            ax.set_ylim(0, np.nanpercentile(candChi2[valid], 99))
        ax.set_xlabel("distance from median position (pixels)")
        ax.set_ylabel("reduced chi2")
        ax.legend(fontsize="small")
        ax.set_title("chi2 vs. detector position")

        ax = fig.add_subplot(gs[0, 2])
        fluxValid = valid & np.isfinite(candFlux) & (candFlux > 0)
        fu, fr = fluxValid & ~candReserved, fluxValid & candReserved
        ax.scatter(candFlux[fu], candChi2[fu], s=4, alpha=0.3, rasterized=True, label="used")
        ax.scatter(candFlux[fr], candChi2[fr], s=6, alpha=0.6, rasterized=True, label="reserved")
        if fluxValid.sum() > 10:
            order = np.argsort(candFlux[fluxValid])
            sortedFlux = candFlux[fluxValid][order]
            sortedChi2 = candChi2[fluxValid][order]
            edges = np.quantile(sortedFlux, np.linspace(0, 1, 11))
            centers, medians = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (sortedFlux >= lo) & (sortedFlux <= hi)
                if sel.sum() > 0:
                    centers.append(np.median(sortedFlux[sel]))
                    medians.append(np.median(sortedChi2[sel]))
            ax.plot(centers, medians, color="k", linewidth=2, label="running median")
        if fluxValid.sum() > 0:
            ax.set_xscale("log")
        if valid.sum() > 0:
            ax.set_ylim(0, np.nanpercentile(candChi2[valid], 99))
        ax.set_xlabel("line flux (instrumental)")
        ax.set_ylabel("reduced chi2")
        ax.legend(fontsize="small")
        ax.set_title("chi2 vs. line flux (brighter-fatter check)")

        ax = fig.add_subplot(gs[1, 0])
        if np.any(np.isfinite(stackedResidual)):
            vmax = np.nanpercentile(np.abs(stackedResidual), 99)
            im = ax.imshow(stackedResidual, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            fig.colorbar(im, ax=ax, label="mean chi (data-model)/sigma", fraction=0.046, pad=0.04)
        ax.set_title("Stacked residual (used candidates, mean chi)")
        ax.set_xlabel("stamp x (pixel)")
        ax.set_ylabel("stamp y (pixel)")

        ax = fig.add_subplot(gs[1, 1:])
        n = len(sampleStamps)
        if n > 0:
            ncols = int(np.ceil(np.sqrt(n)))
            nrows = int(np.ceil(n / ncols))
            size = sampleStamps.shape[1]
            mosaic = np.full((nrows * size, ncols * size), np.nan)
            for i in range(n):
                r, c = divmod(i, ncols)
                mosaic[r * size:(r + 1) * size, c * size:(c + 1) * size] = sampleStamps[i]
            im = ax.imshow(mosaic, cmap="RdBu_r", vmin=-5, vmax=5, origin="lower")
            for i in range(n):
                r, c = divmod(i, ncols)
                if sampleReserved[i]:
                    ax.add_patch(
                        Rectangle(
                            (c * size, r * size), size, size, fill=False, edgecolor="lime", linewidth=1.5
                        )
                    )
            fig.colorbar(im, ax=ax, label="chi", fraction=0.046, pad=0.04)
        ax.set_title(f"Sample residual stamps (chi; green border = held-out, n={n})")
        ax.set_xticks([])
        ax.set_yticks([])

        return fig

    def makeSpatialPlot(self, psf, bbox: geom.Box2I, rng) -> Figure:
        """Build the spatial-variation QA figure

        Parameters
        ----------
        psf : `lsst.afw.detection.Psf`
            Fitted PSF.
        bbox : `lsst.geom.Box2I`
            Detector bounding box to sample across.
        rng : `numpy.random.RandomState`
            Unused (reserved for future use); kept for signature symmetry.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
        """
        width, height = bbox.getWidth(), bbox.getHeight()
        nx, ny = self.config.gridNx, self.config.gridNy
        margin = self.config.gridMargin
        gridXEdges = np.linspace(margin, width - margin, nx)
        gridYEdges = np.linspace(margin, height - margin, ny)

        gridFwhmX = np.full((ny, nx), np.nan)
        gridFwhmY = np.full((ny, nx), np.nan)
        gridE1 = np.full((ny, nx), np.nan)
        gridE2 = np.full((ny, nx), np.nan)
        gridImages = None

        for iy, yy in enumerate(gridYEdges):
            for ix, xx in enumerate(gridXEdges):
                point = geom.Point2D(xx, yy)
                try:
                    shape = psf.computeShape(point)
                    ixx, iyy, ixy = shape.getIxx(), shape.getIyy(), shape.getIxy()
                    gridFwhmX[iy, ix] = 2.3548200450309493 * np.sqrt(ixx)
                    gridFwhmY[iy, ix] = 2.3548200450309493 * np.sqrt(iyy)
                    gridE1[iy, ix] = (ixx - iyy) / (ixx + iyy)
                    gridE2[iy, ix] = 2 * ixy / (ixx + iyy)
                except Exception as exc:
                    self.log.debug("computeShape failed at (%s,%s): %s", xx, yy, exc)
                try:
                    image = psf.computeImage(point)
                    array = image.array
                    if gridImages is None:
                        gridImages = np.full((ny, nx) + array.shape, np.nan)
                    if array.shape == gridImages.shape[2:]:
                        gridImages[iy, ix] = array
                except Exception as exc:
                    self.log.debug("computeImage failed at (%s,%s): %s", xx, yy, exc)

        fig = Figure(figsize=(14, 9), layout="constrained")
        fig.suptitle("FitImagePsfTask QA: spatial variation", fontweight="bold")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])

        ax = fig.add_subplot(gs[:, 0])
        if gridImages is not None:
            stampH, stampW = gridImages.shape[2:]
            mosaic = np.full((ny * stampH, nx * stampW), np.nan)
            for iy in range(ny):
                for ix in range(nx):
                    img = gridImages[iy, ix]
                    if np.all(np.isfinite(img)):
                        peak = np.nanmax(img)
                        if peak > 0:
                            img = img / peak
                    mosaic[(ny - 1 - iy) * stampH:(ny - iy) * stampH, ix * stampW:(ix + 1) * stampW] = img
            im = ax.imshow(mosaic, cmap="viridis", origin="lower")
            fig.colorbar(im, ax=ax, label="normalized amplitude", fraction=0.03, pad=0.02)
        ax.set_title(f"PSF image across the detector ({nx}x{ny} grid, peak-normalized)")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(gs[0, 1])
        fwhm = np.sqrt(gridFwhmX * gridFwhmY)
        im = ax.pcolormesh(gridXEdges, gridYEdges, fwhm, cmap="viridis", shading="nearest")
        fig.colorbar(im, ax=ax, label="FWHM (pixels, geometric mean of x,y)", fraction=0.046, pad=0.04)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        ax.set_title("PSF size across the detector")

        ax = fig.add_subplot(gs[1, 1])
        e = np.hypot(gridE1, gridE2)
        theta = 0.5 * np.arctan2(gridE2, gridE1)
        scale = 2500.0
        xx, yy = np.meshgrid(gridXEdges, gridYEdges)
        dx = scale * e * np.cos(theta)
        dy = scale * e * np.sin(theta)
        ax.quiver(
            xx, yy, dx, dy, e, cmap="magma", angles="xy", scale_units="xy", scale=1,
            headwidth=0, headlength=0, headaxislength=0, pivot="mid",
        )
        sc = ax.scatter(xx, yy, c=e, cmap="magma", s=10)
        fig.colorbar(sc, ax=ax, label="|e| = sqrt(e1^2+e2^2)", fraction=0.046, pad=0.04)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        ax.set_title("PSF ellipticity across the detector (whisker length/color = |e|)")

        return fig
