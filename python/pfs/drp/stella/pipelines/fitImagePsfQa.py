from typing import Iterable, List

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import lsst.geom as geom
import lsst.meas.algorithms as measAlg
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

from ..centroidLines import CentroidLinesTask
from ..datamodel import PfsConfig
from ..DetectorMapContinued import DetectorMap
from ..readLineList import ReadLineListTask
from ..referenceLine import ReferenceLineStatus
from .fitImagePsf import maskNeighboringFibers, selectGoodSources

__all__ = ("FitImagePsfQaTask",)


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
        doc="Automatically set readLineList.exclusionRadius from makePsfCandidates.kernelSize and the "
        "detectorMap's dispersion, matching FitImagePsfTask's own behaviour.",
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

        if self.config.computeExclusionRadius:
            detMap0 = detectorMaps[0]
            midFiber = detMap0.fiberId[len(detMap0.fiberId) // 2]
            dispersion = detMap0.getDispersionAtCenter(midFiber)
            exclusionRadius = self.config.makePsfCandidates.kernelSize * dispersion
            self.log.info("Setting readLineList.exclusionRadius = %.3f nm", exclusionRadius)
            self.readLineList.config.exclusionRadius = exclusionRadius

        allCandidates: List = []
        goodStarCat = SourceCatalog(self.centroidLines.schema)
        nextId = 1
        residuals = []

        for exposure, pfsConfig, detectorMap in zip(exposures, pfsConfigs, detectorMaps):
            refLines = self.readLineList.run(detectorMap, exposure.getMetadata())
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
            except Exception:
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
