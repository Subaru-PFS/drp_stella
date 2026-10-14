from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from lsst.daf.base import PropertyList
from lsst.pex.config import ConfigurableField, DictField, Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection

from ..datamodel import PfsArm
from ..DetectorMapContinued import DetectorMap
from ..fitLsfLines import centroidZeroPoint, findBroadLines, measureLsfMany, pinnedCentres
from ..gen3 import readDatasetRefs
from ..lsfLine import LsfLineFlag, LsfLineSet
from ..lsfModel import ArmLsfModel, CUMULANT_NAMES, FiberSpline
from ..readLineList import ReadLineListTask
from ..referenceLine import ReferenceLineStatus
from ..smoothSurface import DetectorCoords, fitSmoothSurface

__all__ = ("FitLsfConfig", "FitLsfTask")


class FitLsfConnections(PipelineTaskConnections, dimensions=("instrument", "arm", "spectrograph")):
    """Connections for FitLsfTask"""

    pfsArm = InputConnection(
        name="pfsArm",
        doc="Extracted arc spectra",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    detectorMap = InputConnection(
        name="detectorMap",
        doc="Mapping between fiberId,wavelength and x,y used to extract the pfsArm",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    metadata = InputConnection(
        name="raw.metadata",
        doc="Exposure header, for the lamps",
        storageClass="PropertyList",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    lsfModel = OutputConnection(
        name="lsfModel",
        doc="Model of the line-spread function over the detector",
        storageClass="ArmLsfModel",
        dimensions=("instrument", "arm", "spectrograph"),
    )
    lsfLines = OutputConnection(
        name="lsfLines",
        doc="Per-line fits of the line-spread function",
        storageClass="LsfLineSet",
        dimensions=("instrument", "arm", "spectrograph"),
    )


class FitLsfConfig(PipelineTaskConfig, pipelineConnections=FitLsfConnections):
    """Configuration for FitLsfTask"""

    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read the arc line lists")
    # --- line selection
    minFrac = Field(dtype=float, default=0.02,
                    doc="Ignore lines fainter than this fraction of the lamp's brightest line in range")
    minThroughput = Field(dtype=float, default=0.3,
                          doc="Ignore lines where the quartz throughput (pfsArm.norm) < this x its median "
                              "(the dichroic roll-off)")
    isolation = Field(dtype=float, default=12.0,
                      doc="Reject a line with a catalogue neighbour within this many pixels brighter "
                          "than blendFrac x its intensity")
    blendFrac = Field(dtype=float, default=0.05, doc="See isolation")
    coreSep = Field(dtype=float, default=4.0,
                    doc="Reject a line with a catalogue neighbour within this many pixels brighter "
                        "than coreFrac x its intensity")
    coreFrac = Field(dtype=float, default=0.02, doc="See coreSep")
    edge = Field(dtype=int, default=25,
                 doc="Ignore lines within this many pixels of the ends of the spectrum")
    excludeStatus = ListField(dtype=str, default=["BLEND"],
                              doc="Don't fit lines with these catalogue status bits (they are still fit as "
                                  "companions of other lines)")
    fiberStep = Field(dtype=int, default=1, doc="Fit every fiberStep-th fibre")
    # --- per-line fits
    blurOrder = Field(dtype=int, default=4,
                      doc="Edgeworth order of the blur kernel: 2 is a Gaussian; 4 adds skewness and kurtosis")
    extraction = Field(dtype=str, default="optimal",
                       doc="How the spectra were extracted ('optimal' or 'boxcar'): changes the disk's "
                           "profile")
    fitLorentz = Field(dtype=bool, default=False,
                       doc="Also convolve the profile with a Lorentzian of free FWHM (an extended wing)? "
                           "Off by default: the wing may be residual scattered light (ScatteredLightTask)")
    halfWindow = Field(dtype=int, default=12, doc="Fit pixels within this distance of each line")
    coreHalf = Field(dtype=float, default=1.5,
                     doc="Masked pixels within this distance of a line's centre count against maxBadCore")
    maxBadCore = Field(dtype=int, default=0,
                       doc="A line with more masked pixels than this in its core is saturated: not fit")
    fitCompanions = Field(dtype=bool, default=True,
                          doc="Fit catalogue lines in each line's window at the same time (same profile)?")
    companionFrac = Field(dtype=float, default=0.002,
                          doc="Only fit companions brighter than this fraction of the line")
    satDropFrac = Field(dtype=float, default=0.05,
                        doc="Don't use a line in any fibre if it is saturated in more than this fraction "
                            "of them")
    # --- broad lines: intrinsically broader than the LSF (e.g. self-absorbed in the lamp)
    doFindBroad = Field(dtype=bool, default=True,
                        doc="Find and exclude lines broader than their neighbours "
                            "(see fitLsfLines.findBroadLines)")
    broadFrac = Field(dtype=float, default=0.025, doc="Fractional rms-width excess for a broad line")
    broadWindow = Field(dtype=float, default=50.0, doc="Compare widths with neighbours within this many nm")
    broadNSigma = Field(dtype=float, default=3.0, doc="... and the excess must be this significant")
    excludeCatalogueBroad = Field(dtype=bool, default=True,
                                  doc="Also drop lines flagged BROAD in the catalogue? They are still fit, "
                                      "so that disagreements with our measurement are reported (so don't "
                                      "put BROAD in excludeStatus, which drops lines before they are fit)")
    # --- centroids
    doPinCentroids = Field(dtype=bool, default=True,
                           doc="Refit every line with its centre held at the free fit's plus the zero point "
                               "(the clipped mean of data - model first moments), so that the models' first "
                               "moments match the data's on average?")
    centroidHalf = Field(dtype=float, default=4.0,
                         doc="Half-width (pixels) of the window for the data and model first moments")
    centroidNSigma = Field(dtype=float, default=3.0, doc="Clipping threshold for the centroid zero point")
    # --- smooth fits over the detector
    degrees = DictField(keytype=str, itemtype=str,
                        default=dict(centroid="2,3", radius="2,3", blur="3,4", S="2,3", K="2,2",
                                     lam5="1,1", lam6="1,1", lorentz="2,3"),
                        doc="Chebyshev degree 'dx,dy' of each parameter's smooth fit over the detector")
    groupBy = DictField(keytype=str, itemtype=str, default=dict(centroid="visit,fiberId", default="visit"),
                        doc="Factors with constant offsets in each parameter's smooth fit ('default' for "
                            "parameters not listed; '' for none): per visit (focus, brightness), and per "
                            "fibre in the centroid (a wavelength zero-point error)")
    nIter = Field(dtype=int, default=3, doc="Clip-and-refit iterations of the smooth fits")
    nSigma = Field(dtype=float, default=4.0, doc="Clipping threshold (robust sigma) of the smooth fits")
    # --- model
    traceStep = Field(dtype=int, default=32,
                      doc="Spacing (rows) of the spline knots of each fibre's x(row) and wavelength(row) "
                          "in the model")
    # --- execution
    nWorkers = Field(dtype=int, default=1, doc="Worker processes for the line fits (1: serial)")

    def validate(self):
        super().validate()
        if self.blurOrder != 2 and not 3 <= self.blurOrder <= 6:
            raise ValueError(f"blurOrder must be 2 or in 3..6, not {self.blurOrder}")
        for name in self.excludeStatus:
            if not hasattr(ReferenceLineStatus, name):
                raise ValueError(f"Unknown ReferenceLineStatus {name!r} in excludeStatus")


class FitLsfTask(PipelineTask):
    """Fit a model of the line-spread function to arc spectra of one detector

    Every bright, isolated arc line in every fibre of each arc exposure is fit
    with a disk (x) Edgeworth LSF (`pfs.drp.stella.fitLsfLines`). Lines
    saturated in many fibres, with an excluded catalogue status, or broader
    than their neighbours are dropped. The lines' centres are then pinned so
    that the models' first moments match the data's on average, and each
    parameter is fit as a smooth function of position on the detector, giving
    an `ArmLsfModel`; the per-line fits are written as an `LsfLineSet`.
    """

    ConfigClass = FitLsfConfig
    _DefaultName = "fitLsf"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        """Entry point with butler I/O

        Parameters
        ----------
        butler : `QuantumContext`
            Data butler, specialised to operate in the context of a quantum.
        inputRefs : `InputQuantizedConnection`
            Container with attributes that are data references for the various
            input connections.
        outputRefs : `OutputQuantizedConnection`
            Container with attributes that are data references for the various
            output connections.
        """
        arm = inputRefs.pfsArm[0].dataId.arm.name
        spectrograph = inputRefs.pfsArm[0].dataId.spectrograph.num
        data = readDatasetRefs(butler, inputRefs, "pfsArm", "detectorMap", "metadata")
        visits = [ref.dataId["visit"] for ref in inputRefs.pfsArm]
        outputs = self.run(arm, spectrograph, visits, data.pfsArm, data.detectorMap, data.metadata)
        butler.put(outputs, outputRefs)

    def run(
        self,
        arm: str,
        spectrograph: int,
        visits: List[int],
        pfsArmList: List[PfsArm],
        detectorMapList: List[DetectorMap],
        metadataList: List[PropertyList],
    ) -> Struct:
        """Fit the LSF model

        Parameters
        ----------
        arm : `str`
            Arm (``b``, ``r``, ``m``, ``n``).
        spectrograph : `int`
            Spectrograph module.
        visits : `list` of `int`
            Visit of each exposure.
        pfsArmList : `list` of `PfsArm`
            Extracted arc spectra of each exposure.
        detectorMapList : `list` of `DetectorMap`
            Mapping of fiberId,wavelength to x,y used for each exposure.
        metadataList : `list` of `PropertyList`
            Header of each exposure (for the lamps).

        Returns
        -------
        lsfModel : `ArmLsfModel`
            Model of the LSF.
        lsfLines : `LsfLineSet`
            Per-line fits.
        """
        exposures = []
        for visit, pfsArm, detMap, metadata in zip(visits, pfsArmList, detectorMapList, metadataList):
            lines = self.readLineList.run(detectorMap=detMap, metadata=metadata)
            lamps = "+".join(sorted(self.readLineList.getLampInfo(metadata).lamps))
            exposures.append(Struct(visit=int(visit), pfsArm=pfsArm, detMap=detMap, lines=lines, lamps=lamps))
            self.log.info("Visit %d (%s): %d catalogue lines", visit, lamps, len(lines))
        exposures.sort(key=lambda ee: ee.visit)

        # Free fits
        free = self.fitLines(exposures)
        allFree = [ff for ffs in free.fits.values() for ff in ffs]
        self.log.info("Free fits: %d lines in %d spectra", len(allFree), len(free.fits))

        # Lines to drop everywhere: saturated in too many fibres; excluded status; broad
        nFit = Counter(self.lineKey(ff) for ff in allFree)
        nSat = Counter((lamps, round(lam, 3)) for keys in free.saturated.values() for lamps, lam in keys)
        dropSat = {key for key, nn in nSat.items() if nn > self.config.satDropFrac*(nn + nFit.get(key, 0))}
        excludeMask = 0
        for name in self.config.excludeStatus:
            excludeMask |= int(getattr(ReferenceLineStatus, name))
        dropStatus = {self.lineKey(ff) for ff in allFree if ff.status & excludeMask}
        broad = set()
        status = {self.lineKey(ff): ff.status for ff in allFree}
        if self.config.excludeCatalogueBroad:
            broad |= {key for key, st in status.items() if st & int(ReferenceLineStatus.BROAD)}
        if self.config.doFindBroad:
            excess, measuredBroad = findBroadLines(
                [ff for ff in allFree if self.lineKey(ff) not in dropSat | dropStatus],
                window=self.config.broadWindow, broadFrac=self.config.broadFrac,
                nSigma=self.config.broadNSigma,
            )
            self.checkBroad(excess, measuredBroad, status)
            broad |= measuredBroad
        for key in sorted(dropSat, key=lambda kk: kk[1]):
            self.log.info("Line %s %.3f is saturated in %d fibres (fit in %d): not used",
                          key[0], key[1], nSat[key], nFit.get(key, 0))
        dropped = dropSat | dropStatus | broad

        def isUsed(ff):
            return self.lineKey(ff) not in dropped

        # Centroid zero point, and pinned fits
        zeroPoint, zeroPointErr, scatter, used = centroidZeroPoint(
            [ff for ff in allFree if isUsed(ff)], nSigma=self.config.centroidNSigma
        )
        self.log.info("Centroid zero point %+.5f +/- %.5f pixels (clipped mean of %d lines; scatter %.4f)",
                      zeroPoint, zeroPointErr, used.sum(), scatter)
        fits = free.fits
        if self.config.doPinCentroids:
            fixCentres = {key: pinnedCentres(ffs, zeroPoint) for key, ffs in free.fits.items()}
            pinned = self.fitLines(exposures, fixCentres)
            fits = pinned.fits
            freeChi2 = {(key, round(ff.catWavelength, 4)): ff.chi2
                        for key, ffs in free.fits.items() for ff in ffs}
            dchi2 = [ff.chi2 - freeChi2[(key, round(ff.catWavelength, 4))] for key, ffs in fits.items()
                     for ff in ffs if (key, round(ff.catWavelength, 4)) in freeChi2]
            self.log.info("Pinned fits: %d lines; median chi^2 change %+.3f",
                          sum(len(ff) for ff in fits.values()), np.median(dchi2) if dchi2 else np.nan)

        # Smooth fits over the detector
        detMap = exposures[0].detMap
        coords = DetectorCoords.fromDetectorMap(detMap)
        self.log.info("Detector coordinates: %s", coords)
        rows = []
        for (visit, fiberId), ffs in fits.items():
            rows += [(visit, ff) for ff in ffs]
        usedRows = [ii for ii, (_, ff) in enumerate(rows) if isUsed(ff)]
        surfaces, rejected = self.fitSurfaces([rows[ii] for ii in usedRows], coords, detMap)
        rejectedFlags = defaultdict(int)
        for jj, flags in rejected.items():
            rejectedFlags[usedRows[jj]] = flags

        lsfModel = self.makeModel(arm, spectrograph, exposures, surfaces, coords, zeroPoint)
        lsfModel.metadata.update(zeroPointErr=zeroPointErr, zeroPointScatter=scatter)

        flags = {}
        for ii, (visit, ff) in enumerate(rows):
            flag = rejectedFlags[ii]
            key = self.lineKey(ff)
            if key in dropSat:
                flag |= LsfLineFlag.SATURATED_ELSEWHERE
            if key in dropStatus:
                flag |= LsfLineFlag.EXCLUDED_STATUS
            if key in broad:
                flag |= LsfLineFlag.BROAD
            if ff.pinned:
                flag |= LsfLineFlag.PINNED
            flags[ii] = flag
        lsfLines = self.makeLineSet(exposures, rows, flags, free.saturated)
        return Struct(lsfModel=lsfModel, lsfLines=lsfLines)

    def checkBroad(self, excess, broad, status):
        """Report the broad lines, and disagreements with the catalogue's BROAD flags

        Parameters
        ----------
        excess : `dict` [(`str`, `float`), (`float`, `float`, `int`)]
            Median fractional rms-width excess of each line, its error and the
            number of fibres (from `findBroadLines`).
        broad : `set` [(`str`, `float`)]
            Keys of the lines found to be broad.
        status : `dict` [(`str`, `float`), `int`]
            Catalogue status of each line that was fit.
        """
        flagBroad = int(ReferenceLineStatus.BROAD)
        flagBlend = int(ReferenceLineStatus.BLEND)
        use = "not used" if self.config.excludeCatalogueBroad else "used"
        for key in sorted(set(status) | broad, key=lambda kk: kk[1]):
            flagged = bool(status.get(key, 0) & flagBroad)
            if key in broad:
                med, err, num = excess[key]
                if flagged:
                    self.log.info("Broad line %s %.3f: rms width excess %+.3f +/- %.3f (%d fibres)",
                                  key[0], key[1], med, err, num)
                else:
                    self.log.warning("Broad line %s %.3f: rms width excess %+.3f +/- %.3f (%d fibres), "
                                     "but not flagged BROAD%s in the catalogue", key[0], key[1], med, err,
                                     num, " (it is flagged BLEND)" if status.get(key, 0) & flagBlend else "")
            elif flagged:
                if key in excess:
                    med, err, num = excess[key]
                    self.log.warning("Line %s %.3f is flagged BROAD in the catalogue, but its rms width "
                                     "excess is only %+.3f +/- %.3f (%d fibres; threshold %.3f): %s",
                                     key[0], key[1], med, err, num, self.config.broadFrac, use)
                else:
                    self.log.info("Line %s %.3f is flagged BROAD in the catalogue, but is in too few "
                                  "fibres to test: %s", key[0], key[1], use)

    @staticmethod
    def lineKey(fit):
        """Key identifying a catalogue line: (lamps, wavelength)"""
        return (fit.element, round(fit.catWavelength, 3))

    def fitLines(self, exposures, fixCentres=None):
        """Fit the lines in every fibre of every exposure

        Parameters
        ----------
        exposures : `list` of `Struct`
            Exposures, with ``visit``, ``pfsArm``, ``detMap``, ``lines`` and
            ``lamps``.
        fixCentres : `dict` [(`int`, `int`), `dict`], optional
            ``(visit, fiberId) -> fixCentres`` (see `fitLsfLines.measureLsf`):
            hold the lines' centres fixed.

        Returns
        -------
        fits : `dict` [(`int`, `int`), `list` of `DiskLineFit`]
            Successful fits for each ``(visit, fiberId)``, with ``x``, ``y`` set.
        saturated : `dict` [(`int`, `int`), `list` of (`str`, `float`)]
            Lamps and catalogue wavelengths of the saturated lines.
        """
        config = self.config
        excludeMask = 0
        for name in config.excludeStatus:
            excludeMask |= int(getattr(ReferenceLineStatus, name))
        keys = []
        tasks = []
        for exp in exposures:
            pfsArm = exp.pfsArm
            for index in range(0, len(pfsArm), config.fiberStep):
                fiberId = int(pfsArm.fiberId[index])
                key = (exp.visit, fiberId)
                if fixCentres is not None and not fixCentres.get(key):
                    continue
                flux = pfsArm.flux[index].astype(float)
                flux[pfsArm.mask[index] != 0] = np.nan
                kwargs = dict(
                    element=exp.lamps,
                    fiberId=fiberId,
                    blurOrder=config.blurOrder,
                    extraction=config.extraction,
                    halfWindow=config.halfWindow,
                    coreHalf=config.coreHalf,
                    maxBadCore=config.maxBadCore,
                    fitCompanions=config.fitCompanions,
                    fitLorentz=config.fitLorentz,
                    companionFrac=config.companionFrac,
                    fixCentres=None if fixCentres is None else fixCentres[key],
                    centroidHalf=config.centroidHalf,
                    keepPixels=False,
                    isolation=config.isolation,
                    blendFrac=config.blendFrac,
                    coreSep=config.coreSep,
                    coreFrac=config.coreFrac,
                    minFrac=config.minFrac,
                    norm=pfsArm.norm[index].astype(float),
                    minThroughput=config.minThroughput,
                    edge=config.edge,
                    excludeMask=excludeMask if fixCentres is None else 0,
                )
                keys.append(key)
                tasks.append((pfsArm.wavelength[index].astype(float), flux,
                              pfsArm.variance[index].astype(float), exp.lines, kwargs))
        results = measureLsfMany(tasks, nWorkers=config.nWorkers)

        detMaps = {exp.visit: exp.detMap for exp in exposures}
        fits = {}
        saturated = {}
        for key, (ffs, sat) in zip(keys, results):
            if ffs:
                xy = detMaps[key[0]].findPoint(key[1], np.array([ff.wavelength for ff in ffs], dtype=float))
                for ff, (xx, yy) in zip(ffs, np.asarray(xy).reshape(-1, 2)):
                    ff.x, ff.y = float(xx), float(yy)
            fits[key] = ffs
            lamps = next(exp.lamps for exp in exposures if exp.visit == key[0])
            saturated[key] = [(lamps, lam) for lam in sat]
        return Struct(fits=fits, saturated=saturated)

    def fitSurfaces(self, rows, coords, detMap):
        """Fit each parameter as a smooth function over the detector

        Parameters
        ----------
        rows : `list` of (`int`, `DiskLineFit`)
            Visit and fit of each line to use.
        coords : `DetectorCoords`
            Detector pixels to the frame of the fits.
        detMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y (for the detector's extent).

        Returns
        -------
        surfaces : `dict` [`str`, `SmoothSurface`]
            Fitted surfaces.
        rejected : `dict` [`int`, `int`]
            ``LsfLineFlag.REJECTED_*`` flags of the lines (indices into
            ``rows``) clipped by the fits.
        """
        bbox = detMap.getBBox()
        xc, yc = np.meshgrid([bbox.getMinX(), bbox.getMaxX()], [bbox.getMinY(), bbox.getMaxY()])
        uc, vc = coords(xc.ravel(), yc.ravel())
        domain = ((uc.min(), uc.max()), (vc.min(), vc.max()))

        fits = [ff for _, ff in rows]
        u, v = coords(np.array([ff.x for ff in fits]), np.array([ff.y for ff in fits]))
        labels = dict(visit=np.array([str(visit) for visit, _ in rows]),
                      fiberId=np.array([str(ff.fiberId) for ff in fits]))
        params = dict(
            centroid=([ff.pixel - ff.catPixel for ff in fits], [ff.pixelErr for ff in fits]),
            radius=([ff.radius for ff in fits], [ff.radiusErr for ff in fits]),
            blur=([ff.blur for ff in fits], [ff.blurErr for ff in fits]),
        )
        for jj, name in enumerate(CUMULANT_NAMES[:max(0, self.config.blurOrder - 2)]):
            params[name] = ([ff.cumulants[jj] for ff in fits], [ff.cumulantsErr[jj] for ff in fits])
        if self.config.fitLorentz:
            params["lorentz"] = ([ff.lorentz for ff in fits], [ff.lorentzErr for ff in fits])

        surfaces = {}
        rejected = defaultdict(int)
        for name, (value, error) in params.items():
            if name not in self.config.degrees:
                continue
            degree = tuple(int(dd) for dd in self.config.degrees[name].split(","))
            factors = self.config.groupBy.get(name, self.config.groupBy.get("default", ""))
            groups = {ff: labels[ff] for ff in factors.split(",") if ff}
            surface, used, residual = fitSmoothSurface(
                u, v, value, error, degree, domain=domain, name=name, nIter=self.config.nIter,
                nSigma=self.config.nSigma, groups=groups,
            )
            surfaces[name] = surface
            for jj in np.flatnonzero(~used):
                rejected[int(jj)] |= LsfLineFlag.rejected(name)
            self.log.info("Smooth fit of %s: degree %s, %d/%d lines used, rms %.4f, chi^2/dof %.1f", name,
                          degree, surface.nUsed, len(fits), surface.rms, surface.chi2/max(1, surface.ndof))
        return surfaces, rejected

    def makeModel(self, arm, spectrograph, exposures, surfaces, coords, zeroPoint):
        """Construct the LSF model

        Parameters
        ----------
        arm : `str`
            Arm.
        spectrograph : `int`
            Spectrograph module.
        exposures : `list` of `Struct`
            Exposures; the first one's detectorMap provides each fibre's x(row)
            and wavelength(row).
        surfaces : `dict` [`str`, `SmoothSurface`]
            Fitted surfaces.
        coords : `DetectorCoords`
            Detector pixels to the frame of the surfaces.
        zeroPoint : `float`
            Centroid zero point (pixels).

        Returns
        -------
        lsfModel : `ArmLsfModel`
            Model of the LSF.
        """
        detMap = exposures[0].detMap
        fiberId = np.array(sorted(set(int(ff) for exp in exposures for ff in exp.pfsArm.fiberId)), dtype=int)
        length = exposures[0].pfsArm.wavelength.shape[1]
        xCenter = FiberSpline.fromFunction(fiberId, length, self.config.traceStep, detMap.getXCenter)
        wavelength = FiberSpline.fromFunction(fiberId, length, self.config.traceStep, detMap.findWavelength)
        metadata: Dict = dict(visits=[exp.visit for exp in exposures], lamps=[exp.lamps for exp in exposures],
                              config=self.config.toDict())
        return ArmLsfModel(arm, spectrograph, fiberId, length, surfaces, coords, xCenter, wavelength,
                           extraction=self.config.extraction, centroidZeroPoint=zeroPoint, metadata=metadata)

    def makeLineSet(self, exposures, rows, flags, saturated):
        """Construct the table of per-line fits

        Parameters
        ----------
        exposures : `list` of `Struct`
            Exposures.
        rows : `list` of (`int`, `DiskLineFit`)
            Visit and fit of each line.
        flags : `dict` [`int`, `int`]
            ``LsfLineFlag`` of each of ``rows``.
        saturated : `dict` [(`int`, `int`), `list` of (`str`, `float`)]
            Saturated lines in each ``(visit, fiberId)``.

        Returns
        -------
        lsfLines : `LsfLineSet`
            Per-line fits.
        """
        catalogue = {}
        for exp in exposures:
            data = exp.lines.data
            for wl, desc, trans, status in zip(data.wavelength, data.description, data.transition,
                                               data.status):
                catalogue[(exp.visit, round(float(wl), 3))] = (str(desc), str(trans), int(status))
        nCum = max(0, self.config.blurOrder - 2)
        columns = defaultdict(list)
        identity = ("visit", "fiberId", "wavelength", "status", "description", "transition", "flag")

        def add(visit, fiberId, wavelength, flag, **values):
            desc, trans, status = catalogue.get((visit, round(wavelength, 3)), ("UNKNOWN", "UNKNOWN", -1))
            columns["visit"].append(visit)
            columns["fiberId"].append(fiberId)
            columns["wavelength"].append(wavelength)
            columns["status"].append(status)
            columns["description"].append(desc)
            columns["transition"].append(trans)
            columns["flag"].append(int(flag))
            for name, column in LsfLineSet.DamdClass.getSchemaDict().items():
                if name not in identity:
                    columns[name].append(values.get(name, column.default))

        for ii, (visit, ff) in enumerate(rows):
            cumulants = {}
            for jj, name in enumerate(CUMULANT_NAMES[:nCum]):
                cumulants[name] = ff.cumulants[jj]
                cumulants[name + "Err"] = ff.cumulantsErr[jj]
            if self.config.fitLorentz:
                cumulants.update(lorentz=ff.lorentz, lorentzErr=ff.lorentzErr)
            add(visit, ff.fiberId, ff.catWavelength, flags[ii], x=ff.x, y=ff.y, pixel=ff.pixel,
                pixelErr=ff.pixelErr, refPixel=ff.catPixel, dispersion=ff.dispersion,
                dataCentroid=ff.dataCentroid, modelCentroid=ff.modelCentroid, radius=ff.radius,
                radiusErr=ff.radiusErr, blur=ff.blur, blurErr=ff.blurErr, flux=ff.amplitude,
                fluxErr=ff.amplitudeErr, peak=ff.peak, chi2=ff.chi2, ndof=ff.ndof, nBadCore=ff.nBadCore,
                **cumulants)
        for (visit, fiberId), keys in saturated.items():
            for _, wavelength in keys:
                add(visit, fiberId, wavelength, LsfLineFlag.SATURATED)

        dtypes = {name: column.dtype for name, column in LsfLineSet.DamdClass.getSchemaDict().items()}
        return LsfLineSet.fromColumns(**{name: np.array(values, dtype=dtypes[name] if dtypes[name] is not str
                                                        else object)
                                         for name, values in columns.items()})
