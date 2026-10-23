from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Dict, List

import numpy as np

from lsst.daf.base import PropertyList
from lsst.pex.config import ConfigField, ConfigurableField, DictField, Field, ListField
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
from ..lsfModel import ArmLsfModel, CUMULANT_NAMES, FiberSpline, addBlur
from ..readLineList import ReadLineListTask
from ..referenceLine import ReferenceLineStatus
from ..scatteredLight import ScatteredLightConfig
from ..smoothSurface import DetectorCoords, SmoothSurface, fitSmoothSurface, maxGap, supportedDegree

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
    pfiRadius = Field(dtype=float, default=None, optional=True,
                      doc="Hold the science (PFI) fibres' disk radius at this value (pixels; the image of "
                          "the fibre core) rather than fitting it, so that defocus goes into the blur kernel "
                          "(rms, cumulants), which the IIS fibres share. None: fit it")
    doPfiRadiusModel = Field(dtype=bool, default=True,
                             doc="Hold the science (PFI) fibres' disk radius at the optical model's image of "
                                 "the fibre core (pfiRadiusModel), rather than fitting it? Overrides "
                                 "pfiRadius")
    pfiRadiusModel = DictField(
        keytype=str, itemtype=str,
        default=dict(b="27.985,0.418,0.226,0.191,0.016,0.230",
                     r="27.990,0.440,0.308,0.248,0.022,0.224",
                     m="27.990,0.440,0.308,0.248,0.022,0.224",
                     n="28.170,0.598,0.317,0.250,0.028,0.212"),
        doc="Per-arm model of the radius of the image of a science fibre's core on the detector (microns): "
            "coefficients c00,c01,c02,c10,c11,c12 of R = sum_ij c_ij u^(2i) v^j with u = x/30mm along the "
            "slit and v = y/30mm in dispersion (increasing with wavelength), measured from the detector "
            "centre. From Jim Gunn's ray traces (fiberspots.csv) at nominal focus, which fit to 0.06% rms; "
            "the m arm uses the r camera")
    pfiRadiusModelScale = Field(dtype=float, default=1.0,
                                doc="Multiply pfiRadiusModel by this factor (e.g. for a different focus)")
    pixelSize = Field(dtype=float, default=15.0, doc="Detector pixel size (microns), for pfiRadiusModel")
    fitHalo = Field(dtype=bool, default=True,
                    doc="Add a halo to the profile: a fraction haloFrac of the light in the disk blurred by "
                        "a broad Gaussian of rms haloWidth (the light within a few pixels that the "
                        "scattered-light correction leaves behind; so the model depends on scatteredLight)")
    haloMinSN = Field(dtype=float, default=300.0,
                      doc="The halo is measured from lines with at least this median S/N, and then held "
                          "fixed (at its smooth fit) in the second fits of all the lines")
    doIisRadius = Field(dtype=bool, default=True,
                        doc="Measure the engineering (IIS) fibres' disk radius from any IIS arcs (lamps "
                            "'*_eng') among the inputs, with the science fibres' blur kernel held fixed?")
    iisRadiusRatio = Field(dtype=float, default=0.51, optional=True,
                           doc="Ratio of the engineering (IIS) fibres' disk radius to the science (PFI) "
                               "fibres': a property of the fibres, so the same for every detector. The "
                               "default is the ratio of the cores' diameters (65um, FBP065162.5190; "
                               "127.5um). None: use each detector's measured ratio (iisRadiusScaleDegree)")
    iisRadiusScaleDegree = Field(dtype=str, default="0,0",
                                 doc="Chebyshev degree 'dx,dy' of the measured IIS/PFI disk-radius ratio "
                                     "over the detector ('0,0': a constant, the median); used if "
                                     "iisRadiusRatio is None")
    doIisHalo = Field(dtype=bool, default=False,
                      doc="Measure the engineering (IIS) fibres' halo fraction relative to the science "
                          "fibres' (from the IIS lines with median S/N >= haloMinSN, with the PFI core held "
                          "fixed)? If not (or iisHaloScale), their halo is the PFI fibres'. Off: the few "
                          "bright IIS lines are saturated, or broadened by the extra IIS blur")
    iisSatDropFrac = Field(dtype=float, default=0.0,
                           doc="Don't use an IIS line if it is saturated in more than this fraction of the "
                               "engineering fibres: their images are narrow, so the fibres in which a line "
                               "isn't flagged as saturated are close to full scale")
    doIisExtraBlur = Field(dtype=bool, default=True,
                           doc="Measure the extra blur of the engineering (IIS) fibres' images (a variance "
                               "and a fourth cumulant, convolved with the PFI kernel)? Their images differ "
                               "from the PFI kernel convolved with their (smaller) disk: on b they are wider "
                               "(by ~0.35 px rms), with stronger wings; on n slightly narrower")
    iisExtraBlur = DictField(keytype=str, itemtype=str,
                             default=dict(b="0.04285,0.07637", r="0.03732,0.07198", m="0.00615,-0.02752",
                                          n="-0.02936,-0.03583"),
                             doc="Per-arm extra blur of the engineering fibres, 'variance,kappa4' "
                                 "(pixels^2, may be negative; pixels^4), measured from all the spectrographs "
                                 "of an arm (the per-line values: metadata iisExtraBlur). Arms not "
                                 "listed use each detector's own measurement (if doIisExtraBlur). The "
                                 "defaults are the medians over spectrographs 1-4 of the 2026-08/09 IIS arcs")
    iisHaloScaleDegree = Field(dtype=str, default="0,0",
                               doc="Chebyshev degree 'dx,dy' of the IIS/PFI halo-fraction ratio ('0,0': a "
                                   "constant)")
    iisHaloScale = DictField(keytype=str, itemtype=float, default=dict(b=2.0),
                             doc="Per-arm ratio of the engineering (IIS) fibres' halo fraction to the "
                                 "science (PFI) fibres' (multiplying the PFI haloFrac surface). A single "
                                 "detector has too few bright IIS lines to measure it, so it is set per arm: "
                                 "b=2 minimises the IIS lines' chi^2 and flux errors on b1-4 (r, m and n are "
                                 "best with the PFI halo). Arms not listed use the PFI halo (or, if "
                                 "doIisHalo, each detector's own measurement)")
    scatteredLight = ConfigField(dtype=ScatteredLightConfig,
                                 doc="The scattered-light correction applied to the input pfsArms (by "
                                     "reduceExposure, and to the fiberProfiles by buildFiberProfiles), which "
                                     "the profile's wings depend on: recorded in the model, not applied")
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
    minLineSN = DictField(keytype=str, itemtype=float, default=dict(b=25.0, default=50.0),
                          doc="Don't use a line in any fibre if its median S/N (flux/fluxErr) over the "
                              "fibres is below this (a per-line cut, so lines aren't selected by fibre "
                              "throughput); per arm, or 'default'. Lower for b, where the lamps are faint")
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
                        default=dict(centroid="2,3", radius="2,3", rms="3,4", S="2,3", K="2,2",
                                     lam5="1,1", lam6="1,1", lorentz="2,3", haloFrac="1,2", haloWidth="1,2"),
                        doc="Chebyshev degree 'dx,dy' of each parameter's smooth fit over the detector")
    groupBy = DictField(keytype=str, itemtype=str, default=dict(centroid="visit,fiberId", default="visit"),
                        doc="Factors with constant offsets in each parameter's smooth fit ('default' for "
                            "parameters not listed; '' for none): per visit (focus, brightness), and per "
                            "fibre in the centroid (a wavelength zero-point error)")
    nIter = Field(dtype=int, default=3, doc="Clip-and-refit iterations of the smooth fits")
    nSigma = Field(dtype=float, default=4.0, doc="Clipping threshold (robust sigma) of the smooth fits")
    doCoverageGuard = Field(dtype=bool, default=True,
                            doc="Lower a smooth fit's degree in x or y where the lines leave gaps longer "
                                "than coverageFactor x (detector size)/(degree + 1)? (A polynomial wanders, "
                                "unconstrained, across a gap that is long compared to its scale)")
    coverageFactor = Field(dtype=float, default=1.0, doc="Tolerance of the coverage guard; larger is laxer")
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
        visits = [pfsArm.identity.visit for pfsArm in data.pfsArm]   # readDatasetRefs consumes inputRefs
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
        # IIS arcs light only the engineering fibres; they measure only those fibres' disk radius
        isIis = [all(ll.endswith("_eng") for ll in exp.lamps.split("+")) for exp in exposures]
        iisExposures = [exp for exp, iis in zip(exposures, isIis) if iis]
        allExposures = exposures
        exposures = [exp for exp, iis in zip(exposures, isIis) if not iis]
        if not exposures:
            raise RuntimeError("No science-fibre (PFI) arcs among the inputs")
        if iisExposures:
            self.log.info("IIS (engineering-fibre) arcs: %s",
                          [(exp.visit, exp.lamps) for exp in iisExposures])

        # Free fits
        free = self.fitLines(exposures, fixRadius=self.pfiRadiusFixed(arm))
        allFree = [ff for ffs in free.fits.values() for ff in ffs]
        self.log.info("Free fits: %d lines in %d spectra", len(allFree), len(free.fits))

        # Lines to drop everywhere: saturated in too many fibres; excluded status; faint; broad
        nFit = Counter(self.lineKey(ff) for ff in allFree)
        nSat = Counter((lamps, round(lam, 3)) for keys in free.saturated.values() for lamps, lam in keys)
        dropSat = {key for key, nn in nSat.items() if nn > self.config.satDropFrac*(nn + nFit.get(key, 0))}
        excludeMask = 0
        for name in self.config.excludeStatus:
            excludeMask |= int(getattr(ReferenceLineStatus, name))
        dropStatus = {self.lineKey(ff) for ff in allFree if ff.status & excludeMask}
        signalToNoise = defaultdict(list)
        for ff in allFree:
            signalToNoise[self.lineKey(ff)].append(ff.amplitude/ff.amplitudeErr)
        medianSN = {key: float(np.nanmedian(sn)) for key, sn in signalToNoise.items()}
        minLineSN = self.config.minLineSN.get(arm, self.config.minLineSN["default"])
        dropLowSN = {key for key, sn in medianSN.items() if not sn >= minLineSN}
        if dropLowSN:
            lowSN = ", ".join(f"{key[0]} {key[1]:.3f} ({medianSN[key]:.0f})"
                              for key in sorted(dropLowSN, key=lambda kk: kk[1]))
            self.log.info("%d of %d lines have median S/N < %g: not used (%s)", len(dropLowSN), len(medianSN),
                          minLineSN, lowSN)
        broad = set()
        status = {self.lineKey(ff): ff.status for ff in allFree}
        if self.config.excludeCatalogueBroad:
            broad |= {key for key, st in status.items() if st & int(ReferenceLineStatus.BROAD)}
        if self.config.doFindBroad:
            excess, measuredBroad = findBroadLines(
                [ff for ff in allFree if self.lineKey(ff) not in dropSat | dropStatus | dropLowSN],
                window=self.config.broadWindow, broadFrac=self.config.broadFrac,
                nSigma=self.config.broadNSigma,
            )
            self.checkBroad(excess, measuredBroad, status)
            broad |= measuredBroad
        for key in sorted(dropSat, key=lambda kk: kk[1]):
            self.log.info("Line %s %.3f is saturated in %d fibres (fit in %d): not used",
                          key[0], key[1], nSat[key], nFit.get(key, 0))
        dropped = dropSat | dropStatus | dropLowSN | broad

        def isUsed(ff):
            return self.lineKey(ff) not in dropped

        # Centroid zero point, and pinned fits
        zeroPoint, zeroPointErr, scatter, used = centroidZeroPoint(
            [ff for ff in allFree if isUsed(ff)], nSigma=self.config.centroidNSigma
        )
        self.log.info("Centroid zero point %+.5f +/- %.5f pixels (clipped mean of %d lines; scatter %.4f)",
                      zeroPoint, zeroPointErr, used.sum(), scatter)
        detMap = exposures[0].detMap
        coords = DetectorCoords.fromDetectorMap(detMap)
        self.log.info("Detector coordinates: %s", coords)

        # The halo: measured from the bright lines, then held fixed at its smooth fit
        haloSurfaces = {}
        haloRejected = {}
        fixHalos = None
        if self.config.fitHalo:
            bright = {self.lineKey(ff) for ff in allFree if isUsed(ff)
                      and medianSN[self.lineKey(ff)] >= self.config.haloMinSN}
            haloRows = [(key[0], ff) for key, ffs in free.fits.items() for ff in ffs
                        if isUsed(ff) and self.lineKey(ff) in bright]
            self.log.info("Fitting the halo to %d bright lines (%d catalogue lines with median S/N >= %g)",
                          len(haloRows), len(bright), self.config.haloMinSN)
            haloSurfaces, rejected = self.fitSurfaces(haloRows, coords, detMap,
                                                      names=("haloFrac", "haloWidth"))
            haloRejected = {(visit, ff.fiberId, round(ff.catWavelength, 4)): rejected[jj]
                            for jj, (visit, ff) in enumerate(haloRows) if jj in rejected}
            fixHalos = {}
            for key, ffs in free.fits.items():
                ffs = [ff for ff in ffs if isUsed(ff)]
                if not ffs:
                    continue
                u, v = coords(np.array([ff.x for ff in ffs]), np.array([ff.y for ff in ffs]))
                frac = np.clip(haloSurfaces["haloFrac"](u, v), 0.0, 1.0)
                width = np.clip(haloSurfaces["haloWidth"](u, v), 1.0e-3, None)
                fixHalos[key] = {round(float(ff.catWavelength), 4): (float(fr), float(wi))
                                 for ff, fr, wi in zip(ffs, frac, width)}

        fits = free.fits
        if self.config.doPinCentroids or self.config.fitHalo:
            # Only the lines that will be used are refit; the others keep their free fits (in lsfLines)
            fixCentres = None
            if self.config.doPinCentroids:
                fixCentres = {key: pinnedCentres([ff for ff in ffs if isUsed(ff)], zeroPoint)
                              for key, ffs in free.fits.items()}
            pinned = self.fitLines(exposures, fixCentres, fixHalos, fixRadius=self.pfiRadiusFixed(arm))
            if self.config.fitHalo:     # lsfLines reports each line's measured (free) halo
                freeFits = {(key, round(ff.catWavelength, 4)): ff
                            for key, ffs in free.fits.items() for ff in ffs}
                for key, ffs in pinned.fits.items():
                    for ff in ffs:
                        ffFree = freeFits.get((key, round(ff.catWavelength, 4)))
                        if ffFree is not None:
                            ff.haloFrac, ff.haloFracErr = ffFree.haloFrac, ffFree.haloFracErr
                            ff.haloWidth, ff.haloWidthErr = ffFree.haloWidth, ffFree.haloWidthErr
            freeChi2 = {(key, round(ff.catWavelength, 4)): ff.chi2
                        for key, ffs in free.fits.items() for ff in ffs}
            dchi2 = [ff.chi2 - freeChi2[(key, round(ff.catWavelength, 4))] for key, ffs in pinned.fits.items()
                     for ff in ffs if (key, round(ff.catWavelength, 4)) in freeChi2]
            self.log.info("Pinned fits: %d lines; median chi^2 change %+.3f",
                          sum(len(ff) for ff in pinned.fits.values()), np.median(dchi2) if dchi2 else np.nan)
            fits = {key: pinned.fits.get(key, []) + [ff for ff in ffs if not isUsed(ff)]
                    for key, ffs in free.fits.items()}

        # Smooth fits over the detector
        rows = []
        for (visit, fiberId), ffs in fits.items():
            rows += [(visit, ff) for ff in ffs]
        usedRows = [ii for ii, (_, ff) in enumerate(rows) if isUsed(ff)]
        surfaces, rejected = self.fitSurfaces(
            [rows[ii] for ii in usedRows], coords, detMap,
            names=[nn for nn in self.config.degrees if nn not in haloSurfaces],
        )
        surfaces.update(haloSurfaces)
        rejectedFlags = defaultdict(int)
        for jj, flags in rejected.items():
            rejectedFlags[usedRows[jj]] = flags
        for ii, (visit, ff) in enumerate(rows):
            rejectedFlags[ii] |= haloRejected.get((visit, ff.fiberId, round(ff.catWavelength, 4)), 0)

        iis = None
        if iisExposures and self.config.doIisRadius:
            iis = self.fitIisRadius(arm, iisExposures, surfaces, coords, detMap)

        lsfModel = self.makeModel(arm, spectrograph, exposures, surfaces, coords, zeroPoint,
                                  extraExposures=iisExposures)
        lsfModel.metadata.update(zeroPointErr=zeroPointErr, zeroPointScatter=scatter)
        if iis is not None:
            lsfModel.engineeringFiberId = iis.fiberId
            lsfModel.iisRadiusScale = iis.scale
            lsfModel.iisHaloScale = iis.haloScale
            lsfModel.iisExtraBlur = iis.extraBlur
            lsfModel.metadata.update(iisVisits=[exp.visit for exp in iisExposures],
                                     iisLamps=[exp.lamps for exp in iisExposures],
                                     iisRadiusScale=iis.summary, iisHaloScale=iis.haloSummary,
                                     iisExtraBlur=iis.blurSummary)

        flags = {}
        for ii, (visit, ff) in enumerate(rows):
            flag = rejectedFlags[ii]
            key = self.lineKey(ff)
            if key in dropSat:
                flag |= LsfLineFlag.SATURATED_ELSEWHERE
            if key in dropStatus:
                flag |= LsfLineFlag.EXCLUDED_STATUS
            if key in dropLowSN:
                flag |= LsfLineFlag.LOW_SN
            if key in broad:
                flag |= LsfLineFlag.BROAD
            if ff.pinned:
                flag |= LsfLineFlag.PINNED
            flags[ii] = flag
        saturated = dict(free.saturated)
        if iis is not None:
            for ii, (visit, ff) in enumerate(iis.rows):
                flags[len(rows) + ii] = iis.flags[ii]
            rows = rows + iis.rows
            saturated.update(iis.saturated)
        lsfLines = self.makeLineSet(allExposures, rows, flags, saturated)
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

    def pfiRadiusFixed(self, arm):
        """How to hold the PFI disk radius: `None` (fit it), a value, or ``"model"``"""
        if self.config.doPfiRadiusModel:
            if arm not in self.config.pfiRadiusModel:
                raise RuntimeError(f"No pfiRadiusModel for arm {arm}")
            return "model"
        return self.config.pfiRadius

    def modelRadius(self, exp, fiberId):
        """The optical model's image radius (pixels) of each catalogue line in a fibre

        Parameters
        ----------
        exp : `Struct`
            Exposure (see `run`): ``pfsArm``, ``detMap``, ``lines``.
        fiberId : `int`
            Fibre.

        Returns
        -------
        radius : `dict` [`float`, `float`]
            ``round(catalogue wavelength, 4) -> radius`` (pixels).
        """
        config = self.config
        coeffs = np.array([float(cc) for cc in config.pfiRadiusModel[exp.pfsArm.identity.arm].split(",")])
        coeffs = coeffs.reshape(2, 3)
        wavelength = exp.lines.data.wavelength.to_numpy().astype(float)
        point = np.asarray(exp.detMap.findPoint(fiberId, wavelength))
        bbox = exp.detMap.getBBox()
        x0, y0 = 0.5*(bbox.getMinX() + bbox.getMaxX()), 0.5*(bbox.getMinY() + bbox.getMaxY())
        good = np.isfinite(point).all(axis=1)
        sign = 1.0
        if good.sum() > 1:      # v increases with wavelength
            sign = np.sign(np.polyfit(wavelength[good], point[good, 1], 1)[0]) or 1.0
        u = (point[:, 0] - x0)*config.pixelSize/30.0e3
        v = sign*(point[:, 1] - y0)*config.pixelSize/30.0e3
        radius = sum(coeffs[ii, jj]*u**(2*ii)*v**jj for ii in range(2) for jj in range(3))
        radius *= config.pfiRadiusModelScale/config.pixelSize
        return {round(float(ww), 4): float(rr) for ww, rr, gg in zip(wavelength, radius, good) if gg}

    def fitLines(self, exposures, fixCentres=None, fixHalos=None, fixShapes=None, fixRadius=None):
        """Fit the lines in every fibre of every exposure

        Parameters
        ----------
        exposures : `list` of `Struct`
            Exposures, with ``visit``, ``pfsArm``, ``detMap``, ``lines`` and
            ``lamps``.
        fixCentres : `dict` [(`int`, `int`), `dict`], optional
            ``(visit, fiberId) -> fixCentres`` (see `fitLsfLines.measureLsf`):
            hold the lines' centres fixed.
        fixHalos : `dict` [(`int`, `int`), `dict`], optional
            ``(visit, fiberId) -> fixHalos`` (see `fitLsfLines.measureLsf`):
            hold the lines' halos fixed.
        fixShapes : `dict` [(`int`, `int`), `dict`], optional
            ``(visit, fiberId) -> fixShapes`` (see `fitLsfLines.measureLsf`):
            hold the lines' blur kernels fixed.
        fixRadius : `float`, optional
            Hold every line's disk radius at this value (pixels), or (``"model"``)
            at `modelRadius`, or (a `dict`) at ``fixRadius[(visit, fiberId)]``
            (``round(catalogue wavelength, 4) -> radius``), fitting only those
            lines.

        Only the lines in ``fixCentres``, ``fixHalos`` or ``fixShapes`` are
        fit, if any is given.

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
                given = [dd for dd in (fixCentres, fixHalos, fixShapes) if dd is not None]
                if isinstance(fixRadius, dict):
                    given.append(fixRadius)
                refit = bool(given)
                if refit and not all(dd.get(key) for dd in given):
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
                    fitHalo=config.fitHalo,
                    companionFrac=config.companionFrac,
                    fixCentres=None if fixCentres is None else fixCentres[key],
                    fixHalos=None if fixHalos is None else fixHalos[key],
                    fixShapes=None if fixShapes is None else fixShapes[key],
                    fixRadius=(self.modelRadius(exp, fiberId) if isinstance(fixRadius, str) else
                               fixRadius[key] if isinstance(fixRadius, dict) else fixRadius),
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
                    excludeMask=0 if refit else excludeMask,
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

    def fitSurfaces(self, rows, coords, detMap, names=None):
        """Fit each parameter as a smooth function over the detector

        Parameters
        ----------
        rows : `list` of (`int`, `DiskLineFit`)
            Visit and fit of each line to use.
        coords : `DetectorCoords`
            Detector pixels to the frame of the fits.
        detMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y (for the detector's extent).
        names : iterable of `str`, optional
            Only fit these parameters.

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
            rms=([ff.rms for ff in fits], [ff.rmsErr for ff in fits]),
        )
        for jj, name in enumerate(CUMULANT_NAMES[:max(0, self.config.blurOrder - 2)]):
            params[name] = ([ff.cumulants[jj] for ff in fits], [ff.cumulantsErr[jj] for ff in fits])
        if self.config.fitLorentz:
            params["lorentz"] = ([ff.lorentz for ff in fits], [ff.lorentzErr for ff in fits])
        if self.config.fitHalo:
            params["haloFrac"] = ([ff.haloFrac for ff in fits], [ff.haloFracErr for ff in fits])
            params["haloWidth"] = ([ff.haloWidth for ff in fits], [ff.haloWidthErr for ff in fits])

        surfaces = {}
        rejected = defaultdict(int)
        for name, (value, error) in params.items():
            if name not in self.config.degrees or (names is not None and name not in names):
                continue
            if name == "radius" and self.config.doPfiRadiusModel:
                # The radii are the model's: fit them (equal weights) as a surface, for the model to evaluate
                error = np.ones_like(np.asarray(value, dtype=float))
            elif name == "radius" and self.config.pfiRadius is not None:
                surfaces[name] = SmoothSurface(name=name, degree=(0, 0), domain=domain,
                                               coeffs=np.array([self.config.pfiRadius]), offsets={},
                                               reference={}, nUsed=len(fits))
                self.log.info("Radius held at %.3f (pfiRadius)", self.config.pfiRadius)
                continue
            degree = tuple(int(dd) for dd in self.config.degrees[name].split(","))
            if self.config.doCoverageGuard:
                value = np.asarray(value, dtype=float)
                good = np.isfinite(value) & np.isfinite(np.asarray(error, dtype=float))
                supported = tuple(supportedDegree(pos[good], dom, dd, self.config.coverageFactor)
                                  for pos, dom, dd in zip((u, v), domain, degree))
                if supported != degree:
                    gaps = [maxGap(pos[good], dom) for pos, dom in zip((u, v), domain)]
                    self.log.warning("Lowering the degree of the smooth fit of %s from %s to %s: the lines "
                                     "leave gaps of %.0f, %.0f pixels in x, y", name, degree, supported,
                                     *gaps)
                    degree = supported
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

    def fitIisRadius(self, arm, iisExposures, surfaces, coords, detMap):
        """Measure the engineering (IIS) fibres' disk radius relative to the science fibres'

        The IIS fibres have smaller cores than the PFI fibres, but share the
        spectrograph, so their blur kernel (rms, cumulants, halo) is taken from
        the PFI surfaces. Each usable IIS line is refit with that kernel held
        fixed at its position; only its centre, amplitude, continuum,
        companions and disk radius are free. The ratio of that radius to the
        PFI radius surface there is fit as a smooth surface (equal weights; a
        constant is the median).

        The ratio is a property of the fibres, so if ``iisRadiusRatio`` is set
        that global value is used instead (the measured ratio is kept in the
        summary as a diagnostic): the lines are refit with the radius held at
        ``iisRadiusRatio`` times the PFI radius, and the halo scale (from the
        bright lines) is measured with the radius held there too.

        Parameters
        ----------
        arm : `str`
            Arm.
        iisExposures : `list` of `Struct`
            IIS exposures (see `run`).
        surfaces : `dict` [`str`, `SmoothSurface`]
            The PFI surfaces.
        coords : `DetectorCoords`
            Detector pixels to the frame of the surfaces.
        detMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y (for the detector's extent).

        Returns
        -------
        scale : `SmoothSurface`
            Ratio of the IIS to the PFI disk radius.
        fiberId : `numpy.ndarray` of `int`
            The engineering fibres.
        rows : `list` of (`int`, `DiskLineFit`)
            Visit and (fixed-kernel) fit of each IIS line.
        flags : `list` of `int`
            ``LsfLineFlag`` of each of ``rows``.
        saturated : `dict` [(`int`, `int`), `list` of (`str`, `float`)]
            Saturated lines.
        summary : `dict`
            Summary statistics, for the model's metadata.
        """
        config = self.config
        free = self.fitLines(iisExposures)
        allFree = [ff for ffs in free.fits.values() for ff in ffs]
        nFit = Counter(self.lineKey(ff) for ff in allFree)
        nSat = Counter((lamps, round(lam, 3)) for keys in free.saturated.values() for lamps, lam in keys)
        excludeMask = 0
        for name in config.excludeStatus:
            excludeMask |= int(getattr(ReferenceLineStatus, name))
        if config.excludeCatalogueBroad:
            excludeMask |= int(ReferenceLineStatus.BROAD)
        signalToNoise = defaultdict(list)
        for ff in allFree:
            signalToNoise[self.lineKey(ff)].append(ff.amplitude/ff.amplitudeErr)
        minLineSN = config.minLineSN.get(arm, config.minLineSN["default"])
        dropped = {key for key, nn in nSat.items() if nn > config.iisSatDropFrac*(nn + nFit.get(key, 0))}
        dropped |= {self.lineKey(ff) for ff in allFree if ff.status & excludeMask}
        dropped |= {key for key, sn in signalToNoise.items() if not np.nanmedian(sn) >= minLineSN}

        ratioFixed = config.iisRadiusRatio
        extraBlur, blurSummary = None, {}

        def radiusAt(ffs):
            """The PFI disk radius at each fit's position"""
            return surfaces["radius"](*coords(np.array([ff.x for ff in ffs]), np.array([ff.y for ff in ffs])))

        def kernelAt(ffs):
            """The PFI kernel ((rms, cumulants), (haloFrac, haloWidth) or None) at each fit's position"""
            u, v = coords(np.array([ff.x for ff in ffs]), np.array([ff.y for ff in ffs]))
            rms = surfaces["rms"](u, v)
            cumulants = np.array([surfaces[nn](u, v) for nn in CUMULANT_NAMES[:max(0, config.blurOrder - 2)]])
            halo = None
            if "haloFrac" in surfaces:
                halo = (np.clip(surfaces["haloFrac"](u, v), 0.0, 1.0),
                        np.clip(surfaces["haloWidth"](u, v), 1.0e-3, None))
            return rms, cumulants.reshape(-1, len(ffs)), halo

        bbox = detMap.getBBox()
        xc, yc = np.meshgrid([bbox.getMinX(), bbox.getMaxX()], [bbox.getMinY(), bbox.getMaxY()])
        uc, vc = coords(xc.ravel(), yc.ravel())
        domain = ((uc.min(), uc.max()), (vc.min(), vc.max()))

        def scaleSurface(u, v, ratio, degreeStr, name):
            """Fit a ratio as a smooth surface (with the coverage guard), robustly

            The lines get equal weights (their formal errors are dominated by
            systematics, and a few lines with small errors would otherwise
            dominate), and a constant is the median of the lines not clipped.
            """
            degree = tuple(int(dd) for dd in degreeStr.split(","))
            if config.doCoverageGuard:
                degree = tuple(supportedDegree(pos, dom, dd, config.coverageFactor)
                               for pos, dom, dd in zip((u, v), domain, degree))
            try:
                surface, used, _ = fitSmoothSurface(u, v, ratio, np.ones_like(ratio), degree, domain=domain,
                                                    name=name, nIter=config.nIter, nSigma=config.nSigma)
            except RuntimeError as exc:     # too few lines survive the clipping: use the median
                self.log.warning("Fitting %s as a constant (the median): %s", name, exc)
                degree = (0, 0)
                surface = constantSurface(np.median(ratio), name)
                used = np.ones_like(ratio, dtype=bool)
            if degree == (0, 0) and used.any():
                surface.coeffs = np.array([np.median(ratio[used])])
            return surface, used, degree

        def constantSurface(value, name):
            """A constant `SmoothSurface`"""
            return SmoothSurface(name=name, degree=(0, 0), domain=domain, coeffs=np.array([float(value)]),
                                 offsets={}, reference={})

        # The IIS halo: from the bright lines, with the PFI core (rms, cumulants) held fixed and the halo free
        haloScale = None
        haloSummary = {}
        if config.doIisHalo and "haloFrac" in surfaces:
            bright = {key for key, sn in signalToNoise.items()
                      if key not in dropped and np.nanmedian(sn) >= config.haloMinSN}
            fixShapes = {}
            for key, ffs in free.fits.items():
                ffs = [ff for ff in ffs if self.lineKey(ff) in bright]
                if ffs:
                    rms, cumulants, _ = kernelAt(ffs)
                    radius = ratioFixed*radiusAt(ffs) if ratioFixed is not None else None
                    fixShapes[key] = {round(float(ff.catWavelength), 4):
                                      (float(rms[jj]), cumulants[:, jj])
                                      + (() if radius is None else (float(radius[jj]),))
                                      for jj, ff in enumerate(ffs)}
            haloFits = [ff for ffs in self.fitLines(iisExposures, fixShapes=fixShapes).fits.values()
                        for ff in ffs]
            if len(haloFits) > 5:
                u, v = coords(np.array([ff.x for ff in haloFits]), np.array([ff.y for ff in haloFits]))
                pfiHalo = np.clip(surfaces["haloFrac"](u, v), 1.0e-4, None)
                haloRatio = np.array([ff.haloFrac for ff in haloFits])/pfiHalo
                haloScale, used, degree = scaleSurface(u, v, haloRatio, config.iisHaloScaleDegree,
                                                       "iisHaloScale")
                haloSummary = dict(median=float(np.median(haloRatio[used])), nLines=int(used.sum()),
                                   nCatalogueLines=len(bright), degree=list(degree), rms=haloScale.rms,
                                   ratios=[float(rr) for rr in haloRatio],
                                   lines=[[ff.element, round(float(ff.catWavelength), 4)] for ff in haloFits],
                                   fiberId=[int(ff.fiberId) for ff in haloFits])
                self.log.info("IIS halo scale: median %.3f over %d lines (%d catalogue lines; degree %s, "
                              "rms %.3f)",
                              haloSummary["median"], used.sum(), len(bright), degree, haloScale.rms)
            else:
                self.log.warning("Only %d bright IIS lines: not measuring the IIS halo scale", len(haloFits))
        if "haloFrac" in surfaces and arm in config.iisHaloScale:
            value = config.iisHaloScale[arm]
            self.log.info("Using the IIS halo scale %.3f for arm %s (iisHaloScale), not the measured %s",
                          value, arm, "%.3f" % haloSummary["median"] if haloSummary else "(none)")
            haloScale = constantSurface(value, "iisHaloScale")
            haloSummary["used"] = value

        # The IIS radius: every usable line, with the PFI kernel held fixed (its halo scaled by haloScale)
        fixShapes, fixHalos = {}, {}
        for key, ffs in free.fits.items():
            ffs = [ff for ff in ffs if self.lineKey(ff) not in dropped]
            if not ffs:
                continue
            rms, cumulants, halo = kernelAt(ffs)
            fixShapes[key] = {round(float(ff.catWavelength), 4): (float(rms[jj]), cumulants[:, jj])
                              for jj, ff in enumerate(ffs)}
            if halo is not None:
                frac = halo[0]
                if haloScale is not None:
                    u, v = coords(np.array([ff.x for ff in ffs]), np.array([ff.y for ff in ffs]))
                    frac = np.clip(frac*haloScale(u, v), 0.0, 1.0)
                fixHalos[key] = {round(float(ff.catWavelength), 4): (float(frac[jj]), float(halo[1][jj]))
                                 for jj, ff in enumerate(ffs)}
        fixed = self.fitLines(iisExposures, fixHalos=fixHalos if fixHalos else None, fixShapes=fixShapes)

        rows = [(key[0], ff) for key, ffs in fixed.fits.items() for ff in ffs]
        fits = [ff for _, ff in rows]
        u, v = coords(np.array([ff.x for ff in fits]), np.array([ff.y for ff in fits]))
        ratio = np.array([ff.radius for ff in fits])/radiusAt(fits)
        scale, used, degree = scaleSurface(u, v, ratio, config.iisRadiusScaleDegree, "iisRadiusScale")

        # Diagnostics: is only the radius different (chi^2 of the fixed-kernel vs free fits)? Does the ratio
        # depend on brightness (the IIS images are narrower, so brighter per pixel)?
        freeChi2 = {(key, round(ff.catWavelength, 4)): ff.chi2/max(1, ff.ndof)
                    for key, ffs in free.fits.items() for ff in ffs}
        dchi2 = np.array([ff.chi2/max(1, ff.ndof) - freeChi2.get((key, round(ff.catWavelength, 4)), np.nan)
                          for key, ffs in fixed.fits.items() for ff in ffs])
        logPeak = np.log10(np.maximum([ff.peak for ff in fits], 1.0))
        good = used & np.isfinite(ratio)
        slope = np.nan
        if good.sum() > 3:
            slope = np.polyfit(logPeak[good] - np.median(logPeak[good]), ratio[good], 1)[0]
        pointScale = float(np.median(ratio[good])) if good.any() else np.nan
        summary = dict(median=pointScale, nLines=int(good.sum()), degree=list(degree), rms=scale.rms,
                       medianChi2PerDofChange=float(np.nanmedian(dchi2)) if dchi2.size else np.nan,
                       slopePerDexOfPeak=float(slope),
                       used=ratioFixed if ratioFixed is not None else "measured")
        self.log.info("IIS radius scale: median %.4f over %d lines (%d catalogue lines; degree %s, "
                      "rms %.4f); fixed-kernel minus free chi^2/dof median %+.2f; slope %+.4f per dex of "
                      "peak",
                      pointScale, good.sum(), len({self.lineKey(ff) for ff in fits}), degree, scale.rms,
                      summary["medianChi2PerDofChange"], slope)
        flags = [int(LsfLineFlag.IIS | (0 if uu else LsfLineFlag.REJECTED_IIS)) for uu in used]
        if ratioFixed is not None:
            # The radius is held at the global ratio: refit with everything but the centre, amplitude,
            # continuum and companions fixed (these are the IIS lines in lsfLines); the measured ratio above
            # is a diagnostic
            self.log.info("Using the IIS/PFI radius ratio %.3f (iisRadiusRatio), not the measured %.4f",
                          ratioFixed, pointScale)
            radii = {}
            for key, shapes in fixShapes.items():
                ffs = [ff for ff in free.fits[key] if round(float(ff.catWavelength), 4) in shapes]
                radii[key] = {round(float(ff.catWavelength), 4): float(rr)
                              for ff, rr in zip(ffs, ratioFixed*radiusAt(ffs))}

            # The extra blur: the IIS lines with their radius (and halo) held, and the kernel free
            if config.doIisExtraBlur:
                blurFits = [ff for ffs in self.fitLines(iisExposures, fixHalos=fixHalos if fixHalos else None,
                                                        fixRadius=radii).fits.values() for ff in ffs]
                if len(blurFits) > 5:
                    rmsPfi, cumPfi, _ = kernelAt(blurFits)
                    rmsIis = np.array([ff.rms for ff in blurFits])
                    extraVariance = rmsIis**2 - rmsPfi**2
                    kappa4 = np.zeros_like(rmsIis)
                    if config.blurOrder >= 4:
                        kurtosis = np.array([ff.cumulants[1] for ff in blurFits])
                        kappa4 = kurtosis*rmsIis**4 - cumPfi[1]*rmsPfi**4
                    good = np.isfinite(extraVariance) & np.isfinite(kappa4)
                    extraBlur = (float(np.median(extraVariance[good])), float(np.median(kappa4[good])))
                    mad = 1.4826*np.median(np.abs(extraVariance[good] - np.median(extraVariance[good])))
                    blurSummary = dict(variance=extraBlur[0], kappa4=extraBlur[1], nLines=int(good.sum()),
                                       extraVarianceScatter=float(mad),
                                       extraVariance=[float(vv) for vv in extraVariance],
                                       kappa4s=[float(kk) for kk in kappa4],
                                       lines=[[ff.element, round(float(ff.catWavelength), 4)]
                                              for ff in blurFits],
                                       fiberId=[int(ff.fiberId) for ff in blurFits])
                    self.log.info("IIS extra blur: variance %+.4f +- %.4f, kappa4 %+.4f over %d lines",
                                  extraBlur[0], mad, extraBlur[1],
                                  good.sum())
                else:
                    self.log.warning("Only %d IIS lines: not measuring the IIS extra blur", len(blurFits))
            if arm in config.iisExtraBlur:
                extraBlur = tuple(float(vv) for vv in config.iisExtraBlur[arm].split(","))
                self.log.info("Using the IIS extra blur variance %+.4f, kappa4 %+.4f for arm %s "
                              "(iisExtraBlur)",
                              *extraBlur, arm)
                blurSummary["used"] = list(extraBlur)

            for key, shapes in fixShapes.items():
                for lam, (rms, cumulants) in list(shapes.items()):
                    if extraBlur is not None:
                        rms, cumulants = addBlur(rms, cumulants, *extraBlur)
                    shapes[lam] = (float(rms), np.asarray(cumulants, dtype=float), radii[key][lam])
            final = self.fitLines(iisExposures, fixHalos=fixHalos if fixHalos else None, fixShapes=fixShapes)
            rows = [(key[0], ff) for key, ffs in final.fits.items() for ff in ffs]
            flags = [int(LsfLineFlag.IIS)]*len(rows)
            scale = constantSurface(ratioFixed, "iisRadiusScale")
        fiberId = np.array(sorted({int(ff) for exp in iisExposures for ff in exp.pfsArm.fiberId}), dtype=int)
        return Struct(scale=scale, haloScale=haloScale, fiberId=fiberId, rows=rows, flags=flags,
                      saturated=free.saturated, summary=summary, haloSummary=haloSummary, extraBlur=extraBlur,
                      blurSummary=blurSummary)

    def makeModel(self, arm, spectrograph, exposures, surfaces, coords, zeroPoint, extraExposures=()):
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
        extraExposures : `list` of `Struct`
            Further exposures (e.g. IIS arcs) whose fibres the model covers.

        Returns
        -------
        lsfModel : `ArmLsfModel`
            Model of the LSF.
        """
        detMap = exposures[0].detMap
        fiberId = np.array(sorted(set(int(ff) for exp in list(exposures) + list(extraExposures)
                                      for ff in exp.pfsArm.fiberId)), dtype=int)
        length = exposures[0].pfsArm.wavelength.shape[1]
        xCenter = FiberSpline.fromFunction(fiberId, length, self.config.traceStep, detMap.getXCenter)
        wavelength = FiberSpline.fromFunction(fiberId, length, self.config.traceStep, detMap.findWavelength)
        metadata: Dict = dict(visits=[exp.visit for exp in exposures], lamps=[exp.lamps for exp in exposures],
                              config=self.config.toDict(),
                              scatteredLight=asdict(self.config.scatteredLight.getModel(arm, spectrograph)))
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
            if self.config.fitHalo:
                cumulants.update(haloFrac=ff.haloFrac, haloFracErr=ff.haloFracErr, haloWidth=ff.haloWidth,
                                 haloWidthErr=ff.haloWidthErr)
            add(visit, ff.fiberId, ff.catWavelength, flags[ii], x=ff.x, y=ff.y, pixel=ff.pixel,
                pixelErr=ff.pixelErr, refPixel=ff.catPixel, dispersion=ff.dispersion,
                dataCentroid=ff.dataCentroid, modelCentroid=ff.modelCentroid, radius=ff.radius,
                radiusErr=ff.radiusErr, rms=ff.rms, rmsErr=ff.rmsErr, flux=ff.amplitude,
                fluxErr=ff.amplitudeErr, peak=ff.peak, chi2=ff.chi2, ndof=ff.ndof, nBadCore=ff.nBadCore,
                **cumulants)
        for (visit, fiberId), keys in saturated.items():
            for _, wavelength in keys:
                add(visit, fiberId, wavelength, LsfLineFlag.SATURATED)

        dtypes = {name: column.dtype for name, column in LsfLineSet.DamdClass.getSchemaDict().items()}
        return LsfLineSet.fromColumns(**{name: np.array(values, dtype=dtypes[name] if dtypes[name] is not str
                                                        else object)
                                         for name, values in columns.items()})
