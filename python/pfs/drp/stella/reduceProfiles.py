from collections import defaultdict
from typing import Iterable, List

import numpy as np

from lsst.daf.persistence import ButlerDataRef
from lsst.pex.config import Config, Field, ConfigurableField, ConfigField, ListField, ChoiceField
from lsst.pipe.base import CmdLineTask, TaskRunner, ArgumentParser, Struct
from lsst.afw.image import Exposure, makeExposure
from lsst.afw.math import StatisticsControl, stringToStatisticsProperty, statisticsStack

from pfs.datamodel import FiberStatus, TargetType, CalibIdentity
from .adjustDetectorMap import AdjustDetectorMapTask
from .buildFiberProfiles import BuildFiberProfilesTask
from .reduceExposure import ReduceExposureTask
from .centroidTraces import CentroidTracesTask, tracesToLines
from .constructSpectralCalibs import setCalibHeader
from .blackSpotCorrection import BlackSpotCorrectionTask
from . import FiberProfileSet


class ReduceProfilesTaskRunner(TaskRunner):
    """Split data by spectrograph arm and extract normalisation/dark visits"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        dataRefs = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.id.refList:
            spectrograph = ref.dataId["spectrograph"]
            arm = ref.dataId["arm"]
            dataRefs[spectrograph][arm].append(ref)

        normRefs = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.normId.refList:
            spectrograph = ref.dataId["spectrograph"]
            arm = ref.dataId["arm"]
            if spectrograph not in dataRefs or arm not in dataRefs[spectrograph]:
                raise RuntimeError(f"No profile inputs provided for norm {ref.dataId}")
            normRefs[spectrograph][arm].append(ref)

        darkRefs = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.darkId.refList:
            spectrograph = ref.dataId["spectrograph"]
            arm = ref.dataId["arm"]
            if spectrograph not in dataRefs or arm not in dataRefs[spectrograph]:
                raise RuntimeError(f"No profile inputs provided for dark {ref.dataId}")
            darkRefs[spectrograph][arm].append(ref)

        targets = []
        for spectrograph in dataRefs:
            for arm in dataRefs[spectrograph]:
                if spectrograph not in normRefs or arm not in normRefs[spectrograph]:
                    raise RuntimeError(f"No norm provided for spectrograph={spectrograph} arm={arm}")
                dataList = sorted(dataRefs[spectrograph][arm], key=lambda ref: ref.dataId["visit"])
                normList = sorted(normRefs[spectrograph][arm], key=lambda ref: ref.dataId["visit"])
                darkList = sorted(darkRefs[spectrograph][arm], key=lambda ref: ref.dataId["visit"])
                targets.append((dataList, dict(normRefList=normList, darkRefList=darkList, **kwargs)))
        return targets


class CombineConfig(Config):
    """Configuration for combining images"""
    mask = ListField(dtype=str, default=["BAD", "SAT", "INTRP", "CR", "NO_DATA"],
                     doc="Mask planes to reject from combination")
    combine = ChoiceField(dtype=str, default="MEANCLIP",
                          allowed=dict(MEAN="Sample mean", MEANCLIP="Clipped mean", MEDIAN="Sample median"),
                          doc="Statistic to use for combination (from lsst.afw.math)")
    rejThresh = Field(dtype=float, default=3.0, doc="Clipping threshold for combination")
    rejIter = Field(dtype=int, default=3, doc="Clipping iterations for combination")
    maxVisitsToCalcErrorFromInputVariance = Field(
        dtype=int, default=2,
        doc="Maximum number of visits to estimate variance from input variance, not per-pixel spread"
    )


class ReduceProfilesConfig(Config):
    """Configuration for FiberTrace construction"""
    reduceExposure = ConfigurableField(target=ReduceExposureTask, doc="Reduce single exposure")
    profiles = ConfigurableField(target=BuildFiberProfilesTask, doc="Build fiber profiles")
    mask = ListField(dtype=str, default=["BAD_FLAT", "CR", "SAT", "NO_DATA"],
                     doc="Mask planes to exclude from fiberTrace")
    combine = ConfigField(dtype=CombineConfig, doc="Combination configuration")
    doAdjustDetectorMap = Field(dtype=bool, default=True, doc="Adjust detectorMap using trace positions?")
    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Adjust detectorMap")
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")
    traceSpectralError = Field(dtype=float, default=1.0,
                               doc="Error in the spectral dimension to give trace centroids (pixels)")
    fiberStatus = ListField(
        dtype=str,
        default=["GOOD", "BROKENFIBER", "BLACKSPOT"],
        doc="Fiber status for which to build profiles",
    )
    targetType = ListField(
        dtype=str,
        default=[
            "SCIENCE",
            "SKY",
            "FLUXSTD",
            "UNASSIGNED",
            "SUNSS_IMAGING",
            "SUNSS_DIFFUSE",
            "HOME",
            "BLACKSPOT",
        ],
        doc="Target type for which to build profiles",
    )
    blackspots = ConfigurableField(target=BlackSpotCorrectionTask, doc="Black spot correction")
    replaceFibers = ListField(
        dtype=int,
        default=[
            84,  # One of three in a row with bad cobras
            85,  # One of three in a row with bad cobras
            86,  # One of three in a row with bad cobras
            114,  # Broken fiber; always blank
        ],
        doc="Replace the profiles for these fibers",
    )
    replaceNearest = Field(dtype=int, default=2, doc="Replace profiles with average of this many near fibers")
    darkDitherMax = Field(dtype=float, default=0.001, doc="Maximum distance for accepting dark dither")

    def setDefaults(self):
        super().setDefaults()
        self.reduceExposure.doMeasureLines = False
        self.reduceExposure.doMeasurePsf = False
        self.reduceExposure.doSubtractSky2d = False
        self.reduceExposure.doExtractSpectra = False
        self.reduceExposure.doWriteArm = False
        self.profiles.doBlindFind = False  # We have good detectorMaps and pfsConfig, so we know what's where
        self.profiles.profileOversample = 3
        self.adjustDetectorMap.minSignalToNoise = 0  # We don't measure S/N


class ReduceProfilesTask(CmdLineTask):
    """Task to construct the fiber trace"""
    ConfigClass = ReduceProfilesConfig
    _DefaultName = "reduceProfiles"
    RunnerClass = ReduceProfilesTaskRunner

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("reduceExposure")
        self.makeSubtask("profiles")
        self.makeSubtask("centroidTraces")
        self.makeSubtask("adjustDetectorMap")
        self.makeSubtask("blackspots")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="raw",
                               help="identifiers for profile, e.g., --id visit=123^456 ccd=7")
        parser.add_id_argument("--normId", datasetType="raw",
                               help="identifiers for normalisation, e.g., --id visit=98 ccd=7")
        parser.add_id_argument("--darkId", datasetType="raw",
                               help="identifiers for dot-roach darks, e.g., --id visit=56 ccd=7")
        return parser

    def runDataRef(
        self,
        dataRefList: Iterable[ButlerDataRef],
        normRefList: Iterable[ButlerDataRef],
        darkRefList: Iterable[ButlerDataRef],
    ) -> FiberProfileSet:
        """Construct the ``fiberProfiles`` calib

        Parameters
        ----------
        dataRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for exposures.
        normRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for normalisation exposures.
        darkRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for dark exposures.
        """
        if not dataRefList:
            raise RuntimeError("No input exposures")
        if not normRefList:
            raise RuntimeError("No normalisation exposures")
        darkList = [self.processDark(ref) for ref in darkRefList]
        dataList = [self.processExposure(ref, darkList) for ref in dataRefList]

        exposureList = [data.exposure for data in dataList]
        arm = dataRefList[0].dataId["arm"]
        identity = CalibIdentity(
            obsDate=exposureList[0].visitInfo.getDate().toPython().isoformat(),
            spectrograph=dataRefList[0].dataId["spectrograph"],
            arm=arm,
            visit0=dataRefList[0].dataId["visit"],
        )
        detectorMapList = [data.detectorMap for data in dataList]
        pfsConfigList = [
            ref.get("pfsConfig").select(fiberId=detMap.fiberId)
            for ref, detMap in zip(dataRefList, detectorMapList)
        ]

        # Get the union of all available fiberIds
        fibers = set()
        fiberStatus = [FiberStatus.fromString(fs) for fs in self.config.fiberStatus]
        targetType = [TargetType.fromString(tt) for tt in self.config.targetType]
        for pfsConfig in pfsConfigList:
            fibers.update(pfsConfig.select(fiberStatus=fiberStatus, targetType=targetType).fiberId)

        if darkList:
            # Select only fibers that are not hidden (i.e., finite pfiCenter)
            exposedFibers = set()
            for pfsConfig in pfsConfigList:
                hasFiniteCenter = np.logical_and.reduce(np.isfinite(pfsConfig.pfiCenter), axis=1)
                exposedFibers.update(pfsConfig.fiberId[hasFiniteCenter])
            fibers.intersection_update(exposedFibers)

        fiberId = np.array(list(sorted(fibers)))
        pfsConfigList = [pfsConfig.select(fiberId=fiberId) for pfsConfig in pfsConfigList]

        profiles = self.profiles.runMultiple(exposureList, identity, detectorMapList, pfsConfigList).profiles
        profiles.replaceFibers(self.config.replaceFibers, self.config.replaceNearest)

        normList = [self.processExposure(ref, darkList) for ref in normRefList]
        normExp = self.combine([norm.exposure for norm in normList])

        if self.config.doAdjustDetectorMap:
            detectorMap = normList[0].detectorMap
            traces = self.centroidTraces.run(normExp, detectorMap, normList[0].pfsConfig)
            lines = tracesToLines(detectorMap, traces, self.config.traceSpectralError)
            detectorMap = self.adjustDetectorMap.run(
                detectorMap, lines, arm, normExp.visitInfo.id
            ).detectorMap
            normRefList[0].put(detectorMap, "detectorMap_used")

        spectra = profiles.extractSpectra(
            normExp.maskedImage, detectorMap, normExp.mask.getPlaneBitMask(self.config.mask)
        )
        self.blackspots.run(normList[0].pfsConfig, spectra)

        for ss in spectra:
            good = (ss.mask.array[0] & ss.mask.getPlaneBitMask("NO_DATA")) == 0
            profiles[ss.fiberId].norm = np.where(good, ss.flux/ss.norm, np.nan)

        self.write(dataRefList[0], profiles, [dataRef.dataId["visit"] for dataRef in dataRefList],
                   [dataRef.dataId["visit"] for dataRef in normRefList])

        return profiles

    def getExposureDither(self, dataRef: ButlerDataRef) -> float:
        """Get the dither for an exposure

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.

        Returns
        -------
        dither : `float`
            Dither for exposure.
        """
        results = dataRef.getButler().queryMetadata("raw", ("dither",), dataRef.dataId)
        if len(results) != 1:
            raise RuntimeError(f"Expected exactly one dither, got {results}")
        return results[0]

    def processExposure(self, dataRef: ButlerDataRef, darkList: List[Struct]) -> Struct:
        """Process an exposure

        We read existing data from the butler, if available. Otherwise, we
        construct it by running ``reduceExposure``.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.

        Returns
        -------
        data : `lsst.pipe.base.Struct`
            Struct with ``exposure``, ``detectorMap``, ``pfsConfig`` and
            ``dither``.
        """
        require = ("calexp", "detectorMap_used")
        if all(dataRef.datasetExists(name) for name in require):
            self.log.info("Reading existing data for %s", dataRef.dataId)
            data = Struct(
                exposure=dataRef.get("calexp"),
                detectorMap=dataRef.get("detectorMap_used"),
                pfsConfig=dataRef.get("pfsConfig"),
            )
        else:
            data = self.reduceExposure.runDataRef(dataRef)

        if not darkList:
            self.log.warn("No darks provided; not performing dark subtraction")
            return data

        # Find the dark exposure with the closest dither
        dither = self.getExposureDither(dataRef)
        dark = min(darkList, key=lambda data: abs(data.dither - dither))
        if abs(dark.dither - dither) > self.config.darkDitherMax:
            raise RuntimeError(
                f"No dark dithers close enough to {dither} (darkDitherMax={self.config.darkDitherMax})"
            )
        self.log.info(
            "Using dark %d (dither=%f) for %d (dither=%f)",
            dark.exposure.getInfo().getId(),
            dark.dither,
            data.exposure.getInfo().getId(),
            dither,
        )
        darkTime = dark.exposure.getInfo().getVisitInfo().getExposureTime()
        dataTime = data.exposure.getInfo().getVisitInfo().getExposureTime()
        data.exposure.maskedImage.scaledMinus(dataTime/darkTime, dark.exposure.maskedImage)

        return data

    def processDark(self, dataRef: ButlerDataRef) -> Struct:
        """Process a dot-roach dark exposure

        We perform ISR only.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.

        Returns
        -------
        data : `lsst.pipe.base.Struct`
            Struct with ``exposure`` and ``dither``.
        """
        dataset = "postISRCCD"
        if dataRef.datasetExists(dataset):
            exposure = dataRef.get(dataset)
        else:
            exposure = self.reduceExposure.isr.runDataRef(dataRef)
        return Struct(exposure=exposure, dither=self.getExposureDither(dataRef))

    def write(self, dataRef: ButlerDataRef, profiles: FiberProfileSet, dataVisits: Iterable[int],
              normVisits: Iterable[int]):
        """Write outputs

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for output.
        profiles : `pfs.drp.stella.FiberProfileSet`
            Fiber profiles.
        dataVisits : iterable of `int`
            List of visits used to construct the fiber profiles.
        normVisits : iterable of `int`
            List of visits used to measure the normalisation.
        """
        self.log.info("Writing output for %s", dataRef.dataId)
        visit0 = dataRef.dataId["visit"]

        outputId = dict(
            visit0=visit0,
            calibDate=dataRef.dataId["dateObs"],
            calibTime=dataRef.dataId["taiObs"],
            arm=dataRef.dataId["arm"],
            spectrograph=dataRef.dataId["spectrograph"],
            ccd=dataRef.dataId["ccd"],
            filter=dataRef.dataId["filter"],
        )

        setCalibHeader(profiles.metadata, "fiberProfiles", dataVisits, outputId)
        for ii, vv in enumerate(sorted(set(normVisits))):
            profiles.metadata.set(f"CALIB_NORM_{ii}", vv)

        dataRef.put(profiles, "fiberProfiles", **outputId)

    def combine(self, exposureList: Iterable[Exposure]) -> Exposure:
        """Combine multiple exposures.

        Parameters
        ----------
        exposureList : iterable of `lsst.afw.image.Exposure`
            List of exposures to combine.

        Returns
        -------
        combined : `lsst.afw.image.Exposure`
            Combined exposure.
        """
        firstExp = exposureList[0]
        if len(exposureList) == 1:
            return firstExp  # That was easy!
        dimensions = firstExp.getDimensions()
        for exp in exposureList[1:]:
            if exp.getDimensions() != dimensions:
                raise RuntimeError(f"Dimension difference: {exp.getDimensions()} vs {dimensions}")

        config = self.config.combine
        combineStat = stringToStatisticsProperty(config.combine)
        ctrl = StatisticsControl(config.rejThresh, config.rejIter, firstExp.mask.getPlaneBitMask(config.mask))
        numImages = len(exposureList)
        if numImages < 1:
            raise RuntimeError("No valid input data")
        if numImages < self.config.combine.maxVisitsToCalcErrorFromInputVariance:
            ctrl.setCalcErrorFromInputVariance(True)

        # Combine images
        combined = makeExposure(statisticsStack([exp.maskedImage for exp in exposureList], combineStat, ctrl))
        combined.setMetadata(firstExp.getMetadata())
        combined.getInfo().setVisitInfo(firstExp.getInfo().getVisitInfo())
        return combined

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
