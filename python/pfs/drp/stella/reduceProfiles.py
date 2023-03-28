from collections import defaultdict
from typing import Iterable

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
    """Split data by spectrograph arm and extract normalisation visit"""
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

        targets = []
        for spectrograph in dataRefs:
            for arm in dataRefs[spectrograph]:
                if spectrograph not in normRefs or arm not in normRefs[spectrograph]:
                    raise RuntimeError(f"No norm provided for spectrograph={spectrograph} arm={arm}")
                targets.append(
                    (dataRefs[spectrograph][arm], dict(normRefList=normRefs[spectrograph][arm], **kwargs))
                )
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
        default=["GOOD", "BROKENFIBER"],
        doc="Fiber status for which to build profiles",
    )
    targetType = ListField(
        dtype=str,
        default=["SCIENCE", "SKY", "FLUXSTD", "UNASSIGNED", "SUNSS_IMAGING", "SUNSS_DIFFUSE"],
        doc="Target type for which to build profiles",
    )
    blackspots = ConfigurableField(target=BlackSpotCorrectionTask, doc="Black spot correction")

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
        return parser

    def runDataRef(self, dataRefList: Iterable[ButlerDataRef], normRefList: Iterable[ButlerDataRef]
                   ) -> FiberProfileSet:
        """Construct the ``fiberProfiles`` calib

        Parameters
        ----------
        expRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for exposures.
        butler : `lsst.daf.persistence.Butler`
            Data butler.
        calibId : `dict`
            Data identifier keyword-value pairs to use for the calib.
        """
        if not dataRefList:
            raise RuntimeError("No input exposures")
        if not normRefList:
            raise RuntimeError("No normalisation exposures")
        dataList = [self.processExposure(ref) for ref in dataRefList]

        exposureList = [data.exposureList[0] for data in dataList]
        identity = CalibIdentity(
            obsDate=exposureList[0].visitInfo.getDate().toPython().isoformat(),
            spectrograph=dataRefList[0].dataId["spectrograph"],
            arm=dataRefList[0].dataId["arm"],
            visit0=dataRefList[0].dataId["visit"],
        )
        detectorMapList = [data.detectorMapList[0] for data in dataList]
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
        fiberId = np.array(list(sorted(fibers)))
        pfsConfigList = [pfsConfig.select(fiberId=fiberId) for pfsConfig in pfsConfigList]

        profiles = self.profiles.runMultiple(exposureList, identity, detectorMapList, pfsConfigList).profiles

        normList = [self.processExposure(ref) for ref in normRefList]
        normExp = self.combine([norm.exposureList[0] for norm in normList])

        if self.config.doAdjustDetectorMap:
            detectorMap = normList[0].detectorMapList[0]
            traces = self.centroidTraces.run(normExp, detectorMap, normList[0].pfsConfig)
            lines = tracesToLines(detectorMap, traces, self.config.traceSpectralError)
            detectorMap = self.adjustDetectorMap.run(detectorMap, lines, normExp.visitInfo.id).detectorMap
            normRefList[0].put(detectorMap, "detectorMap_used")

        spectra = profiles.extractSpectra(
            normExp.maskedImage, detectorMap, normExp.mask.getPlaneBitMask(self.config.mask)
        )
        self.blackspots.run(normList[0].pfsConfig, spectra)

        for ss in spectra:
            good = (ss.mask.array[0] & ss.mask.getPlaneBitMask(self.config.mask)) == 0
            profiles[ss.fiberId].norm = np.where(good, ss.flux/ss.norm, np.nan)

        self.write(dataRefList[0], profiles, [dataRef.dataId["visit"] for dataRef in dataRefList],
                   [dataRef.dataId["visit"] for dataRef in normRefList])

        return profiles

    def processExposure(self, dataRef: ButlerDataRef) -> Struct:
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
            Struct with ``exposureList``, ``detectorMapList``, and
            ``pfsConfig``.
        """
        require = ("calexp", "detectorMap_used")
        if all(dataRef.datasetExists(name) for name in require):
            self.log.info("Reading existing data for %s", dataRef.dataId)
            return Struct(
                exposureList=[dataRef.get("calexp")],
                detectorMapList=[dataRef.get("detectorMap_used")],
                pfsConfig=dataRef.get("pfsConfig"),
            )
        return self.reduceExposure.runDataRef([dataRef])

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

        dataRef.put(profiles, "fiberProfiles", visit0=visit0)

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
