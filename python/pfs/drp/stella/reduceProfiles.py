from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np

from lsst.daf.persistence import ButlerDataRef
from lsst.pex.config import Config, Field, ConfigurableField, ConfigField, ListField, ChoiceField
from lsst.pipe.base import CmdLineTask, TaskRunner, ArgumentParser, Struct
from lsst.afw.image import Exposure, makeExposure, MaskedImage, Image, VisitInfo
from lsst.afw.math import StatisticsControl, stringToStatisticsProperty, statisticsStack
from lsst.geom import Box2I
from lsst.meas.algorithms.subtractBackground import SubtractBackgroundTask

from pfs.datamodel import FiberStatus, TargetType, CalibIdentity, PfsConfig
from .adjustDetectorMap import AdjustDetectorMapTask
from .buildFiberProfiles import BuildFiberProfilesTask
from .reduceExposure import ReduceExposureTask
from .centroidTraces import CentroidTracesTask, tracesToLines
from .constructSpectralCalibs import setCalibHeader
from .blackSpotCorrection import BlackSpotCorrectionTask
from .images import getIndices
from . import DetectorMap, FiberProfileSet


ButlerRefsDict = Dict[int, Dict[str, ButlerDataRef]]

def getDataRefs(
    refList: List[ButlerDataRef], checkRefs: Optional[ButlerRefsDict] = None, kind: Optional[str] = ""
) -> ButlerRefsDict:
    """Extract data reference lists indexed by spectrograph and arm

    Parameters
    ----------
    refList : list of `lsst.daf.persistence.ButlerDataRef`
        List of data references.
    checkRefs : `ButlerRefsDict`, optional
        Data references to check against. If the spectrograph and arm are not
        present in ``checkRefs``, an exception is raised.
    kind : `str`, optional
        Kind of data reference (e.g., "norm", "dark", "background").

    Returns
    -------
    dataRefs : `ButlerRefsDict`
        Data references indexed by spectrograph and arm.
    """
    result = defaultdict(lambda: defaultdict(list))
    for ref in refList:
        spectrograph = ref.dataId["spectrograph"]
        arm = ref.dataId["arm"]
        if checkRefs and (spectrograph not in checkRefs or arm not in checkRefs[spectrograph]):
            raise RuntimeError(f"No profile inputs provided for {kind} {ref.dataId}")
        result[spectrograph][arm].append(ref)
    return result


class ReduceProfilesTaskRunner(TaskRunner):
    """Split data by spectrograph arm and extract normalisation/dark visits"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        dataRefs = getDataRefs(parsedCmd.id.refList)
        normRefs = getDataRefs(parsedCmd.normId.refList, dataRefs, "norm")
        darkRefs = getDataRefs(parsedCmd.darkId.refList, dataRefs, "dark")

        targets = []
        for spectrograph in dataRefs:
            for arm in dataRefs[spectrograph]:
                if spectrograph not in normRefs or arm not in normRefs[spectrograph]:
                    raise RuntimeError(f"No norm provided for spectrograph={spectrograph} arm={arm}")
                dataList = sorted(dataRefs[spectrograph][arm], key=lambda ref: ref.dataId["visit"])
                normList = sorted(normRefs[spectrograph][arm], key=lambda ref: ref.dataId["visit"])
                darkList = sorted(darkRefs[spectrograph][arm], key=lambda ref: ref.dataId["visit"])
                targetKwargs = dict(
                    normRefList=normList,
                    darkRefList=darkList,
                    bgRefList=[],
                    bgDarkRefList=[],
                    **kwargs,
                )
                targets.append((dataList, targetKwargs))
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
    doAdjustDetectorMap = Field(dtype=bool, default=False, doc="Adjust detectorMap using trace positions?")
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
#            84,  # One of three in a row with bad cobras
#            85,  # One of three in a row with bad cobras
#            86,  # One of three in a row with bad cobras
#            114,  # Broken fiber; always blank
        ],
        doc="Replace the profiles for these fibers",
    )
    replaceNearest = Field(dtype=int, default=2, doc="Replace profiles with average of this many near fibers")
    darkFiberWidth = Field(dtype=int, default=3, doc="Width around fibers to measure dark subtraction")
    bgMaskWidth = Field(dtype=int, default=50, doc="Width of mask around fibers for background estimation")
    background = ConfigurableField(target=SubtractBackgroundTask, doc="Background estimation")

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
        self.background.binSize = 128
        self.background.statisticsProperty = "MEDIAN"
        self.background.ignoredPixelMask = ["BAD", "SAT", "INTRP", "CR", "NO_DATA", "FIBERTRACE"]


def getIlluminatedFibers(pfsConfig: PfsConfig, actual=True) -> np.ndarray:
    """Return the list of illuminated fibers

    Parameters
    ----------
    pfsConfig : `pfs.datamodel.PfsConfig`
        Fiber configuration.
    actual : `bool`, optional
        Use actual fiber positions, rather than nominal/intended?

    Returns
    -------
    fiberId : `numpy.ndarray` of `int`
        List of fiber identifiers.
    """
    position = pfsConfig.pfiCenter if actual else pfsConfig.pfiNominal
    select = np.logical_and.reduce(np.isfinite(position), axis=1)
    return pfsConfig.fiberId[select]


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
        self.makeSubtask("background")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="raw",
                               help="identifiers for profile, e.g., --id visit=123^456 ccd=7")
        parser.add_id_argument("--normId", datasetType="raw",
                               help="identifiers for normalisation, e.g., --id visit=98 ccd=7")
        parser.add_id_argument("--darkId", datasetType="raw",
                               help="identifiers for dot-roach darks, e.g., --id visit=56 ccd=7")
        parser.add_id_argument("--bgId", datasetType="raw",
                               help="identifiers for background images, e.g., --id visit=135 ccd=7")
        parser.add_id_argument("--bgDarkId", datasetType="raw",
                               help="identifiers for background dot-roach darks, e.g., --id visit=35 ccd=7")
        return parser

    def runDataRef(
        self,
        dataRefList: Iterable[ButlerDataRef],
        normRefList: Iterable[ButlerDataRef],
        darkRefList: Iterable[ButlerDataRef],
        bgRefList: Iterable[ButlerDataRef],
        bgDarkRefList: Iterable[ButlerDataRef],
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
        bgRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for background exposures.
        bgDarkRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for background dark exposures.
        """
        if not dataRefList:
            raise RuntimeError("No input exposures")
        if not normRefList:
            raise RuntimeError("No normalisation exposures")

        # Create the background image
        # The background data have their own dot-roach darks
        bgImage = None
        if bgRefList:
            bg = self.processCombined(bgRefList)
            bgFiberId = set(getIlluminatedFibers(bg.pfsConfig))
            if bgDarkRefList:
                bgDark = self.processCombined(bgDarkRefList)
                bgTime = bg.exposure.getInfo().getVisitInfo().getExposureTime()
                bgDarkTime = bgDark.exposure.getInfo().getVisitInfo().getExposureTime()
                bg.exposure.maskedImage.scaledMinus(bgTime/bgDarkTime, bgDark.exposure.maskedImage)
                bgDarkFiberId = getIlluminatedFibers(bgDark.pfsConfig)
                bgFiberId.difference_update(bgDarkFiberId)
                # Mask out the bright fibers
                bgDarkMask = self.maskFibers(
                    bgDark.exposure.getBBox(), bgDark.detectorMap, bgDarkFiberId, self.config.darkFiberWidth
                )
                bg.exposure.mask.array[bgDarkMask] |= bgDark.exposure.mask.getPlaneBitMask("FIBERTRACE")

            bgImage = self.makeBackground(bg.exposure, bg.detectorMap, bgFiberId)

        darkList = [self.processDark(ref, bgImage) for ref in darkRefList]
        dataList = [self.processExposure(ref, bgImage, darkList) for ref in dataRefList]

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

        norm = self.processCombined(normRefList, bgImage, darkList)

        spectra = profiles.extractSpectra(
            norm.exposure.maskedImage, norm.detectorMap, norm.exposure.mask.getPlaneBitMask(self.config.mask)
        )
        self.blackspots.run(norm.pfsConfig, spectra)

        for ss in spectra:
            good = (ss.mask.array[0] & ss.mask.getPlaneBitMask("NO_DATA")) == 0
            profiles[ss.fiberId].norm = np.where(good, ss.flux/ss.norm, np.nan)

        self.write(dataRefList[0], profiles, [dataRef.dataId["visit"] for dataRef in dataRefList],
                   [dataRef.dataId["visit"] for dataRef in normRefList])

        return profiles

    def processExposure(
            self,
            dataRef: ButlerDataRef,
            bgImage: Optional[Image] = None,
            darkList: Optional[List[Struct]] = None,
        ) -> Struct:
        """Process an exposure

        We read existing data from the butler, if available. Otherwise, we
        construct it by running ``reduceExposure``.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
        bgImage : `lsst.afw.image.Image`
            Background image.
        darkList : list of `lsst.pipe.base.Struct`
            List of dark processing results.

        Returns
        -------
        data : `lsst.pipe.base.Struct`
            Struct with ``exposure``, ``detectorMap``, ``pfsConfig``.
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

        data.pfsConfig = data.pfsConfig.select(spectrograph=dataRef.dataId["spectrograph"])

        if not bgImage:
            self.log.warn("No background image provided; not performing background subtraction")
        else:
            fiberId = getIlluminatedFibers(data.pfsConfig)
            bgScale = self.getBackgroundScaling(data.exposure.getInfo().getVisitInfo(), fiberId)
            data.exposure.image.scaledMinus(bgScale, bgImage)

        self.subtractDarks(data.exposure, data.detectorMap, darkList)

        return data

    def subtractDarks(
        self,
        exposure: Exposure,
        detectorMap: DetectorMap,
        darkList: List[Struct]
     ) -> None:
        if not darkList:
            self.log.warn("No darks provided; not performing dark subtraction")
            return

        fiberId = set()
        for dark in darkList:
            fiberId.update(dark.fiberId)
        fiberId.intersection_update(detectorMap.fiberId)

        select = self.maskFibers(exposure.getBBox(), detectorMap, fiberId, self.config.darkFiberWidth)
        dataArray = exposure.image.array[select]
        darkArrays = [dark.exposure.image.array[select] for dark in darkList]

        numDarks = len(darkList)
        matrix = np.zeros((numDarks, numDarks), dtype=float)
        vector = np.zeros(numDarks, dtype=float)
        for ii, iDark in enumerate(darkArrays):
            vector[ii] = np.sum(iDark*dataArray)
            matrix[ii, ii] = np.sum(iDark*iDark)
            for jj, jDark in enumerate(darkArrays[ii + 1:], ii + 1):
                value = np.sum(iDark*jDark)
                matrix[ii, jj] = value
                matrix[jj, ii] = value

        coeffs = np.linalg.solve(matrix, vector)
        self.log.info("Dark coefficients: %s", coeffs)
        for value, dark in zip(coeffs, darkList):
            exposure.maskedImage.scaledMinus(value, dark.exposure.maskedImage)

    def processDark(self, dataRef: ButlerDataRef, bgImage: Optional[Image] = None) -> Struct:
        """Process a dot-roach dark exposure

        We perform ISR only.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
        bgImage : `lsst.afw.image.Image`
            Background image.

        Returns
        -------
        data : `lsst.pipe.base.Struct`
            Struct with ``exposure`` and ``dither``.
        """
        dataset = "postISRCCD"
        if dataRef.datasetExists(dataset):
            self.log.info("Reading existing data for %s", dataRef.dataId)
            exposure = dataRef.get(dataset)
        else:
            exposure = self.reduceExposure.isr.runDataRef(dataRef).exposure

        pfsConfig = dataRef.get("pfsConfig").select(spectrograph=dataRef.dataId["spectrograph"])
        fiberId = getIlluminatedFibers(pfsConfig)

        if bgImage:
            bgScale = self.getBackgroundScaling(exposure.getInfo().getVisitInfo(), fiberId)
            exposure.image.scaledMinus(bgScale, bgImage)

        return Struct(
            exposure=exposure, pfsConfig=pfsConfig, fiberId=fiberId
        )

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

    def processCombined(
            self,
            dataRefList: List[ButlerDataRef],
            bgImage: Optional[Image] = None,
            darkList: Optional[List[Struct]] = None,
        ) -> Struct:
        """Process a combined exposure

        Combines the input exposures, and optionally adjusts the detectorMap.

        Parameters
        ----------
        dataRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for exposures.
        bgImage : `lsst.afw.image.Image`
            Background image.
        darkList : list of `lsst.pipe.base.Struct`
            List of dark processing results.

        Returns
        -------
        data : `lsst.pipe.base.Struct`
            Struct with ``exposure``, ``detectorMap``, ``pfsConfig``.
        """
        dataList = [self.processExposure(ref, bgImage, darkList) for ref in dataRefList]
        combined = self.combine([data.exposure for data in dataList])

        if self.config.doAdjustDetectorMap:
            detectorMap = dataList[0].detectorMap
            arm = dataRefList[0].dataId["arm"]
            traces = self.centroidTraces.run(combined, detectorMap, dataList[0].pfsConfig)
            lines = tracesToLines(detectorMap, traces, self.config.traceSpectralError)
            detectorMap = self.adjustDetectorMap.run(
                detectorMap, lines, arm, combined.visitInfo.id
            ).detectorMap
        else:
            detectorMap = dataList[0].detectorMap

        return Struct(exposure=combined, detectorMap=detectorMap, pfsConfig=dataList[0].pfsConfig)

    def maskFibers(
        self, bbox: Box2I, detectorMap: DetectorMap, fiberId: Iterable[int], radius: int
    ) -> np.ndarray:
        """Mask fibers in an image

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of image.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Detector map.
        fiberId : iterable of `int`
            Fiber identifiers.
        radius : `int`
            Radius of mask.

        Returns
        -------
        mask : `numpy.ndarray` of `bool`
            Mask: ``True`` means the pixel is near a fiber.
        """
        mask = np.zeros((bbox.getHeight(), bbox.getWidth()), dtype=bool)
        xx, _ = getIndices(bbox, int)
        for ff in fiberId:
            xCenter = detectorMap.getXCenter(ff)
            mask[np.abs(xx - xCenter[:, np.newaxis]) < radius] = True
        return mask

    def makeBackground(
            self, exp: Exposure, detectorMap: DetectorMap, fiberId: List[int]
        ) -> Image:
        """Make the background image

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Detector map.
        fiberId : list of `int`
            Fiber identifiers of illuminated fibers.

        Returns
        -------
        background : `lsst.afw.image.Image`
            Background image.
        """
        mask = self.maskFibers(exp.getBBox(), detectorMap, fiberId, self.config.bgMaskWidth)
        exp.mask.array[mask] |= exp.mask.getPlaneBitMask("FIBERTRACE")

        bgModel = self.background.run(exp).background
        bgImage = bgModel.getImage()
        bgImage /= self.getBackgroundScaling(exp.getInfo().getVisitInfo(), fiberId)
        return bgImage

    def getBackgroundScaling(self, visitInfo: VisitInfo, fiberId: List[int]) -> float:
        """Return the scaling for the background

        The background per unit exposure time appears to vary with the number
        of fibers illuminated.

        Parameters
        ----------
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information.
        fiberId : list of `int`
            Fiber identifiers of illuminated fibers.

        Returns
        -------
        scaling : `float`
            Scaling factor for background.
        """
        expTime = visitInfo.getExposureTime()
        numFibers = len(fiberId)
        return expTime*numFibers

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
