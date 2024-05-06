from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np

from lsst.daf.persistence import ButlerDataRef
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.pipe.base import CmdLineTask, TaskRunner, ArgumentParser, Struct
from lsst.afw.image import Exposure
from lsst.geom import Box2I

from pfs.datamodel import FiberStatus, TargetType, CalibIdentity, PfsConfig
from .buildFiberProfiles import BuildFiberProfilesTask
from .reduceExposure import ReduceExposureTask
from .images import getIndices
from .normalizeFiberProfiles import NormalizeFiberProfilesTask
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
                dataList = sorted(dataRefs[spectrograph][arm], key=lambda ref: ref.dataId["visit"])
                normList = sorted(normRefs[spectrograph][arm], key=lambda ref: ref.dataId["visit"])
                if spectrograph not in normRefs or arm not in normRefs[spectrograph]:
                    raise RuntimeError(f"No norm provided for spectrograph={spectrograph} arm={arm}")
                darkList = sorted(darkRefs[spectrograph][arm], key=lambda ref: ref.dataId["visit"])
                targetKwargs = dict(
                    normRefList=normList,
                    darkRefList=darkList,
                    **kwargs,
                )
                targets.append((dataList, targetKwargs))
        return targets


class ReduceProfilesConfig(Config):
    """Configuration for FiberTrace construction"""
    reduceExposure = ConfigurableField(target=ReduceExposureTask, doc="Reduce single exposure")
    profiles = ConfigurableField(target=BuildFiberProfilesTask, doc="Build fiber profiles")
    normalize = ConfigurableField(target=NormalizeFiberProfilesTask, doc="Normalize fiber profiles")
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
    replaceFibers = ListField(dtype=int, default=[], doc="Replace the profiles for these fibers")
    replaceNearest = Field(dtype=int, default=2, doc="Replace profiles with average of this many near fibers")
    darkFiberWidth = Field(dtype=int, default=3, doc="Width around fibers to measure dark subtraction")
    darkMask = ListField(
        dtype=str,
        default=["BAD", "SAT", "CR", "NO_DATA"],
        doc="Mask planes to ignore when subtracting darks",
    )

    def setDefaults(self):
        super().setDefaults()
        for base in (self.reduceExposure, self.normalize.reduceExposure):
            base.doApplyFiberNorms = False
            base.doMeasureLines = False
            base.doMeasurePsf = False
            base.doSubtractSky2d = False
            base.doWriteArm = False
        self.reduceExposure.doExtractSpectra = False
        self.profiles.doBlindFind = False  # We have good detectorMaps and pfsConfig, so we know what's where
        self.profiles.profileOversample = 3
        self.profiles.profileRejIter = 0


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
        self.makeSubtask("normalize")

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

            # Remove fibers that are in the darks
            darkFibers = set()
            for dark in darkList:
                hasFiniteCenter = np.logical_and.reduce(np.isfinite(dark.pfsConfig.pfiCenter), axis=1)
                darkFibers.update(dark.pfsConfig.fiberId[hasFiniteCenter])
            fibers.difference_update(darkFibers)

        fiberId = np.array(list(sorted(fibers)))
        pfsConfigList = [pfsConfig.select(fiberId=fiberId) for pfsConfig in pfsConfigList]

        profiles = self.profiles.runMultiple(exposureList, identity, detectorMapList, pfsConfigList).profiles
        profiles.replaceFibers(self.config.replaceFibers, self.config.replaceNearest)

        dataVisits = [dataRef.dataId["visit"] for dataRef in dataRefList]
        if normRefList:
            self.normalize.run(profiles, normRefList, dataVisits)
        else:
            self.normalize.write(dataRefList[0], profiles, dataVisits, [])

        return profiles

    def processExposure(
            self,
            dataRef: ButlerDataRef,
            darkList: Optional[List[Struct]] = None,
    ) -> Struct:
        """Process an exposure

        We read existing data from the butler, if available. Otherwise, we
        construct it by running ``reduceExposure``.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
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

        self.subtractDarks(data.exposure, data.detectorMap, darkList)

        return data

    def subtractDarks(
        self,
        exposure: Exposure,
        detectorMap: DetectorMap,
        darkList: List[Struct],
    ) -> None:
        """Subtract darks from an exposure

        We subtract a linear combination of darks from the exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure from which to subtract darks (modified).
        detectorMap : `pfs.drp.stella.DetectorMap`
            Detector map.
        darkList : list of `lsst.pipe.base.Struct`
            List of dark processing results.
        """
        if not darkList:
            self.log.warn("No darks provided; not performing dark subtraction")
            return

        fiberId = set()
        for dark in darkList:
            fiberId.update(dark.fiberId)
        fiberId.intersection_update(detectorMap.fiberId)

        select = self.maskFibers(exposure.getBBox(), detectorMap, fiberId, self.config.darkFiberWidth)
        select &= (exposure.mask.array & exposure.mask.getPlaneBitMask(self.config.darkMask)) == 0
        select &= np.isfinite(exposure.image.array)
        for dark in darkList:
            badBitMask = dark.exposure.mask.getPlaneBitMask(self.config.darkMask)
            select &= (dark.exposure.mask.array & badBitMask) == 0
            select &= np.isfinite(dark.exposure.image.array)

        dataArray = exposure.image.array[select]
        darkArrays = [dark.exposure.image.array[select] for dark in darkList]

        numDarks = len(darkList)
        numParams = numDarks + 1  # darks plus background
        bgIndex = numDarks
        matrix = np.zeros((numParams, numParams), dtype=float)
        vector = np.zeros(numParams, dtype=float)
        for ii, iDark in enumerate(darkArrays):
            vector[ii] = np.sum(iDark*dataArray)
            matrix[ii, ii] = np.sum(iDark*iDark)
            vector[bgIndex] = np.sum(dataArray)
            matrix[bgIndex, bgIndex] = dataArray.size
            matrix[ii, bgIndex] = matrix[bgIndex, ii] = np.sum(iDark)
            for jj, jDark in enumerate(darkArrays[ii + 1:], ii + 1):
                value = np.sum(iDark*jDark)
                matrix[ii, jj] = value
                matrix[jj, ii] = value

        coeffs = np.linalg.solve(matrix, vector)
        self.log.info("Dark coefficients: %s", coeffs)
        for value, dark in zip(coeffs, darkList):
            exposure.maskedImage.scaledMinus(value, dark.exposure.maskedImage)

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
            self.log.info("Reading existing data for %s", dataRef.dataId)
            exposure = dataRef.get(dataset)
        else:
            exposure = self.reduceExposure.runIsr(dataRef)
        if self.reduceExposure.config.doRepair:
            self.reduceExposure.repairExposure(exposure)

        pfsConfig = dataRef.get("pfsConfig").select(spectrograph=dataRef.dataId["spectrograph"])
        fiberId = getIlluminatedFibers(pfsConfig)

        return Struct(
            exposure=exposure, pfsConfig=pfsConfig, fiberId=fiberId
        )

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

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
