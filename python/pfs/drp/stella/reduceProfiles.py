from functools import partial
from typing import Dict, Iterable, List

import numpy as np

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from lsst.pex.config import Field, ConfigurableField, ListField
from lsst.pipe.base import Struct
from lsst.afw.image import Exposure
from lsst.geom import Box2I

from pfs.datamodel import FiberStatus, TargetType, CalibIdentity, PfsConfig
from .buildFiberProfiles import BuildFiberProfilesTask
from .calibs import setCalibHeader
from .images import getIndices
from . import DetectorMap
from .pipelines.measureCentroids import (
    MeasureDetectorMapTask, MeasureDetectorMapConfig, MeasureDetectorMapConnections
)

__all__ = (
    "ProfilesFitDetectorMapConfig",
    "ProfilesFitDetectorMapTask",
    "ReduceProfilesConfig",
    "ReduceProfilesTask",
)


class ProfilesFitDetectorMapConnections(
    MeasureDetectorMapConnections,
    dimensions=("instrument", "exposure", "arm", "spectrograph", "profiles_group"),
):
    """Connections for ProfilesFitDetectorMapTask"""
    data = PrerequisiteConnection(
        name="profiles_exposures",
        doc="List of bright and dark exposures",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "profiles_group"),
    )


class ProfilesFitDetectorMapConfig(
    MeasureDetectorMapConfig, pipelineConnections=ProfilesFitDetectorMapConnections
):
    pass


class ProfilesFitDetectorMapTask(MeasureDetectorMapTask):
    ConfigClass = ProfilesFitDetectorMapConfig

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
        data = butler.get(inputRefs.data)
        del inputRefs.data

        if inputRefs.exposure.dataId["exposure"] in data["dark"]:
            return

        super().runQuantum(butler, inputRefs, outputRefs)


class ReduceProfilesConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "profiles_group", "arm", "spectrograph"),
):
    """Connections for ReduceProfilesTask"""
    exposure = InputConnection(
        name="postISRCCD",
        doc="ISR-processed exposure",
        storageClass="Exposure",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
        multiple=True,
    )
    data = PrerequisiteConnection(
        name="profiles_exposures",
        doc="List of bright and dark exposures",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "profiles_group"),
    )
    detectorMap = InputConnection(
        name="detectorMap",
        doc="Detector map",
        storageClass="DetectorMap",
        dimensions=("instrument", "exposure", "arm", "spectrograph"),
        multiple=True,
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "exposure"),
        multiple=True,
    )

    fiberProfiles = OutputConnection(
        name="fiberProfiles_group",
        doc="Fiber profiles for group",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "profiles_group", "arm", "spectrograph"),
    )


class ReduceProfilesConfig(PipelineTaskConfig, pipelineConnections=ReduceProfilesConnections):
    """Configuration for FiberTrace construction"""
    profiles = ConfigurableField(target=BuildFiberProfilesTask, doc="Build fiber profiles")
    fiberStatus = ListField(
        dtype=str,
        default=["GOOD", "BROKENFIBER", "BLACKSPOT", "BROKENCOBRA"],
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
        Use actual fiber positions, rather than nominal/intended and
        fiberStatus/targetType?

    Returns
    -------
    fiberId : `numpy.ndarray` of `int`
        List of fiber identifiers.
    """
    position = pfsConfig.pfiCenter if actual else pfsConfig.pfiNominal
    select = np.logical_and.reduce(np.isfinite(position), axis=1)
    if not actual:
        select &= (pfsConfig.fiberStatus != FiberStatus.BLACKSPOT)
        select &= (pfsConfig.targetType != TargetType.BLACKSPOT)
    return pfsConfig.fiberId[select]


class ReduceProfilesTask(PipelineTask):
    """Task to construct the fiber trace"""
    ConfigClass = ReduceProfilesConfig
    _DefaultName = "reduceProfiles"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("profiles")

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
        data = butler.get(inputRefs.data)
        brightIds = set(data["bright"])
        darkIds = set(data["dark"])

        brightList = {}
        darkList = {}
        for ref in inputRefs.exposure:
            expId = ref.dataId["exposure"]
            if expId in darkIds:
                darkList[expId] = butler.get(ref)
                darkIds.remove(expId)
            elif expId in brightIds:
                brightList[expId] = butler.get(ref)
                brightIds.remove(expId)
            else:
                raise RuntimeError(f"Exposure {expId} not in bright or dark list")
        if brightIds or darkIds:
            raise RuntimeError(f"Unmatched bright or dark exposures: bright={brightIds}, dark={darkIds}")
        pfsConfigList = {ref.dataId["exposure"]: butler.get(ref) for ref in inputRefs.pfsConfig}
        detectorMapList = {ref.dataId["exposure"]: butler.get(ref) for ref in inputRefs.detectorMap}

        expId = min(brightList.keys())

        identity = CalibIdentity(
            visit0=expId,
            arm=outputRefs.fiberProfiles.dataId["arm"],
            spectrograph=outputRefs.fiberProfiles.dataId["spectrograph"],
            obsDate=brightList[expId].visitInfo.getDate().toPython().isoformat(),
        )

        outputs = self.run(
            identity=identity,
            brightList=brightList,
            darkList=darkList,
            pfsConfigList=pfsConfigList,
            detectorMapList=detectorMapList,
        )
        butler.put(outputs, outputRefs)

    def run(
        self,
        identity: CalibIdentity,
        brightList: Dict[int, Exposure],
        darkList: Dict[int, Exposure],
        pfsConfigList: Dict[int, PfsConfig],
        detectorMapList: Dict[int, DetectorMap],
    ) -> Struct:
        """Construct the ``fiberProfiles`` calib

        Parameters
        ----------
        identity : `pfs.datamodel.CalibIdentity`
            Identity for the profiles.
        brightList : `dict` [`int`, `lsst.afw.image.Exposure`]
            Bright exposures (fibers of interest exposed), indexed by expId.
        darkList : `dict` [`int`, `lsst.afw.image.Exposure`]
            Dark exposures (all fibers hidden), indexed by expId.
        pfsConfigList : `dict` [`int`, `pfs.datamodel.PfsConfig`]
            Top-end fiber configurations, indexed by expId.
        detectorMapList : `dict` [`int`, `pfs.drp.stella.DetectorMap`]
            Detector maps, indexed by expId.

        Returns
        -------
        outputs : `lsst.pipe.base.Struct`
            Struct with the fiber profiles.
        """
        for expId in pfsConfigList:
            pfsConfigList[expId] = pfsConfigList[expId].select(spectrograph=identity.spectrograph)

        # Get the list of fiberIds that are illuminated in the darks
        darkFibers = set()
        for expId in darkList:
            pfsConfig = pfsConfigList[expId]
            darkFibers.update(getIlluminatedFibers(pfsConfig))

        for expId in brightList:
            self.subtractDarks(brightList[expId], detectorMapList[expId], darkList.values(), darkFibers)

        # Get the union of all available fiberIds
        fibers = set()
        fiberStatus = [FiberStatus.fromString(fs) for fs in self.config.fiberStatus]
        targetType = [TargetType.fromString(tt) for tt in self.config.targetType]
        for expId in brightList:
            pfsConfig = pfsConfigList[expId]
            fibers.update(pfsConfig.select(fiberStatus=fiberStatus, targetType=targetType).fiberId)

        if darkList:
            # Select only fibers that are not hidden (i.e., finite pfiCenter)
            exposedFibers = set()
            for expId in brightList:
                exposedFibers.update(getIlluminatedFibers(pfsConfigList[expId], actual=True))
            fibers.intersection_update(exposedFibers)

            # Remove fibers that are in the darks
            fibers.difference_update(darkFibers)

        fiberId = np.array(list(sorted(fibers)))

        pfsConfigList = {
            expId: pfsConfig.select(fiberId=fiberId) for expId, pfsConfig in pfsConfigList.items()
        }

        result = self.profiles.runMultiple(
            list(brightList.values()),
            identity,
            [detectorMapList[expId] for expId in brightList],
            [pfsConfigList[expId].select(fiberId=fiberId) for expId in brightList],
        )
        profiles = result.profiles
        profiles.replaceFibers(self.config.replaceFibers, self.config.replaceNearest)

        header = profiles.metadata
        setCalibHeader(header, "fiberProfiles", brightList, identity.toDict())
        # Clobber any existing CALIB_DARK_*
        names = list(header.keys())
        for key in names:
            if key.startswith("CALIB_DARK_"):
                header.remove(key)
        # Set new CALIB_DARK_*
        for ii, vv in enumerate(sorted(set(darkList.keys()))):
            header[f"CALIB_DARK_{ii}"] = vv

        return Struct(fiberProfiles=profiles)

    def subtractDarks(
        self,
        exposure: Exposure,
        detectorMap: DetectorMap,
        darkList: List[Exposure],
        fiberId: Iterable[int],
    ) -> None:
        """Subtract darks from an exposure

        We subtract a linear combination of darks from the exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure from which to subtract darks (modified).
        detectorMap : `pfs.drp.stella.DetectorMap`
            Detector map.
        darkList : list of `lsst.afw.image.Exposure`
            List of dark exposures.
        darkFibers : iterable of `int`
            Fiber identifiers that are illuminated in the darks.
        """
        if not darkList:
            self.log.warn("No darks provided; not performing dark subtraction")
            return

        select = self.maskFibers(exposure.getBBox(), detectorMap, fiberId, self.config.darkFiberWidth)
        select &= (exposure.mask.array & exposure.mask.getPlaneBitMask(self.config.darkMask)) == 0
        select &= np.isfinite(exposure.image.array)
        for dark in darkList:
            badBitMask = dark.mask.getPlaneBitMask(self.config.darkMask)
            select &= (dark.mask.array & badBitMask) == 0
            select &= np.isfinite(dark.image.array)

        dataArray = exposure.image.array[select]
        darkArrays = [dark.image.array[select] for dark in darkList]

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
            exposure.maskedImage.scaledMinus(value, dark.maskedImage)

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
