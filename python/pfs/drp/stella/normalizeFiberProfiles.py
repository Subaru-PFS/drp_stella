from typing import Iterable, List

import numpy as np

from lsst.pex.config import Config, ConfigurableField, Field, ListField
from lsst.pipe.base import Task, Struct
from lsst.daf.persistence import ButlerDataRef

from .reduceExposure import ReduceExposureTask
from .combineImages import CombineImagesTask
from .adjustDetectorMap import AdjustDetectorMapTask
from .blackSpotCorrection import BlackSpotCorrectionTask
from .fiberProfileSet import FiberProfileSet
from .centroidTraces import CentroidTracesTask, tracesToLines
from .constructSpectralCalibs import setCalibHeader

__all__ = ("NormalizeFiberProfilesConfig", "NormalizeFiberProfilesTask")


class NormalizeFiberProfilesConfig(Config):
    """Configuration for normalizing fiber profiles"""
    reduceExposure = ConfigurableField(target=ReduceExposureTask, doc="Reduce single exposure")
    combine = ConfigurableField(target=CombineImagesTask, doc="CombineImages")
    doAdjustDetectorMap = Field(dtype=bool, default=False, doc="Adjust detectorMap using trace positions?")
    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Adjust detectorMap")
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")
    traceSpectralError = Field(dtype=float, default=1.0,
                               doc="Error in the spectral dimension to give trace centroids (pixels)")
    mask = ListField(dtype=str, default=["BAD_FLAT", "CR", "SAT", "NO_DATA"],
                     doc="Mask planes to exclude from fiberTrace")
    blackspots = ConfigurableField(target=BlackSpotCorrectionTask, doc="Black spot correction")

    def setDefaults(self):
        self.reduceExposure.doMeasureLines = False
        self.reduceExposure.doMeasurePsf = False
        self.reduceExposure.doSubtractSky2d = False
        self.reduceExposure.doExtractSpectra = False
        self.reduceExposure.doWriteArm = False
        self.adjustDetectorMap.minSignalToNoise = 0  # We don't measure S/N


class NormalizeFiberProfilesTask(Task):
    """Task to normalize fiber profiles"""
    ConfigClass = NormalizeFiberProfilesConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("reduceExposure")
        self.makeSubtask("combine")
        self.makeSubtask("centroidTraces")
        self.makeSubtask("adjustDetectorMap")
        self.makeSubtask("blackspots")

    def run(self, profiles: FiberProfileSet, normRefList: List[ButlerDataRef], visitList: List[int]):
        combined = self.makeCombinedExposure(normRefList)
        spectra = profiles.extractSpectra(
            combined.exposure.maskedImage,
            combined.detectorMap,
            combined.exposure.mask.getPlaneBitMask(self.config.mask),
        )
        self.blackspots.run(combined.pfsConfig, spectra)

        for ss in spectra:
            good = (ss.mask.array[0] & ss.mask.getPlaneBitMask("NO_DATA")) == 0
            profiles[ss.fiberId].norm = np.where(good, ss.flux, np.nan)

        self.write(normRefList[0], profiles, visitList, [dataRef.dataId["visit"] for dataRef in normRefList])

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
        return data

    def makeCombinedExposure(self, dataRefList: List[ButlerDataRef]) -> Struct:
        """Generate a combined exposure

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
        dataList = [self.processExposure(ref) for ref in dataRefList]
        combined = self.combine.run([data.exposure for data in dataList])

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

        outputId = dict(
            visit0=profiles.identity.visit0,
            calibDate=profiles.identity.obsDate.split("T")[0],
            calibTime=profiles.identity.obsDate,
            arm=profiles.identity.arm,
            spectrograph=profiles.identity.spectrograph,
            ccd=dataRef.dataId["ccd"],
            filter=profiles.identity.arm,
        )

        setCalibHeader(profiles.metadata, "fiberProfiles", dataVisits, outputId)
        for ii, vv in enumerate(sorted(set(normVisits))):
            profiles.metadata.set(f"CALIB_NORM_{ii}", vv)

        dataRef.put(profiles, "fiberProfiles", **outputId)
