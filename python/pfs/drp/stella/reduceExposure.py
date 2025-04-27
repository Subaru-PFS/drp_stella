#
# LSST Data Management System
# Copyright 2008-2016 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import numpy as np

from lsst.pex.config import Field, ConfigurableField, DictField, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import PrerequisiteInput as PrerequisiteConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.daf.butler import DataCoordinate
from lsst.afw.image import Exposure

from .datamodel.pfsConfig import PfsConfig
from .DetectorMapContinued import DetectorMap


from lsst.obs.pfs.utils import getLamps
from pfs.datamodel import FiberStatus, TargetType, Identity
from pfs.datamodel.pfsFiberNorms import PfsFiberNorms
from .extractSpectraTask import ExtractSpectraTask
from .lsf import GaussianLsf, LsfDict
from .readLineList import ReadLineListTask
from .centroidLines import CentroidLinesTask
from .photometerLines import PhotometerLinesTask
from .centroidTraces import CentroidTracesTask, tracesToLines
from .adjustDetectorMap import AdjustDetectorMapTask
from .blackSpotCorrection import BlackSpotCorrectionTask
from .arcLine import ArcLineSet
from .fiberProfile import FiberProfile
from .fiberProfileSet import FiberProfileSet
from .utils.sysUtils import metadataToHeader, getPfsVersions
from .screen import ScreenResponseTask
from .barycentricCorrection import calculateBarycentricCorrection
from .pipelines.lookups import lookupFiberNorms
from .fitFluxCal import applyFiberNorms
from .fitDistortedDetectorMap import FittingError

__all__ = ["ReduceExposureConfig", "ReduceExposureTask"]


class ReduceExposureConnections(
    PipelineTaskConnections, dimensions=("instrument", "visit", "arm", "spectrograph")
):
    exposure = InputConnection(
        name="calexp",
        doc="Exposure to reduce",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    pfsConfig = PrerequisiteConnection(
        name="pfsConfig",
        doc="Top-end fiber configuration",
        storageClass="PfsConfig",
        dimensions=("instrument", "visit"),
    )
    fiberProfiles = PrerequisiteConnection(
        name="fiberProfiles",
        doc="Profile of fibers",
        storageClass="FiberProfileSet",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )
    fiberNorms = PrerequisiteConnection(
        name="fiberNorms_calib",
        doc="Fiber normalisations",
        storageClass="PfsFiberNorms",
        dimensions=("instrument", "arm"),
        isCalibration=True,
        lookupFunction=lookupFiberNorms,
    )
    detectorMap = PrerequisiteConnection(
        name="detectorMap_calib",
        doc="Mapping from fiberId,wavelength to x,y",
        storageClass="DetectorMap",
        dimensions=("instrument", "arm", "spectrograph"),
        isCalibration=True,
    )

    pfsArm = OutputConnection(
        name="pfsArm",
        doc="Extracted spectra from arm",
        storageClass="PfsArm",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    lsf = OutputConnection(
        name="pfsArmLsf",
        doc="1D line-spread function",
        storageClass="LsfDict",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    lines = OutputConnection(
        name="lines",
        doc="Emission line measurements",
        storageClass="ArcLineSet",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    apCorr = OutputConnection(
        name="apCorr",
        doc="Aperture correction for line photometry",
        storageClass="FocalPlaneFunction",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    detectorMapUsed = OutputConnection(
        name="detectorMap",
        doc="DetectorMap used for extraction",
        storageClass="DetectorMap",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config:
            return
        if self.config.doBoxcarExtraction:
            self.prerequisiteInputs.remove("fiberProfiles")
        if not self.config.doApplyFiberNorms:
            self.prerequisiteInputs.remove("fiberNorms")


class ReduceExposureConfig(PipelineTaskConfig, pipelineConnections=ReduceExposureConnections):
    """Config for ReduceExposure"""
    doAdjustDetectorMap = Field(dtype=bool, default=True,
                                doc="Apply a low-order correction to the detectorMap?")
    readLineList = ConfigurableField(target=ReadLineListTask,
                                     doc="Read line lists for detectorMap adjustment")
    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Measure slit offsets")
    requireAdjustDetectorMap = Field(dtype=bool, default=False,
                                     doc="Require detectorMap adjustment to succeed?")
    centroidLines = ConfigurableField(target=CentroidLinesTask, doc="Centroid lines")
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")
    traceSpectralError = Field(dtype=float, default=5.0,
                               doc="Error in the spectral dimension to give trace centroids (pixels)")
    doForceTraces = Field(dtype=bool, default=True, doc="Force use of traces for non-continuum data?")
    doPhotometerLines = Field(dtype=bool, default=True, doc="Measure photometry for lines?")
    photometerLines = ConfigurableField(target=PhotometerLinesTask, doc="Photometer lines")
    doBoxcarExtraction = Field(dtype=bool, default=False, doc="Extract with a boxcar of width boxcarWidth")
    boxcarWidth = Field(dtype=float, default=5,
                        doc="Extract with a boxcar of width boxcarWidth if doBoxcarExtraction is True")
    doDetectIIS = Field(dtype=bool, default=True,
                        doc="If ~ENGINEERING fibres is requested but the IIS is illuminated, "
                        "use a boxcar of boxcarWidth to extract the ENGINEERING fibres")
    doBoxcarForIIS = Field(dtype=bool, default=True,
                           doc="Enable boxcar extractions when ENGINEERING fibres are illuminated?")
    gaussianLsfWidth = DictField(keytype=str, itemtype=float,
                                 doc="Gaussian sigma (nm) for LSF as a function of the spectrograph arm",
                                 default=dict(b=0.081, r=0.109, m=0.059, n=0.109))
    extractSpectra = ConfigurableField(target=ExtractSpectraTask, doc="Extract spectra from exposure")
    doApplyScreenResponse = Field(dtype=bool, default=True, doc="Apply screen response correction to quartz?")
    screen = ConfigurableField(target=ScreenResponseTask, doc="Screen response correction")
    targetType = ListField(dtype=str, default=["^ENGINEERING"],
                           doc="Target type for which to extract spectra")
    doBarycentricCorrection = Field(dtype=bool, default=True, doc="Calculate barycentric correction?")
    doBlackSpotCorrection = Field(dtype=bool, default=True, doc="Correct for black spot penumbra?")
    blackSpotCorrection = ConfigurableField(target=BlackSpotCorrectionTask, doc="Black spot correction")
    spatialOffset = Field(dtype=float, default=0.0, doc="Spatial offset to add")
    spectralOffset = Field(dtype=float, default=0.0, doc="Spectral offset to add")
    doApplyFiberNorms = Field(dtype=bool, default=True, doc="Apply fiber norms to extracted spectra?")
    doCheckFiberNormsHashes = Field(dtype=bool, default=True, doc="Check hashes in fiberNorms?")


class ReduceExposureTask(PipelineTask):
    r"""!Reduce a PFS exposures, generating pfsArm files

    @anchor ReduceExposureTask_

    @section drp_stella_reduceExposure_Contents  Contents

     - @ref drp_stella_reduceExposure_Purpose
     - @ref drp_stella_reduceExposure_Initialize
     - @ref drp_stella_reduceExposure_IO
     - @ref drp_stella_reduceExposure_Config
     - @ref drp_stella_reduceExposure_Debug
     - @ref drp_stella_reduceExposure_Example

    @section drp_stella_reduceExposure_Purpose  Description

    Perform the following operations:
    - Call extractSpectra to extract the spectra
    and then apply the wavelength solution from the DetectorMap

    @section drp_stella_reduceExposure_Initialize  Task initialisation

    @copydoc \_\_init\_\_

    @section drp_stella_reduceExposure_IO  Invoking the Task

    This task is primarily designed to be run from the command line.

    The main method is `run`, which takes a single butler data reference for the raw input data.

    @section drp_stella_reduceExposure_Config  Configuration parameters

    See @ref ReduceExposureConfig

    @section drp_stella_reduceExposure_Debug  Debug variables

    ReduceExposureTask has no debug output, but its subtasks do.

    @section drp_stella_reduceExposure_Example   A complete example of using ReduceExposureTask

    The following commands will process all raw data in obs_test's data repository.
    Note: be sure to specify an `--output` that does not already exist:

        setup obs_test
        setup pipe_tasks
        processCcd.py $DRP_STELLA_TEST_DIR --rerun you/tmp -c doWriteCalexp=True --id visit=5699

    The data is read from the small repository in the `drp_stella_test` package and written to `rerun/you/tmp`
    """
    ConfigClass = ReduceExposureConfig
    _DefaultName = "reduceExposure"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")
        self.makeSubtask("centroidLines")
        self.makeSubtask("centroidTraces")
        self.makeSubtask("photometerLines")
        self.makeSubtask("adjustDetectorMap")
        self.makeSubtask("extractSpectra")
        self.makeSubtask("screen")
        self.makeSubtask("blackSpotCorrection")

    def runQuantum(
        self,
        butler: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection
    ) -> None:
        inputs = butler.get(inputRefs)
        dataId = inputRefs.exposure.dataId
        if self.config.doBoxcarExtraction:
            inputs["fiberProfiles"] = None
            inputs["fiberNorms"] = None
        if not self.config.doApplyFiberNorms:
            inputs["fiberNorms"] = None
        outputs = self.run(**inputs, dataId=dataId)
        if outputs.apCorr is None:  # e.g., for a quartz
            del outputRefs.apCorr
        butler.put(outputs, outputRefs)
        return outputs

    def run(
        self,
        exposure: Exposure,
        pfsConfig: PfsConfig,
        fiberProfiles: FiberProfileSet | None,
        fiberNorms: PfsFiberNorms | None,
        detectorMap: DetectorMap,
        dataId: dict[str, str] | DataCoordinate,
    ) -> Struct:
        """Process an arm exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure data to reduce.
        pfsConfig : `pfs.datamodel.PfsConfig`
            PFS fiber configuration.
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profiles of fibers.
        fiberNorms : `pfs.drp.stella.PfsFiberNorms`
            Normalization of fibers.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        dataId : `dict` [`str`, `str`] or `DataCoordinate`
            Data identifier.

        Returns
        -------
        pfsArm : `pfs.drp.stella.datamodel.PfsArm`
            Extracted spectra.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        detectorMapUsed : `pfs.drp.stella.DetectorMap`
            Mapping of wl,fiber to detector position; used for extraction.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured lines.
        apCorr : `pfs.drp.stella.FocalPlaneFunction`
            Measured aperture correction.
        lsf : LSF
            Line-spread function.
        """

        arm = dataId["arm"]
        spectrograph = dataId["spectrograph"]

        spatialOffset = self.config.spatialOffset
        spectralOffset = self.config.spectralOffset
        if spatialOffset != 0.0 or spectralOffset != 0.0:
            self.log.info("Adjusting detectorMap slit offset by %f,%f", spatialOffset, spectralOffset)
            detectorMap.applySlitOffset(spatialOffset, spectralOffset)

        check = self.checkPfsConfig(pfsConfig, detectorMap, spectrograph)
        pfsConfig = check.pfsConfig
        boxcarWidth = check.boxcarWidth

        if boxcarWidth > 0:
            fiberProfiles = FiberProfileSet.makeEmpty(None)
            for fid in pfsConfig.fiberId:
                # the Gaussian will be replaced by a boxcar, so params don't matter
                fiberProfiles[fid] = FiberProfile.makeGaussian(1, exposure.getHeight(), 5, 1)

        measurements = self.measure(exposure, pfsConfig, fiberProfiles, detectorMap, boxcarWidth, arm)

        lsf = self.defaultLsf(arm, pfsConfig.fiberId, detectorMap)

        fiberId = np.array(sorted(set(pfsConfig.fiberId) & set(detectorMap.fiberId)))
        spectra = self.extractSpectra.run(
            exposure.maskedImage,
            measurements.fiberTraces,
            measurements.detectorMap,
            fiberId,
            True if boxcarWidth > 0 else False,
        ).spectra

        if self.config.doBlackSpotCorrection:
            self.blackSpotCorrection.run(pfsConfig, spectra)

        visitInfo = exposure.visitInfo
        identity = Identity(
            visit=dataId["visit"],
            arm=arm,
            spectrograph=spectrograph,
            pfsDesignId=dataId["pfs_design_id"],
            obsTime=visitInfo.date.toString(visitInfo.date.TAI),
            expTime=visitInfo.exposureTime,
        )
        pfsArm = spectra.toPfsArm(identity)

        if self.config.doApplyScreenResponse:
            self.screen.run(exposure.getMetadata(), pfsArm, pfsConfig)
        if self.config.doBarycentricCorrection and not getLamps(exposure.getMetadata()):
            self.log.info("Calculating barycentric correction")
            calculateBarycentricCorrection(pfsArm, pfsConfig)
        pfsArm.metadata.update(metadataToHeader(exposure.getMetadata()))
        if fiberProfiles is not None:
            pfsArm.metadata["PFS.HASH.FIBERPROFILES"] = fiberProfiles.hash

        if self.config.doApplyFiberNorms:
            if fiberNorms is None:
                raise RuntimeError("fiberNorms required but not provided")
            missingFiberIds = applyFiberNorms(pfsArm, fiberNorms, self.config.doCheckFiberNormsHashes)
            if missingFiberIds:
                self.log.warn("Missing fiberIds in fiberNorms: %s", list(missingFiberIds))

        metadata = exposure.getMetadata()
        versions = getPfsVersions()
        for key, value in versions.items():
            metadata.set(key, value)
            pfsArm.metadata[key] = value

        return Struct(
            outputExposure=exposure,
            pfsArm=pfsArm,
            fiberTraces=measurements.fiberTraces,
            detectorMapUsed=measurements.detectorMap,
            lines=measurements.lines,
            apCorr=measurements.apCorr,
            lsf=lsf,
        )

    def checkPfsConfig(self, pfsConfig: PfsConfig, detectorMap: DetectorMap, spectrograph: int) -> PfsConfig:
        """Check that the PfsConfig is consistent with the DetectorMap

        Parameters
        ----------
        pfsConfig : `pfs.datamodel.PfsConfig`
            PFS fiber configuration.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.

        Returns
        -------
        pfsConfig : `pfs.datamodel.PfsConfig`
            PFS fiber configuration with active fibers selected.
        boxcarWidth : `int`
            Width of boxcar extraction; use fiberProfiles if <= 0.
        """
        kwargs = dict(spectrograph=spectrograph)
        if self.config.targetType:
            kwargs.update(targetType=TargetType.fromList(self.config.targetType))

        # Handle the IIS fibres for the user
        boxcarWidth = self.config.boxcarWidth if self.config.doBoxcarExtraction else -1
        if set(pfsConfig.select(targetType=TargetType.ENGINEERING).fiberStatus) == set([FiberStatus.GOOD]):
            if self.config.doDetectIIS:
                if len(set(kwargs["targetType"]) ^ set(~TargetType.ENGINEERING)) == 0:
                    kwargs["targetType"] = [TargetType.ENGINEERING]
                    self.log.info("~TargetType.ENGINEERING requested but IIS is on; assuming ENGINEERING")

            if self.config.doBoxcarForIIS:
                boxcarWidth = self.config.boxcarWidth

        pfsConfig = pfsConfig.select(**kwargs)
        if len(pfsConfig) == 0:
            raise RuntimeError(f"Selection {kwargs} returns no fibers")

        need = set(pfsConfig.fiberId)
        haveDetMap = set(detectorMap.fiberId)
        missingDetMap = need - haveDetMap
        if missingDetMap:
            raise RuntimeError(f"detectorMap does not include fibers: {list(sorted(missingDetMap))}")

        return Struct(pfsConfig=pfsConfig, boxcarWidth=boxcarWidth)

    def measure(
        self,
        exposure: Exposure,
        pfsConfig: PfsConfig,
        fiberProfiles: FiberProfileSet | None,
        detectorMap: DetectorMap,
        boxcarWidth: int,
        arm: str,
    ) -> Struct:
        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap, boxcarWidth)
        refLines = self.readLineList.run(detectorMap, exposure.getMetadata())
        seed = exposure.visitInfo.id if exposure.visitInfo is not None else 0
        lines = ArcLineSet.empty()
        if len(refLines) > 0:
            lines = self.centroidLines.run(
                exposure, refLines, detectorMap, pfsConfig, fiberTraces, seed=seed
            )
        if self.config.doForceTraces or not lines:
            traces = self.centroidTraces.run(exposure, detectorMap, pfsConfig)
            lines.extend(tracesToLines(detectorMap, traces, self.config.traceSpectralError))

        if self.config.doAdjustDetectorMap:
            try:
                detectorMap = self.adjustDetectorMap.run(
                    detectorMap,
                    lines,
                    arm,
                    exposure.visitInfo,
                    exposure.metadata,
                    seed=seed,
                ).detectorMap
            except (FittingError, RuntimeError) as exc:
                if self.config.requireAdjustDetectorMap:
                    raise
                self.log.warn("DetectorMap adjustment failed: %s", exc)

            if fiberProfiles is not None:
                # make fiberTraces with new detectorMap
                fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap, boxcarWidth)

        # Update photometry using best detectorMap
        notTrace = lines.description != "Trace"
        if not self.config.doPhotometerLines:
            apCorr = None
        else:
            phot = self.photometerLines.run(exposure, lines[notTrace], detectorMap, pfsConfig, fiberTraces)
            apCorr = phot.apCorr

            # Copy results to the one list of lines that we return
            lines.flux[notTrace] = phot.lines.flux
            lines.fluxErr[notTrace] = phot.lines.fluxErr
            lines.fluxNorm[notTrace] = phot.lines.fluxNorm
            lines.flag[notTrace] |= phot.lines.flag

        return Struct(
            refLines=refLines,
            lines=lines,
            apCorr=apCorr,
            detectorMap=detectorMap,
            fiberTraces=fiberTraces,
        )

    def defaultLsf(self, arm, fiberId, detectorMap):
        """Generate a default LSF for this exposure

        Parameters
        ----------
        arm : `str`
            Name of the spectrograph arm (one of ``b``, ``r``, ``m``, ``n``).
        fiberId : iterable of `int`
            Fiber identifiers.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.

        Returns
        -------
        lsf : `LsfDict`
            Line-spread functions, indexed by fiber identifier.
        """
        length = detectorMap.bbox.getHeight()
        sigma = self.config.gaussianLsfWidth[arm]
        return LsfDict(
            {ff: GaussianLsf(length, sigma/detectorMap.getDispersionAtCenter(ff)) for ff in fiberId}
        )
