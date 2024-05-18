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
from datetime import datetime
import numpy as np
import lsstDebug

from lsst.pex.config import Config, Field, ConfigurableField, DictField, ListField
from lsst.pipe.base import CmdLineTask, Struct
from lsst.obs.pfs.isrTask import PfsIsrTask
from lsst.afw.display import Display
from pfs.datamodel import FiberStatus, TargetType
from .measurePsf import MeasurePsfTask
from .extractSpectraTask import ExtractSpectraTask
from .subtractSky2d import SubtractSky2dTask
from .fitContinuum import FitContinuumTask
from .lsf import ExtractionLsf, GaussianLsf
from .readLineList import ReadLineListTask
from .centroidLines import CentroidLinesTask
from .photometerLines import PhotometerLinesTask
from .centroidTraces import CentroidTracesTask, tracesToLines
from .adjustDetectorMap import AdjustDetectorMapTask
from .fitDistortedDetectorMap import FittingError
from .constructSpectralCalibs import setCalibHeader
from .repair import PfsRepairTask, maskLines
from .blackSpotCorrection import BlackSpotCorrectionTask
from .background import DichroicBackgroundTask
from .arcLine import ArcLineSet
from .fiberProfile import FiberProfile
from .fiberProfileSet import FiberProfileSet
from .utils.sysUtils import metadataToHeader, getPfsVersions, processConfigListFromCmdLine


__all__ = ["ReduceExposureConfig", "ReduceExposureTask"]


class ReduceExposureConfig(Config):
    """Config for ReduceExposure"""
    isr = ConfigurableField(target=PfsIsrTask, doc="Instrumental signature removal")
    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    repair = ConfigurableField(target=PfsRepairTask, doc="Task to repair artifacts")
    doMeasureLines = Field(dtype=bool, default=True, doc="Measure emission lines (sky, arc)?")
    doAdjustDetectorMap = Field(dtype=bool, default=True,
                                doc="Apply a low-order correction to the detectorMap?")
    requireAdjustDetectorMap = Field(dtype=bool, default=False,
                                     doc="Require detectorMap adjustment to succeed?")
    readLineList = ConfigurableField(target=ReadLineListTask,
                                     doc="Read line lists for detectorMap adjustment")
    doMaskLines = Field(dtype=bool, default=True, doc="Mask reference lines on image?")
    maskRadius = Field(dtype=int, default=2, doc="Radius around reference lines to mask (pixels)")
    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Measure slit offsets")
    centroidLines = ConfigurableField(target=CentroidLinesTask, doc="Centroid lines")
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")
    traceSpectralError = Field(dtype=float, default=5.0,
                               doc="Error in the spectral dimension to give trace centroids (pixels)")
    doForceTraces = Field(dtype=bool, default=True, doc="Force use of traces for non-continuum data?")
    photometerLines = ConfigurableField(target=PhotometerLinesTask, doc="Photometer lines")
    doBoxcarExtraction = Field(dtype=bool, default=False, doc="Extract with a boxcar of width boxcarWidth")
    boxcarWidth = Field(dtype=float, default=5,
                        doc="Extract with a boxcar of width boxcarWidth if doBoxcarExtraction is True")
    doDetectIIS = Field(dtype=bool, default=True,
                        doc="If ~ENGINEERING fibres is requested but the IIS is illuminated, "
                        "use a boxcar of boxcarWidth to extract the ENGINEERING fibres")
    doBoxcarForIIS = Field(dtype=bool, default=True,
                           doc="Enable boxcar extractions when ENGINEERING fibres are illuminated?")
    doSkySwindle = Field(dtype=bool, default=False,
                         doc="Do the Sky Swindle (subtract the exact sky)? "
                             "This only works with Simulator files produced with the --allOutput flag")
    doMeasurePsf = Field(dtype=bool, default=False, doc="Measure PSF?")
    measurePsf = ConfigurableField(target=MeasurePsfTask, doc="Measure PSF")
    gaussianLsfWidth = DictField(keytype=str, itemtype=float,
                                 doc="Gaussian sigma (nm) for LSF as a function of the spectrograph arm",
                                 default=dict(b=0.081, r=0.109, m=0.059, n=0.109))
    doSubtractSky2d = Field(dtype=bool, default=False, doc="Subtract sky on 2D image?")
    subtractSky2d = ConfigurableField(target=SubtractSky2dTask, doc="2D sky subtraction")
    doExtractSpectra = Field(dtype=bool, default=True, doc="Extract spectra from exposure?")
    extractSpectra = ConfigurableField(target=ExtractSpectraTask, doc="Extract spectra from exposure")
    doSubtractSpectra = Field(dtype=bool, default=False, doc="Subtract extracted spectra from exposure?")
    doSubtractContinuum = Field(dtype=bool, default=False, doc="Subtract continuum as part of extraction?")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum for subtraction")
    doWriteCalexp = Field(dtype=bool, default=True, doc="Write corrected frame?")
    doWriteLsf = Field(dtype=bool, default=True, doc="Write line-spread function?")
    doWriteArm = Field(dtype=bool, default=True, doc="Write PFS arm file?")
    usePostIsrCcd = Field(dtype=bool, default=False, doc="Use existing postISRCCD, if available?")
    useCalexp = Field(dtype=bool, default=False, doc="Use existing calexp, if available?")
    targetType = ListField(dtype=str, default=["^ENGINEERING"],
                           doc="""Target type for which to extract spectra
N.b. you can exclude a set of types, e.g. `["^ENGINEERING", "^UNASSIGNED"]` which is interpreted as
"neither ENGINEERING nor UNASSIGNED"   (empty: `["^ENGINEERING"]`)""")
    windowed = Field(dtype=bool, default=False,
                     doc="Reduction of windowed data, for real-time acquisition? Implies "
                     "doAdjustDetectorMap=False doMeasureLines=False isr.overscanFitType=MEDIAN")
    doApplyFiberNorms = Field(dtype=bool, default=True, doc="Apply fiber normalizations?")
    doBlackSpotCorrection = Field(dtype=bool, default=True, doc="Correct for black spot penumbra?")
    blackSpotCorrection = ConfigurableField(target=BlackSpotCorrectionTask, doc="Black spot correction")
    doBackground = Field(dtype=bool, default=False, doc="Subtract background?")
    background = ConfigurableField(target=DichroicBackgroundTask, doc="Background subtraction")
    spatialOffset = Field(dtype=float, default=0.0, doc="Spatial offset to add")
    spectralOffset = Field(dtype=float, default=0.0, doc="Spectral offset to add")

    def validate(self):
        if not self.doExtractSpectra and self.doWriteArm:
            raise ValueError("You may not specify doWriteArm if doExtractSpectra is False")
        if self.windowed:
            self.doAdjustDetectorMap = False
            self.doMeasureLines = False
            self.isr.overscanFitType = "MEDIAN"

        # Handle setting lists of strings on the command line
        self.targetType = processConfigListFromCmdLine(self.targetType)

        #
        # Parse the target types.  Note that types may start ^ to exclude that type
        badTargetTypes = []
        exclude = []                    # list of types to exclude
        for tt in self.targetType:
            if tt.startswith('^'):
                exclude.append(tt[1:])
                tt = tt[1:]
            elif exclude and not tt.startswith('^'):
                raise RuntimeError(f"If you exclude any type you must exclude all; saw {tt}")

            try:
                TargetType.fromString(tt)
            except AttributeError:
                badTargetTypes.append(tt)

        if badTargetTypes:
            raise ValueError(f"Unrecognised TargetTypes: {', '.join(badTargetTypes)}")

        if exclude:
            self.targetType = []
            for tt in [str(_) for _ in TargetType]:
                if tt not in exclude:
                    self.targetType.append(tt)

        super().validate()


class ReduceExposureTask(CmdLineTask):
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
    - Call isr to unpersist raw data and assemble it into a post-ISR exposure
    - Call repair to repair cosmic rays
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
        self.makeSubtask("isr")
        self.makeSubtask("repair")
        self.makeSubtask("readLineList")
        self.makeSubtask("centroidLines")
        self.makeSubtask("centroidTraces")
        self.makeSubtask("photometerLines")
        self.makeSubtask("adjustDetectorMap")
        self.makeSubtask("measurePsf")
        self.makeSubtask("subtractSky2d")
        self.makeSubtask("extractSpectra")
        self.makeSubtask("fitContinuum")
        self.makeSubtask("blackSpotCorrection")
        self.makeSubtask("background")
        self.debugInfo = lsstDebug.Info(__name__)

    def runDataRef(self, sensorRef):
        """Process an arm exposure

        The sequence of operations is:
        - remove instrument signature
        - measure PSF
        - subtract sky from the image
        - extract the spectra from the fiber traces
        - write the outputs

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for sensors to process.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Exposure data for sensor.
        psf : PSF
            Point-spread function; if ``doMeasurePsf`` is set.
        lsf : LSF
            Line-spread function; if ``doMeasurePsf`` is set.
        sky2d : `pfs.drp.stella.FocalPlaneFunction`
            2D sky subtraction solution.
        spectra : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        original : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra before continuum subtraction.
            Will be identical to ``spectra`` if continuum subtraction
            was not performed.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of wl,fiber to detector position.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured lines.
        apCorr : `pfs.drp.stella.FocalPlaneFunction`
            Measured aperture correction.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.
        """

        self.log.info("Processing %s", sensorRef.dataId)

        pfsConfig = sensorRef.get("pfsConfig")
        exposure = None
        fiberTraces = None
        detectorMap = None
        refLines = None
        lines = None
        apCorr = None
        psf = None
        lsf = None
        skyImage = None
        sky2d = None

        boxcarWidth = self.config.boxcarWidth if self.config.doBoxcarExtraction else -1
        if self.config.useCalexp:
            if sensorRef.datasetExists("calexp"):
                self.log.info("Reading existing calexp for %s", sensorRef.dataId)
                exposure = sensorRef.get("calexp")
                exposure.mask &= ~exposure.mask.getPlaneBitMask("REFLINE")  # not related to ISR etc.
            else:
                self.log.warn("Not retrieving calexp, despite 'useCalexp' config, since it is missing")

        if not exposure:
            exposure = self.runIsr(sensorRef)
            if self.config.doRepair:
                self.repairExposure(exposure)
            if self.config.doSkySwindle:
                self.skySwindle(sensorRef, exposure.image)

        calibs = self.getSpectralCalibs(sensorRef, exposure, pfsConfig, boxcarWidth)

        if self.config.doBackground:
            bg = self.background.run(
                exposure.maskedImage, sensorRef.dataId["arm"], calibs.detectorMap, pfsConfig
            )
            sensorRef.put(bg, "background")

        if self.config.doMaskLines:
            maskLines(exposure.mask, calibs.detectorMap, calibs.refLines, self.config.maskRadius)
        if self.config.doRepair:
            self.repairExposure(exposure)

        detectorMap = calibs.detectorMap
        fiberTraces = calibs.fiberTraces
        refLines = calibs.refLines
        lines = calibs.lines
        apCorr = calibs.apCorr
        pfsConfig = calibs.pfsConfig
        psf = calibs.psf
        lsf = calibs.lsf

        if self.config.doSubtractSky2d:
            self.log.warn("Performing 2D sky subtraction on single arm")
            skyResults = self.subtractSky2d.run(
                [exposure], pfsConfig, [psf], [fiberTraces], [detectorMap], [lines], [apCorr]
            )
            skyImage = skyResults.imageList[0]
            sky2d = skyResults.sky2d

        metadata = exposure.getMetadata()
        versions = getPfsVersions()
        for key, value in versions.items():
            metadata.set(key, value)

        results = Struct(
            exposure=exposure,
            fiberTraces=fiberTraces,
            detectorMap=detectorMap,
            psf=psf,
            lsf=lsf,
            lines=lines,
            apCorr=apCorr,
            pfsConfig=pfsConfig,
            sky2d=sky2d,
            skyImage=skyImage,
        )

        original = None
        spectra = None
        if self.config.doExtractSpectra:
            self.log.info("Extracting spectra from %(visit)d%(arm)s%(spectrograph)d", sensorRef.dataId)
            maskedImage = exposure.maskedImage
            fiberId = np.array(sorted(set(pfsConfig.fiberId) & set(detectorMap.fiberId)))
            spectra = self.extractSpectra.run(maskedImage, fiberTraces, detectorMap, fiberId,
                                              True if boxcarWidth > 0 else False, calibs.fiberNorms).spectra
            original = spectra

            if self.config.doSubtractSpectra:
                self.log.info("Subtracting spectra from exposure")
                sub = exposure.clone()
                sub.maskedImage -= spectra.makeImage(maskedImage.getBBox(), fiberTraces)
                sensorRef.put(sub, "subtracted")

            if self.config.doSubtractContinuum:
                continua = self.fitContinuum.run(spectra, refLines)
                maskedImage -= continua.makeImage(exposure.getBBox(), fiberTraces)
                spectra = self.extractSpectra.run(maskedImage, fiberTraces, detectorMap, fiberId).spectra
                # Set sky flux from continuum
                for ss, cc in zip(spectra, continua):
                    ss.background += cc.spectrum/cc.norm*ss.norm

            if self.config.doBlackSpotCorrection:
                self.blackSpotCorrection.run(pfsConfig, spectra)

            if skyImage is not None:
                skySpectra = self.extractSpectra.run(skyImage, fiberTraces, detectorMap, fiberId).spectra
                for spec, skySpec in zip(spectra, skySpectra):
                    spec.background += skySpec.spectrum/skySpec.norm*spec.norm

            results.original = original
            results.spectra = spectra

        if self.debugInfo.plotSpectra:
            self.plotSpectra(results.spectra)

        self.write(sensorRef, results)
        return results

    def runIsr(self, sensorRef):
        """Run Instrument Signature Removal (ISR)

        This method wraps the ISR call to allow us to post-process the ISR-ed
        image.

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for sensor.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Sensor image after ISR has been applied.
        """
        if self.config.usePostIsrCcd:
            if sensorRef.datasetExists("postISRCCD"):
                self.log.info("Reusing existing postISRCCD for %s", sensorRef.dataId)
                return sensorRef.get("postISRCCD")
            self.log.warn("Not retrieving postISRCCD for %s, despite 'usePostIsrCcd' config, "
                          "since it is missing", sensorRef.dataId)
        exposure = self.isr.runDataRef(sensorRef).exposure
        # Remove negative variance (because we can have very low count levels)
        bad = np.where(exposure.variance.array < 0)
        exposure.variance.array[bad] = np.inf
        exposure.mask.array[bad] |= exposure.mask.getPlaneBitMask("BAD")
        return exposure

    def write(self, sensorRef, results):
        """Write out results

        Respects the various ``doWrite`` entries in the configuation.

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for writing.
        results : `lsst.pipe.base.Struct`
            Conglomeration of results to be written out.
        """
        if self.config.doWriteCalexp:
            sensorRef.put(results.exposure, "calexp")
        if self.config.doWriteLsf:
            if results.lsf is None:
                self.log.warn("Can't write LSF for %s" % (sensorRef.dataId,))
            else:
                sensorRef.put(results.lsf, "pfsArmLsf")
        if self.config.doSubtractSky2d:
            self.log.warn("Writing sky2d for single arm")
            sensorRef.put(results.sky2d, "sky2d")

        if self.config.doWriteArm:
            pfsArm = results.spectra.toPfsArm(sensorRef.dataId)
            pfsArm.metadata.update(metadataToHeader(results.exposure.getMetadata()))
            sensorRef.put(pfsArm, "pfsArm")
            if results.lines is not None:
                sensorRef.put(results.lines, "arcLines")

        return results

    def repairExposure(self, exposure):
        """Repair CCD defects in the exposure

        Uses the PSF specified in the config.
        """
        modelPsfConfig = self.config.repair.interp.modelPsf
        psf = modelPsfConfig.apply()
        exposure.setPsf(psf)
        self.repair.run(exposure)

    def skySwindle(self, sensorRef, image):
        """Perform the sky swindle

        The 'sky swindle' is where we subtract the known sky signal from the
        image. The resultant sky-subtracted image contains the noise from the
        sky, but no sky and no systematic sky subtraction residuals.

        This is a dirty hack, intended only to allow a direct comparison with
        the 1D simulator, which uses the same swindle.

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for sensor data.
        image : `lsst.afw.image.Image`
            Exposure from which to subtract the sky.
        """
        self.log.warn("Applying sky swindle")
        import astropy.io.fits
        filename = sensorRef.getUri("raw")
        with astropy.io.fits.open(filename) as fits:
            image.array -= fits["SKY"].data

    def getSpectralCalibs(self, sensorRef, exposure, pfsConfig, boxcarWidth=0):
        """Provide suitable spectral calibrations

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for sensor data.
        exposure : `lsst.afw.image.Exposure`
            Image of spectra. Required for measuring a slit offset.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration. Required for measuring a slit offset.
        boxcarWidth: `int`
            Width of boxcar extraction; use fiberProfiles if <= 0

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profile for each fiber.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Trace for each fiber (defined as a boxcar if boxcarWidth > 0).
        refLines : `pfs.drp.stella.ReferenceLineSet`
            Reference lines.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured lines.
        apCorr : `pfs.drp.stella.FocalPlaneFunction`
            Aperture correction.
        traces : `dict` [`int`: `list` of `pfs.drp.stella.TracePeak`]
            Peaks for each trace, indexed by fiberId.
        psf : `pfs.drp.stella.SpectralPsf`
            Two-dimensional point-spread function.
        lsf : `pfs.drp.stella.Lsf`
            One-dimensional line-spread function.
        """
        detectorMap = sensorRef.get("detectorMap")
        spatialOffset = self.config.spatialOffset
        spectralOffset = self.config.spectralOffset
        if spatialOffset != 0.0 or spectralOffset != 0.0:
            self.log.info("Adjusting detectorMap slit offset by %f,%f", spatialOffset, spectralOffset)
            detectorMap.applySlitOffset(spatialOffset, spectralOffset)

        refLines = self.readLineList.run(detectorMap, exposure.getMetadata())

        # Check that the detectorMap includes all the expected fibers
        kwargs = dict(spectrograph=sensorRef.dataId["spectrograph"])
        if self.config.targetType:
            kwargs.update(targetType=[TargetType.fromString(tt) for tt in self.config.targetType])

        # Handle the IIS fibres for the user
        if set(pfsConfig.select(targetType=TargetType.ENGINEERING).fiberStatus) == set([FiberStatus.GOOD]):
            if self.config.doDetectIIS:
                if len(set(kwargs["targetType"]) ^ set(~TargetType.ENGINEERING)) == 0:
                    kwargs["targetType"] = [TargetType.ENGINEERING]
                    self.log.info("~TargetType.ENGINEERING requested but IIS is on; assuming ENGINEERING")

            if self.config.doBoxcarForIIS:
                boxcarWidth = self.config.boxcarWidth

        select = pfsConfig.getSelection(**kwargs)
        if not select.any():
            raise RuntimeError(f"Selection {kwargs} returns no fibres for dataId "
                               f"{'%(visit)d %(arm)s%(spectrograph)d}' % sensorRef.dataId}")
        need = set(pfsConfig.fiberId[select])
        haveDetMap = set(detectorMap.fiberId)
        missingDetMap = need - haveDetMap
        if missingDetMap:
            uri = sensorRef.getUri("detectorMap")
            raise RuntimeError(f"detectorMap ({uri}) does not include fibers: {list(sorted(missingDetMap))}")

        pfsConfig = pfsConfig[select]

        fiberProfiles = None
        fiberTraces = None
        if (
            self.config.doMeasureLines or
            (self.config.doAdjustDetectorMap and len(refLines) > 0) or
            self.config.doExtractSpectra or
            self.config.doMeasurePsf
        ):
            if boxcarWidth <= 0 and sensorRef.datasetExists("fiberProfiles"):
                fiberProfiles = sensorRef.get("fiberProfiles")

            if fiberProfiles is None:
                assert boxcarWidth > 0
                fiberProfiles = FiberProfileSet.makeEmpty(None)
                for fid in need:        # the Gaussian will be replaced by a boxcar, so params don't matter
                    fiberProfiles[fid] = FiberProfile.makeGaussian(1, exposure.getHeight(), 5, 1)

            fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap, boxcarWidth)

        lines = ArcLineSet.empty()
        traces = None
        if self.config.doAdjustDetectorMap or self.config.doMeasureLines:
            if len(refLines) > 0:
                lines = self.centroidLines.run(
                    exposure, refLines, detectorMap, pfsConfig, fiberTraces, seed=exposure.visitInfo.id
                )
            if (
                self.config.doForceTraces
                or not lines
            ):
                traces = self.centroidTraces.run(exposure, detectorMap, pfsConfig)
                lines.extend(tracesToLines(detectorMap, traces, self.config.traceSpectralError))

            if self.config.doAdjustDetectorMap:
                if self.debugInfo.detectorMap:
                    fiberId = pfsConfig.fiberId  # a set of fibres to display
                    display = Display(frame=1)
                    display.mtv(exposure)
                    detectorMap.display(display, fiberId=fiberId, wavelengths=refLines.wavelength,
                                        ctype="red", plotTraces=False)

                try:
                    detectorMap = self.adjustDetectorMap.run(
                        detectorMap,
                        lines,
                        sensorRef.dataId["arm"],
                        seed=exposure.visitInfo.id if exposure.visitInfo is not None else 0,
                    ).detectorMap
                except (FittingError, RuntimeError) as exc:
                    if self.config.requireAdjustDetectorMap:
                        raise
                    self.log.warn("DetectorMap adjustment failed: %s", exc)
                except RuntimeError as exc:
                    self.log.warn("DetectorMap adjustment failed: %s", exc)

                if self.debugInfo.detectorMap:
                    detectorMap.display(display, fiberId=fiberId[::5], wavelengths=refLines.wavelength,
                                        ctype="green", plotTraces=False)

                outputId = sensorRef.dataId.copy()
                outputId["visit0"] = outputId["visit"]
                outputId["calibDate"] = outputId["dateObs"]
                outputId["calibTime"] = outputId["taiObs"]
                setCalibHeader(detectorMap.metadata, "detectorMap", [sensorRef.dataId["visit"]], outputId)
                date = datetime.now().isoformat()
                history = f"reduceExposure on {date} with visit={sensorRef.dataId['visit']}"
                detectorMap.metadata.add("HISTORY", history)

                if fiberProfiles is not None:
                    # make fiberTraces with new detectorMap
                    fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap, boxcarWidth)

        sensorRef.put(detectorMap, "detectorMap_used")

        fiberNorms = None
        if self.config.doApplyFiberNorms:
            fiberNorms = sensorRef.get("fiberNorms")

        if self.config.doMeasurePsf:
            psf = self.measurePsf.runSingle(exposure, detectorMap)
            lsf = self.calculateLsf(psf, fiberTraces, exposure.getHeight())
        else:
            psf = None
            lsf = self.defaultLsf(sensorRef.dataId["arm"], pfsConfig.fiberId, detectorMap)

        # Update photometry using best detectorMap, PSF
        apCorr = None
        if self.config.doMeasureLines:
            notTrace = lines.description != "Trace"
            phot = self.photometerLines.run(
                exposure, lines[notTrace], detectorMap, pfsConfig, fiberTraces, fiberNorms
            )
            apCorr = phot.apCorr

            # Copy results to the one list of lines that we return
            lines.flux[notTrace] = phot.lines.flux
            lines.fluxErr[notTrace] = phot.lines.fluxErr
            lines.fluxNorm[notTrace] = phot.lines.fluxNorm
            lines.flag[notTrace] |= phot.lines.flag

            if apCorr is not None:
                sensorRef.put(phot.apCorr, "apCorr")

        return Struct(detectorMap=detectorMap, fiberProfiles=fiberProfiles, fiberTraces=fiberTraces,
                      pfsConfig=pfsConfig, refLines=refLines, lines=lines, apCorr=apCorr, psf=psf, lsf=lsf,
                      fiberNorms=fiberNorms)

    def calculateLsf(self, psf, fiberTraceSet, length):
        """Calculate the LSF for this exposure

        Parameters
        ----------
        psf : `pfs.drp.stella.SpectralPsf`
            Point-spread function for spectral data.
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Traces for each fiber.
        length : `int`
            Array length.

        Returns
        -------
        lsf : `dict` (`int`: `pfs.drp.stella.ExtractionLsf`)
            Line-spread functions, indexed by fiber identifier.
        """
        return {ft.fiberId: ExtractionLsf(psf, ft, length) for ft in fiberTraceSet}

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
        lsf : `dict` (`int`: `pfs.drp.stella.GaussianLsf`)
            Line-spread functions, indexed by fiber identifier.
        """
        length = detectorMap.bbox.getHeight()
        sigma = self.config.gaussianLsfWidth[arm]
        return {ff: GaussianLsf(length, sigma/detectorMap.getDispersionAtCenter(ff)) for ff in fiberId}

    def plotSpectra(self, spectra):
        """Plot spectra

        The spectra from different fibers are shown as different colors.
        Points with non-zero mask are drawn as dotted lines, while good points
        are solid.

        Parameters
        ----------
        spectra : `pfs.drp.stella.SpectrumSet`
            Extracted spectra from spectrograph arm.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm

        fiberId = set()
        fiberId.update(set(ss.fiberId for ss in spectra))
        fiberId = dict(zip(fiberId, matplotlib.cm.rainbow(np.linspace(0, 1, len(fiberId)))))

        figure, axes = plt.subplots()
        for ii, ss in enumerate(spectra):
            color = fiberId[ss.fiberId]
            axes.plot(ss.wavelength, ss.spectrum, ls="solid", color=color)
            bad = (ss.mask.array[0] & ss.mask.getPlaneBitMask(["BAD_FLAT", "CR", "NO_DATA", "SAT"])) != 0
            if np.any(bad):
                axes.plot(ss.wavelength[bad], ss.spectrum[bad], ".", color=color)

        axes.set_xlabel("Wavelength (nm)")
        axes.set_ylabel("Flux")
        figure.show()
        input("Hit ENTER to continue... ")

    def _getMetadataName(self):
        return None
