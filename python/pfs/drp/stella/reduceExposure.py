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
from collections import defaultdict
import numpy as np
import lsstDebug

from lsst.pex.config import Config, Field, ConfigurableField, DictField, ListField
from lsst.pipe.base import CmdLineTask, TaskRunner, Struct
from lsst.ip.isr import IsrTask
from lsst.afw.display import Display
from lsst.pipe.tasks.repair import RepairTask
from lsst.obs.pfs.utils import getLampElements
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

__all__ = ["ReduceExposureConfig", "ReduceExposureTask"]


class ReduceExposureConfig(Config):
    """Config for ReduceExposure"""
    isr = ConfigurableField(target=IsrTask, doc="Instrumental signature removal")
    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    repair = ConfigurableField(target=RepairTask, doc="Task to repair artifacts")
    doMeasureLines = Field(dtype=bool, default=True, doc="Measure emission lines (sky, arc)?")
    doAdjustDetectorMap = Field(dtype=bool, default=True,
                                doc="Apply a low-order correction to the detectorMap?")
    requireAdjustDetectorMap = Field(dtype=bool, default=False,
                                     doc="Require detectorMap adjustment to succeed?")
    readLineList = ConfigurableField(target=ReadLineListTask,
                                     doc="Read line lists for detectorMap adjustment")
    adjustDetectorMap = ConfigurableField(target=AdjustDetectorMapTask, doc="Measure slit offsets")
    centroidLines = ConfigurableField(target=CentroidLinesTask, doc="Centroid lines")
    centroidTraces = ConfigurableField(target=CentroidTracesTask, doc="Centroid traces")
    traceSpectralError = Field(dtype=float, default=1.0,
                               doc="Error in the spectral dimension to give trace centroids (pixels)")
    photometerLines = ConfigurableField(target=PhotometerLinesTask, doc="Photometer lines")
    doSkySwindle = Field(dtype=bool, default=False,
                         doc="Do the Sky Swindle (subtract the exact sky)? "
                             "This only works with Simulator files produced with the --allOutput flag")
    doMeasurePsf = Field(dtype=bool, default=False, doc="Measure PSF?")
    measurePsf = ConfigurableField(target=MeasurePsfTask, doc="Measure PSF")
    gaussianLsfWidth = DictField(keytype=str, itemtype=float,
                                 doc="Gaussian sigma (nm) for LSF as a function of the spectrograph arm",
                                 default=dict(b=0.21, r=0.27, m=0.16, n=0.24))
    doSubtractSky2d = Field(dtype=bool, default=False, doc="Subtract sky on 2D image?")
    subtractSky2d = ConfigurableField(target=SubtractSky2dTask, doc="2D sky subtraction")
    doExtractSpectra = Field(dtype=bool, default=True, doc="Extract spectra from exposure?")
    extractSpectra = ConfigurableField(target=ExtractSpectraTask, doc="Extract spectra from exposure")
    doSubtractContinuum = Field(dtype=bool, default=False, doc="Subtract continuum as part of extraction?")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum for subtraction")
    doWriteCalexp = Field(dtype=bool, default=True, doc="Write corrected frame?")
    doWriteLsf = Field(dtype=bool, default=True, doc="Write line-spread function?")
    doWriteArm = Field(dtype=bool, default=True, doc="Write PFS arm file?")
    usePostIsrCcd = Field(dtype=bool, default=False, doc="Use existing postISRCCD, if available?")
    useCalexp = Field(dtype=bool, default=False, doc="Use existing calexp, if available?")
    targetType = ListField(dtype=str, default=["SCIENCE", "SKY", "FLUXSTD", "SUNSS_IMAGING", "SUNSS_DIFFUSE"],
                           doc="Target type for which to extract spectra")
    windowed = Field(dtype=bool, default=False,
                     doc="Reduction of windowed data, for real-time acquisition? Implies "
                     "doAdjustDetectorMap=False doMeasureLines=False isr.overscanFitType=MEDIAN")

    def validate(self):
        if not self.doExtractSpectra and self.doWriteArm:
            raise ValueError("You may not specify doWriteArm if doExtractSpectra is False")
        if self.windowed:
            self.doAdjustDetectorMap = False
            self.doMeasureLines = False
            self.isr.overscanFitType = "MEDIAN"
        super().validate()


class ReduceExposureRunner(TaskRunner):
    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        exposures = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.id.refList:
            visit = ref.dataId["visit"]
            arm = ref.dataId["arm"]
            exposures[visit][arm].append(ref)
        return [(exps, kwargs) for arms in exposures.values() for exps in arms.values()]


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
    RunnerClass = ReduceExposureRunner

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
        self.debugInfo = lsstDebug.Info(__name__)

    def runDataRef(self, sensorRefList):
        """Process all arms of the same kind within an exposure

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
        exposureList : `list` of `lsst.afw.image.Exposure`
            Exposure data for sensors.
        psfList : `list` of PSFs
            Point-spread functions; if ``doMeasurePsf`` is set.
        lsfList : `list` of LSFs
            Line-spread functions; if ``doMeasurePsf`` is set.
        sky2d : `pfs.drp.stella.FocalPlaneFunction`
            2D sky subtraction solution.
        spectraList : `list` of `pfs.drp.stella.SpectrumSet`
            Sets of extracted spectra.
        originalList : `list` of `pfs.drp.stella.SpectrumSet`
            Sets of extracted spectra before continuum subtraction.
            Will be identical to ``spectra`` if continuum subtraction
            was not performed.getSpectral
        fiberTraceList : `list` of `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        detectorMapList : `list` of `pfs.drp.stella.DetectorMap`
            Mappings of wl,fiber to detector position.
        linesList : `list` of `pfs.drp.stella.ArcLineSet`
            Measured lines.
        apCorrList : `list` of `pfs.drp.stella.FocalPlaneFunction`
            Measured aperture corrections.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.
        """
        self.log.info("Processing %s" % ([sensorRef.dataId for sensorRef in sensorRefList]))

        # Check that we were provided data from the same visit
        visits = set([ref.dataId["visit"] for ref in sensorRefList])
        if len(visits) > 1:
            raise RuntimeError(f"Data references from multiple visits provided: {list(visits)}")

        pfsConfig = sensorRefList[0].get("pfsConfig")
        exposureList = []
        fiberTraceList = []
        detectorMapList = []
        linesList = []
        apCorrList = []
        psfList = []
        lsfList = []
        skyResults = None
        if self.config.useCalexp:
            if all(sensorRef.datasetExists("calexp") for sensorRef in sensorRefList):
                self.log.info("Reading existing calexps")
                exposureList = [sensorRef.get("calexp") for sensorRef in sensorRefList]
                if self.config.doExtractSpectra:
                    calibs = [self.getSpectralCalibs(ref, exp, pfsConfig) for
                              ref, exp in zip(sensorRefList, exposureList)]
                    detectorMapList = [cal.detectorMap for cal in calibs]
                    fiberTraceList = [cal.fiberTraces for cal in calibs]
                psfList = [exp.getPsf() for exp in exposureList]
                lsfList = [sensorRef.get("pfsArmLsf") for sensorRef in sensorRefList]
                linesList = [sensorRef.get("arcLines") if sensorRef.datasetExists("arcLines") else None
                             for sensorRef in sensorRefList]
            else:
                self.log.warn("Not retrieving calexps, despite 'useCalexp' config, since some are missing")

        if not exposureList:
            for sensorRef in sensorRefList:
                exposure = self.runIsr(sensorRef)
                if self.config.doRepair:
                    self.repairExposure(exposure)
                if self.config.doSkySwindle:
                    self.skySwindle(sensorRef, exposure.image)

                exposureList.append(exposure)
                calibs = self.getSpectralCalibs(sensorRef, exposure, pfsConfig)
                detectorMapList.append(calibs.detectorMap)
                fiberTraceList.append(calibs.fiberTraces)
                linesList.append(calibs.lines)
                apCorrList.append(calibs.apCorr)
                psfList.append(calibs.psf)
                lsfList.append(calibs.lsf)

            if self.config.doSubtractSky2d:
                skyResults = self.subtractSky2d.run(exposureList, pfsConfig, psfList,
                                                    fiberTraceList, detectorMapList,
                                                    linesList, apCorrList)

        skyImageList = skyResults.imageList if skyResults is not None else [None]*len(exposureList)
        results = Struct(
            exposureList=exposureList,
            fiberTraceList=fiberTraceList,
            detectorMapList=detectorMapList,
            psfList=psfList,
            lsfList=lsfList,
            linesList=linesList,
            apCorrList=apCorrList,
            pfsConfig=pfsConfig,
            sky2d=skyResults.sky2d if skyResults is not None else None,
            skyImageList=skyImageList,
        )

        if self.config.doExtractSpectra:
            originalList = []
            spectraList = []
            for exposure, fiberTraces, detectorMap, skyImage, sensorRef in zip(exposureList, fiberTraceList,
                                                                               detectorMapList, skyImageList,
                                                                               sensorRefList):
                self.log.info("Extracting spectra from %(visit)d %(arm)s%(spectrograph)d" % sensorRef.dataId)
                maskedImage = exposure.maskedImage
                fiberId = np.array(sorted(set(pfsConfig.fiberId) & set(detectorMap.fiberId)))
                spectra = self.extractSpectra.run(maskedImage, fiberTraces, detectorMap, fiberId).spectra
                originalList.append(spectra)

                if self.config.doSubtractContinuum:
                    continua = self.fitContinuum.run(spectra)
                    maskedImage -= continua.makeImage(exposure.getBBox(), fiberTraces)
                    spectra = self.extractSpectra.run(maskedImage, fiberTraces, detectorMap, fiberId).spectra
                    # Set sky flux from continuum
                    for ss, cc in zip(spectra, continua):
                        ss.background += cc.spectrum/cc.norm*ss.norm

                if skyImage is not None:
                    skySpectra = self.extractSpectra.run(skyImage, fiberTraces, detectorMap, fiberId).spectra
                    for spec, skySpec in zip(spectra, skySpectra):
                        spec.background += skySpec.spectrum/skySpec.norm*spec.norm

                spectraList.append(spectra)

            results.originalList = originalList
            results.spectraList = spectraList

        if self.debugInfo.plotSpectra:
            self.plotSpectra(results.spectraList)

        self.write(sensorRefList, results)
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

    def write(self, sensorRefList, results):
        """Write out results

        Respects the various ``doWrite`` entries in the configuation, and
        iterates through the arrays of outputs.

        Parameters
        ----------
        sensorRefList : iterable of `lsst.daf.persistence.ButlerDataRef`
            Data references for writing.
        results : `lsst.pipe.base.Struct`
            Conglomeration of results to be written out.
        """
        if self.config.doWriteCalexp:
            for sensorRef, exposure in zip(sensorRefList, results.exposureList):
                sensorRef.put(exposure, "calexp")
        if self.config.doWriteLsf:
            for sensorRef, lsf in zip(sensorRefList, results.lsfList):
                if lsf is None:
                    self.log.warn("Can't write PSF for %s" % (sensorRef.dataId,))
                    continue
                sensorRef.put(lsf, "pfsArmLsf")
        if self.config.doSubtractSky2d:
            sensorRefList[0].put(results.sky2d, "sky2d")

        if self.config.doWriteArm:
            for sensorRef, spectra, lines in zip(sensorRefList, results.spectraList, results.linesList):
                sensorRef.put(spectra.toPfsArm(sensorRef.dataId), "pfsArm")
                if lines is not None:
                    sensorRef.put(lines, "arcLines")

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

    def getSpectralCalibs(self, sensorRef, exposure, pfsConfig):
        """Provide suitable spectral calibrations

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for sensor data.
        exposure : `lsst.afw.image.Exposure`
            Image of spectra. Required for measuring a slit offset.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration. Required for measuring a slit offset.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        fiberProfiles : `pfs.drp.stella.FiberProfileSet`
            Profile for each fiber.
        fiberTraces : `pfs.drp.stella.FiberTraceSet`
            Trace for each fiber.
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
        fiberProfiles = sensorRef.get("fiberProfiles")

        # Check that the calibs have the expected number of fibers
        select = pfsConfig.getSelection(fiberStatus=FiberStatus.GOOD,
                                        targetType=[TargetType.fromString(tt) for
                                                    tt in self.config.targetType],
                                        spectrograph=sensorRef.dataId["spectrograph"])
        fiberId = pfsConfig.fiberId[select]
        need = set(fiberId)
        haveDetMap = set(detectorMap.fiberId)
        haveProfiles = set(fiberProfiles.fiberId)
        missingDetMap = need - haveDetMap
        missingProfiles = need - haveProfiles
        if missingDetMap:
            uri = sensorRef.getUri("detectorMap")
            raise RuntimeError(f"detectorMap ({uri}) does not include fibers: {list(sorted(missingDetMap))}")
        if need - haveProfiles:
            uri = sensorRef.getUri("fiberProfiles")
            raise RuntimeError(
                f"fiberProfiles ({uri}) does not include fibers: {list(sorted(missingProfiles))}"
            )

        fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)

        refLines = None
        lines = None
        traces = None
        if self.config.doAdjustDetectorMap or self.config.doMeasureLines:
            refLines = self.readLineList.run(detectorMap, exposure.getMetadata())
            lines = self.centroidLines.run(exposure, refLines, detectorMap, pfsConfig, fiberTraces)
            if not lines or "Continuum" in getLampElements(exposure.getMetadata()):
                traces = self.centroidTraces.run(exposure, detectorMap, pfsConfig)
                lines.extend(tracesToLines(detectorMap, traces, self.config.traceSpectralError))

            if self.config.doAdjustDetectorMap:
                if self.debugInfo.detectorMap:
                    display = Display(frame=1)
                    display.mtv(exposure)
                    detectorMap.display(display, fiberId=fiberId, wavelengths=refLines.wavelength,
                                        ctype="red", plotTraces=False)

                try:
                    detectorMap = self.adjustDetectorMap.run(detectorMap, lines).detectorMap
                except FittingError as exc:
                    if self.config.requireAdjustDetectorMap:
                        raise
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

                sensorRef.put(detectorMap, "detectorMap_used")
                fiberTraces = fiberProfiles.makeFiberTracesFromDetectorMap(detectorMap)  # use new detectorMap

        if self.config.doMeasurePsf:
            psf = self.measurePsf.runSingle(sensorRef, exposure, detectorMap)
            lsf = self.calculateLsf(psf, fiberTraces, exposure.getHeight())
        else:
            psf = None
            lsf = self.defaultLsf(sensorRef.dataId["arm"], fiberTraces.fiberId, detectorMap)

        # Update photometry using best detectorMap, PSF
        apCorr = None
        if self.config.doMeasureLines:
            phot = self.photometerLines.run(exposure, lines[lines.description != "Trace"], detectorMap,
                                            pfsConfig, fiberTraces)
            apCorr = phot.apCorr
            lines = phot.lines
            if apCorr is not None:
                sensorRef.put(phot.apCorr, "apCorr")

        return Struct(detectorMap=detectorMap, fiberProfiles=fiberProfiles, fiberTraces=fiberTraces,
                      refLines=refLines, lines=lines, apCorr=apCorr, psf=psf, lsf=lsf)

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

    def plotSpectra(self, spectraList):
        """Plot spectra

        The spectra from different fibers are shown as different colors.
        Points with non-zero mask are drawn as dotted lines, while good points
        are solid.

        Parameters
        ----------
        spectraList : iterable of `pfs.drp.stella.SpectrumSet`
            Extracted spectra from spectrograph arms for a single exposure.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm

        fiberId = set()
        for spectra in spectraList:
            fiberId.update(set(ss.fiberId for ss in spectra))
        fiberId = dict(zip(fiberId, matplotlib.cm.rainbow(np.linspace(0, 1, len(fiberId)))))

        figure, axes = plt.subplots()
        for spectra in spectraList:
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
