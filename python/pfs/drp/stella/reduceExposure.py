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
from collections import defaultdict
import numpy as np

from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import CmdLineTask, TaskRunner, Struct
from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.repair import RepairTask
from .measurePsf import MeasurePsfTask
from .extractSpectraTask import ExtractSpectraTask
from .subtractSky2d import SubtractSky2dTask
from .fitContinuum import FitContinuumTask
from .lsf import ExtractionLsf

__all__ = ["ReduceExposureConfig", "ReduceExposureTask"]


class ReduceExposureConfig(Config):
    """Config for ReduceExposure"""
    isr = ConfigurableField(target=IsrTask, doc="Instrumental signature removal")
    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    repair = ConfigurableField(target=RepairTask, doc="Task to repair artifacts")
    doSkySwindle = Field(dtype=bool, default=False,
                         doc="Do the Sky Swindle (subtract the exact sky)? "
                             "This only works with Simulator files produced with the --allOutput flag")
    doMeasurePsf = Field(dtype=bool, default=False, doc="Measure PSF?")
    measurePsf = ConfigurableField(target=MeasurePsfTask, doc="Measure PSF")
    doSubtractSky2d = Field(dtype=bool, default=False, doc="Subtract sky on 2D image?")
    subtractSky2d = ConfigurableField(target=SubtractSky2dTask, doc="2D sky subtraction")
    doExtractSpectra = Field(dtype=bool, default=True, doc="Extract spectra from exposure?")
    extractSpectra = ConfigurableField(target=ExtractSpectraTask, doc="Extract spectra from exposure")
    doSubtractContinuum = Field(dtype=bool, default=False, doc="Subtract continuum as part of extraction?")
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum for subtraction")
    fiberDx = Field(doc="DetectorMap slit offset in x", dtype=float, default=0)
    fiberDy = Field(doc="DetectorMap slit offset in y", dtype=float, default=0)
    doWriteCalexp = Field(dtype=bool, default=False, doc="Write corrected frame?")
    doWriteLsf = Field(dtype=bool, default=False, doc="Write line-spread function?")
    doWriteArm = Field(dtype=bool, default=True, doc="Write PFS arm file?")
    usePostIsrCcd = Field(dtype=bool, default=False, doc="Use existing postISRCCD, if available?")
    useCalexp = Field(dtype=bool, default=False, doc="Use existing calexp, if available?")


class ReduceExposureRunner(TaskRunner):
    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        exposures = defaultdict(lambda: defaultdict(list))
        for ref in parsedCmd.id.refList:
            visit = ref.dataId["visit"]
            arm = ref.dataId["arm"]
            exposures[arm][visit].append(ref)
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
        self.makeSubtask("measurePsf")
        self.makeSubtask("subtractSky2d")
        self.makeSubtask("extractSpectra")
        self.makeSubtask("fitContinuum")

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
            was not performed.
        fiberTraceList : `list` of `pfs.drp.stella.FiberTraceSet`
            Fiber traces.
        detectorMapList : `list` of `pfs.drp.stella.DetectorMap`
            Mappings of wl,fiber to detector position.
        """
        self.log.info("Processing %s" % ([sensorRef.dataId for sensorRef in sensorRefList]))
        fiberTraceList = [self.getFiberTraces(sensorRef) for sensorRef in sensorRefList]
        detectorMapList = [self.getDetectorMap(sensorRef) for sensorRef in sensorRefList]
        pfsConfig = sensorRefList[0].get("pfsConfig")

        exposureList = []
        psfList = []
        lsfList = []
        skyResults = None
        if self.config.useCalexp:
            if all(sensorRef.datasetExists("calexp") for sensorRef in sensorRefList):
                self.log.info("Reading existing calexps")
                exposureList = [sensorRef.get("calexp") for sensorRef in sensorRefList]
                psfList = [exp.getPsf() for exp in exposureList]
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

            if self.config.doMeasurePsf:
                psfList = self.measurePsf.run(sensorRefList, exposureList, detectorMapList)
                lsfList = [self.calculateLsf(psf, ft, exp.getHeight()) for
                           psf, ft, exp in zip(psfList, fiberTraceList, exposureList)]
            else:
                psfList = [None]*len(sensorRefList)
                lsfList = [None]*len(sensorRefList)

            if self.config.doSubtractSky2d:
                skyResults = self.subtractSky2d.run(exposureList, pfsConfig, psfList,
                                                    fiberTraceList, detectorMapList)

        skyImageList = skyResults.imageList if skyResults is not None else [None]*len(exposureList)
        results = Struct(
            exposureList=exposureList,
            fiberTraceList=fiberTraceList,
            detectorMapList=detectorMapList,
            psfList=psfList,
            lsfList=lsfList,
            pfsConfig=pfsConfig,
            sky2d=skyResults.sky2d if skyResults is not None else None,
            skyImageList=skyImageList,
        )

        if self.config.doExtractSpectra:
            originalList = []
            spectraList = []
            for exposure, fiberTraces, detectorMap, skyImage in zip(exposureList, fiberTraceList,
                                                                    detectorMapList, skyImageList):
                spectra = self.extractSpectra.run(exposure.maskedImage, fiberTraces, detectorMap).spectra
                originalList.append(spectra)

                if self.config.doSubtractContinuum:
                    continua = self.fitContinuum.run(spectra)
                    exposure.maskedImage -= continua.makeImage(exposure.getBBox(), fiberTraces)
                    spectra = self.extractSpectra.run(exposure.maskedImage, fiberTraces,
                                                      detectorMap).spectra
                    # Set sky flux from continuum
                    for ss, cc in zip(spectra, continua):
                        ss.background += cc.spectrum

                if skyImage is not None:
                    skySpectra = self.extractSpectra.run(skyImage, fiberTraces, detectorMap).spectra
                    for spec, skySpec in zip(spectra, skySpectra):
                        spec.background += skySpec.spectrum

                spectraList.append(spectra)

            results.originalList = originalList
            results.spectraList = spectraList

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
                sensorRef.put(lsf, "lsf")
        if self.config.doSubtractSky2d:
            sensorRefList[0].put(results.sky2d, "sky2d")

        if self.config.doWriteArm:
            for sensorRef, spectra in zip(sensorRefList, results.spectraList):
                sensorRef.put(spectra.toPfsArm(sensorRef.dataId), "pfsArm")

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

    def getDetectorMap(self, sensorRef):
        """Get the appropriate detectorMap

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for sensor data.

        Returns
        -------
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        """
        detectorMap = sensorRef.get("detectorMap")
        if self.config.fiberDx != 0.0 or self.config.fiberDy != 0.0:
            slitOffsets = detectorMap.getSlitOffsets()
            slitOffsets[detectorMap.DX] += self.config.fiberDx
            slitOffsets[detectorMap.DY] += self.config.fiberDy
            detectorMap.setSlitOffsets(slitOffsets)
        return detectorMap

    def getFiberTraces(self, sensorRef):
        """Get the appropriate fiber trace set

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for sensor data.

        Returns
        -------
        fiberTraceSet : `pfs.drp.stella.FiberTraceSet`
            Set of fiber traces.
        """
        return sensorRef.get('fiberTrace')

    def calculateLsf(self, psf, fiberTraceSet, length):
        """Calculate the LSF for this exposure

        psf : `pfs.drp.stella.SpectralPsf`
            Point-spread function for spectral data.
        fiberTrace : `pfs.drp.stella.FiberTrace`
            Fiber profile.
        length : `int`
            Array length.

        Returns
        -------
        lsf : `dict` (`int`: `pfs.drp.stella.ExtractionLsf`)
            Line-spread functions, indexed by fiber identifier.
        """
        return {ft.fiberId: ExtractionLsf(psf, ft, length) for ft in fiberTraceSet}

    def _getMetadataName(self):
        return None
