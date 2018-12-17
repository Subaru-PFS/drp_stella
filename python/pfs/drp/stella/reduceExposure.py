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
from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.repair import RepairTask
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from .extractSpectraTask import ExtractSpectraTask
from .fitContinuum import FitContinuumTask

__all__ = ["ReduceExposureConfig", "ReduceExposureTask"]


class ReduceExposureConfig(pexConfig.Config):
    """Config for ReduceExposure"""
    isr = pexConfig.ConfigurableField(target=IsrTask, doc="Instrumental signature removal")
    doRepair = pexConfig.Field(dtype=bool, default=True, doc="Repair artifacts?")
    repair = pexConfig.ConfigurableField(target=RepairTask, doc="Task to repair artifacts")
    doWriteCalexp = pexConfig.Field(dtype=bool, default=False, doc="Write corrected frame?")
    doWriteArm = pexConfig.Field(dtype=bool, default=True, doc="Write PFS arm file?")
    doExtractSpectra = pexConfig.Field(dtype=bool, default=True, doc="Extract spectra from exposure?")
    extractSpectra = pexConfig.ConfigurableField(
        target=ExtractSpectraTask,
        doc="Task to extract spectra using the fibre traces",
    )
    doSubtractContinuum = pexConfig.Field(dtype=bool, default=False,
                                          doc="Subtract continuum as part of extraction?")
    fitContinuum = pexConfig.ConfigurableField(target=FitContinuumTask, doc="Fit continuum for subtraction")
    fiberDy = pexConfig.Field(doc="Offset to add to all FIBER_DY values (used when bootstrapping)",
                              dtype=float, default=0)
    doSkySwindle = pexConfig.Field(dtype=bool, default=False,
                                   doc="Apply sky swindle? We subtract the known sky from the simulator.")


## \addtogroup LSST_task_documentation
## \{
## \page ReduceExposureTask
## \ref ReduceExposureTask_ "ReduceExposureTask"
## \copybrief ReduceExposureTask
## \}


class ReduceExposureTask(pipeBase.CmdLineTask):
    """!Reduce a PFS exposures, generating pfsArm files

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
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("repair")
        self.makeSubtask("extractSpectra")
        self.makeSubtask("fitContinuum")

    @pipeBase.timeMethod
    def run(self, sensorRef, lines=None):
        """Process one CCD

        The sequence of operations is:
        - remove instrument signature
        - extract the spectra from the fiber traces
        - write the outputs

        Parameters
        ----------
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for sensort to process.
        lines : `list` of `pfs.drp.stella.ReferenceLine`, optional
            Reference lines.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Exposure data for sensor.
        spectra : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra.
        original : `pfs.drp.stella.SpectrumSet`
            Set of extracted spectra before continuum subtraction.
            Will be identical to ``spectra`` if continuum subtraction
            was not performed.
        detectorMap : `pfs.drp.stella.utils.DetectorMap`
            Mapping of wl,fiber to detector position.
        """
        self.log.info("Processing %s" % (sensorRef.dataId))

        exposure = self.isr.runDataRef(sensorRef).exposure

        if self.config.doSkySwindle:
            self.subtractSky(exposure, sensorRef)

        if self.config.doRepair:
            self.repairExposure(exposure)

        results = pipeBase.Struct(exposure=exposure)

        if self.config.doExtractSpectra:
            fiberTraces = self.getFiberTraces(sensorRef)
            detectorMap = self.getDetectorMap(sensorRef)
            spectra = self.extractSpectra.run(exposure.maskedImage, fiberTraces, detectorMap, lines).spectra
            results.original = spectra
            results.detectorMap = detectorMap

            if self.config.doSubtractContinuum:
                continua = self.fitContinuum.run(spectra)
                exposure.maskedImage -= continua.makeImage(exposure.getBBox(), fiberTraces)
                spectra = self.extractSpectra.run(exposure.maskedImage, fiberTraces,
                                                  detectorMap, lines).spectra

            results.spectra = spectra

        if self.config.doWriteCalexp:
            sensorRef.put(exposure, "calexp")
        if self.config.doWriteArm:
            # XXX set exposure.getMetadata() in spectra
            sensorRef.put(results.spectra, "pfsArm")

        return results

    def subtractSky(self, exposure, sensorRef):
        """Subtract the known sky image from the simulator

        This implements the "sky swindle", whereby we get the noise from the
        sky without the burden of actually measuring and subtracting the
        sky.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure from which to subtract the sky. Sky will be subtracted
            in-place.
        sensorRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for sensor.
        """
        import astropy.io.fits
        filename = sensorRef.getUri("raw")
        with astropy.io.fits.open(filename) as fd:
            sky = fd["SKY"].data
        exposure.image.array -= sky

    def repairExposure(self, exposure):
        """Repair CCD defects in the exposure

        Uses the PSF specified in the config.
        """
        modelPsfConfig = self.config.repair.interp.modelPsf
        psf = modelPsfConfig.apply()
        exposure.setPsf(psf)
        self.repair.run(exposure)

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
        detectorMap = sensorRef.get("detectormap")
        if self.config.fiberDy != 0.0:
            slitOffsets = detectorMap.getSlitOffsets()
            slitOffsets[detectorMap.FIBER_DY] += self.config.fiberDy
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
        return sensorRef.get('fibertrace')

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
