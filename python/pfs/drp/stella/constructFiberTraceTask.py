import math
import numpy as np
from scipy.interpolate import UnivariateSpline

import lsst.daf.base as dafBase
import lsst.daf.persistence as dafPersist
import lsst.afw.detection as afwDet
import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
from lsst.ctrl.pool.pool import NODE
import lsst.meas.algorithms as measAlg
from lsst.pex.config import Field, ConfigurableField
from lsst.pipe.drivers.constructCalibs import CalibConfig, CalibTask
from lsst.pipe.tasks.repair import RepairTask
from lsst.pipe.drivers.utils import getDataRef
from .findAndTraceAperturesTask import FindAndTraceAperturesTask
from .extractSpectraTask import ExtractSpectraTask


class ConstructFiberTraceConfig(CalibConfig):
    """Configuration for FiberTrace construction"""
    rerunISR = Field(
        dtype=bool,
        default=True,
        doc="Rerun ISR even if postISRCCD is available (may be e.g. not flat fielded)"
    )
    crGrow = Field(
        dtype=int,
        default=2,
        doc="Grow radius for CR (pixels)"
    )
    doRepair = Field(
        dtype=bool,
        default=True,
        doc="Repair artifacts?"
    )
    psfFwhm = Field(
        dtype=float,
        default=3.0,
        doc="Repair PSF FWHM (pixels)"
    )
    psfSize = Field(
        dtype=int,
        default=21,
        doc="Repair PSF size (pixels)"
    )
    repair = ConfigurableField(
        target=RepairTask,
        doc="Task to repair artifacts"
    )
    trace = ConfigurableField(
        target=FindAndTraceAperturesTask,
        doc="Task to trace apertures"
    )
    extractSpectra = ConfigurableField(
        target=ExtractSpectraTask,
        doc="Task to extract spectra using the fibre traces",
    )
    requireZeroSlitOffset = Field(
        dtype=bool,
        default=True,
        doc="""Require a zero slit offset value?

        The fiber trace should have the same slit offset as the data, which is usually zero.
        """,
    )

    def setDefaults(self):
        CalibConfig.setDefaults(self)
        self.doCameraImage = False  # We don't produce 2D images


class ConstructFiberTraceTask(CalibTask):
    """Task to construct the normalized Flat"""
    ConfigClass = ConstructFiberTraceConfig
    _DefaultName = "fiberTrace"
    calibName = "fibertrace"

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")
        self.makeSubtask("trace")
        self.makeSubtask("extractSpectra")

        import lsstDebug
        self.debugInfo = lsstDebug.Info(__name__)

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for FiberTrace construction"""
        config.isr.doFringe = False

    def processSingle(self, sensorRef):
        """Process a single CCD

        Besides the regular ISR, also masks cosmic-rays.
        """
        if not self.config.rerunISR:
            try:
                exposure = sensorRef.get('postISRCCD')
                self.log.debug("Obtained postISRCCD from butler for %s" % sensorRef.dataId)
                return exposure
            except dafPersist.NoResults:
                pass                    # ah well.  We'll have to run the ISR

        exposure = CalibTask.processSingle(self, sensorRef)

        if self.config.doRepair:
            psf = measAlg.DoubleGaussianPsf(self.config.psfSize, self.config.psfSize,
                                            self.config.psfFwhm/(2*math.sqrt(2*math.log(2))))
            exposure.setPsf(psf)
            self.repair.run(exposure, keepCRs=False)
            if self.config.crGrow > 0:
                mask = exposure.getMaskedImage().getMask().clone()
                mask &= mask.getPlaneBitMask("CR")
                fpSet = afwDet.FootprintSet(mask, afwDet.Threshold(0.5))
                fpSet = afwDet.FootprintSet(fpSet, self.config.crGrow, True)
                fpSet.setMask(exposure.getMaskedImage().getMask(), "CR")

        if self.debugInfo.display and self.debugInfo.inputs_frame >= 0:
            disp = afwDisplay.Display(frame=self.debugInfo.inputs_frame)
            disp.mtv(exposure, "raw %(visit)d" % sensorRef.dataId)

        return exposure

    def run(self, expRefList, butler, calibId):
        if self.config.requireZeroSlitOffset:
            # Only run for Flats with slitOffset == 0.0
            rejected = []
            newExpRefList = []
            for expRef in expRefList:
                slitOffset = expRef.getButler().queryMetadata("raw", "slitOffset", expRef.dataId)
                assert len(slitOffset) == 1, "Expect a single answer for this single dataset"
                slitOffset = slitOffset.pop()
                if slitOffset == 0.0:
                    newExpRefList.append(expRef)
                else:
                    rejected.apped(expRef.dataId)
            if rejected:
                self.log.warn("Rejected the following exposures with non-zero slitOffset: %s", rejected)
                self.log.warn("To overcome this, either set 'requireZeroSlitOffset=False' or select only "
                              "input exposures with zero slitOffset")
            expRefList = newExpRefList

        if not expRefList:
            raise RuntimeError("No input exposures")

        return CalibTask.run(self, expRefList, butler, calibId)

    def getOutputId(self, expRefList, calibId):
        """Generate the data identifier for the output calib

        The mean date and the common filter are included, using keywords
        from the configuration.  The CCD-specific part is not included
        in the data identifier.

        This override implementation adds ``visit0`` to the output identifier.

        Parameters
        ----------
        expRefList : `list` of `lsst.daf.persistence.ButlerDataRef`
            List of data references for input exposures.
        calibId : `dict`
            Data identifier elements for the calib, provided by the user.

        Returns
        -------
        outputId : `dict`
            Data identifier for output.
        """
        outputId = CalibTask.getOutputId(self, expRefList, calibId)
        outputId["visit0"] = expRefList[0].dataId['visit']
        return outputId

    def combine(self, cache, struct, outputId):
        """!Combine multiple exposures of a particular CCD and write the output

        Only the slave nodes execute this method.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        struct : `lsst.pipe.base.Struct`
            Parameters for the combination, which has the following components:

            - ``ccdName`` (`tuple`): Name tuple for CCD.
            - ``ccdIdList`` (`list`): List of data identifiers for combination.
            - ``scales``: Unused by this implementation.

        Returns
        -------
        outputId : `dict`
            Data identifier for combined image (exposure part only).
        """
        # Check if we need to look up any keys that aren't in the output dataId
        fullOutputId = {k: struct.ccdName[i] for i, k in enumerate(self.config.ccdKeys)}
        self.addMissingKeys(fullOutputId, cache.butler)
        fullOutputId.update(outputId)  # must be after the call to queryMetadata
        outputId = fullOutputId
        del fullOutputId

        dataRefList = [getDataRef(cache.butler, dataId) if dataId is not None else None for
                       dataId in struct.ccdIdList]

        self.log.info("Combining %s on %s" % (outputId, NODE))
        calib = self.combination.run(dataRefList, expScales=struct.scales.expScales,
                                     finalScale=struct.scales.ccdScale)
        calExp = afwImage.makeExposure(calib)

        self.interpolateNans(calExp)

        if self.debugInfo.display and self.debugInfo.combined_frame >= 0:
            disp = afwDisplay.Display(frame=self.debugInfo.combined_frame)
            disp.mtv(calExp, "Combined")

        detMap = dataRefList[0].get('detectormap')

        traces = self.trace.run(calExp.maskedImage, detMap)
        self.log.info('%d fiber traces found on combined flat' % (traces.size(),))

        if self.debugInfo.display and self.debugInfo.combined_frame >= 0:
            disp = afwDisplay.Display(frame=self.debugInfo.combined_frame)
            traces.applyToMask(calExp.getMaskedImage().getMask())
            disp.setMaskTransparency(50)
            disp.mtv(calExp, "Traces")
        #
        # And write it
        #
        visitInfo = calExp.getInfo().getVisitInfo()
        if visitInfo is None:
            dateObs = dafBase.DateTime('%sT00:00:00Z' % dataRefList[0].dataId['dateObs'],
                                       dafBase.DateTime.UTC)
            visitInfo = afwImage.VisitInfo(date=dateObs)

        # Clear out metadata to avoid conflicts with existing keywords when we set the stuff we need
        for key in detMap.metadata.names():
            detMap.metadata.remove(key)
        detMap.setVisitInfo(visitInfo)
        self.recordCalibInputs(cache.butler, detMap, struct.ccdIdList, outputId)
        detMap.getMetadata().set("OBSTYPE", "detectormap")  # Overwrite "fibertrace"

        dataRefList[0].put(detMap, 'detectormap', visit0=dataRefList[0].dataId['visit'])

        self.recordCalibInputs(cache.butler, traces, struct.ccdIdList, outputId)
        self.write(cache.butler, traces, outputId)
