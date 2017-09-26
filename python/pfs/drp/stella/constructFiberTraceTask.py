#!/usr/bin/env python
import math
import os

import lsst.afw.detection as afwDet
import lsst.afw.display as afwDisplay
import lsst.afw.image as afwImage
from lsst.ctrl.pool.pool import NODE
import lsst.meas.algorithms as measAlg
from lsst.pex.config import Field, ConfigurableField
from lsst.pipe.drivers.constructCalibs import CalibConfig, CalibTask
from lsst.pipe.tasks.repair import RepairTask
from lsst.pipe.drivers.utils import getDataRef
from lsst.utils import getPackageDir
from pfs.datamodel.pfsFiberTrace import PfsFiberTrace
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import readWavelengthFile, addFiberTraceSetToMask
from pfs.drp.stella.findAndTraceAperturesTask import FindAndTraceAperturesTask
from pfs.drp.stella.datamodelIO import PfsFiberTraceIO

class ConstructFiberTraceConfig(CalibConfig):
    """Configuration for FiberTrace construction"""
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
    fiberPixelFile = Field(
        dtype=str,
        default=os.path.join(getPackageDir("obs_pfs"), "pfs/RedFiberPixels.fits.gz"),
        doc="File containing the map for fiber pixels"
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
    xOffsetHdrKeyWord = Field(
        dtype=str,
        default='sim.slit.xoffset',
        doc="Header keyword for fiber offset in input files"
    )

    def setDefaults(self):
        CalibConfig.setDefaults(self)

class ConstructFiberTraceTask(CalibTask):
    """Task to construct the normalized Flat"""
    ConfigClass = ConstructFiberTraceConfig
    _DefaultName = "constructFiberTrace"
    calibName = "fibertrace"

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")
        self.makeSubtask("trace")

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

        if self.debugInfo.display and self.debugInfo.display_inputs:
            disp = afwDisplay.Display(frame=self.debugInfo.inputs_frame)

            visit = sensorRef.dataId['visit']
            disp.mtv(exposure, "raw %d" % (visit))

        return exposure

    def run(self, expRefList, butler, calibId):
        #Only run for Flats with xOffset == 0.0
        newExpRefList = []
        for expRef in expRefList:
            exposure = expRef.get('raw')
            md = exposure.getMetadata()

            if self.config.xOffsetHdrKeyWord not in md.names():
                self.log.warn("Keyword %s not found in metadata; ignoring flat for %s" %
                              (self.config.xOffsetHdrKeyWord, expRef.dataId))
            else:
                if md.get(self.config.xOffsetHdrKeyWord) == 0.:
                    newExpRefList.append(expRef)

        if len(newExpRefList) == 0:
            self.log.fatal("No flats were found with valid xOffset keyword %s" %
                           self.config.xOffsetHdrKeyWord)
            raise RuntimeError("Unable to find any valid flats")

        return CalibTask.run(self, newExpRefList, butler, calibId)


    def combine(self, cache, struct, outputId):
        """!Combine multiple exposures of a particular CCD and write the output

        Only the slave nodes execute this method.

        @param cache  Process pool cache
        @param struct  Parameters for the combination, which has the following components:
            * ccdName     Name tuple for CCD
            * ccdIdList   List of data identifiers for combination
            * scales      Scales to apply (expScales are scalings for each exposure,
                               ccdScale is final scale for combined image)
        @param outputId    Data identifier for combined image (exposure part only)
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

        self.recordCalibInputs(cache.butler, calExp, struct.ccdIdList, outputId)

        self.interpolateNans(calExp)

        if self.debugInfo.display:
            disp = afwDisplay.Display(frame=self.debugInfo.combined_frame)

            disp.mtv(calExp, "Combined")

        fts = self.trace.run(calExp)
        self.log.info('%d FiberTraces found on combined flat' % (fts.size()))

        # assign trace IDs
        xCenters, wavelengths, traceIds = readWavelengthFile(self.config.fiberPixelFile)
        fts.assignTraceIDs(traceIds, xCenters)

        if self.debugInfo.display:
            disp = afwDisplay.Display(frame=self.debugInfo.combined_frame)

            addFiberTraceSetToMask(calExp.getMaskedImage().getMask(), fts.getTraces(), disp)
            disp.setMaskTransparency(50)
            disp.mtv(calExp, "Traced")
        #
        # Package up fts (a FiberTraceSet) into a pfsFiberTrace for I/O;  these two classes
        # should be merged at some point
        #
        pfsFT = PfsFiberTrace(
            outputId['calibDate'],
            outputId['spectrograph'],
            outputId['arm']
        )

        for iFt in range(fts.size()):
            ft = fts.getFiberTrace(iFt)
            pfsFT.fiberId.append(ft.getITrace() + 1)
            pfsFT.traces.append(ft.getTrace())

        pfsFiberTrace = PfsFiberTraceIO(pfsFT)
        self.write(cache.butler, pfsFiberTrace, outputId)
