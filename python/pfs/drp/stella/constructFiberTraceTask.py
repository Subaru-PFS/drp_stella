#!/usr/bin/env python
from __future__ import division
import math
import os
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
from lsst.utils import getPackageDir
from pfs.datamodel.pfsFiberTrace import PfsFiberTrace
from .utils import addFiberTraceSetToMask, makeDetectorMapIO
from .findAndTraceAperturesTask import FindAndTraceAperturesTask
from .datamodelIO import PfsFiberTraceIO
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
        doc="""Task to extract spectra using the fibre traces""",
    )
    xOffsetHdrKeyWord = Field(
        dtype=str,
        default=None,
        doc="Header keyword for fiber offset in input files (set in obs_pfs)"
    )
    wavelengthForThroughputIsPercent = Field(
        dtype=bool,
        default=True,
        doc="""Interpret wavelengthForThroughput[01] as percentages of spectral range"""
    )
    wavelengthForThroughput0 = Field(
        dtype=float,
        default=20,
        doc="""Starting wavelength for integral of flux over wavelength used to estimate throughput.

        If wavelengthForThroughputIsPercent is True, interpret value as percentage of spectral range
        """
    )
    wavelengthForThroughput1 = Field(
        dtype=float,
        default=80,
        doc="""Ending wavelength for integral of flux over wavelength used to estimate throughput.

        If wavelengthForThroughputIsPercent is True, interpret value as percentage of spectral range
        """
    )

    def setDefaults(self):
        CalibConfig.setDefaults(self)
        self.doCameraImage = False  # We don't produce 2D images


class ConstructFiberTraceTask(CalibTask):
    """Task to construct the normalized Flat"""
    ConfigClass = ConstructFiberTraceConfig
    _DefaultName = "constructFiberTrace"
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

        if self.debugInfo.display and self.debugInfo.combined_frame >= 0:
            disp = afwDisplay.Display(frame=self.debugInfo.combined_frame)

            disp.mtv(calExp, "Combined")

        detMap = dataRefList[0].get('detectormap') 

        fts = self.trace.run(calExp, detMap)
        self.log.info('%d FiberTraces found on combined flat' % (fts.getNtrace()))

        if self.debugInfo.display and self.debugInfo.combined_frame >= 0:
            disp = afwDisplay.Display(frame=self.debugInfo.combined_frame)

            addFiberTraceSetToMask(calExp.getMaskedImage().getMask(), fts.getTraces(), disp)
            disp.setMaskTransparency(50)
            disp.mtv(calExp, "Traced")
        #
        # Use our new FiberTraceSet to extract the spectra so we can measure relative throughputs
        #
        spectrumSet = self.extractSpectra.run(calExp, fts, detMap).spectrumSet
        #
        # Integrate each spectrum over a fixed wavelength interval
        # to estimate the (relative) throughput
        #
        wavelengthForThroughput0 = self.config.wavelengthForThroughput0
        wavelengthForThroughput1 = self.config.wavelengthForThroughput1

        if self.config.wavelengthForThroughputIsPercent:
            spec = spectrumSet[len(spectrumSet)//2]
            wavelength = spec.getWavelength()
            
            wavelengthForThroughput0 = wavelength[int(0.01*wavelengthForThroughput0*len(wavelength))]
            wavelengthForThroughput1 = wavelength[int(0.01*wavelengthForThroughput1*len(wavelength))]

        throughput = {}
        for spec in spectrumSet:
            spl = UnivariateSpline(spec.wavelength, spec.spectrum)
            throughput[spec.getFiberId()] = spl.integral(wavelengthForThroughput0, wavelengthForThroughput1)

        med = np.median(list(throughput.values()))
        for fiberId in throughput:
            throughput[fiberId] /= med
        #
        # Update the DetectorMap
        #
        for fiberId in throughput:
            detMap.setThroughput(fiberId, throughput[fiberId])

        if self.debugInfo.display and self.debugInfo.plotTraces:
            import matplotlib.pyplot as plt
            for spec in spectrumSet:
                fiberId = spec.getFiberId()
                plt.plot(spec.getWavelength(), spec.getSpectrum()/detMap.getThroughput(fiberId),
                         label=fiberId)
            plt.legend(loc='best')
            plt.show()
        #
        # And write it
        #
        visitInfo = calExp.getInfo().getVisitInfo()
        if visitInfo is None:
            dateObs = dafBase.DateTime('%sT00:00:00Z' % dataRefList[0].dataId['dateObs'], dafBase.DateTime.UTC)
            visitInfo = afwImage.VisitInfo(date=dateObs)
        
        detMapIO = makeDetectorMapIO(detMap, visitInfo)
        dataRefList[0].put(detMapIO, 'detectormap', visit0=dataRefList[0].dataId['visit'])
        #
        # Package up fts (a FiberTraceSet) into a pfsFiberTrace for I/O;  these two classes
        # should be merged at some point
        #
        pfsFT = PfsFiberTrace(
            outputId['calibDate'],
            outputId['spectrograph'],
            outputId['arm']
        )

        for ft in fts:
            pfsFT.fiberId.append(ft.getFiberId())
            pfsFT.traces.append(ft.getTrace())

        # Add suitable metadata
        md = dafBase.PropertyList()
        for i, dataRef in enumerate(dataRefList):
            dataId = dataRef.dataId
            if i == 0:
                md.set('arm', dataId['arm'])
                md.set('spectrograph', dataId['spectrograph'])
            md.set('visit%03d' % i, dataId['visit'])

        pfsFiberTrace = PfsFiberTraceIO(pfsFT, md)
        self.write(cache.butler, pfsFiberTrace, outputId)
