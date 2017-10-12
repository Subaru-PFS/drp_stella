#!/usr/bin/env python
import os
from lsst.utils import getPackageDir
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
from lsst.ctrl.pool.pool import NODE
import lsst.meas.algorithms as measAlg
import lsst.daf.persistence as dafPersist
from lsst.pex.config import Field, ConfigurableField
from lsst.pipe.drivers.constructCalibs import CalibTask
from pfs.drp.stella.constructFiberTraceTask import ConstructFiberTraceConfig
from lsst.pipe.drivers.utils import getDataRef
from lsst.pipe.tasks.repair import RepairTask
import math
import numpy as np
import pfs.drp.stella.utils as dsUtils

class ConstructFiberFlatConfig(ConstructFiberTraceConfig):
    """Configuration for flat construction"""
    minSNR = Field(
        doc = "Minimum Signal-to-Noise Ratio for normalized Flat pixels",
        dtype = float,
        default = 100.,
        check = lambda x : x > 0.)

class ConstructFiberFlatTask(CalibTask):
    """Task to construct the normalized Flat"""
    ConfigClass = ConstructFiberFlatConfig
    _DefaultName = "constructFiberFlat"
    calibName = "flat"

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")
        self.makeSubtask("trace")

        import lsstDebug
        self.debugInfo = lsstDebug.Info(__name__)

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for flat construction"""
        config.isr.doFlat = False
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
        return exposure

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
        self.log.info('len(dataRefList) = %d' % len(dataRefList))

        sumFlats, sumVariances = None, None

        detMap = dsUtils.makeDetectorMap(cache.butler, dataRefList[0].dataId, self.config.wavelengthFile)

        xOffsets = []
        for expRef in dataRefList:
            exposure = expRef.get('postISRCCD')
            md = exposure.getMetadata()

            if self.config.xOffsetHdrKeyWord not in md.names():
                self.log.warn("Keyword %s not found in metadata; ignoring flat for %s" %
                              (self.config.xOffsetHdrKeyWord, expRef.dataId))
                continue

            xOffsets.append(md.get(self.config.xOffsetHdrKeyWord))

            fts = self.trace.run(exposure, detMap)
            self.log.info('%d FiberTraces found for arm %d%s, visit %d' %
                          (fts.getNtrace(),
                           expRef.dataId['spectrograph'], expRef.dataId['arm'], expRef.dataId['visit']))

            if sumFlats is None:
                sumFlats = exposure.image
                sumVariances = exposure.variance

                sumRecIm = afwImage.ImageF(sumFlats.getDimensions())
                sumVarIm = afwImage.ImageF(sumFlats.getDimensions())
            else:
                sumFlats += exposure.image
                sumVariances += exposure.variance

            # Add all reconstructed FiberTraces of all dithered flats to one
            # reconstructed image 'sumRecIm'

            maskedImage = exposure.maskedImage
            for ft in fts.getTraces():
                profile = ft.getTrace()

                spectrum = ft.extractSpectrum(maskedImage, useProfile=True)
                recFt = ft.getReconstructed2DSpectrum(spectrum)
                if False:
                    recFt.array[profile.image.array <= 0] = 0.0

                bbox = profile.getBBox()
                if ft.getITrace() == -315:
                    import matplotlib.pyplot as plt

                    cen, hwidth = 2010, 4
                    plt.plot(maskedImage.image[cen - hwidth: cen + hwidth + 1, :].array.sum(axis=1), label='im')
                    plt.plot(spectrum.getSpectrum(), label='spec')
                    plt.plot(np.arange(recFt.getBBox().getMinY(), recFt.getBBox().getMaxY() + 1),
                             recFt.array.sum(axis=1), label='trace')
                    plt.legend(loc='best')
                    plt.show()

                    import pdb; pdb.set_trace() 

                sumRecIm[bbox] += recFt
                sumVarIm[bbox] += profile.variance

        if sumFlats is None:
            self.log.fatal("No flats were found with valid xOffset keyword %s" %
                           self.config.xOffsetHdrKeyWord)
            raise RuntimeError("Unable to find any valid flats")

        self.log.info('xOffsets = %s' % (xOffsets))

        sumVariances = sumVariances.array
        sumVariances[sumVariances <= 0.0] = 0.1
        snrArr = sumFlats.array/np.sqrt(sumVariances)
        #
        # Find and mask bad flat field pixels
        #
        with np.errstate(divide='ignore'):
            normalizedFlat = sumFlats.array/sumRecIm.array

        msk = np.zeros_like(normalizedFlat, dtype=afwImage.MaskPixel)

        bad = np.logical_or(np.logical_not(np.isfinite(snrArr)),
                            snrArr < self.config.minSNR)

        normalizedFlat[bad] = 1.0
        msk[bad] |= (1 << afwImage.Mask.addMaskPlane("BAD_FLAT"))

        normalizedFlat = afwImage.makeMaskedImage(afwImage.ImageF(normalizedFlat), afwImage.Mask(msk))

        import lsstDebug
        di = lsstDebug.Info(__name__)
        if di.display:
            import lsst.afw.display as afwDisplay

            if di.frames_flat >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_flat)
                display.mtv(normalizedFlat, title='normalized Flat')
                if di.zoomPan:
                    display.zoom(*zoomPan)

            if di.frames_meanFlats >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_meanFlats)
                display.mtv(afwImage.ImageF(sumFlats.array/len(dataRefList)), title='mean(Flats)')
                if di.zoomPan:
                    display.zoom(*zoomPan)

            if di.frames_meanTraces >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_meanTraces)
                display.mtv(afwImage.ImageF(sumRecIm.array/len(dataRefList)), title='mean(Traces)')
                if di.zoomPan:
                    display.zoom(*zoomPan)

            if di.frames_ratio >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_ratio)
                rat = sumFlats.array/sumRecIm.array
                rat[msk != 0] = np.nan
                display.mtv(afwImage.MaskedImageF(afwImage.ImageF(rat), normalizedFlat.mask),
                            title='mean(Flats)/mean(Traces)')
                if di.zoomPan:
                    display.zoom(*zoomPan)

        #Write fiber flat
        normFlatOut = afwImage.makeExposure(normalizedFlat)
        self.recordCalibInputs(cache.butler, normFlatOut, struct.ccdIdList, outputId)
        self.interpolateNans(normFlatOut)
        self.write(cache.butler, normFlatOut, outputId)
