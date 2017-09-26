#!/usr/bin/env python
import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
from lsst.ctrl.pool.pool import NODE
import lsst.meas.algorithms as measAlg
from lsst.pex.config import Field, ConfigurableField
from lsst.pipe.drivers.constructCalibs import CalibConfig, CalibTask
from lsst.pipe.drivers.utils import getDataRef
from lsst.pipe.tasks.repair import RepairTask
import math
import numpy as np
from pfs.drp.stella.findAndTraceAperturesTask import FindAndTraceAperturesTask

class ConstructFiberFlatConfig(CalibConfig):
    """Configuration for flat construction"""
    crGrow = Field(dtype=int,
        default=2,
        doc="Grow radius for CR (pixels)")
    doRepair = Field(dtype=bool,
        default=True,
        doc="Repair artifacts?")
    minSNR = Field(
        doc = "Minimum Signal-to-Noise Ratio for normalized Flat pixels",
         dtype = float,
        default = 100.,
         check = lambda x : x > 0.)
    psfFwhm = Field(dtype=float,
        default=3.0,
        doc="Repair PSF FWHM (pixels)")
    psfSize = Field(dtype=int,
        default=21,
        doc="Repair PSF size (pixels)")
    repair = ConfigurableField(target=RepairTask,
        doc="Task to repair artifacts")
    trace = ConfigurableField(target=FindAndTraceAperturesTask,
        doc="Task to trace apertures")
    xOffsetHdrKeyWord = Field(dtype=str,
                              default='sim.slit.xoffset',
                              doc="Header keyword for fiber offset in input files")

class ConstructFiberFlatTask(CalibTask):
    """Task to construct the normalized Flat"""
    ConfigClass = ConstructFiberFlatConfig
    _DefaultName = "constructFiberFlat"
    calibName = "flat"

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")
        self.makeSubtask("trace")

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for flat construction"""
        config.isr.doFlat = False
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

        allFts = []
        xOffsets = []
        for expRef in dataRefList:
            exposure = expRef.get('postISRCCD')
            md = exposure.getMetadata()

            if self.config.xOffsetHdrKeyWord not in md.names():
                self.log.warn("Keyword %s not found in metadata; ignoring flat for %s" %
                              (self.config.xOffsetHdrKeyWord, expRef.dataId))
                continue

            xOffsets.append(md.get(self.config.xOffsetHdrKeyWord))

            fts = self.trace.run(exposure)
            self.log.info('%d FiberTraces found for arm %d%s, visit %d' %
                          (fts.size(),
                           expRef.dataId['spectrograph'], expRef.dataId['arm'], expRef.dataId['visit']))
            allFts.append(fts)

            if sumFlats is None:
                sumFlats = exposure.getMaskedImage().getImage().getArray()
                sumVariances = exposure.getMaskedImage().getVariance().getArray()
            else:
                sumFlats += exposure.getMaskedImage().getImage().getArray()
                sumVariances += exposure.getMaskedImage().getVariance().getArray()

        if sumFlats is None:
            self.log.fatal("No flats were found with valid xOffset keyword %s" %
                           self.config.xOffsetHdrKeyWord)
            raise RuntimeError("Unable to find any valid flats")

        self.log.info('xOffsets = %s' % (xOffsets))

        sumRec = np.zeros(shape=sumFlats.shape, dtype='float32')
        sumRecIm = afwImage.ImageF(sumRec)
        sumVar = np.zeros(shape=sumFlats.shape, dtype='float32')
        sumVarIm = afwImage.ImageF(sumVar)

        # Add all reconstructed FiberTraces of all dithered flats to one
        # reconstructed image 'sumRecIm'
        for iFts in range(len(allFts)):
            fts = allFts[iFts]
            maskedImage = dataRefList[iFts].get('postISRCCD').getMaskedImage()
            for ft in fts.getTraces():
                spectrum = ft.extractFromProfile(maskedImage)
                recFt = ft.getReconstructed2DSpectrum(spectrum)
                recFtArr = recFt.getArray()
                imArr = ft.getTrace().getImage().getArray()
                recFtArr[imArr <= 0] = 0.0
                bbox = ft.getTrace().getBBox()
                sumRecIm[bbox] += recFt
                sumVarIm[bbox] = ft.getTrace().getVariance()
                    
        sumVariances[sumVariances <= 0.0] = 0.1
        snrArr = sumFlats / np.sqrt(sumVariances)

        sumRecImArr = sumRecIm.getArray()
        #to avoid division by zero and remove non-physical negative values,
        #we set the reconstructed values <= 0.0 to 0.01. We later set the normalized
        #Flat to unity where sumRecImArr <= 0.01, as well as all other pixels
        #with a SNR < self.config.minSNR
        sumRecImArr[sumRecImArr <= 0.0] = 0.01
        normalizedFlat = sumFlats / sumRecImArr
        msk = np.zeros_like(normalizedFlat, dtype=afwImage.MaskPixel)

        bad = np.logical_or.reduce([sumRecImArr <= 0.01,
                                    snrArr < self.config.minSNR,
                                    sumFlats <= 0.0,
                                    #sumVariances <= 0.0, # explicitly set to 0.1 above
        ])

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

            if di.frames_meanFlats >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_sumFlats)
                display.mtv(afwImage.ImageF(sumFlats/len(dataRefList)), title='mean(Flats)')

            if di.frames_meanTraces >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_sumTraces)
                display.mtv(afwImage.ImageF(sumRecImArr/len(dataRefList)), title='mean(Traces)')

            if di.frames_residuals >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_residuals)
                display.mtv(afwImage.ImageF((sumFlats - sumRecImArr)/len(dataRefList)),
                            title='mean(Flats - Traces)')

        #Write fiber flat
        normFlatOut = afwImage.makeExposure(normalizedFlat)
        self.recordCalibInputs(cache.butler, normFlatOut, struct.ccdIdList, outputId)
        self.interpolateNans(normFlatOut)
        self.write(cache.butler, normFlatOut, outputId)
