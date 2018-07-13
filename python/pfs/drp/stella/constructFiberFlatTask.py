import math
import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
from lsst.ctrl.pool.pool import NODE
import lsst.meas.algorithms as measAlg
import lsst.daf.persistence as dafPersist
from lsst.pex.config import Field
from lsst.pipe.drivers.constructCalibs import CalibTask
from lsst.pipe.drivers.utils import getDataRef

from pfs.drp.stella.constructFiberTraceTask import ConstructFiberTraceConfig

__all__ = ["ConstructFiberFlatConfig", "ConstructFiberFlatTask"]


class ConstructFiberFlatConfig(ConstructFiberTraceConfig):
    """Configuration for flat construction"""
    minSNR = Field(
        doc="Minimum Signal-to-Noise Ratio for normalized Flat pixels",
        dtype=float,
        default=50.,
        check=lambda x: x > 0.
    )


class ConstructFiberFlatTask(CalibTask):
    """Task to construct the normalized flat"""
    ConfigClass = ConstructFiberFlatConfig
    _DefaultName = "fiberFlat"
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
        """Combine multiple exposures of a particular CCD and write the output

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
        self.log.info('len(dataRefList) = %d' % len(dataRefList))

        sumFlats, sumVariances = None, None

        detMap = dataRefList[0].get('detectormap')

        xOffsets = []
        for expRef in dataRefList:
            exposure = expRef.get('postISRCCD')

            slitOffset = expRef.getButler().queryMetadata("raw", "slitOffset", expRef.dataId)
            assert len(slitOffset) == 1, "Expect a single answer for this single dataset"
            xOffsets.append(slitOffset.pop())

            traces = self.trace.run(exposure.maskedImage, detMap)
            self.log.info('%d FiberTraces found for %s' % (traces.size(), expRef.dataId))

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
            for ft in traces:
                profile = ft.getTrace()

                spectrum = ft.extractSpectrum(maskedImage, useProfile=True)
                recFt = ft.constructImage(spectrum)

                bbox = profile.getBBox()
                sumRecIm[bbox] += recFt
                sumVarIm[bbox] += profile.variance

        self.log.info('xOffsets = %s' % (xOffsets))
        if sumFlats is None:
            raise RuntimeError("Unable to find any valid flats")

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
                    display.zoom(*di.zoomPan)

            if di.frames_meanFlats >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_meanFlats)
                display.mtv(afwImage.ImageF(sumFlats.array/len(dataRefList)), title='mean(Flats)')
                if di.zoomPan:
                    display.zoom(*di.zoomPan)

            if di.frames_meanTraces >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_meanTraces)
                display.mtv(afwImage.ImageF(sumRecIm.array/len(dataRefList)), title='mean(Traces)')
                if di.zoomPan:
                    display.zoom(*di.zoomPan)

            if di.frames_ratio >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_ratio)
                rat = sumFlats.array/sumRecIm.array
                rat[msk != 0] = np.nan
                display.mtv(afwImage.MaskedImageF(afwImage.ImageF(rat), normalizedFlat.mask),
                            title='mean(Flats)/mean(Traces)')
                if di.zoomPan:
                    display.zoom(*di.zoomPan)

        # Write fiber flat
        normFlatOut = afwImage.makeExposure(normalizedFlat)
        self.recordCalibInputs(cache.butler, normFlatOut, struct.ccdIdList, outputId)
        self.interpolateNans(normFlatOut)
        self.write(cache.butler, normFlatOut, outputId)
