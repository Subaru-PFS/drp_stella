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

        sumFlats, sumTrace = None, None

        detMap = dataRefList[0].get('detectormap')

        xOffsets = []
        for ii, expRef in enumerate(dataRefList):
            exposure = expRef.get('postISRCCD')

            slitOffset = expRef.getButler().queryMetadata("raw", "slitOffset", expRef.dataId)
            assert len(slitOffset) == 1, "Expect a single answer for this single dataset"
            xOffsets.append(slitOffset.pop())

            traces = self.trace.run(exposure.maskedImage, detMap)
            self.log.info('%d FiberTraces found for %s' % (traces.size(), expRef.dataId))

            maskVal = exposure.mask.getPlaneBitMask(["BAD", "SAT", "CR"])

            traceImage = afwImage.ImageF(exposure.getBBox())
            traceImage.set(0)

            for ft in traces:
                spectrum = ft.extractSpectrum(exposure.maskedImage, useProfile=True)
                reconstructed = ft.constructImage(spectrum)
                bbox = reconstructed.getBBox()
                traceImage[bbox] += reconstructed

                bad = (reconstructed.array <= 0.0) | (exposure.mask[bbox].array & maskVal > 0)
                sub = exposure.maskedImage[bbox]                
                sub.image.array[bad] = 0.0
                sub.variance.array[bad] = 0.0
                traceImage[bbox].array[bad] = 0.0

            unused = traceImage.array == 0.0
            exposure.image.array[unused] = 0.0
            exposure.variance.array[unused] = 0.0

            if sumFlats is None:
                sumFlats = exposure.maskedImage
                sumTrace = traceImage
            else:
                sumFlats += exposure.maskedImage
                sumTrace += traceImage

        self.log.info('xOffsets = %s' % (xOffsets))
        if sumFlats is None:
            raise RuntimeError("Unable to find any valid flats")

        # Divide through by the number of good contributions, but avoid dividing by zero
        empty = sumTrace.array <= 0
        sumTrace.array[empty] = 1.0
        sumTrace.array[empty] = 1.0
        sumFlats.image.array[empty] = 1.0
        sumFlats.variance.array[empty] = 1.0
        sumFlats /= sumTrace
        normalizedFlat = sumFlats

        # Mask bad pixels
        snr = normalizedFlat.image.array/np.sqrt(normalizedFlat.variance.array)
        bad = (snr < self.config.minSNR) | ~np.isfinite(snr)
        normalizedFlat.image.array[bad] = 1.0
        normalizedFlat.mask.array[bad] = (1 << normalizedFlat.mask.addMaskPlane("BAD_FLAT"))

        import lsstDebug
        di = lsstDebug.Info(__name__)
        if di.display:
            import lsst.afw.display as afwDisplay

            if di.frames_flat >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_flat)
                display.mtv(normalizedFlat, title='normalized Flat')
                if di.zoomPan:
                    display.zoom(*di.zoomPan)

        # Write fiber flat
        normFlatOut = afwImage.makeExposure(normalizedFlat)
        self.recordCalibInputs(cache.butler, normFlatOut, struct.ccdIdList, outputId)
        self.interpolateNans(normFlatOut)
        self.write(cache.butler, normFlatOut, outputId)
