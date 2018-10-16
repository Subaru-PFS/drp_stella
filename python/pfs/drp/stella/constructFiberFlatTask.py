import math
import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.detection as afwDet
from lsst.ctrl.pool.pool import NODE
import lsst.meas.algorithms as measAlg
import lsst.daf.persistence as dafPersist
from lsst.pex.config import Field, ConfigurableField
from lsst.pipe.drivers.constructCalibs import CalibTask
from lsst.pipe.drivers.utils import getDataRef

from pfs.drp.stella.constructFiberTraceTask import ConstructFiberTraceConfig
from pfs.drp.stella import Spectrum
from pfs.drp.stella.fitContinuum import FitContinuumTask

__all__ = ["ConstructFiberFlatConfig", "ConstructFiberFlatTask"]


class ConstructFiberFlatConfig(ConstructFiberTraceConfig):
    """Configuration for flat construction"""
    minSNR = Field(
        doc="Minimum Signal-to-Noise Ratio for normalized Flat pixels",
        dtype=float,
        default=50.,
        check=lambda x: x > 0.
    )
    fitContinuum = ConfigurableField(target=FitContinuumTask, doc="Fit continuum")


class ConstructFiberFlatTask(CalibTask):
    """Task to construct the normalized flat"""
    ConfigClass = ConstructFiberFlatConfig
    _DefaultName = "fiberFlat"
    calibName = "flat"

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")
        self.makeSubtask("trace")
        self.makeSubtask("fitContinuum")

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

        sumFlat = None  # Sum of flat-fields
        sumExpect = None  # Sum of what we expect
        xOffsets = []
        for ii, expRef in enumerate(dataRefList):
            exposure = expRef.get('postISRCCD')

            slitOffset = expRef.getButler().queryMetadata("raw", "slitOffset", expRef.dataId)
            assert len(slitOffset) == 1, "Expect a single answer for this single dataset"
            xOffsets.append(slitOffset.pop())

            detMap = expRef.get('detectormap')
            traces = self.trace.run(exposure.maskedImage, detMap)
            self.log.info('%d FiberTraces found for %s' % (traces.size(), expRef.dataId))
            spectra = traces.extractSpectra(exposure.maskedImage, detMap, True)
            # Get median spectrum across rows.
            # This is not the same as averaging over wavelength, and it matters: small features like
            # absorption lines are a function of wavelength, so can show up on different rows. We'll
            # work around this by fitting the continuum and using that instead of the average. That
            # also avoids having sharp features in the average spectrum, which should mean that the
            # flux calibration shouldn't have to include sharp features either (except for telluric lines).
            average = Spectrum(spectra.getLength())
            for ss in spectra:
                ss.spectrum[:] = np.where(np.isfinite(ss.spectrum), ss.spectrum, 0.0)
            average.spectrum[:] = np.median([ss.spectrum for ss in spectra], axis=0)
            average.spectrum = self.fitContinuum.fitContinuum(average)

            expect = afwImage.ImageF(exposure.getBBox())
            expect.set(0.0)

            for ft in traces:
                expect[ft.trace.getBBox()] += ft.constructImage(average)

            maskVal = exposure.mask.getPlaneBitMask(["BAD", "SAT", "CR"])
            bad = (expect.array <= 0.0) | (exposure.mask.array & maskVal > 0)
            exposure.image.array[bad] = 0.0
            exposure.variance.array[bad] = 0.0
            expect.array[bad] = 0.0

            if sumFlat is None:
                sumFlat = exposure.maskedImage
                sumExpect = expect
            else:
                sumFlat += exposure.maskedImage
                sumExpect += expect

        self.log.info('xOffsets = %s' % (xOffsets,))
        if sumFlat is None:
            raise RuntimeError("Unable to find any valid flats")
        if np.all(sumExpect.array == 0.0):
            raise RuntimeError("No good pixels")

        # Avoid NANs when dividing
        empty = sumExpect.array == 0
        sumFlat.image.array[empty] = 1.0
        sumFlat.variance.array[empty] = 1.0
        sumExpect.array[empty] = 1.0
        sumFlat.mask.addMaskPlane("BAD_FLAT")
        badFlat = sumFlat.mask.getPlaneBitMask("BAD_FLAT")

        sumFlat /= sumExpect
        sumFlat.mask.array[empty] |= badFlat

        # Mask bad pixels
        snr = sumFlat.image.array/np.sqrt(sumFlat.variance.array)
        bad = (snr < self.config.minSNR) | ~np.isfinite(snr)
        sumFlat.image.array[bad] = 1.0
        sumFlat.mask.array[bad] |= badFlat

        import lsstDebug
        di = lsstDebug.Info(__name__)
        if di.display:
            import lsst.afw.display as afwDisplay

            if di.frames_flat >= 0:
                display = afwDisplay.getDisplay(frame=di.frames_flat)
                display.mtv(sumFlat, title='normalized Flat')
                if di.zoomPan:
                    display.zoom(*di.zoomPan)

        # Write fiber flat
        flatExposure = afwImage.makeExposure(sumFlat)
        self.recordCalibInputs(cache.butler, flatExposure, struct.ccdIdList, outputId)
        self.interpolateNans(flatExposure)
        self.write(cache.butler, flatExposure, outputId)
