import numpy as np

import lsst.afw.image as afwImage
from lsst.pex.config import Field, ConfigurableField

from .constructSpectralCalibs import SpectralCalibConfig, SpectralCalibTask
from .findAndTraceAperturesTask import FindAndTraceAperturesTask

__all__ = ["ConstructFiberFlatConfig", "ConstructFiberFlatTask"]


class ConstructFiberFlatConfig(SpectralCalibConfig):
    """Configuration for flat construction"""
    minSNR = Field(
        doc="Minimum Signal-to-Noise Ratio for normalized Flat pixels",
        dtype=float,
        default=50.,
        check=lambda x: x > 0.
    )
    trace = ConfigurableField(target=FindAndTraceAperturesTask, doc="Task to trace apertures")


class ConstructFiberFlatTask(SpectralCalibTask):
    """Task to construct the normalized flat"""
    ConfigClass = ConstructFiberFlatConfig
    _DefaultName = "fiberFlat"
    calibName = "flat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("trace")

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for flat construction"""
        config.isr.doFlat = False
        config.isr.doFringe = False

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
        combineResults = super().combine(cache, struct, outputId)
        dataRefList = combineResults.dataRefList
        outputId = combineResults.outputId

        sumFlat = None  # Sum of flat-fields
        sumExpect = None  # Sum of what we expect
        xOffsets = []
        for ii, expRef in enumerate(dataRefList):
            exposure = expRef.get('postISRCCD')

            dither = expRef.getButler().queryMetadata("raw", "dither", expRef.dataId)
            assert len(dither) == 1, "Expect a single answer for this single dataset"
            xOffsets.append(dither.pop())

            detMap = expRef.get('detectormap')
            traces = self.trace.run(exposure.maskedImage, detMap)
            self.log.info('%d FiberTraces found for %s' % (traces.size(), expRef.dataId))
            spectra = traces.extractSpectra(exposure.maskedImage, detMap, True)

            expect = spectra.makeImage(exposure.getBBox(), traces)

            maskVal = exposure.mask.getPlaneBitMask(["BAD", "SAT", "CR"])
            with np.errstate(invalid="ignore"):
                bad = (expect.array <= 0.0) | ((exposure.mask.array & maskVal) > 0)
            exposure.image.array[bad] = 0.0
            exposure.variance.array[bad] = 0.0
            expect.array[bad] = 0.0

            if sumFlat is None:
                sumFlat = exposure.maskedImage
                sumExpect = expect
            else:
                sumFlat += exposure.maskedImage
                sumExpect += expect

        self.log.info('Dither values = %s' % (xOffsets,))
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
        with np.errstate(invalid="ignore"):
            bad = (snr < self.config.minSNR) | ~np.isfinite(snr)
        sumFlat.image.array[bad] = 1.0
        sumFlat.mask.array[bad] |= badFlat

        import lsstDebug
        di = lsstDebug.Info(__name__)
        if di.display:
            import lsst.afw.display as afwDisplay

            if di.framesFlat >= 0:
                display = afwDisplay.getDisplay(frame=di.framesFlat)
                display.mtv(sumFlat, title='normalized Flat')
                if di.zoomPan:
                    display.zoom(*di.zoomPan)

        # Write fiber flat
        flatExposure = afwImage.makeExposure(sumFlat)
        self.recordCalibInputs(cache.butler, flatExposure, struct.ccdIdList, outputId)
        self.interpolateNans(flatExposure)
        self.write(cache.butler, flatExposure, outputId)
