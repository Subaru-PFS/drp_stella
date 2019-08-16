from collections import defaultdict
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

        # Coadd exposures taken with the same slit dither position to remove CRs
        dithers = defaultdict(list)
        for dataRef in dataRefList:
            value = dataRef.getButler().queryMetadata("raw", "dither", dataRef.dataId)
            assert len(value) == 1, "Expect a single answer for this single dataset"
            dithers[value.pop()].append(dataRef)
        self.log.info("Dither values: %s" % (sorted(dithers.keys()),))
        coadds = {dd: self.combination.run(dithers[dd]) for dd in dithers}

        # Sum coadded dithers to fill in the gaps
        sumFlat = None  # Sum of flat-fields
        sumExpect = None  # Sum of what we expect
        for dd in dithers:
            image = coadds[dd]
            dataRef = dithers[dd][0]  # Representative dataRef

            detMap = dataRef.get('detectormap')
            traces = self.trace.run(image, detMap)
            self.log.info('%d FiberTraces found for %s' % (traces.size(), dataRef.dataId))
            spectra = traces.extractSpectra(image, detMap, True)

            expect = spectra.makeImage(image.getBBox(), traces)

            maskVal = image.mask.getPlaneBitMask(["BAD", "SAT", "CR", "INTRP"])
            with np.errstate(invalid="ignore"):
                bad = (expect.array <= 0.0) | ((image.mask.array & maskVal) > 0)
            image.image.array[bad] = 0.0
            image.variance.array[bad] = 0.0
            expect.array[bad] = 0.0
            image.mask.array &= ~maskVal  # Remove planes we are masking so they don't leak through

            if sumFlat is None:
                sumFlat = image
                sumExpect = expect
            else:
                sumFlat += image
                sumExpect += expect

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
