from collections import defaultdict
import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.pex.config import Field, ConfigurableField

from .constructSpectralCalibs import SpectralCalibConfig, SpectralCalibTask
from .buildFiberProfiles import BuildFiberProfilesTask

__all__ = ["ConstructFiberFlatConfig", "ConstructFiberFlatTask"]


class ConstructFiberFlatConfig(SpectralCalibConfig):
    """Configuration for flat construction"""
    minSNR = Field(
        doc="Minimum Signal-to-Noise Ratio for normalized Flat pixels",
        dtype=float,
        default=50.,
        check=lambda x: x > 0.
    )
    profiles = ConfigurableField(target=BuildFiberProfilesTask, doc="Build fiber profiles")
    ditherRounding = Field(dtype=int, default=4, doc="Number of decimals for rounding the dither value")

    def setDefaults(self):
        self.combination.stats.maxVisitsToCalcErrorFromInputVariance = 5


class ConstructFiberFlatTask(SpectralCalibTask):
    """Task to construct the normalized flat"""
    ConfigClass = ConstructFiberFlatConfig
    _DefaultName = "fiberFlat"
    calibName = "flat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("profiles")

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for flat construction"""
        config.isr.doFlat = False
        config.isr.doFringe = False
        config.profiles.doBlindFind = True  # Because we've dithered the fiber positions

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
            value = value.pop()
            value = np.round(value, self.config.ditherRounding)
            dithers[value].append(dataRef)
        self.log.info("Dither values: %s" % (sorted(dithers.keys()),))

        # Sum coadded dithers to fill in the gaps
        sumFlat = None  # Sum of flat-fields
        sumExpect = None  # Sum of what we expect
        for dd in dithers:
            image = self.combination.run(dithers[dd])
            self.log.info("Combined %d images for dither %s", len(dithers[dd]), dd)

            # NaNs can appear in the image and variance planes from masked areas
            # on the CCD. NaNs can cause problems further downstream, so
            # we will interpolate over them.
            imgIsNan = np.isnan(image.image.array)
            self.interpolateNans(image.image)
            self.interpolateNans(image.variance)
            # Assign mask plane of INTRP.
            mask = image.mask
            mask.array[imgIsNan] |= mask.getPlaneBitMask(['INTRP'])

            # Check and correct for low values in variance
            self.correctLowVarianceImage(image)

            profileData = self.profiles.run(afwImage.makeExposure(image))
            if len(profileData.profiles) == 0:
                self.log.warn("No profiles found for dither %s: skipping", dd)
                continue
            self.log.info("%d fiber profiles found for dither %s", len(profileData.profiles), dd)
            maskVal = image.mask.getPlaneBitMask(["BAD", "SAT", "CR", "INTRP"])
            traces = profileData.profiles.makeFiberTraces(image.getDimensions(), profileData.centers)
            spectra = traces.extractSpectra(image, maskVal)
            self.log.info("Extracted %d for dither %s", len(spectra), dd)

            expect = spectra.makeImage(image.getBBox(), traces)
            # Occasionally NaNs are present in these images,
            # despite the original coadded image containing zero NaNs
            self.interpolateNans(expect)

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
        self.interpolateNans(flatExposure.image)
        self.interpolateNans(flatExposure.variance)
        self.write(cache.butler, flatExposure, outputId)

        return afwMath.binImage(flatExposure.image, self.config.binning)

    def correctLowVarianceImage(self, maskedImage, minVar=0.0):
        """Check the variance plane in the input image
        for low variance values
        and interpolate the variance if necessary.
        A corrsponding mask plane of 'NO_DATA' is assigned.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImageF`
            the input maskedImage, whose variance plane is corrected.
        minVar : `float`
            the minimum variance
        """
        varArr = maskedImage.getVariance().getArray()
        isLowVar = varArr < minVar
        varArr[isLowVar] = np.median(varArr[np.logical_not(isLowVar)])
        mask = maskedImage.getMask()
        mask.getArray()[isLowVar] |= mask.getPlaneBitMask(['NO_DATA'])
