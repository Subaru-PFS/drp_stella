from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import ClassVar, Iterable, List, Type, TYPE_CHECKING

import numpy as np
from lsst.afw.detection import setMaskFromFootprintList
from lsst.afw.image import MaskedImageF
from lsst.ip.isr.isrFunctions import growMasks
from lsst.meas.algorithms import findCosmicRays, FindCosmicRaysConfig, GaussianPsfFactory
from lsst.pex.config import Config, ChoiceField, ConfigField, ConfigurableField, DictField, Field
from lsst.pex.config import makePropertySet
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connections import QuantaAdjuster
from pfs.drp.stella.utils.math import robustRms

from .repair import PfsRepairTask

if TYPE_CHECKING:
    from lsst.daf.butler import Butler, DataCoordinate
    from lsst.afw.image import Exposure, MaskedImage, Mask
    from lsst.afw.detection import Psf

__all__ = ("CosmicRayTask",)


class CosmicRayConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    inputExposures = InputConnection(
        name="postISRCCD",
        doc="Exposure to repair",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    masks = OutputConnection(
        name="crMask",
        doc="Cosmic ray mask",
        storageClass="Mask",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    outputExposures = OutputConnection(
        name="calexp",
        doc="Exposure with cosmic rays masked",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config:
            return
        if not self.config.doWriteExposure:
            self.outputs.remove("outputExposures")

    def groupingAuto(self, instrument: str, visitList: Iterable[int], butler: Butler) -> dict[int, int]:
        """Adjust the quanta using the 'auto' grouping algorithm.

        We group visits by their pfsDesignId and visit0, which are stored in the
        pfsConfig.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument.
        visitList : `iterable` of `int`
            List of visit numbers to group.
        butler : `lsst.daf.butler.Butler`
            Butler to use to retrieve data.

        Returns
        -------
        grouping : `dict` mapping `int` to `int`
            Mapping of visit to group numbers.
        """
        pfsConfigList = {
            visit: butler.get("pfsConfig", instrument=instrument, visit=visit) for visit in visitList
        }
        groupMapping = {
            visit: (
                pfsConfig.pfsDesignId,
                pfsConfig.visit0,
                butler.query_dimension_records("visit", instrument=instrument, visit=visit)[0].lamps
            ) for visit, pfsConfig in pfsConfigList.items()
        }
        groups = set(groupMapping.values())
        groupIds = dict(zip(groups, range(len(groups))))
        return {dataId: groupIds[gg] for dataId, gg in groupMapping.items()}

    def groupingSeparate(self, instrument: str, visitList: Iterable[int], butler: Butler) -> dict[int, int]:
        """Adjust the quanta using the separate grouping algorithm.

        Visits are assigned to separate groups.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument.
        visitList : `iterable` of `int`
            List of visit numbers to group.
        butler : `lsst.daf.butler.Butler`
            Butler to use to retrieve data (unused).

        Returns
        -------
        grouping : `dict` mapping `int` to `int`
            Mapping of visit to group numbers.
        """
        return dict(zip(visitList, itertools.count()))

    def groupingAll(self, instrument: str, visitList: Iterable[int], butler: Butler) -> dict[int, int]:
        """Adjust the quanta using the all grouping algorithm.

        All visits are assigned to the same group.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument.
        visitList : `iterable` of `int`
            List of visit numbers to group.
        butler : `lsst.daf.butler.Butler`
            Butler to use to retrieve data (unused).

        Returns
        -------
        grouping : `dict` mapping `int` to `int`
            Mapping of visit to group numbers.
        """
        return {visit: 0 for visit in visitList}

    def groupingManual(self, instrument: str, visitList: Iterable[int], butler: Butler) -> dict[int, int]:
        """Adjust the quanta using the manual grouping algorithm.

        Visits are assigned to groups according to the ``groups`` configuration
        parameter.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument.
        visitList : `iterable` of `int`
            List of visit numbers to group.
        butler : `lsst.daf.butler.Butler`
            Butler to use to retrieve data (unused).

        Returns
        -------
        grouping : `dict` mapping `int` to `int`
            Mapping of visit to group numbers.
        """
        groups = self.config.groups
        return {visit: groups[visit] for visit in visitList}

    def adjust_all_quanta(self, adjuster: QuantaAdjuster) -> None:
        """Customize the set of quanta predicted for this task during quantum
        graph generation.

        Parameters
        ----------
        adjuster : `QuantaAdjuster`
            A helper object that implementations can use to modify the
            under-construction quantum graph.

        Notes
        -----
        This hook is called before `adjustQuantum`, which is where built-in
        checks for `NoWorkFound` cases and missing prerequisites are handled.
        This means that the set of preliminary quanta seen by this method could
        include some that would normally be dropped later.
        """
        dataIdList = list(adjuster.iter_data_ids())
        if len(dataIdList) == 0:
            return

        instrument = set(dataId["instrument"] for dataId in dataIdList)
        if len(instrument) != 1:
            raise RuntimeError("Cannot group data from different instruments")
        instrument = instrument.pop()

        # Get the groups
        menu = dict(
            auto=self.groupingAuto,
            separate=self.groupingSeparate,
            all=self.groupingAll,
            manual=self.groupingManual,
        )
        method = menu[self.config.grouping]
        visitList = set(dataId["visit"] for dataId in dataIdList)
        grouping: Mapping[int, int] = method(instrument, visitList, adjuster.butler)  # visit -> group
        groups: set[int] = set(grouping.values())

        # Sort the dataId list by arm+spectrograph
        camera: dict[(str, int), list[DataCoordinate]] = defaultdict(list)
        for dataId in dataIdList:
            name = (dataId["arm"], dataId["spectrograph"])
            camera[name].append(dataId)

        # Divide each list of dataIds into groups
        for dataIdList in camera.values():
            target: dict[int, list[DataCoordinate]] = {gg: [] for gg in groups}  # group -> list of dataIds
            for dataId in dataIdList:
                visit = dataId["visit"]
                target[grouping[visit]].append(dataId)
            for coordList in target.values():
                if len(coordList) <= 1:
                    continue
                coordList.sort()
                first = coordList[0]
                for other in coordList[1:]:
                    adjuster.add_input(first, "inputExposures", other)
                    if self.config.doWriteExposure:
                        adjuster.move_output(first, "outputExposures", other)
                    adjuster.move_output(first, "masks", other)
                    adjuster.remove_quantum(other)


class CosmicRayConfig(PipelineTaskConfig, pipelineConnections=CosmicRayConnections):
    """Configuration for CosmicRayTask"""
    minVisitsMedian = Field(dtype=int, default=5, doc="Minimum number of visits to use median")
    modelPsf = GaussianPsfFactory.makeField(doc="Model PSF")
    cosmicray = ConfigField(dtype=FindCosmicRaysConfig, doc="Find cosmic rays")
    grow = Field(dtype=int, default=3, doc="Radius to grow CRs")
    doCosmicRay = Field(dtype=bool, default=True, doc="Remove cosmic rays?")
    doH4MorphologicalCRs = Field(dtype=bool, default=False, doc="""Use morphological CR rejection for H4RGs?
    This is usually handled in the up-the-ramp code""")
    crMinReadsH4 = Field(dtype=int, default=4,
                         doc="Minimum number of reads for up-the-ramp CR rejection in H4RGs")
    repair = ConfigurableField(target=PfsRepairTask, doc="Task to repair artifacts; used for single exposure")
    grouping = ChoiceField(
        dtype=str,
        default="auto",
        allowed=dict(
            auto="Select groupings automatically, using pfsConfig",
            separate="Visits are in separate groups",
            all="Use all visits in a group",
            manual="Specify the grouping manually",
        ),
        doc="Algorithm to use for grouping visits",
    )
    groups = DictField(
        keytype=int,
        itemtype=int,
        default={},
        doc=(
            "Manual specification of visit groups; only used if grouping=manual. "
            "Keys are the visit numbers, and the values are the group numbers (which may be arbitrary)."
        ),
    )
    doWriteExposure = Field(dtype=bool, default=False, doc="Write CR-masked exposure?")
    scaleFluxMinSnr = Field(dtype=float, default=2.0,
                            doc="Minimum S/N threshold for pixels used to compute per-image flux scales.")
    scaleFluxMinExcessFactor = Field(dtype=float, default=3.0,
                                     doc=("Required excess, over the Gaussian-noise expectation "
                                          "of pixels consistently above `scaleFluxMinSnr` "
                                          "in all images for flux scaling to be trusted."))
    scaleFluxMinNPixels = Field(dtype=int, default=500,
                                doc="Minimum number of pixels passing the S/N cuts")
    doNormalizeChiRms = Field(dtype=bool, default=True,
                              doc=("Normalize chi to have RMS = 1.0 (workaround for imperfect noise model; "
                                   "disable once better characterized)."))

    def setDefaults(self):
        super().setDefaults()
        self.repair.interp.modelPsf.defaultFwhm = 1.5  # FWHM of the PSF in pixels, for single exposure
        self.modelPsf.defaultFwhm = 4.0  # Not the real PSF FWHM, but this sets the CR finding sensitivity
        self.cosmicray.nCrPixelMax = 100000000


class CosmicRayTask(PipelineTask):
    """Perform cosmic-ray removal

    We use the standard single-exposure cosmic-ray removal task, but in the
    future we intend to upgrade this to use a more sophisticated algorithm using
    multiple exposures.
    """
    ConfigClass: ClassVar[Type[Config]] = CosmicRayConfig
    config: CosmicRayConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("repair")

    def runQuantum(
            self,
            butler: QuantumContext,
            inputRefs: InputQuantizedConnection,
            outputRefs: OutputQuantizedConnection
    ) -> None:
        """Entry point for running the task under the Gen3 middleware"""
        exposures = butler.get(inputRefs.inputExposures)
        outputs = self.run(exposures)
        butler.put(outputs.masks, outputRefs.masks)
        if self.config.doWriteExposure:
            butler.put(outputs.exposures, outputRefs.outputExposures)

    def run(self, exposures: List["Exposure"]) -> Struct:
        """Perform cosmic-ray removal

        Parameters
        ----------
        exposures : `list` of `lsst.afw.image.Exposure`
            Exposures to repair.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results of cosmic-ray removal.
        """
        result = Struct(exposures=exposures, masks=[exp.mask for exp in exposures])
        if len(exposures) == 0:
            return result
        if not self.config.doCosmicRay:
            return result
        if exposures[0].getDetector().getName().startswith("n") and not self.config.doH4MorphologicalCRs:
            nRead = exposures[0].getMetadata()["W_H4NRED"]
            if nRead >= self.config.crMinReadsH4:  # we already ran the CR rejection code
                self.log.info("Assuming that up-the-ramp CR code was already run (nread = %d)", nRead)
                return result

        # Single exposure cosmic-ray removal: use the repair task
        if len(exposures) == 1:
            self.log.warn("No subtraction possible with single exposure; using sub-optimal CR removal")
            for exp in exposures:
                self.runSingle(exp)
            return result

        # Multiple exposure cosmic-ray removal
        clean = self.calculateCleanImage([exposure.getMaskedImage() for exposure in exposures])
        result.mergeItems(clean, "image", "scales")
        psf = self.config.modelPsf.apply()
        # default rms to 1.0 if doNormalizeChiRms is disabled.
        rms = self.calculateChiRms(exposures, clean) if self.config.doNormalizeChiRms else 1.0
        for exp, scale in zip(exposures, clean.scales):
            self.findCosmicRays(exp.getMaskedImage(), psf, clean.image, scale, rms)
        self.findOverlaps([exp.mask for exp in exposures])

        return result

    def runSingle(self, exposure: "Exposure") -> None:
        """Perform cosmic-ray removal on a single exposure"""
        modelPsfConfig = self.config.repair.interp.modelPsf
        psf = modelPsfConfig.apply()
        exposure.setPsf(psf)
        self.repair.run(exposure)
        return exposure

    def computeFluxScale(self, stack, snrStack, minSnr, minExcessFactor, minNPixels):
        """Compute per-image multiplicative flux scales from a stack.

        Pixels used to determine the scaling are selected using a per-pixel
        S/N map. For each pixel, we require that it is above `minSnr` in
        all images, which rejects pixels contaminated in only one image
        (e.g. cosmic rays).

        The fraction of such high-S/N pixels is compared to the fraction
        expected from a standard normal distribution in S/N units (using
        `minSnr` and the number of images). This uses a Gaussian tail as
        an approximate noise model for S/N.
        If the observed fraction does not exceed this
        expectation by at least `minExcessFactor`, or if the absolute
        number of selected pixels is less than `minNPixels`, the scaling
        is considered unreliable and a RuntimeError is raised.

        Parameters
        ----------
        stack : `numpy.ndarray` or `numpy.ma.MaskedArray`
            Image cube with shape (nImages, ny, nx) containing the input
            images.
        snrStack : `numpy.ndarray` or `numpy.ma.MaskedArray`
            Image cube with shape (nImages, ny, nx) containing the per-image
            S/N maps (image / sqrt(variance)), with masking consistent
            with `stack`.
        minSnr : `float`
            Minimum S/N threshold in a single image. A pixel must exceed
            this threshold in all images to be used for flux scaling.
        minExcessFactor : `float`
            Factor by which the observed fraction of consistently
            high-S/N pixels must exceed the expectation from pure
            Gaussian noise for flux scaling to be considered reliable.
        minNPixels : `int`
            Minimum number of pixels passing the S/N consistency cuts
            required to compute the flux scales.

        Returns
        -------
        fluxes : `numpy.ndarray`
            Array of length nImages containing the multiplicative flux
            scales, normalised so that their mean is 1.

        Raises
        ------
        RuntimeError
            Raised if the number of selected pixels is too small or if
            their fraction is not significantly above the Gaussian-noise
            expectation.
        """

        def gaussianTailProb(minSnr: float, nImages: int = 1) -> float:
            """One-sided tail probability that a pixel exceeds minSnr in all images.

            Assumes each image has independent standard normal noise Z ~ N(0, 1).
            For one image this returns P(Z > minSnr); for multiple images it
            returns P(Z_1 > minSnr, ..., Z_n > minSnr) = P(Z > minSnr)**nImages.
            """
            pSingle = 0.5 * math.erfc(minSnr / math.sqrt(2.0))
            return pSingle ** nImages

        num, nRows, nCols = stack.shape
        # Build a reference image; the median of ratios is used for the scale.
        refImage = np.nanmean(stack, axis=0)
        # Computing a mask where pixels are consistently above minSNR.
        maskSNR = np.isfinite(refImage) & np.isfinite(snrStack) & (snrStack > minSnr)
        maskSNR = maskSNR.sum(axis=0) == num
        # Calculating absolute number and fraction to be compared with a gaussian distribution.
        nPixels = maskSNR.sum()
        fracPixels = nPixels / (nRows * nCols)

        if fracPixels < minExcessFactor * gaussianTailProb(minSnr, num) or nPixels < minNPixels:
            raise RuntimeError("No valid high-flux pixels found to compute flux scales")

        fluxes = np.empty(num, dtype=float)
        for ii in range(num):
            ratios = stack[ii][maskSNR] / refImage[maskSNR]
            fluxes[ii] = np.nanmedian(ratios)

        fluxes /= np.nanmean(fluxes)

        return fluxes

    def calculateCleanImage(self, images: List["MaskedImage"]) -> MaskedImageF:
        """Calculate a cleaned image from a list of images

        Parameters
        ----------
        images : `list` of `lsst.afw.image.MaskedImage`
            List of images to clean.

        Returns
        -------
        image : `lsst.afw.image.MaskedImageF`
            Cleaned image.
        scales : `numpy.ndarray`
            Scaling factor to multiply the cleaned image by to match the
            input image.
        """
        num = len(images)
        dims = images[0].getDimensions()
        for image in images:
            if image.getDimensions() != dims:
                raise ValueError("Images must have the same dimensions")

        stack = np.empty((num, dims.getY(), dims.getX()), dtype=np.float32)
        varStack = np.empty((num, dims.getY(), dims.getX()), dtype=np.float32)
        maskStack = np.empty((num, dims.getY(), dims.getX()), dtype=np.int32)

        for ii, image in enumerate(images):
            stack[ii] = image.image.array.copy()
            varStack[ii] = image.variance.array.copy()
            maskStack[ii] = image.mask.array != 0

        # build S/N cube safely: do not take sqrt of non-positive or non-finite variance
        varSafe = np.where(np.isfinite(varStack) & (varStack > 0), varStack, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            snrStack = stack / np.sqrt(varSafe)

        snrMask = maskStack | ~np.isfinite(varSafe)

        try:
            fluxes = self.computeFluxScale(
                np.ma.array(stack, mask=maskStack),
                np.ma.array(snrStack, mask=snrMask),
                self.config.scaleFluxMinSnr,
                self.config.scaleFluxMinExcessFactor,
                self.config.scaleFluxMinNPixels
            )
        except RuntimeError:
            self.log.warn("Could not calculate per-image scaling falling back to unity...")
            fluxes = np.ones(num)

        # Apply the scales to put all images on a common flux scale
        stack /= fluxes[:, None, None]
        varStack /= fluxes[:, None, None] ** 2

        # set clean image array.
        operation = np.nanmin if num < self.config.minVisitsMedian else np.nanmedian

        clean = MaskedImageF(dims)
        clean.image.array[:] = operation(stack, axis=0)
        # set clean image mask.
        clean.mask.array[:] = 0
        clean.mask.array[~np.isfinite(clean.image.array)] = images[0].mask.getPlaneBitMask("BAD")
        # set clean image variance.
        clean.variance.array[:] = self.calculateCleanImageVariance(stack, varStack)

        return Struct(image=clean, scales=fluxes)

    def calculateCleanImageVariance(self, stack, varStack):
        """Propagate variance for the cleaned image.

        Parameters
        ----------
        stack : `numpy.ndarray`
            Rescaled image cube with shape (nImages, nRows, nCols).
        varStack : `numpy.ndarray`
            Rescaled variance cube with shape (nImages, nRows, nCols),
            already divided by the square of the flux scales.

        Returns
        -------
        cleanVariance : `numpy.ndarray`
            Per-pixel variance for the cleaned image.
        """
        num, nRows, nCols = stack.shape
        cleanVariance = np.full((nRows, nCols), np.nan, dtype=np.float32)

        if num < self.config.minVisitsMedian:
            # variance of the pixel that contributed the minimum value
            safeStack = np.where(np.isfinite(stack), stack, np.inf)
            idxMin = np.argmin(safeStack, axis=0)
            rows, cols = np.indices((nRows, nCols))
            cleanVariance = varStack[idxMin, rows, cols]
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                invVar = np.where(varStack > 0, 1.0 / varStack, 0.0)
            sumInvVar = np.sum(invVar, axis=0)

            positive = sumInvVar > 0
            cleanVariance[positive] = 1.0 / sumInvVar[positive]
            cleanVariance[positive] *= np.pi / 2

        return cleanVariance

    def calculateChiRms(self, images: List["Exposure"], clean) -> float:
        """Estimate the RMS of the chi distribution between scaled exposures.

        This computes an empirical chi field using the difference of two
        scaled exposures and the summed variance from two (possibly other)
        scaled exposures. Using variance from different exposures reduces
        correlations between the noise in the numerator and denominator
        when more than two images are available.

        For a list of images ``images``:

        - the first two entries (indices 0 and 1) provide the images for
          the numerator;
        - the last two entries (indices -2 and -1) provide the variance for
          the denominator (after rescaling).

        With exactly 2 images, the same pair is used for both numerator
        and denominator; with 3 images, the middle one is used in both.
        The method works for any number of images >= 2, but behaves best
        when at least 4 independent exposures are available.

        The chi image is

            chi = (I1/scale[0] - I2/scale[1]) / sqrt(V3/scale[-2]**2 + V4/scale[-1]**2),

        and the returned value is a robust RMS of this chi distribution,
        computed with ``robustRms`` (ignoring NaNs). In the ideal case
        (perfect model and correctly estimated variances), this RMS should
        be close to 1.

        Parameters
        ----------
        images : `list` of `lsst.afw.image.Exposure`
            Input exposures used to estimate the chi RMS. The first two
            entries provide the images for the numerator; the last two
            entries provide the variance for the denominator (after
            rescaling). A minimum of 2 images is required.
        clean : `lsst.pipe.base.Struct`
            Result of `calculateCleanImage`, expected to have a `scales`
            attribute containing the per-image flux scales.

        Returns
        -------
        rms : `float`
            Robust RMS of the chi distribution, in units where an ideal
            model would give chi ~ N(0, 1).
        """
        exp1 = images[0].getMaskedImage()
        exp2 = images[1].getMaskedImage()
        exp3 = images[-2].getMaskedImage()
        exp4 = images[-1].getMaskedImage()

        I1 = exp1.image.array / clean.scales[0]
        I2 = exp2.image.array / clean.scales[1]

        V3 = exp3.variance.array / (clean.scales[-2]) ** 2
        V4 = exp4.variance.array / (clean.scales[-1]) ** 2

        varSum = V3 + V4
        # mark non-positive or non-finite variance as NaN so sqrt does not warn
        varSafe = np.where(np.isfinite(varSum) & (varSum > 0), varSum, np.nan)

        with np.errstate(invalid="ignore", divide="ignore"):
            chi = (I1 - I2) / np.sqrt(varSafe)

        rms = robustRms(chi, nanSafe=True)

        return rms

    # see below (median case)
    def findCosmicRays(self, image: "MaskedImage", psf: "Psf", clean: MaskedImageF, scale: float, rms: float):
        """Find cosmic rays in an image.

        We subtract the clean image and then find cosmic rays in the residual.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image to find cosmic rays in.
        psf : `lsst.afw.detection.Psf`
            PSF model.
        clean : `lsst.afw.image.MaskedImageF`
            Cleaned image.
        scale : `float`
            Scaling factor to multiply the cleaned image by to match the
            input image.
        rms : `float`
            Scaling factor used to renormalise the variance so that the
            residuals are approximately chi-distributed with N(0, 1).
        """
        subtracted = MaskedImageF(image.image.clone(), image.mask, image.variance)
        subtracted.image.array -= clean.image.array * scale
        subtracted.variance.array += clean.variance.array * scale ** 2
        subtracted.variance.array *= rms ** 2

        crBit = image.mask.getPlaneBitMask("CR")
        image.mask &= ~crBit
        cosmicrays = findCosmicRays(subtracted, psf, 0.0, makePropertySet(self.config.cosmicray), True)

        num = 0
        numPixels = 0
        if cosmicrays is not None:
            setMaskFromFootprintList(image.mask, cosmicrays, crBit)
            num = len(cosmicrays)
            numPixels = np.sum((image.mask.array & crBit) != 0)

        self.log.info("Found %d cosmic rays (%d pixels)", num, numPixels)

    def findOverlaps(self, masks: List["Mask"]):
        """Find overlapping cosmic rays in a set of masks

        The cleaned image will still include cosmic rays if they are present in
        all the input images. To remove these, we find the pixels that are
        near identified cosmic rays in all images.

        Parameters
        ----------
        masks : `list` of `lsst.afw.image.Mask`
            Masks to find overlapping cosmic rays in.
        """
        num = len(masks)
        dims = masks[0].getDimensions()
        for mask in masks:
            if mask.getDimensions() != dims:
                raise ValueError("Masks must have the same dimensions")

        count = np.zeros((dims.getY(), dims.getX()), dtype=int)
        for mm in masks:
            mm = mm.clone()
            crBit = mm.getPlaneBitMask("CR")
            mm &= crBit
            growMasks(mm, self.config.grow, "CR", "CR")
            count[mm.array != 0] += 1

        select = (count == num)
        self.log.info("Found %d overlapping cosmic ray pixels", select.sum())
        for mm in masks:
            mm.array[select] |= mm.getPlaneBitMask("CR")


class ApplyCosmicRayMaskConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    inputExposure = InputConnection(
        name="postISRCCD",
        doc="Exposure to repair",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    crMask = InputConnection(
        name="crMask",
        doc="Cosmic ray mask",
        storageClass="Mask",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    outputExposure = OutputConnection(
        name="calexp",
        doc="Exposure with cosmic rays masked",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config:
            return
        if not self.config.doApplyCrMask:
            self.prerequisiteInputs.remove("crMask")


class ApplyCosmicRayMaskConfig(PipelineTaskConfig, pipelineConnections=CosmicRayConnections):
    """Configuration for CosmicRayTask"""
    doApplyCrMask = Field(dtype=bool, default=True, doc="Apply cosmic-ray mask to input exposure?")


class ApplyCosmicRayMaskTask(PipelineTask):
    """Apply cosmic-ray mask to an exposure

    This task applies a cosmic-ray mask to an exposure. This is not usually
    something you want to do in a separate task (since it results in an extra
    copy of the image) but there are cases where it is useful (e.g., when you
    need an input for CalibCombineTask).
    """
    ConfigClass: ClassVar[Type[Config]] = ApplyCosmicRayMaskConfig
    config: ApplyCosmicRayMaskConfig

    def run(self, inputExposure: "Exposure", crMask: Mask | None = None) -> Struct:
        """Apply cosmic-ray mask"""
        if self.config.doApplyCrMask:
            if not crMask:
                raise ValueError("Cosmic-ray mask required but not provided")
            inputExposure.mask |= crMask
        return Struct(outputExposure=inputExposure)
