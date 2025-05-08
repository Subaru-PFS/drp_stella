from typing import ClassVar, List, Type, TYPE_CHECKING

import numpy as np

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pex.config import Config, ConfigField, ConfigurableField, Field, makePropertySet

from lsst.afw.detection import setMaskFromFootprintList
from lsst.afw.image import ImageF, MaskedImageF
from lsst.meas.algorithms import findCosmicRays, FindCosmicRaysConfig, GaussianPsfFactory
from lsst.ip.isr.isrFunctions import growMasks

from .repair import PfsRepairTask

if TYPE_CHECKING:
    from lsst.afw.image import Exposure, MaskedImage, Mask
    from lsst.afw.detection import Psf


__all__ = ("CosmicRayTask", "CompareCosmicRayTask")


class CosmicRayConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "arm", "spectrograph"),
):
    inputExposure = InputConnection(
        name="postISRCCD",
        doc="Exposure to repair",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )
    outputExposure = OutputConnection(
        name="calexp",
        doc="Repaired exposure",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
    )


class CosmicRayConfig(PipelineTaskConfig, pipelineConnections=CosmicRayConnections):
    """Configuration for CosmicRayTask"""
    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    repair = ConfigurableField(target=PfsRepairTask, doc="Task to repair artifacts")

    def setDefaults(self):
        super().setDefaults()
        self.repair.interp.modelPsf.defaultFwhm = 1.5  # FWHM of the PSF in pixels


class CosmicRayTask(PipelineTask):
    """Perform cosmic-ray removal

    We use the standard single-exposure cosmic-ray removal task, but in the
    future we intend to upgrade this to use a more sophisticated algorithm using
    multiple exposures.
    """
    ConfigClass: ClassVar[Type[Config]] = CosmicRayConfig

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
        exposure = butler.get(inputRefs.inputExposure)
        outputs = self.run(exposure)
        butler.put(outputs.exposure, outputRefs.outputExposure)

    def run(self, exposure) -> Struct:
        """Perform cosmic-ray removal"""
        modelPsfConfig = self.config.repair.interp.modelPsf
        psf = modelPsfConfig.apply()
        exposure.setPsf(psf)

        if self.config.doRepair:
            self.repair.run(exposure)

        return Struct(exposure=exposure)


class CompareCosmicRayConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit_group", "arm", "spectrograph"),
):
    """Connections for CompareCosmicRayTask"""
    inputExposures = InputConnection(
        name="postISRCCD",
        doc="Exposure to repair",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )
    outputExposures = OutputConnection(
        name="calexp",
        doc="Repaired exposure",
        storageClass="Exposure",
        dimensions=("instrument", "visit", "arm", "spectrograph"),
        multiple=True,
    )


class CompareCosmicRayConfig(PipelineTaskConfig, pipelineConnections=CompareCosmicRayConnections):
    """Configuration for CompareCosmicRayTask"""
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

    def setDefaults(self):
        super().setDefaults()
        self.repair.interp.modelPsf.defaultFwhm = 1.5  # FWHM of the PSF in pixels, for single exposure
        self.modelPsf.defaultFwhm = 4.0  # Not the real PSF FWHM, but this sets the CR finding sensitivity
        self.cosmicray.nCrPixelMax = 100000000


class CompareCosmicRayTask(PipelineTask):
    """Perform cosmic-ray removal

    We use comparison of multiple exposures of the same targets to identify
    cosmic rays.
    """
    ConfigClass: ClassVar[Type[Config]] = CompareCosmicRayConfig

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
        result = Struct(exposures=exposures)
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
                modelPsfConfig = self.config.repair.interp.modelPsf
                psf = modelPsfConfig.apply()
                exp.setPsf(psf)
                self.repair.run(exp)
            return result

        # Multiple exposure cosmic-ray removal
        clean = self.calculateCleanImage([exposure.getMaskedImage() for exposure in exposures])
        result.mergeItems(clean, "image", "scales")
        psf = self.config.modelPsf.apply()
        for exp, scale in zip(exposures, clean.scales):
            self.findCosmicRays(exp.getMaskedImage(), psf, clean.image, scale)
        self.findOverlaps([exp.mask for exp in exposures])

        return result

    def calculateCleanImage(self, images: List["MaskedImage"]) -> ImageF:
        """Calculate a cleaned image from a list of images

        Parameters
        ----------
        images : `list` of `lsst.afw.image.MaskedImage`
            List of images to clean.

        Returns
        -------
        image : `lsst.afw.image.Image`
            Cleaned image.
        scales : `numpy.ndarray`
            Scaling factor to multiple the cleaned image by to match the
            input image.
        """
        num = len(images)
        dims = images[0].getDimensions()
        for image in images:
            if image.getDimensions() != dims:
                raise ValueError("Images must have the same dimensions")

        fluxes = np.zeros(num, dtype=float)
        stack = np.empty((num, dims.getY(), dims.getX()), dtype=np.float32)
        for ii, image in enumerate(images):
            fluxes[ii] = np.nanmedian(image.image.array)

        fluxes /= np.mean(fluxes)

        for ii, image in enumerate(images):
            stack[ii] = image.image.array / fluxes[ii]

        operation = np.nanmin if num < self.config.minVisitsMedian else np.nanmedian
        clean = ImageF(dims)
        clean.array[:] = operation(stack, axis=0)

        return Struct(image=clean, scales=fluxes)

    def findCosmicRays(self, image: "MaskedImage", psf: "Psf", clean: ImageF, scale: float):
        """Find cosmic rays in an image

        We subtracted the clean image and then find cosmic rays in the residual.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image to find cosmic rays in.
        psf : `lsst.afw.detection.Psf`
            PSF model.
        clean : `lsst.afw.image.ImageF`
            Cleaned image.
        scale : `float`
            Scaling factor to multiple the cleaned image by to match the
            input image.
        """
        subtracted = MaskedImageF(image.image.clone(), image.mask, image.variance)
        subtracted.image.array -= clean.array*scale

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
