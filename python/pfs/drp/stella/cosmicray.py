from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import itertools
from typing import ClassVar, Iterable, List, Type, TYPE_CHECKING

import numpy as np

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Output as OutputConnection
from lsst.pipe.base.connectionTypes import Input as InputConnection
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from lsst.pipe.base.connections import QuantaAdjuster
from lsst.pex.config import Config, ChoiceField, ConfigField, ConfigurableField, DictField, Field
from lsst.pex.config import makePropertySet

from lsst.afw.detection import setMaskFromFootprintList
from lsst.afw.image import ImageF, MaskedImageF
from lsst.meas.algorithms import findCosmicRays, FindCosmicRaysConfig, GaussianPsfFactory
from lsst.ip.isr.isrFunctions import growMasks

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
            visit: (pfsConfig.pfsDesignId, pfsConfig.visit0) for visit, pfsConfig in pfsConfigList.items()
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
        for exp, scale in zip(exposures, clean.scales):
            self.findCosmicRays(exp.getMaskedImage(), psf, clean.image, scale)
        self.findOverlaps([exp.mask for exp in exposures])

        return result

    def runSingle(self, exposure: "Exposure") -> None:
        """Perform cosmic-ray removal on a single exposure"""
        modelPsfConfig = self.config.repair.interp.modelPsf
        psf = modelPsfConfig.apply()
        exposure.setPsf(psf)
        self.repair.run(exposure)
        return exposure

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
