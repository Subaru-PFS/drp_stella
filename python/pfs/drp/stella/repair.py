from typing import ClassVar, Optional, Type

import numpy as np

from lsst.pex.config import Config, Field, makePropertySet
from lsst.pipe.tasks.repair import RepairConfig, RepairTask
from lsst.afw.detection import setMaskFromFootprintList
from lsst.afw.image import Exposure
from lsst.meas.algorithms import findCosmicRays

__all__ = ("PfsRepairConfig", "PfsRepairTask")


class PfsRepairConfig(RepairConfig):
    halfHeight = Field(dtype=int, default=50, doc="Half-height for column background determination")

    def setDefaults(self):
        self.cosmicray.nCrPixelMax = 5000000


class PfsRepairTask(RepairTask):
    ConfigClass: ClassVar[Type[Config]] = PfsRepairConfig

    def cosmicRay(self, exposure: Exposure, keepCRs: Optional[bool] = None):
        """Mask cosmic rays

        Parameters
        ----------
        exposure : `Exposure`
            Exposure to search for CRs. The image and mask are modified.
        keepCRs : `bool` or `None`
            Preserve the CR pixels (rather than interpolating over them)? If
            `None`, then defer to configuration.
        """
        psf = exposure.getPsf()
        if psf is None:
            raise RuntimeError("No PSF in exposure")
        if keepCRs is None:
            keepCRs = self.config.cosmicray.keepCRs
        config = makePropertySet(self.config.cosmicray)

        # Blow away old mask
        mask = exposure.getMaskedImage().getMask()
        crBit = mask.addMaskPlane("CR")
        mask.clearMaskPlane(crBit)

        # Measure background
        width, height = exposure.getDimensions()
        background = np.full((height, width), np.nan, np.float32)
        for yy in range(height):
            low = max(0, yy - self.config.halfHeight)
            high = min(exposure.getHeight(), yy + self.config.halfHeight + 1)
            array = exposure.image.array[low:high, :]
            background[yy, :] = np.median(array, axis=0)

        # Find CRs in background-subtracted image
        exposure.image.array -= background
        try:
            cosmicrays = findCosmicRays(exposure.maskedImage, psf, 0.0, config, keepCRs)
        finally:
            exposure.image.array += background

        num = 0
        if cosmicrays is not None:
            mask = exposure.mask
            setMaskFromFootprintList(mask, cosmicrays, mask.getPlaneBitMask("CR"))
            num = len(cosmicrays)

        self.log.info("Identified %s cosmic rays.", num)
