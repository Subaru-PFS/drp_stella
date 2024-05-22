from typing import ClassVar, Optional, Type

import numpy as np
import scipy.signal

from lsst.pex.config import Config, Field, ListField, makePropertySet
from lsst.pipe.tasks.repair import RepairConfig, RepairTask
from lsst.afw.detection import setMaskFromFootprintList
from lsst.afw.image import Exposure, Mask
from lsst.afw.geom import SpanSet
from lsst.geom import Point2I
from lsst.meas.algorithms import findCosmicRays
from lsst.ip.isr.isrFunctions import growMasks
from lsst.utils.timer import timeMethod
from .DetectorMap import DetectorMap
from .referenceLine import ReferenceLineSet
from .scatteredLight import estimateScatteredLight

__all__ = ("maskLines", "PfsRepairConfig", "PfsRepairTask")


def maskLines(
    mask: Mask, detectorMap: DetectorMap, refLines: ReferenceLineSet, radius: int, maskPlane: str = "REFLINE"
):
    """Mask lines on an exposure

    It is helpful to have bright lines masked so that they can be ignored when
    measuring the background before finding cosmic-rays.

    Parameters
    ----------
    mask : `lsst.afw.image.Mask`
        Mask image on which to mask sky lines; modified.
    detectorMap : `DetectorMap`
        Mapping of fiberId,wavelength to x,y.
    refLines : `ReferenceLineSet`
        Lines to mask.
    radius : `int`
        Radius around reference lines to mask.
    maskPlane : `str`
        Name of mask plane to set.
    """
    bitmask = mask.getPlaneBitMask(maskPlane)
    if detectorMap is not None and refLines is not None and len(refLines) > 0:
        for fiberId in detectorMap.fiberId:
            points = detectorMap.findPoint(fiberId, refLines.wavelength)
            good = np.all(np.isfinite(points), axis=1)
            for xx, yy in points[good]:
                spans = SpanSet.fromShape(radius, offset=Point2I(int(xx + 0.5), int(yy + 0.5)))
                spans.clippedTo(mask.getBBox()).setMask(mask, bitmask)


class PfsRepairConfig(RepairConfig):
    halfHeight = Field(dtype=int, default=35, doc="Half-height for column background determination")
    doNirCosmicRay = Field(dtype=bool, default=True, doc="Do CR finding on NIR data?")
    mask = ListField(
        dtype=str,
        default=["BAD", "SAT", "REFLINE", "NO_DATA"],
        doc="Mask planes to ignore in trace removal",
    )
    crGrow = Field(dtype=int, default=1, doc="Radius to grow CRs")
    subtractScatteredLight = Field(dtype=bool, default=False, doc="Subtract scattered light?")

    def setDefaults(self):
        self.cosmicray.nCrPixelMax = 5000000


class PfsRepairTask(RepairTask):
    ConfigClass: ClassVar[Type[Config]] = PfsRepairConfig

    @timeMethod
    def run(self, exposure, defects=None, keepCRs=None,
            pfsArm=None, detectorMap=None
            ):
        """Repair an Exposure's defects and cosmic rays.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure must have a valid Psf.
            Modified in place.
        defects : `lsst.meas.algorithms.DefectListT` or `None`, optional
            If `None`, do no defect correction.
        keepCRs : `Unknown` or `None`, optional
            Don't interpolate over the CR pixels (defer to ``RepairConfig`` if `None`).

        Raises
        ------
        AssertionError
            Raised if any of the following occur:
            - No exposure provided.
            - The object provided as exposure evaluates to False.
            - No PSF provided.
            - The Exposure has no associated Psf.
        """

        super().run(exposure, defects=defects, keepCRs=keepCRs)

        if self.config.subtractScatteredLight:
            scatteringCoefficients = dict(b3=(-1.75, 0.0, 6.59e-02),
                                          r3=(-1.5, 1.51e-03, 5.07e-02),
                                          n3=(-1.5, 7.51e-04, 6.53e-02),
                                          m3=(-1.5, 4.98e-03, 5.34e-02),
                                          )

            scatteringCoeffs = scatteringCoefficients.get(exposure.getDetector().getName())
            if scatteringCoeffs is not None:
                alpha, c0, c1 = scatteringCoeffs
                smodel = estimateScatteredLight(pfsArm, detectorMap, self.log, alpha=alpha)
                if smodel is not None:
                    self.log.info("Subtracting scattered light")
                    smodel *= c1
                    smodel += c0*np.nanmean(exposure.image.array)
                    exposure.image -= smodel

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

        if not self.config.doNirCosmicRay and exposure.getDetector().getName().startswith("n"):
            self.log.warn("CR finding for NIR data has been disabled")
            return

        # Median filter on columns with mask
        bad = (exposure.mask.array & exposure.mask.getPlaneBitMask(self.config.mask)) != 0
        tmp = exposure.image.array.copy()
        tmp[bad] = 1e10                 # n.b. we'll call np.minimum(), so these values will be ignored
        traces = scipy.signal.medfilt2d(tmp, (self.config.halfHeight, 1))

        # Don't allow the result to exceed the actual value.
        # Negative values in the subtracted image mean a larger gradient,
        # which can be interpreted as a CR, whereas it really just means
        # that we're on an absorption feature or our median has been
        # biased high by real structure in the trace.
        traces = np.minimum(exposure.image.array, traces)

        # Find CRs in traces-subtracted image
        exposure.image.array -= traces

        try:
            cosmicrays = findCosmicRays(exposure.maskedImage, psf, 0.0, config, keepCRs)
        finally:
            exposure.image.array += traces

        num = 0
        numPixels = 0
        numGrown = 0
        if cosmicrays is not None:
            mask = exposure.mask
            setMaskFromFootprintList(mask, cosmicrays, mask.getPlaneBitMask("CR"))
            num = len(cosmicrays)
            numPixels = np.sum(mask.array & mask.getPlaneBitMask("CR") != 0)

            if self.config.crGrow > 0:
                growMasks(mask, self.config.crGrow, "CR", "CR")
                numGrown = np.sum(mask.array & mask.getPlaneBitMask("CR") != 0)

        self.log.info("Identified %d cosmic rays covering %d pixels, grown to %d", num, numPixels, numGrown)
