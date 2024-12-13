from dataclasses import dataclass

import numpy as np
import scipy.signal as signal

from lsst.pex.config import Config, DictField, Field
from lsst.pipe.base import Task, Struct
from lsst.afw.image import Image, ImageF, MaskedImage

from pfs.drp.stella import FiberTrace, FiberTraceSet, SpectrumSet, DetectorMap, LayeredDetectorMap
from .datamodel import PfsArm

__all__ = ("ScatteredLightTask", "ScatteredLightConfig", "ScatteredLightModel")


@dataclass
class ScatteredLightModel:
    scale: float  # Scale factor for the scattered light model
    frac1: float = 1.0  # Fraction of the total power in the first component
    powerLaw1: float = -1.5  # Power-law index (2-D) of the first component
    soften1: float = 1.0  # Softening (pixels) of the first component
    frac2: float = 0.2  # Fraction of the total power in the second component
    powerLaw2: float = -3.0  # Power-law index (2-D) of the second component
    soften2: float = 5.0  # Softening (pixels) of the second component
    halfSize: int = 4096  # Half-size of the kernel

    @property
    def grid(self):
        indices = np.arange(-self.halfSize, self.halfSize + 1)
        return np.meshgrid(indices, indices)

    @staticmethod
    def _makeKernelImpl(frac, powerLaw, soften, grid, doSpectral=True):
        dx, dy = grid
        rr2 = dx**2 + soften**2
        if doSpectral:
            rr2 += dy**2
        kernel = rr2**(powerLaw/2)
        kernel *= frac/np.sum(kernel)
        return kernel

    def makeKernel1(self):
        return self._makeKernelImpl(self.frac1, self.powerLaw1, self.soften1, self.grid, doSpectral=True)

    def makeKernel2(self):
        return self._makeKernelImpl(self.frac2, self.powerLaw2, self.soften2, self.grid, doSpectral=False)

    def makeKernel(self):
        kernel = self.makeKernel1()
        if self.frac2 != 0:
            kernel += self.makeKernel2()
        return kernel

    def calculateImage(self, pfsArm: PfsArm, detectorMap: DetectorMap) -> Image:
        """Calculate the scattered light model image

        Parameters
        ----------
        pfsArm : `PfsArm`
            pfsArm from which to calculate the image.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.

        Returns
        -------
        image : `Image`
            Scattered light model image.
        """
        dims = detectorMap.getBBox().getDimensions()

        traces = FiberTraceSet(len(pfsArm))
        for fid in pfsArm.fiberId:
            centers = detectorMap.getXCenter(fid)
            traces.add(FiberTrace.boxcar(fid, dims, 0.5, centers))

        spectra = SpectrumSet.fromPfsArm(pfsArm)
        model = spectra.makeImage(dims, traces).array

        xGap: int | None = None
        if isinstance(detectorMap, LayeredDetectorMap):
            xGap = - int(detectorMap.rightCcd.getTranslation().getX() + 0.5)
            # Assume that the yGap is small
            height, width = model.shape
            gapped = np.zeros((height, width + xGap))
            gapped[:, 0:width//2] = model[:, 0:width//2]
            gapped[:, -width//2:] = model[:, -width//2:]
            model = gapped

        # Convolve;  zero-padded but that's OK as we're using a model with a background of zero
        scattered = self.scale * signal.convolve(model, self.makeKernel(), mode='same')

        if xGap is not None:
            ungapped = np.zeros((height, width))
            ungapped[:, 0:width//2] = scattered[:, 0:width//2]
            ungapped[:, -width//2:] = scattered[:, -width//2:]
            scattered = ungapped

        return ImageF(scattered.astype(np.float32))


class ScatteredLightConfig(Config):
    scale = DictField(
        keytype=str,
        itemtype=float,
        default=dict(b3=6.6e-2/1.2, r3=5.8e-2/1.2, n3=6.5e-2/1.2, m3=5.8e-2/1.2),
        doc="Scale factor for the scattered light model, indexed by camera name or 'default'",
    )
    frac1 = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=1.0),
        doc="Fraction of the total power in the first component, indexed by camera name or 'default'",
    )
    powerLaw1 = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=-1.5),
        doc="Power-law index (2-D) of the first component, indexed by camera name or 'default'",
    )
    soften1 = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=1.0),
        doc="Softening (pixels) of the first component, indexed by camera name or 'default'",
    )
    frac2 = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=0.2),
        doc="Fraction of the total power in the second component, indexed by camera name or 'default'",
    )
    powerLaw2 = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=-3.0),
        doc="Power-law index (2-D) of the second component, indexed by camera name or 'default'",
    )
    soften2 = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=5.0),
        doc="Softening (pixels) of the second component, indexed by camera name or 'default'",
    )
    halfSize = Field(dtype=int, default=4096, doc="Half-size of the kernel")

    def getValue(self, name: str, camera: str) -> float:
        attr = getattr(self, name)
        if camera in attr:
            return attr[camera]
        return attr["default"]

    def getModel(self, arm: str, spectrograph: int) -> ScatteredLightModel:
        camera = f"{arm}{spectrograph}"
        return ScatteredLightModel(
            scale=self.getValue("scale", camera),
            frac1=self.getValue("frac1", camera),
            powerLaw1=self.getValue("powerLaw1", camera),
            soften1=self.getValue("soften1", camera),
            frac2=self.getValue("frac2", camera),
            powerLaw2=self.getValue("powerLaw2", camera),
            soften2=self.getValue("soften2", camera),
            halfSize=self.halfSize,
        )


class ScatteredLightTask(Task):
    ConfigClass = ScatteredLightConfig
    _DefaultName = "scatteredLight"

    def run(self, image: MaskedImage, pfsArm: PfsArm, detectorMap: DetectorMap) -> Struct:
        """Subtract the scattered light in an image

        Parameters
        ----------
        image : `MaskedImage`
            Image from which to subtract the scattered light; modified.
        pfsArm : `PfsArm`
            Spectra from which to estimate the scattered light.
        detectorMap : `DetectorMap`
            Mapping of fiberId,wavelength to x,y.

        Returns
        -------
        model : `Image`
            Scattered light model image.
        """
        model = self.config.getModel(pfsArm.identity.arm, pfsArm.identity.spectrograph)
        modelImage = model.calculateImage(pfsArm, detectorMap)
        image -= modelImage
        return Struct(model=modelImage)
