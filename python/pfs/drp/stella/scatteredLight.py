from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.signal as signal

from lsst.pex.config import Config, DictField, Field
from lsst.pipe.base import Task, Struct
from lsst.afw.image import Image, ImageF, MaskedImage

from .FiberTraceContinued import FiberTrace
from .FiberTraceSetContinued import FiberTraceSet
from .SpectrumSetContinued import SpectrumSet

if TYPE_CHECKING:
    from .datamodel import PfsArm
    from .DetectorMapContinued import DetectorMap


__all__ = ("ScatteredLightTask", "ScatteredLightConfig", "ScatteredLightModel")


@dataclass
class ScatteredLightModel:
    """Model of the scattered light

    Parameters
    ----------
    top : `float`
        Scale factor for the scattered light model at top.
    bottom : `float`
        Scale factor for the scattered light model at bottom.
    frac1 : `float`
        Fraction of the total power in the first component.
    powerLaw1 : `float`
        Power-law index (2-D) of the first component.
    soften1 : `float`
        Softening (pixels) of the first component.
    frac2 : `float`
        Fraction of the total power in the second component.
    powerLaw2 : `float`
        Power-law index (2-D) of the second component.
    soften2 : `float`
        Softening (pixels) of the second component.
    halfSize : `int`
        Half-size of the kernel (pixels).
    """
    top: float = 1.0        # Scale factor for the scattered light model at top
    bottom: float = 1.0     # Scale factor for the scattered light model at bottom
    frac1: float = 0.048    # Fraction of the total power in the first component
    powerLaw1: float = 1.5  # Power-law index (2-D) of the first component
    soften1: float = 1.0    # Softening (pixels) of the first component
    frac2: float = 0.010    # Fraction of the total power in the second component
    powerLaw2: float = 3.0  # Power-law index (2-D) of the second component
    soften2: float = 5.0    # Softening (pixels) of the second component
    halfSize: int = 4096    # Half-size of the kernel (pixels)

    @property
    def grid(self):
        """Return a grid of indices for the kernel

        Returns
        -------
        dx, dy : `numpy.ndarray`
            Grid of indices in x and y for the kernel.
        """
        indices = np.arange(-self.halfSize, self.halfSize + 1)
        return np.meshgrid(indices, indices)

    @staticmethod
    def _makeKernelImpl(frac, powerLaw, soften, grid, doSpectral=True):
        """Implementation of kernel generation

        Parameters
        ----------
        frac : `float`
            Fraction of the total power in the component.
        powerLaw : `float`
            Power-law index (2-D) of the component.
        soften : `float`
            Softening (pixels) of the component.
        grid : `tuple` of `numpy.ndarray`
            Grid of indices in x and y for the kernel.
        doSpectral : `bool`
            Include the spectral dimension?

        Returns
        -------
        kernel : `numpy.ndarray`
            Kernel for the scattered light model.
        """
        dx, dy = grid
        rr2 = dx**2 + soften**2
        if doSpectral:
            rr2 += dy**2
        rr2 = np.maximum(rr2, 1.0)  # Avoid division by zero
        kernel = rr2**(-powerLaw/2)
        kernel *= frac/np.sum(kernel)
        return kernel

    def makeKernel1(self):
        """Make the kernel for the first component

        Returns
        -------
        kernel : `numpy.ndarray`
            Kernel for the first component of the scattered light model.
        """
        return self._makeKernelImpl(self.frac1, self.powerLaw1, self.soften1, self.grid, doSpectral=True)

    def makeKernel2(self):
        """Make the kernel for the second component

        Returns
        -------
        kernel : `numpy.ndarray`
            Kernel for the second component of the scattered light model.
        """
        return self._makeKernelImpl(self.frac2, self.powerLaw2, self.soften2, self.grid, doSpectral=True)

    def makeKernel(self):
        """Make the kernel for the scattered light model

        Returns
        -------
        kernel : `numpy.ndarray`
            Kernel for the scattered light model.
        """
        kernel = self.makeKernel1()
        if self.frac2 != 0:
            kernel += self.makeKernel2()
        return kernel

    def calculateImage(self, pfsArm: "PfsArm", detectorMap: "DetectorMap") -> Image:
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
            if True:
                tt = FiberTrace.boxcar(fid, dims, 0.5, centers)
            else:
                from .fiberProfile import FiberProfile
                profile = FiberProfile.makeGaussian(2.0, dims.getY(), 7, 3)
                tt = profile.makeFiberTraceFromDetectorMap(detectorMap, fid)
            traces.add(tt)

        if True:
            flux_new = []
            for wav, flx, msk in zip(pfsArm.wavelength, pfsArm.flux, pfsArm.mask):
                bad = msk & pfsArm.flags.get("BAD", "CR", "SAT", "INTRP") != 0
                flx[bad] = np.nan
                invalid_indices = np.where(bad)[0]
                for i in invalid_indices:
                    start = max(0, i - 5)
                    end = min(i + 6, len(invalid_indices))
                    flx[i] = np.nanmedian(flx[start:end])
                flx[bad] = np.nanmedian(flx[~bad])
                flux_new.append(flx)
            pfsArm.flux = np.array(flux_new)

        spectra = SpectrumSet.fromPfsArm(pfsArm)
        model = spectra.makeImage(dims, traces).array

        width = dims.getX()
        height = dims.getY()
        xGap: int | None = None
        from .LayeredDetectorMapContinued import LayeredDetectorMap  # import here to avoid circular import
        if isinstance(detectorMap, LayeredDetectorMap):
            xGap = -int(detectorMap.rightCcd.getTranslation().getX() + 0.5)
            # Assume that the yGap is small
            height, width = model.shape
            gapped = np.zeros((height, width + xGap))
            gapped[:, 0:width//2] = model[:, 0:width//2]
            gapped[:, -width//2:] = model[:, -width//2:]
            model = gapped

        # Convolve; zero-padded but that's OK as we're using a model with a background of zero
        self.scale = np.interp(np.arange(0, height), [0, height-1], [self.bottom, self.top])[:, np.newaxis]
        scattered = self.scale*signal.convolve(model, self.makeKernel(), mode='same')

        if xGap is not None:
            ungapped = np.zeros((height, width))
            ungapped[:, 0:width//2] = scattered[:, 0:width//2]
            ungapped[:, -width//2:] = scattered[:, -width//2:]
            scattered = ungapped

        return ImageF(scattered.astype(np.float32))


class ScatteredLightConfig(Config):
    top = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=1.0, b1=1.0, b2=1.0, b3=1.0, b4=1.0, r1=0.8, r2=0.8, r3=0.8, r4=0.8, n1=1.0, n2=1.0, n3=1.0, n4=1.0, m1=0.9, m2=0.9, m3=0.9, m4=0.9),
        doc="Scale factor for the scattered light model at top, indexed by camera name or 'default'",
    )
    bottom = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=1.0, b1=1.0, b2=1.0, b3=1.0, b4=1.0, r1=1.1, r2=1.1, r3=1.1, r4=1.1, n1=1.0, n2=1.0, n3=1.0, n4=1.0, m1=1.1, m2=1.1, m3=1.1, m4=1.1),
        doc="Scale factor for the scattered light model at bottom, indexed by camera name or 'default'",
    )
    frac1 = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=0.048, b1=0.055, b2=0.050, b3=0.060, b4=0.100, r1=0.037, r2=0.040, r3=0.043, r4=0.070, n1=0.065, n2=0.068, n3=0.050, n4=0.060, m1=0.040, m2=0.040, m3=0.053, m4=0.060),
        doc="Fraction of the total power in the first component, indexed by camera name or 'default'",
    )
    powerLaw1 = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=1.5),
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
        default=dict(default=0.01, b1=0.025, b2=0.055, b3=0.032, b4=0.045, r1=0.025, r2=0.020, r3=0.013, r4=0.021, n1=0.010, n2=0.010, n3=0.010, n4=0.017, m1=0.025, m2=0.020, m3=0.014, m4=0.021),
        doc="Fraction of the total power in the second component, indexed by camera name or 'default'",
    )
    powerLaw2 = DictField(
        keytype=str,
        itemtype=float,
        default=dict(default=3.0),
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
        """Get a value for a camera from the configuration

        Parameters
        ----------
        name : `str`
            Name of the value to get. Options: ``scale``, ``frac1``,
            ``powerLaw1``, ``soften1``, ``frac2``, ``powerLaw2``, ``soften2``.
        camera : `str`
            Name of the camera, e.g., ``r3``, ``n1``, etc.

        Returns
        -------
        value : `float`
            Value for the camera.
        """
        attr = getattr(self, name)
        if camera in attr:
            return attr[camera]
        return attr["default"]

    def getModel(self, arm: str, spectrograph: int) -> ScatteredLightModel:
        """Get the scattered light model for a camera

        Parameters
        ----------
        arm : `str`
            Arm of the spectrograph (``b``, ``r``, ``n``, ``m``).
        spectrograph : `int`
            Spectrograph number (1, 2, 3, 4).

        Returns
        -------
        model : `ScatteredLightModel`
            Scattered light model for the camera.
        """
        camera = f"{arm}{spectrograph}"
        return ScatteredLightModel(
            top=self.getValue("top", camera),
            bottom=self.getValue("bottom", camera),
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

    def run(self, image: MaskedImage, pfsArm: "PfsArm", detectorMap: "DetectorMap") -> Struct:
        """Subtract the scattered light in an image

        No subtraction is performed if the scattered light scale factor is zero.

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
        if model.top == 0.0 and model.bottom == 0.0:
            self.log.warn("Scattered light model scale is zero; not subtracting")
            return Struct(model=None)
        self.log.info("Subtracting scattered light model: %s", model)
        modelImage = model.calculateImage(pfsArm, detectorMap)
        image -= modelImage
        return Struct(model=modelImage)
