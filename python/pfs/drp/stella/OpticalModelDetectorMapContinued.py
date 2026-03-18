import numpy as np
from lsst.utils import continueClass
import lsst.afw.image
import pfs.datamodel.pfsDetectorMap

from .DetectorMapContinued import DetectorMap
from .Distortion import Distortion
from .LayeredDetectorMapContinued import LayeredDetectorMap
from .math import makeAffineTransform, getAffineParameters
from .OpticalModel import SlitModel, OpticsModel, DetectorModel
from .OpticalModelDetectorMap import OpticalModelDetectorMap
from .SplinedDetectorMapContinued import SplinedDetectorMap
from .utils import headerToMetadata

__all__ = ("OpticalModelDetectorMap",)


@continueClass  # noqa: F811 (redefinition)
class OpticalModelDetectorMap:  # noqa: F811 (redefinition)
    @classmethod
    def fromDetectorMap(cls, detectorMap: DetectorMap) -> "OpticalModelDetectorMap":
        """Construct from another DetectorMap

        Instances of different DetectorMap subclasses are upgraded/converted
        to `OpticalModelDetectorMap`.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            DetectorMap to convert.

        Returns
        -------
        self : `OpticalModelDetectorMap`
            OpticalModelDetectorMap representation of the input.
        """
        if isinstance(detectorMap, OpticalModelDetectorMap):
            # Nothing to do
            return detectorMap.clone()

        bbox = detectorMap.bbox
        visitInfo = detectorMap.visitInfo
        metadata = detectorMap.metadata

        if isinstance(detectorMap, SplinedDetectorMap):
            spatialOffsets = detectorMap.getSpatialOffsets()
            spectralOffsets = detectorMap.getSpectralOffsets()
            if not np.all(spatialOffsets == 0.0) or not np.all(spectralOffsets == 0.0):
                raise RuntimeError("Cannot convert a SplinedDetectorMap with non-zero slit offsets")
            slit = SlitModel(detectorMap)
            optics = OpticsModel(detectorMap)
            detector = DetectorModel(bbox)
            return cls(slit, optics, detector, visitInfo, metadata)

        if isinstance(detectorMap, LayeredDetectorMap):
            slit = SlitModel(detectorMap.base)
            optics = OpticsModel(detectorMap.base, detectorMap.distortions)
            if detectorMap.getDividedDetector():
                detector = DetectorModel(bbox, detectorMap.getRightCcd())
            else:
                detector = DetectorModel(bbox)
            new = cls(slit, optics, detector, visitInfo, metadata)
            new.setSlitOffsets(detectorMap.getSpatialOffsets(), detectorMap.getSpectralOffsets())
            return new

        raise RuntimeError(f"Unsupported DetectorMap type: {type(detectorMap)}")

    @classmethod
    def fromDatamodel(cls, detMap):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        detMap : `pfs.datamodel.OpticalModelDetectorMap`
            datamodel representation of OpticalModelDetectorMap.

        Returns
        -------
        self : `pfs.drp.stella.OpticalModelDetectorMap`
            drp_stella representation of OpticalModelDetectorMap.
        """
        if not isinstance(detMap, pfs.datamodel.OpticalModelDetectorMap):
            raise RuntimeError(f"Wrong type: {detMap}")
        slit = SlitModel(
            detMap.fiberId.astype(np.int32),
            detMap.fiberPitch, detMap.wavelengthDispersion,
            detMap.spatialOffsets.astype(np.float64), detMap.spectralOffsets.astype(np.float64),
            [Distortion.fromDatamodel(dd) for dd in detMap.slitDistortions],
        )

        optics = OpticsModel(
            detMap.spatialOptics.astype(np.float64), detMap.spectralOptics.astype(np.float64),
            detMap.xOptics.astype(np.float64), detMap.yOptics.astype(np.float64),
            [Distortion.fromDatamodel(dd) for dd in detMap.opticsDistortions],
        )
        detectorDistortions = [Distortion.fromDatamodel(dd) for dd in detMap.detectorDistortions]
        if detMap.dividedDetector:
            detector = DetectorModel(
                detMap.box.toLsst(),
                makeAffineTransform(detMap.rightCcd.astype(np.float64)),
                detectorDistortions,
            )
        else:
            detector = DetectorModel(detMap.box.toLsst(), distortions=detectorDistortions)
        metadata = headerToMetadata(detMap.metadata)
        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(slit, optics, detector, visitInfo, metadata)

    def toDatamodel(self, identity=None):
        """Convert to the pfs.datamodel representation

        Parameters
        ----------
        identity : `pfs.datamodel.CalibIdentity`, optional
            Identification of the calibration. Providing this is only necessary
            if you intend to write via the datamodel representation's ``write``
            method; other means of writing that provide a filename directly do
            not require providing an ``identity`` here.

        Returns
        -------
        detMap : `pfs.datamodel.OpticalModelDetectorMap`
            Datamodel representation of OpticalModelDetectorMap.
        """
        # Slit
        fiberId = self.getFiberId()
        fiberPitch = self.slitModel.getFiberPitch()
        wavelengthDispersion = self.slitModel.getWavelengthDispersion()
        spatialOffsets = self.slitModel.getSpatialOffsets()
        spectralOffsets = self.slitModel.getSpectralOffsets()
        slitDistortions = [dd.toDatamodel() for dd in self.slitModel.getDistortions()]

        # Optics
        spatialOptics = self.opticsModel.getSpatial()
        spectralOptics = self.opticsModel.getSpectral()
        xOptics = self.opticsModel.getX()
        yOptics = self.opticsModel.getY()
        opticsDistortions = [dd.toDatamodel() for dd in self.opticsModel.getDistortions()]

        # Detector
        box = pfs.datamodel.Box.fromLsst(self.bbox)
        dividedDetector = self.detectorModel.getIsDivided()
        rightCcd = getAffineParameters(self.detectorModel.getRightCcd())
        detectorDistortions = [dd.toDatamodel() for dd in self.detectorModel.getDistortions()]

        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        return pfs.datamodel.OpticalModelDetectorMap(
            identity,
            fiberId, fiberPitch, wavelengthDispersion,
            spatialOffsets, spectralOffsets, slitDistortions,
            spatialOptics, spectralOptics, xOptics, yOptics, opticsDistortions,
            box, dividedDetector, rightCcd, detectorDistortions,
            metadata.toDict(),
        )


DetectorMap.register(OpticalModelDetectorMap)
