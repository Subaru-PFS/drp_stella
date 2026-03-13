import numpy as np
from lsst.utils import continueClass
import lsst.afw.image
import pfs.datamodel.pfsDetectorMap

from .DetectorMapContinued import DetectorMap
from .Distortion import Distortion
from .LayeredDetectorMapContinued import LayeredDetectorMap
from .OpticalModel import SlitModel, OpticsModel, DetectorModel
from .OpticalModelDetectorMap import OpticalModelDetectorMap
from .SplinedDetectorMapContinued import SplinedDetectorMap
from .utils import headerToMetadata

__all__ = ("OpticalModelDetectorMap",)


@continueClass  # noqa: F811 (redefinition)
class OpticalModelDetectorMap:  # noqa: F811 (redefinition)
    @classmethod
    def fromDetectorMap(cls, detectorMap):
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
            optics = OpticsModel(slit)
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
        raise NotImplementedError("This method is not yet implemented")

        if not isinstance(detMap, pfs.datamodel.OpticalModelDetectorMap):
            raise RuntimeError(f"Wrong type: {detMap}")
        box = detMap.box.toLsst()
        spatial = detMap.spatial
        spectral = detMap.spectral
        base = SplinedDetectorMap.fromDatamodel(detMap.base)
        distortions = [Distortion.fromDatamodel(dd) for dd in detMap.distortions]
        dividedDetector = detMap.dividedDetector
        rightCcd = detMap.rightCcd
        metadata = headerToMetadata(detMap.metadata)
        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(box, spatial, spectral, base, distortions, dividedDetector, rightCcd, visitInfo, metadata)

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
        raise NotImplementedError("This method is not yet implemented")

        base = self.getBase().toDatamodel()
        distortions = [dd.toDatamodel() for dd in self.getDistortions()]

        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        return pfs.datamodel.OpticalModelDetectorMap(
            identity,
            pfs.datamodel.Box.fromLsst(self.bbox),
            self.getSpatialOffsets(),
            self.getSpectralOffsets(),
            base,
            distortions,
            self.getDividedDetector(),
            self.getRightCcdParameters(),
            metadata.toDict(),
        )


DetectorMap.register(OpticalModelDetectorMap)
