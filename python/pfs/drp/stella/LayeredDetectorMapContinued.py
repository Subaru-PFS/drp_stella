import numpy as np
from lsst.utils import continueClass
import lsst.afw.image
from lsst.geom import AffineTransform
import pfs.datamodel.pfsDetectorMap

from .DetectorMapContinued import DetectorMap
from .Distortion import Distortion
from .LayeredDetectorMap import LayeredDetectorMap
from .MosaicPolynomialDistortionContinued import MosaicPolynomialDistortion
from .MultipleDistortionsDetectorMapContinued import MultipleDistortionsDetectorMap
from .PolynomialDistortionContinued import PolynomialDistortion
from .SplinedDetectorMapContinued import SplinedDetectorMap
from .utils import headerToMetadata

__all__ = ("LayeredDetectorMap",)


@continueClass  # noqa: F811 (redefinition)
class LayeredDetectorMap:  # noqa: F811 (redefinition)
    @classmethod
    def fromDetectorMap(cls, detectorMap):
        """Construct from another DetectorMap

        Instances of different DetectorMap subclasses are upgraded/converted
        to `LayeredDetectorMap`.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            DetectorMap to convert.

        Returns
        -------
        self : `LayeredDetectorMap`
            LayeredDetectorMap representation of the input.
        """
        if isinstance(detectorMap, LayeredDetectorMap):
            # Nothing to do
            return detectorMap.clone()

        bbox = detectorMap.bbox
        slitOffsets = np.zeros(len(detectorMap), dtype=float)
        base = None  # Need to set this
        distortions = []
        dividedDetector = False
        rightCcd = AffineTransform()  # Identity transform
        visitInfo = detectorMap.visitInfo
        metadata = detectorMap.metadata

        if isinstance(detectorMap, SplinedDetectorMap):
            base = detectorMap.clone()
        elif isinstance(detectorMap, MultipleDistortionsDetectorMap):
            if len(detectorMap.distortions) > 1:
                raise RuntimeError(
                    "Cannot convert a MultipleDistortionsDetectorMap with multiple distortions"
                )
            base = detectorMap.base.clone()
            dist = detectorMap.distortions[0]
            if isinstance(dist, MosaicPolynomialDistortion):
                dividedDetector = True
                rightCcd = dist.getAffine()
                distortions = [
                    PolynomialDistortion(
                        dist.getOrder(), dist.getRange(), dist.getXCoefficients(), dist.getYCoefficients(),
                    ),
                ]
            elif isinstance(dist, PolynomialDistortion):
                distortions = [dist.clone()]
            else:
                raise RuntimeError(f"Unsupported distortion type: {dist}")
        else:
            # Long-deprecated classes
            from .DistortedDetectorMapContinued import DistortedDetectorMap
            from .DoubleDetectorMapContinued import DoubleDetectorMap
            from .PolynomialDetectorMapContinued import PolynomialDetectorMap
            if isinstance(detectorMap, (PolynomialDetectorMap, DoubleDetectorMap, DistortedDetectorMap)):
                base = detectorMap.base.clone()
                distortions = [detectorMap.distortion.clone()]
            else:
                raise RuntimeError(f"Unsupported detectorMap type: {type(detectorMap)}")

        return cls(
            bbox, slitOffsets, slitOffsets, base, distortions, dividedDetector, rightCcd, visitInfo, metadata
        )

    @classmethod
    def fromDatamodel(cls, detMap):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        detMap : `pfs.datamodel.LayeredDetectorMap`
            datamodel representation of LayeredDetectorMap.

        Returns
        -------
        self : `pfs.drp.stella.LayeredDetectorMap`
            drp_stella representation of LayeredDetectorMap.
        """
        if not isinstance(detMap, pfs.datamodel.LayeredDetectorMap):
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
        detMap : `pfs.datamodel.LayeredDetectorMap`
            Datamodel representation of LayeredDetectorMap.
        """
        base = self.getBase().toDatamodel()
        distortions = [dd.toDatamodel() for dd in self.getDistortions()]

        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        return pfs.datamodel.LayeredDetectorMap(
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


DetectorMap.register(LayeredDetectorMap)
