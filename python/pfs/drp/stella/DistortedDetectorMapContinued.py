from lsst.utils import continueClass
import lsst.afw.image
from lsst.geom import Box2D

import pfs.datamodel.pfsDetectorMap

from .DistortedDetectorMap import DistortedDetectorMap
from .DetectorMapContinued import DetectorMap
from .DetectorDistortion import DetectorDistortion
from .SplinedDetectorMapContinued import SplinedDetectorMap
from .utils import headerToMetadata

__all__ = ("DistortedDetectorMap",)


@continueClass  # noqa: F811 (redefinition)
class DistortedDetectorMap:
    @classmethod
    def fromDatamodel(cls, detMap):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        detMap : `pfs.datamodel.DistortedDetectorMap`
            datamodel representation of DistortedDetectorMap.

        Returns
        -------
        self : `pfs.drp.stella.DistortedDetectorMap`
            drp_stella representation of DistortedDetectorMap.
        """
        if not isinstance(detMap, pfs.datamodel.DistortedDetectorMap):
            raise RuntimeError(f"Wrong type: {detMap}")
        base = SplinedDetectorMap.fromDatamodel(detMap.base)
        distortion = DetectorDistortion(detMap.order, Box2D(base.bbox), detMap.xCoeff, detMap.yCoeff,
                                        detMap.rightCcdCoeff)
        metadata = headerToMetadata(detMap.metadata)
        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(base, distortion, visitInfo, metadata)

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
        detMap : `pfs.datamodel.DistortedDetectorMap`
            Datamodel representation of DistortedDetectorMap.
        """
        base = self.getBase().toDatamodel()
        distortion = self.getDistortion()

        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        return pfs.datamodel.DistortedDetectorMap(
            identity, pfs.datamodel.Box.fromLsst(self.bbox), base,
            distortion.getOrder(), distortion.getXCoefficients(), distortion.getYCoefficients(),
            distortion.getRightCcdCoefficients(), metadata.toDict()
        )


DetectorMap.register(DistortedDetectorMap)
