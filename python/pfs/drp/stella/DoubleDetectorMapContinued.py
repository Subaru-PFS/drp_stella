from lsst.utils import continueClass
import lsst.afw.image
from lsst.geom import Box2D

import pfs.datamodel.pfsDetectorMap

from .DoubleDetectorMap import DoubleDetectorMap
from .DetectorMapContinued import DetectorMap
from .DoubleDistortion import DoubleDistortion
from .SplinedDetectorMapContinued import SplinedDetectorMap
from .utils import headerToMetadata

__all__ = ("DoubleDetectorMap",)


@continueClass  # noqa: F811 (redefinition)
class DoubleDetectorMap:
    @classmethod
    def fromDatamodel(cls, detMap):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        detMap : `pfs.datamodel.DoubleDetectorMap`
            datamodel representation of DoubleDetectorMap.

        Returns
        -------
        self : `pfs.drp.stella.DoubleDetectorMap`
            drp_stella representation of DoubleDetectorMap.
        """
        if not isinstance(detMap, pfs.datamodel.DoubleDetectorMap):
            raise RuntimeError(f"Wrong type: {detMap}")
        base = SplinedDetectorMap.fromDatamodel(detMap.base)
        distortion = DoubleDistortion(detMap.order, Box2D(base.bbox), detMap.xLeft, detMap.yLeft,
                                      detMap.xRight, detMap.yRight)
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
        detMap : `pfs.datamodel.DoubleDetectorMap`
            Datamodel representation of DoubleDetectorMap.
        """
        base = self.getBase().toDatamodel()
        distortion = self.getDistortion()

        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        return pfs.datamodel.DoubleDetectorMap(
            identity, pfs.datamodel.Box.fromLsst(self.bbox), base,
            distortion.getOrder(), distortion.getXLeftCoefficients(), distortion.getYLeftCoefficients(),
            distortion.getXRightCoefficients(), distortion.getYRightCoefficients(), metadata.toDict()
        )


DetectorMap.register(DoubleDetectorMap)
