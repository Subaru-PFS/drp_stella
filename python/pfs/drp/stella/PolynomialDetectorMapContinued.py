from lsst.utils import continueClass
import lsst.afw.image
from lsst.geom import Box2D

import pfs.datamodel.pfsDetectorMap

from .PolynomialDetectorMap import PolynomialDetectorMap
from .DetectorMapContinued import DetectorMap
from .PolynomialDistortion import PolynomialDistortion
from .SplinedDetectorMapContinued import SplinedDetectorMap
from .utils import headerToMetadata

__all__ = ("PolynomialDetectorMap",)


@continueClass  # noqa: F811 (redefinition)
class PolynomialDetectorMap:  # noqa: F811 (redefinition)
    @classmethod
    def fromDatamodel(cls, detMap):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        detMap : `pfs.datamodel.PolynomialDetectorMap`
            datamodel representation of PolynomialDetectorMap.

        Returns
        -------
        self : `pfs.drp.stella.PolynomialDetectorMap`
            drp_stella representation of PolynomialDetectorMap.
        """
        if not isinstance(detMap, pfs.datamodel.PolynomialDetectorMap):
            raise RuntimeError(f"Wrong type: {detMap}")
        base = SplinedDetectorMap.fromDatamodel(detMap.base)
        distortion = PolynomialDistortion(detMap.order, Box2D(base.bbox), detMap.xCoeff, detMap.yCoeff)
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
        detMap : `pfs.datamodel.PolynomialDetectorMap`
            Datamodel representation of PolynomialDetectorMap.
        """
        base = self.getBase().toDatamodel()
        distortion = self.getDistortion()

        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        return pfs.datamodel.PolynomialDetectorMap(
            identity, pfs.datamodel.Box.fromLsst(self.bbox), base,
            distortion.getOrder(), distortion.getXCoefficients(), distortion.getYCoefficients(),
            metadata.toDict()
        )


DetectorMap.register(PolynomialDetectorMap)
