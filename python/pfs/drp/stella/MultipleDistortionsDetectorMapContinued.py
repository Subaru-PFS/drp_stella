from lsst.utils import continueClass
import lsst.afw.image
import pfs.datamodel.pfsDetectorMap

from .DetectorMapContinued import DetectorMap
from .Distortion import Distortion
from .MultipleDistortionsDetectorMap import MultipleDistortionsDetectorMap
from .SplinedDetectorMapContinued import SplinedDetectorMap
from .utils import headerToMetadata

__all__ = ("MultipleDistortionsDetectorMap",)


@continueClass  # noqa: F811 (redefinition)
class MultipleDistortionsDetectorMap:  # noqa: F811 (redefinition)
    @classmethod
    def fromDatamodel(cls, detMap):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        detMap : `pfs.datamodel.MultipleDistortionsDetectorMap`
            datamodel representation of MultipleDistortionsDetectorMap.

        Returns
        -------
        self : `pfs.drp.stella.MultipleDistortionsDetectorMap`
            drp_stella representation of MultipleDistortionsDetectorMap.
        """
        if not isinstance(detMap, pfs.datamodel.MultipleDistortionsDetectorMap):
            raise RuntimeError(f"Wrong type: {detMap}")
        base = SplinedDetectorMap.fromDatamodel(detMap.base)
        distortions = [Distortion.fromDatamodel(dd) for dd in detMap.distortions]
        metadata = headerToMetadata(detMap.metadata)
        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(base, distortions, visitInfo, metadata)

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
        detMap : `pfs.datamodel.MultipleDistortionsDetectorMap`
            Datamodel representation of MultipleDistortionsDetectorMap.
        """
        base = self.getBase().toDatamodel()
        distortions = [dd.toDatamodel() for dd in self.getDistortions()]

        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        return pfs.datamodel.MultipleDistortionsDetectorMap(
            identity,
            pfs.datamodel.Box.fromLsst(self.bbox),
            base,
            distortions,
            metadata.toDict(),
        )


DetectorMap.register(MultipleDistortionsDetectorMap)
