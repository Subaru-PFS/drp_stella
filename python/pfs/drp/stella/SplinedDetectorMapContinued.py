import lsst.afw.fits
import lsst.geom

from lsst.utils import continueClass

import pfs.datamodel
from .SplinedDetectorMap import SplinedDetectorMap
from .DetectorMapContinued import DetectorMap
from .utils import headerToMetadata


__all__ = ["SplinedDetectorMap"]


@continueClass  # noqa: F811 (redefinition)
class SplinedDetectorMap:  # noqa: F811 (redefinition)
    @classmethod
    def fromDatamodel(cls, detMap):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        detMap : `pfs.datamodel.SplinedDetectorMap`
            datamodel representation of SplinedDetectorMap.

        Returns
        -------
        self : `pfs.drp.stella.SplinedDetectorMap`
            drp_stella representation of SplinedDetectorMap.
        """
        if not isinstance(detMap, pfs.datamodel.SplinedDetectorMap):
            raise RuntimeError(f"Wrong type: {detMap}")
        bbox = detMap.box.toLsst()
        centerKnots = [ss.knots for ss in detMap.xSplines]
        centerValues = [ss.values for ss in detMap.xSplines]
        wavelengthKnots = [ss.knots for ss in detMap.wavelengthSplines]
        wavelengthValues = [ss.values for ss in detMap.wavelengthSplines]

        metadata = headerToMetadata(detMap.metadata)
        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(bbox, detMap.fiberId, centerKnots, centerValues, wavelengthKnots, wavelengthValues,
                   detMap.spatialOffsets, detMap.spectralOffsets, visitInfo, metadata)

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
        detMap : `pfs.datamodel.SplinedDetectorMap`
            datamodel representation of SplinedDetectorMap.
        """
        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        xSplines = [pfs.datamodel.Spline(self.getXCenterSpline(ff).getX(), self.getXCenterSpline(ff).getY())
                    for ff in self.fiberId]
        wavelengthSplines = [pfs.datamodel.Spline(self.getWavelengthSpline(ff).getX(),
                                                  self.getWavelengthSpline(ff).getY())
                             for ff in self.fiberId]

        return pfs.datamodel.SplinedDetectorMap(
            identity, pfs.datamodel.Box.fromLsst(self.bbox), self.fiberId, xSplines, wavelengthSplines,
            self.getSpatialOffsets(), self.getSpectralOffsets(), metadata.toDict()
        )


DetectorMap.register(SplinedDetectorMap)
