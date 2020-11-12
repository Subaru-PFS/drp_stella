import lsst.afw.fits
import lsst.geom

from lsst.utils import continueClass

import pfs.datamodel
from .GlobalDetectorMap import GlobalDetectorMap, GlobalDetectorModel, GlobalDetectorModelScaling, FiberMap
from .DetectorMapContinued import DetectorMap
from .utils import headerToMetadata

__all__ = ["GlobalDetectorMap", "GlobalDetectorModel", "GlobalDetectorModelScaling", "FiberMap"]


@continueClass  # noqa: F811 (redefinition)
class GlobalDetectorMap:
    @classmethod
    def fromDatamodel(cls, detMap):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        detMap : `pfs.datamodel.GlobalDetectorMap`
            datamodel representation of GlobalDetectorMap.

        Returns
        -------
        self : `pfs.drp.stella.GlobalDetectorMap`
            drp_stella representation of GlobalDetectorMap.
        """
        if not isinstance(detMap, pfs.datamodel.GlobalDetectorMap):
            raise RuntimeError(f"Wrong type: {detMap}")
        bbox = detMap.box.toLsst()
        scalingKwargs = {name: getattr(detMap.scaling, name) for name in
                         ("fiberPitch", "dispersion", "wavelengthCenter", "minFiberId", "maxFiberId",
                          "height", "buffer")}
        scaling = GlobalDetectorModelScaling(**scalingKwargs)
        model = GlobalDetectorModel(bbox, detMap.order, detMap.fiberId, scaling,
                                    detMap.xCoeff, detMap.yCoeff, detMap.rightCcdCoeff,
                                    detMap.spatialOffsets, detMap.spectralOffsets)
        metadata = headerToMetadata(detMap.metadata)
        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(bbox, model, visitInfo, metadata)

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
        detMap : `pfs.datamodel.GlobalDetectorMap`
            datamodel representation of GlobalDetectorMap.
        """
        model = self.getModel()
        scaling = model.getScaling()
        scalingKwargs = {name: getattr(scaling, name) for name in
                         ("fiberPitch", "dispersion", "wavelengthCenter", "minFiberId", "maxFiberId",
                          "height", "buffer")}

        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        return pfs.datamodel.GlobalDetectorMap(
            identity, pfs.datamodel.Box.fromLsst(self.bbox), self.getDistortionOrder(), self.fiberId,
            pfs.datamodel.GlobalDetectorMapScaling(**scalingKwargs),
            model.getXCoefficients(), model.getYCoefficients(), model.getRightCcdCoefficients(),
            self.getSpatialOffsets(), self.getSpectralOffsets(), metadata.toDict()
        )


DetectorMap.register(GlobalDetectorMap)
