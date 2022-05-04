from lsst.utils import continueClass
import lsst.afw.image

import pfs.datamodel.pfsDetectorMap

from .DifferentialDetectorMap import DifferentialDetectorMap
from .DetectorMapContinued import DetectorMap
from .GlobalDetectorModel import GlobalDetectorModel, GlobalDetectorModelScaling
from .SplinedDetectorMapContinued import SplinedDetectorMap
from .utils import headerToMetadata

__all__ = ("DifferentialDetectorMap",)


@continueClass  # noqa: F811 (redefinition)
class DifferentialDetectorMap:  # type: ignore [no-redef] # noqa: F811 (redefinition)
    @classmethod
    def fromDatamodel(cls, detMap):
        """Construct from the pfs.datamodel representation

        Parameters
        ----------
        detMap : `pfs.datamodel.DifferentialDetectorMap`
            datamodel representation of DifferentialDetectorMap.

        Returns
        -------
        self : `pfs.drp.stella.DifferentialDetectorMap`
            drp_stella representation of DifferentialDetectorMap.
        """
        if not isinstance(detMap, pfs.datamodel.DifferentialDetectorMap):
            raise RuntimeError(f"Wrong type: {detMap}")
        base = SplinedDetectorMap.fromDatamodel(detMap.base)
        scalingKwargs = {name: getattr(detMap.scaling, name) for name in
                         ("fiberPitch", "dispersion", "wavelengthCenter", "minFiberId", "maxFiberId",
                          "height", "buffer")}
        scaling = GlobalDetectorModelScaling(**scalingKwargs)
        model = GlobalDetectorModel(detMap.order, scaling, detMap.fiberCenter,
                                    detMap.xCoeff, detMap.yCoeff, detMap.highCcdCoeff)
        metadata = headerToMetadata(detMap.metadata)
        visitInfo = lsst.afw.image.VisitInfo(metadata)
        lsst.afw.image.stripVisitInfoKeywords(metadata)

        return cls(base, model, visitInfo, metadata)

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
        detMap : `pfs.datamodel.DifferentialDetectorMap`
            Datamodel representation of DifferentialDetectorMap.
        """
        base = self.getBase().toDatamodel()
        model = self.getModel()
        scaling = model.getScaling()
        scalingKwargs = {name: getattr(scaling, name) for name in
                         ("fiberPitch", "dispersion", "wavelengthCenter", "minFiberId", "maxFiberId",
                          "height", "buffer")}

        metadata = self.metadata.deepCopy()
        if self.visitInfo is not None:
            lsst.afw.image.setVisitInfoMetadata(metadata, self.visitInfo)

        return pfs.datamodel.DifferentialDetectorMap(
            identity, pfs.datamodel.Box.fromLsst(self.bbox), base, model.getDistortionOrder(),
            pfs.datamodel.GlobalDetectorModelScaling(**scalingKwargs), model.getFiberCenter(),
            model.getXCoefficients(), model.getYCoefficients(), model.getHighCcdCoefficients(),
            metadata.toDict()
        )


DetectorMap.register(DifferentialDetectorMap)
