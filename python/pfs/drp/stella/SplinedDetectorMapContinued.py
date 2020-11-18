import numpy as np

import lsst.afw.fits
import lsst.geom

from lsst.utils import continueClass

import pfs.datamodel
from .SplinedDetectorMap import SplinedDetectorMap
from .DetectorMapContinued import DetectorMap
from .utils import headerToMetadata


__all__ = ["SplinedDetectorMap"]


@continueClass  # noqa: F811 (redefinition)
class SplinedDetectorMap:
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
        detMap : `pfs.datamodel.GlobalDetectorMap`
            datamodel representation of GlobalDetectorMap.
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

    def measureSlitOffsets(self, fiberId, wavelength, x, y, xErr, yErr):
        """Measure and apply slit offsets

        This implementation shadows a fake implementation in C++.

        We measure the weighted mean of the offset for each fiber.

        Parameters
        ----------
        fiberId : `numpy.ndarray` of `int`, shape ``(N,)``
            Fiber identifiers for reference lines.
        wavelength : `numpy.ndarray` of `float`, shape ``(N,)``
            Wavelengths of reference lines.
        x, y : `numpy.ndarray` of `float`, shape ``(N,)``
            Positions of reference lines (pixels).
        xErr, yErr : `numpy.ndarray` of `float`, shape ``(N,)``
            Errors in positions of reference lines (pixels).
        """
        for ff in set(fiberId):
            select = (fiberId == ff)
            points = self.findPoint(ff, wavelength.astype(np.float32)[select])
            dx = x[select] - points[:, 0]
            dy = y[select] - points[:, 1]
            xWeight = 1.0/xErr[select]**2
            yWeight = 1.0/yErr[select]**2
            spatial = np.sum(dx*xWeight)/np.sum(xWeight)
            spectral = np.sum(dy*yWeight)/np.sum(yWeight)
            self.setSlitOffsets(ff, self.getSpatialOffset(ff) + spatial,
                                self.getSpectralOffset(ff) + spectral)


DetectorMap.register(SplinedDetectorMap)
