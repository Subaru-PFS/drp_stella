from typing import TYPE_CHECKING, Type

import numpy as np

from lsst.afw.image import VisitInfo
from lsst.daf.base import PropertyList
from lsst.geom import Box2D

from . import ReferenceLineStatus
from .calibs import setCalibHeader
from .DetectorMapContinued import DetectorMap
from .DistortionContinued import Distortion
from .OpticalModelDetectorMapContinued import OpticalModelDetectorMap
from .PolynomialDistortionContinued import PolynomialDistortion
from .fitDetectorMap import (
    ArcLineResidualsSet, FitDetectorMapTask, FitDetectorMapConfig
)

if TYPE_CHECKING:
    from .arcLine import ArcLineSet

__all__ = ("AdjustDetectorMapConfig", "AdjustDetectorMapTask")


class AdjustDetectorMapConfig(FitDetectorMapConfig):
    """Configuration for AdjustDetectorMapTask"""
    def setDefaults(self):
        self.exclusionRadius = 4.0
        self.order = 1


class AdjustDetectorMapTask(FitDetectorMapTask):
    ConfigClass = AdjustDetectorMapConfig
    _DefaultName = "adjustDetectorMap"

    def run(
        self,
        detectorMap: DetectorMap,
        lines: "ArcLineSet",
        arm: str,
        visitInfo: VisitInfo,
        metadata: PropertyList | None = None,
        seed: int = 0,
    ):
        """Adjust a DistortedDetectorMap to fit arc line measurements

        We fit only the lowest order terms.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured line positions. The ``status`` member will be updated to
            indicate which lines were used and reserved.
        arm : `str`
            Spectrograph arm in use (``b``, ``r``, ``n``, ``m``).
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information for exposure.
        metadata : `lsst.daf.base.PropertyList`, optional
            DetectorMap metadata (FITS header).
        seed : `int`
            Seed for random number generator used for selecting reserved lines.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Adjusted detectorMap.
        model : `pfs.drp.stella.GlobalDetectorModel`
            Model that was fit to the data.
        xResid, yResid : `numpy.ndarray` of `float`
            Fit residual in x,y for each of the ``lines`` (pixels).
        xRms, yRms : `float`
            Residual RMS in x,y (pixels)
        chi2 : `float`
            Fit chi^2.
        soften : `float`
            Systematic error that was applied to measured errors (pixels).
        used : `numpy.ndarray` of `bool`
            Array indicating which lines were used in the fit.
        reserved : `numpy.ndarray` of `bool`
            Array indicating which lines were reserved from the fit.

        Raises
        ------
        pfs.drp.stella.fitDetectorMap.FittingError
            If the data is not of sufficient quality to fit.
        """
        if metadata is None:
            metadata = detectorMap.metadata.deepCopy()
        # Strip CALIB_INPUT_* from base detectorMap metadata (because it gets written to the PHDU
        # along with the adjusted detectorMap's metadata); other keywords will get overwritten,
        # but these will not because there's only one input now.
        for key in detectorMap.metadata.names():
            if key.startswith("CALIB_INPUT_"):
                detectorMap.metadata.remove(key)
        setCalibHeader(
            metadata,
            "detectorMap",
            [visitInfo.id],
            dict(arm=arm, spectrograph=detectorMap.metadata.get("SPECTROGRAPH", "unknown")),
        )

        base = self.getBaseDetectorMap(detectorMap, arm)
        DistortionClass = self.getDistortionClass(arm)
        dispersion = base.getDispersionAtCenter(base.fiberId[len(base)//2])
        needNumLines = PolynomialDistortion.getNumDistortionForOrder(self.config.order)
        good = self.getGoodLines(lines, detectorMap.getDispersionAtCenter())
        numGoodLines = good.sum()

        if numGoodLines < needNumLines:
            raise RuntimeError(f"Insufficient good lines: {numGoodLines} vs {needNumLines}")
        for ii in range(self.config.traceIterations or 1):
            self.log.debug("Commencing trace iteration %d", ii)
            residuals = self.calculateBaseResiduals(base, lines)
            weights = self.calculateWeights(lines)
            fit = self.fitDistortion(
                base.bbox,
                residuals,
                weights,
                dispersion,
                seed=seed,
                DistortionClass=DistortionClass,
            )
            detectorMap = self.makeDetectorMap(base, fit.distortion, visitInfo, metadata)

            numParameters = fit.numParameters
            if self.config.doSlitOffsets:
                for _ in range(self.config.slitOffsetIterations):
                    offsets = self.measureSlitOffsets(detectorMap, lines, fit.selection, weights)
                numParameters += offsets.numParameters

            if not self.updateTraceWavelengths(lines, detectorMap):
                break

        results = self.measureQuality(lines, detectorMap, fit.selection, numParameters)
        results.detectorMap = detectorMap

        lines.status[fit.selection] |= ReferenceLineStatus.DETECTORMAP_USED
        lines.status[fit.reserved] |= ReferenceLineStatus.DETECTORMAP_RESERVED

        if self.debugInfo.finalResiduals:
            self.plotResiduals(residuals, fit.xResid, fit.yResid, fit.selection, fit.reserved, dispersion)
        if self.debugInfo.lineQa:
            self.lineQa(lines, results.detectorMap)
        if self.debugInfo.wlResid:
            self.plotWavelengthResiduals(results.detectorMap, lines, fit.selection, fit.reserved)
        return results

    def getDistortionClass(self, arm: str) -> Type:
        """Return the class to use for the distortion

        Parameters
        ----------
        arm : `str`
            Spectrograph arm in use (one of ``b``, ``r``, ``n``, ``m``).

        Returns
        -------
        distortionClass : `type`
            Class to use for the distortion.
        """
        return PolynomialDistortion

    def getBaseDetectorMap(self, detectorMap, arm: str):
        """Get detectorMap to use as a base

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        arm : `str`
            Spectrograph arm in use.

        Returns
        -------
        base : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        """
        return OpticalModelDetectorMap.fromDetectorMap(detectorMap)

    def calculateBaseResiduals(self, detectorMap, lines):
        """Calculate position residuals w.r.t. base detectorMap

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Base detectorMap.
        lines : `ArcLineSet`
            Original line measurements (NOT the residuals).

        Returns
        -------
        residuals : `ArcLineResidualsSet`
            Arc line position residuals.
        """
        # Convert to "pre-slit" coordinates, which are slit coordinates (spatial,spectral) in units of pixels
        # We don't apply any slit distortions, since we're going to fit those.
        modelSlit = detectorMap.slitModel.withoutDistortion().spectrographToPreSlit(
            lines.fiberId, lines.wavelength
        )
        measSlit = detectorMap.slitModel.slitToPreSlit(
            detectorMap.opticsModel.detectorToSlit(
                detectorMap.detectorModel.pixelsToDetector(lines.x, lines.y)
            )
        )

        slope = np.zeros(len(lines), dtype=float)
        isTrace = lines.description == "Trace"
        if False and np.any(isTrace):
            delta = 1.0
            dySlit = detectorMap.slitModel.slitToPreSlit(
                detectorMap.opticsModel.detectorToSlit(
                    detectorMap.detectorModel.pixelsToDetector(lines.x[isTrace], lines.y[isTrace] + delta)
                )
            )
            slope[isTrace] = (dySlit[:, 0] - measSlit[isTrace, 0])  # dx/dy

        # Measurement minus model: our fitted distortions get added to the model
        xx = measSlit[:, 0]
        yy = measSlit[:, 1]
        xBase = modelSlit[:, 0]
        yBase = modelSlit[:, 1]
        dx = xx - xBase
        dy = yy - yBase
        xErr = lines.xErr
        yErr = lines.yErr

        return ArcLineResidualsSet.fromColumns(
            fiberId=lines.fiberId,
            wavelength=lines.wavelength,
            x=dx,
            y=dy,
            xOrig=xx,
            yOrig=yy,
            slope=slope,
            xBase=xBase,
            yBase=yBase,
            xErr=xErr,
            yErr=yErr,
            xx=lines.xx,
            yy=lines.yy,
            xy=lines.xy,
            flux=lines.flux,
            fluxErr=lines.fluxErr,
            fluxNorm=lines.fluxNorm,
            flag=lines.flag | np.any(np.isnan(modelSlit), axis=1),
            status=lines.status,
            description=lines.description,
            transition=lines.transition,
            source=lines.source,
        )

    def makeDetectorMap(
        self,
        base: DetectorMap,
        distortion: Distortion,
        visitInfo: VisitInfo,
        metadata: PropertyList,
    ) -> DetectorMap:
        """Make a DetectorMap from a base and distortion

        Parameters
        ----------
        base : `pfs.drp.stella.DetectorMap`
            Base detectorMap.
        distortion : `pfs.drp.stella.Distortion`
            Distortion to apply.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information for exposure.
        metadata : `lsst.daf.base.PropertyList`
            DetectorMap metadata (FITS header).

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            DetectorMap with distortion applied.
        """
        if not isinstance(base, OpticalModelDetectorMap):
            raise RuntimeError(f"Require OpticalModelDetectorMap instead of {type(base)}")
        slit = base.slitModel.withoutDistortion().withDistortion(distortion)
        return OpticalModelDetectorMap(
            slit, base.opticsModel.copy(), base.detectorModel.copy(), visitInfo, metadata
        )

    def fitModelImpl(
        self,
        bbox: Box2D,
        xBase: np.ndarray,
        yBase: np.ndarray,
        xMeas: np.ndarray,
        yMeas: np.ndarray,
        xErr: np.ndarray,
        yErr: np.ndarray,
        slope : np.ndarray,
        isLine: np.ndarray,
        DistortionClass: Type[Distortion],
    ) -> Distortion:
        """Implementation for fitting the distortion model

        We've gathered and formatted all the data; this method encapsulates the
        actual fitting, allowing it to be easily overridden by subclasses.

        Parameters
        ----------
        bbox : `lsst.geom.Box2D`
            Bounding box for detector.
        xBase, yBase : `numpy.ndarray` of `float`
            Base position for each line (pixels).
        xMeas, yMeas : `numpy.ndarray` of `float`
            Measured position for each line (pixels).
        xErr, yErr : `numpy.ndarray` of `float`
            Error in measured position for each line (pixels).
        slope : `numpy.ndarray` of `float`
            (Inverse) slope of trace (dx/dy; pixels per pixel). Only set for
            lines where ``useForWavelength`` is `False`.
        isLine : `numpy.ndarray` of `bool`
            Is this point a line? Otherwise, it's a trace.
        DistortionClass : `type`
            Class to use for the distortion.

        Returns
        -------
        distortion : `pfs.drp.stella.Distortion`
            Distortion model that was fit to the data.
        """
        threshold = self.config.lsqThreshold
        forced = self.config.forced or None
        parameters = self.config.parameters or None
        return DistortionClass.fit(
            self.config.order,
            bbox,
            xBase,
            yBase,
            xMeas,
            yMeas,
            xErr,
            yErr,
            isLine,
            slope,
            threshold,
            forced,
            parameters,
        )
