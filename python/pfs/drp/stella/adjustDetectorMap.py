from typing import Type

import numpy as np

from lsst.geom import Box2D

from . import SplinedDetectorMap
from . import ReferenceLineStatus
from .DistortionContinued import Distortion
from .DoubleDetectorMapContinued import DoubleDetectorMap
from .DistortedDetectorMapContinued import DistortedDetectorMap
from .MultipleDistortionsDetectorMapContinued import MultipleDistortionsDetectorMap
from .PolynomialDetectorMapContinued import PolynomialDetectorMap
from .RotScaleDistortionContinued import RotScaleDistortion
from .fitDistortedDetectorMap import FitDistortedDetectorMapTask, FitDistortedDetectorMapConfig

__all__ = ("AdjustDetectorMapConfig", "AdjustDetectorMapTask")


class AdjustDetectorMapConfig(FitDistortedDetectorMapConfig):
    """Configuration for AdjustDetectorMapTask"""
    def setDefaults(self):
        self.exclusionRadius = 4.0


class AdjustDetectorMapTask(FitDistortedDetectorMapTask):
    ConfigClass = AdjustDetectorMapConfig
    _DefaultName = "adjustDetectorMap"

    def run(self, detectorMap, lines, arm: str, seed=0):
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
        pfs.drp.stella.fitDistortedDetectorMap.FittingError
            If the data is not of sufficient quality to fit.
        """
        base = self.getBaseDetectorMap(detectorMap, arm)  # NB: not SplinedDetectorMap
        DistortionClass = self.getDistortionClass(arm)
        dispersion = base.getDispersionAtCenter(base.fiberId[len(base)//2])
        needNumLines = 8  # RotScaleDistortion
        good = self.getGoodLines(lines, detectorMap.getDispersionAtCenter())
        numGoodLines = good.sum()

        if numGoodLines < needNumLines:
            raise RuntimeError(f"Insufficient good lines: {numGoodLines} vs {needNumLines}")
        for ii in range(self.config.traceIterations):
            self.log.debug("Commencing trace iteration %d", ii)
            residuals = self.calculateBaseResiduals(base, lines)
            weights = self.calculateWeights(lines)
            fit = self.fitDistortion(
                detectorMap.bbox, residuals, weights, dispersion, seed=seed, DistortionClass=DistortionClass
            )
            detectorMap = self.constructAdjustedDetectorMap(base, fit.distortion)
            if not self.updateTraceWavelengths(lines, detectorMap):
                break

        results = self.measureQuality(lines, detectorMap, fit.selection, fit.numParameters)
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
        return RotScaleDistortion

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
        visitInfo = detectorMap.visitInfo
        metadata = detectorMap.metadata
        # Promote other detectorMap classes to MultipleDistortionsDetectorMap
        if isinstance(detectorMap, SplinedDetectorMap):
            distortions = []
            base = detectorMap
        elif isinstance(detectorMap, (PolynomialDetectorMap, DoubleDetectorMap, DistortedDetectorMap)):
            distortions = [detectorMap.distortion.clone()]
            base = detectorMap.base
        elif isinstance(detectorMap, MultipleDistortionsDetectorMap):
            distortions = [dd.clone() for dd in detectorMap.distortions]
            base = detectorMap.base
        else:
            raise RuntimeError(f"Unrecognized detectorMap type: {type(detectorMap)}")
        return MultipleDistortionsDetectorMap(base, distortions, visitInfo, metadata)

    def constructAdjustedDetectorMap(self, base, distortion):
        """Construct an adjusted detectorMap

        The adjusted detectorMap uses the low-order coefficients that we've just
        fit, and the high-order coefficients from the original detectorMap.

        Parameters
        ----------
        base : `pfs.drp.stella.MultipleDistortionsDetectorMap`
            Mapping from fiberId,wavelength --> x,y with low-order coefficients
            zeroed out.
        distortion : `pfs.drp.stella.Distortion`
            Low-order distortion to apply.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y with both low- and
            high-order coefficients set.
        """
        return MultipleDistortionsDetectorMap(
            base.base, base.distortions + [distortion], base.visitInfo, base.metadata
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
            bbox, xBase, yBase, xMeas, yMeas, xErr, yErr, slope, isLine, threshold, forced, parameters
        )
