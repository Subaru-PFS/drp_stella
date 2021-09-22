import numpy as np

from lsst.pex.config import Field
from lsst.geom import Box2D

from . import SplinedDetectorMap
from . import ReferenceLineStatus
from .arcLine import ArcLine, ArcLineSet
from .fitDistortedDetectorMap import FitDistortedDetectorMapTask, FitDistortedDetectorMapConfig

__all__ = ("AdjustDetectorMapConfig", "AdjustDetectorMapTask")


class AdjustDetectorMapConfig(FitDistortedDetectorMapConfig):
    """Configuration for AdjustDetectorMapTask"""
    traceSpectralError = Field(dtype=float, default=1.0,
                               doc="Error in the spectral dimension to give trace centroids")

    def setDefaults(self):
        super().setDefaults()
        self.order = 2  # Fit low-order coefficients only


class AdjustDetectorMapTask(FitDistortedDetectorMapTask):
    ConfigClass = AdjustDetectorMapConfig
    _DefaultName = "adjustDetectorMap"

    def run(self, detectorMap, lines=None, *, traces=None):
        """Adjust a DistortedDetectorMap to fit arc line measurements

        We fit only the lowest order terms.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured arc lines.
        traces : `dict` mapping `int` to `list` of `pfs.drp.stella.TracePeak`
            Measured peak positions for each row, indexed by (identified)
            fiberId. These are only used if we don't have lines.

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
        """
        base = self.getBaseDetectorMap(detectorMap)  # NB: DistortionBasedDetectorMap not SplinedDetectorMap
        needNumLines = self.Distortion.getNumParametersForOrder(self.config.order)
        numGoodLines = self.countGoodLines(lines)
        if numGoodLines < needNumLines:
            if traces is not None:
                lines = self.generateLines(detectorMap, traces)
            else:
                raise RuntimeError(f"Insufficient good lines: {numGoodLines} vs {needNumLines}")
        residuals = self.calculateBaseResiduals(base, lines)
        dispersion = base.getDispersion(base.fiberId[len(base)//2])
        results = self.fitDistortion(detectorMap.bbox, residuals, dispersion,
                                     seed=detectorMap.visitInfo.getExposureId(), fitStatic=False,
                                     Distortion=type(base.getDistortion()))
        results.detectorMap = self.constructAdjustedDetectorMap(base, results.distortion)

        if self.debugInfo.lineQa:
            self.lineQa(lines, results.detectorMap)
        if self.debugInfo.wlResid:
            self.plotWavelengthResiduals(results.detectorMap, lines, results.used, results.reserved)
        return results

    def getBaseDetectorMap(self, detectorMap):
        """Get detectorMap to use as a base

        We need to ensure that the detectorMap is indeed a
        ``DistortionBasedDetectorMap`` with a sufficient polynomial order.
        We zero out the low-order coefficients to make it easy to construct the
        adjusted version.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.

        Returns
        -------
        base : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        """
        visitInfo = detectorMap.visitInfo
        metadata = detectorMap.metadata
        if isinstance(detectorMap, SplinedDetectorMap):
            # Promote SplinedDetectorMap to DistortionBasedDetectorMap
            numCoeff = self.Distortion.getNumParametersForOrder(self.config.order)
            coeff = np.zeros(numCoeff, dtype=float)
            distortion = self.Distortion(self.config.order, Box2D(detectorMap.bbox), coeff)
            Class = self.DetectorMap
        else:
            distortion = detectorMap.distortion.removeLowOrder(self.config.order)
            Class = type(detectorMap)
            detectorMap = detectorMap.base
        return Class(detectorMap, distortion, visitInfo, metadata)

    def countGoodLines(self, lines):
        """Count the number of good lines

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineSet`
            Measured arc lines.

        Returns
        -------
        num : `int`
            Number of good lines.
        """
        if lines is None:
            return 0
        good = lines.flag == 0
        good &= (lines.status & ReferenceLineStatus.fromNames(*self.config.lineFlags)) == 0
        good &= np.isfinite(lines.x) & np.isfinite(lines.y)
        good &= np.isfinite(lines.xErr) & np.isfinite(lines.yErr)
        return good.sum()

    def isSufficientGoodLines(self, lines):
        """Return whether we have enough good lines

        to be able to fit using lines only.

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineSet`
            Measured arc lines.

        Returns
        -------
        isSufficient : `bool`
            Do we have enough good lines?
        """
        needNumLines = self.Distortion.getNumParametersForOrder(self.config.order)
        numGoodLines = self.countGoodLines(lines)
        return numGoodLines > needNumLines

    def generateLines(self, detectorMap, traces):
        """Convert traces to lines

        Well, they're not really lines, but we have measurements on where the
        traces are in x, so that will allow us to fit some distortion
        parameters. If there aren't any lines, we won't be able to update the
        wavelength solution much, but we're probably working with a quartz so
        that doesn't matter.

        Parameters
        ----------
        traces : `dict` mapping `int` to `list` of `pfs.drp.stella.TracePeak`
            Measured peak positions for each row, indexed by (identified)
            fiberId. These are only used if we don't have lines.

        Returns
        -------
        lines : `pfs.drp.stella.ArcLineSet`
            Line measurements, treating every trace row with a centroid as a
            line.
        """
        lines = ArcLineSet.empty()
        for fiberId in traces:
            row = np.array([tt.row for tt in traces[fiberId]], dtype=float)
            wavelength = detectorMap.findWavelength(fiberId, row)
            lines.extend([ArcLine(fiberId, wl, tt.peak, yy, tt.peakErr, self.config.traceSpectralError,
                                  1.0, 0.0, False, ReferenceLineStatus.GOOD, "Trace") for
                          wl, yy, tt in zip(wavelength, row, traces[fiberId])])
        return lines

    def constructAdjustedDetectorMap(self, base, distortion):
        """Construct an adjusted detectorMap

        The adjusted detectorMap uses the low-order coefficients that we've just
        fit, and the high-order coefficients from the original detectorMap.

        Parameters
        ----------
        base : `pfs.drp.stella.DistortionBasedDetectorMap`
            Mapping from fiberId,wavelength --> x,y with low-order coefficients
            zeroed out.
        distortion : `pfs.drp.stella.BaseDistortion`
            Low-order distortion to apply.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DistortionBasedDetectorMap`
            Mapping from fiberId,wavelength --> x,y with both low- and
            high-order coefficients set.
        """
        return type(base)(base.getBase(), base.getDistortion().merge(distortion),
                          base.visitInfo, base.metadata)
