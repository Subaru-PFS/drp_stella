import numpy as np

from lsst.pex.config import Field
from lsst.geom import Box2D

from . import SplinedDetectorMap
from . import DistortedDetectorMap, DetectorDistortion
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
        detectorMap : `pfs.drp.stella.DistortedDetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured arc lines.
        traces : `dict` mapping `int` to `list` of `pfs.drp.stella.TracePeak`
            Measured peak positions for each row, indexed by (identified)
            fiberId. These are only used if we don't have lines.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DistortedDetectorMap`
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
        base = self.getBaseDetectorMap(detectorMap)  # NB: a DistortedDetectorMap, not SplinedDetectorMap
        needNumLines = DetectorDistortion.getNumDistortion(self.config.order)
        numGoodLines = self.countGoodLines(lines)
        if numGoodLines < needNumLines:
            if traces is not None:
                lines = self.generateLines(detectorMap, traces)
            else:
                raise RuntimeError(f"Insufficient good lines: {numGoodLines} vs {needNumLines}")
        residuals = self.calculateBaseResiduals(base, lines)
        dispersion = base.getDispersion(base.fiberId[len(base)//2])
        results = self.fitDetectorDistortion(detectorMap.bbox, residuals, False, dispersion,
                                             seed=detectorMap.visitInfo.getExposureId())
        results.detectorMap = self.constructAdjustedDetectorMap(base, results.distortion)

        numCoeff = DetectorDistortion.getNumDistortion(self.config.order)
        if isinstance(detectorMap, DistortedDetectorMap):
            xCoeff = detectorMap.distortion.getXCoefficients()
            yCoeff = detectorMap.distortion.getYCoefficients()
        else:
            xCoeff = np.zeros(numCoeff, dtype=float)
            yCoeff = np.zeros(numCoeff, dtype=float)
        xDiff = results.detectorMap.distortion.getXCoefficients() - xCoeff
        yDiff = results.detectorMap.distortion.getYCoefficients() - yCoeff
        self.log.info("Adjustment in x: %s", xDiff[:numCoeff])
        self.log.info("Adjustment in y: %s", yDiff[:numCoeff])

        if self.debugInfo.lineQa:
            self.lineQa(lines, results.detectorMap)
        if self.debugInfo.wlResid:
            self.plotWavelengthResiduals(results.detectorMap, lines, results.used, results.reserved)
        return results

    def getBaseDetectorMap(self, detectorMap):
        """Get detectorMap to use as a base

        We need to ensure that the detectorMap is indeed a
        ``DistortedDetectorMap`` with a sufficient polynomial order.
        We zero out the low-order coefficients to make it easy to construct the
        adjusted version.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DistortedDetectorMap`
            Mapping from fiberId,wavelength --> x,y.

        Returns
        -------
        base : `pfs.drp.stella.DistortedDetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        """
        if isinstance(detectorMap, SplinedDetectorMap):
            # Promote SplinedDetectorMap to DistortedDetectorMap
            numCoeff = DetectorDistortion.getNumDistortion(self.config.order)
            coeff = np.zeros(numCoeff, dtype=float)
            rightCcd = np.zeros(6, dtype=float)
            distortion = DetectorDistortion(self.config.order, Box2D(detectorMap.bbox), coeff, coeff,
                                            rightCcd)
            return DistortedDetectorMap(detectorMap, distortion, detectorMap.visitInfo, detectorMap.metadata)
        if not isinstance(detectorMap, DistortedDetectorMap):
            raise NotImplementedError(f"Unsupported detectorMap class: {detectorMap.__class__}")
        distortion = detectorMap.distortion
        orderHave = distortion.getDistortionOrder()
        orderWant = self.config.order
        numHave = distortion.getNumDistortion(orderHave)
        numWant = distortion.getNumDistortion(orderWant)
        if numHave > numWant:
            xCoeff = detectorMap.distortion.getXCoefficients()
            yCoeff = detectorMap.distortion.getYCoefficients()
            # Taking advantage of the fact that the coefficients are stored in order of increasing order:
            # e.g., 1 x y x^2 xy y^2 x^3 x^2y xy^2 y^3
            xCoeff[:numWant] = 0.0
            yCoeff[:numWant] = 0.0
        else:
            xCoeff = np.zeros(numWant, dtype=float)
            yCoeff = np.zeros(numWant, dtype=float)
            orderHave = orderWant
            numHave = numWant

        adjDistortion = DetectorDistortion(orderHave, distortion.getRange(), xCoeff, yCoeff,
                                           distortion.getRightCcdCoefficients())
        return DistortedDetectorMap(detectorMap.base, adjDistortion, detectorMap.visitInfo,
                                    detectorMap.metadata)

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
        needNumLines = DetectorDistortion.getNumDistortion(self.config.order)
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
                                  np.nan, np.nan, False, ReferenceLineStatus.GOOD, "Trace") for
                          wl, yy, tt in zip(wavelength, row, traces[fiberId])])
        return lines

    def constructAdjustedDetectorMap(self, base, distortion):
        """Construct an adjusted detectorMap

        The adjusted detectorMap uses the low-order coefficients that we've just
        fit, and the high-order coefficients from the original detectorMap.

        Parameters
        ----------
        base : `pfs.drp.stella.DistortedDetectorMap`
            Mapping from fiberId,wavelength --> x,y with low-order coefficients
            zeroed out.
        distortion : `pfs.drp.stella.DetectorDistortion`
            Low-order distortion to apply.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DistortedDetectorMap`
            Mapping from fiberId,wavelength --> x,y with both low- and
            high-order coefficients set.
        """
        numCoeff = DetectorDistortion.getNumDistortion(self.config.order)
        xCoeff = base.distortion.getXCoefficients()
        yCoeff = base.distortion.getYCoefficients()

        # Taking advantage of the fact that the coefficients are stored in order of increasing order:
        # e.g., 1 x y x^2 xy y^2 x^3 x^2y xy^2 y^3
        xCoeff[:numCoeff] = distortion.getXCoefficients()
        yCoeff[:numCoeff] = distortion.getYCoefficients()

        original = base.distortion
        distortion = DetectorDistortion(original.getDistortionOrder(), original.getRange(), xCoeff, yCoeff,
                                        original.getRightCcdCoefficients())
        return DistortedDetectorMap(base.getBase(), distortion, base.visitInfo, base.metadata)
