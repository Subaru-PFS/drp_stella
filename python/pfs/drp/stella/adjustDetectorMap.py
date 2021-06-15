from functools import partial

import numpy as np
import scipy.optimize

import lsstDebug

from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task, Struct
from lsst.geom import Box2D

from . import SplinedDetectorMap
from . import DifferentialDetectorMap, GlobalDetectorModel
from . import DistortedDetectorMap, DetectorDistortion
from . import ReferenceLineStatus

__all__ = ("AdjustDetectorMapConfig", "AdjustDetectorMapTask")


class AdjustDetectorMapConfig(Config):
    """Configuration for FitDifferentialDetectorMapTask"""
    iterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejection = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    order = Field(dtype=int, default=1, doc="Distortion order")
    lineFlags = ListField(dtype=str, default=["BAD"], doc="ReferenceLineStatus flags for lines to ignore")
    tracesSoften = Field(dtype=float, default=0.05, doc="Systematic error to apply to traces (pixels)")
    linesSoften = Field(dtype=float, default=0.05, doc="Systematic error to apply to lines (pixels)")


class AdjustDetectorMapTask(Task):
    ConfigClass = AdjustDetectorMapConfig
    _DefaultName = "adjustDetectorMap"

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def getDetectorMap(self, detectorMap):
        """Return a suitable detectorMap for adjusting

        A `pfs.drp.stella.SplinedDetectorMap` needs to be promoted to a
        `pfs.drp.stella.DistortedDetectorMap`.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y (that can be adjusted).
        """
        if isinstance(detectorMap, (DifferentialDetectorMap, DistortedDetectorMap)):
            return detectorMap
        if not isinstance(detectorMap, SplinedDetectorMap):
            raise RuntimeError(f"Unrecognised detectorMap class: {type(detectorMap)}")
        # Promote SplinedDetectorMap to DistortedDetectorMap
        numCoeff = DetectorDistortion.getNumDistortion(self.config.order)
        coeff = np.zeros(numCoeff, dtype=float)
        rightCcd = np.zeros(6, dtype=float)
        distortion = DetectorDistortion(self.config.order, Box2D(detectorMap.bbox), coeff, coeff, rightCcd)
        return DistortedDetectorMap(detectorMap, distortion, detectorMap.visitInfo, detectorMap.metadata)

    def run(self, detectorMap, lines, traces):
        """Fit for low-order terms in the detectorMap to trace positions from the detectorMap

        This provides a more accurate functional form, and therefore a better
        fit to the trace positions and consequently more accurate centroids to
        use for the profiles. This is especially important in the low S/N
        regions affected by the dichroic.

        We fit for a low-order correction to the detectorMap in order to match
        the positions of reference lines and/or the continuum.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength --> x,y.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured reference lines.
        traces : `dict` mapping `int` to `list` of `pfs.drp.stella.TracePeak`
            Measured peak positions for each row, indexed by (identified)
            fiberId.

        Returns
        -------
        detectorMap : `pfs.drp.stella.DetectorMap`
            An adjusted detectorMap.
        params : `numpy.ndarray` of `float`
            Fit parameters.
        traceResid : `numpy.ndarray` of `float`
            Residual values for traces.
        traceRobustRms : `float`
            Measured robust RMS from ``traceResid``.
        traceWeightedRms : `float`
            Measured weighted RMS from ``traceResid``.
        traceChi2 : `float`
            chi2 value for traces.
        xResid : `numpy.ndarray` of `float`
            Residual values for line x positions.
        xRobustRms : `float`
            Measured robust RMS from ``xChi``.
        xWeightedRms : `float`
            Measured weighted RMS from ``xChi``.
        xChi2 : `float`
            chi2 value for line x positions.
        yResid : `numpy.ndarray` of `float`
            Residual values for line y positions.
        yRobustRms : `float`
            Measured robust RMS from ``yChi``.
        yWeightedRms : `float`
            Measured weighted RMS from ``yChi``.
        yChi2 : `float`
            chi2 value for line y positions.
        """
        # Extract traces into convenient representation
        fiberIdTraces = np.array(sum(([ff]*len(traces[ff]) for ff in traces), []), dtype=np.int32)
        xTraces = np.array(sum(([pp.peak for pp in traces[ff]] for ff in traces), []), dtype=float)
        xErrTraces = np.array(sum(([pp.peakErr for pp in traces[ff]] for ff in traces), []), dtype=float)
        yTraces = np.array(sum(([pp.row for pp in traces[ff]] for ff in traces), []), dtype=float)
        numTraces = sum(len(traces[ff]) for ff in traces)
        goodTraces = np.isfinite(xTraces) & np.isfinite(yTraces) & np.isfinite(xErrTraces)
        goodTraces &= (xTraces >= detectorMap.bbox.getMinX()) & (xTraces < detectorMap.bbox.getMaxX())
        xSoftErrTraces = np.hypot(xErrTraces, self.config.tracesSoften)
        xInvErr2Traces = 1.0/(xErrTraces**2 + self.config.tracesSoften**2)

        # Extract lines into convenient representation
        xLines = lines.x
        xErrLines = lines.xErr
        yLines = lines.y
        yErrLines = lines.yErr
        numLines = len(lines)
        goodLines = np.isfinite(xLines) & np.isfinite(yLines)
        goodLines &= np.isfinite(xErrLines) & np.isfinite(yErrLines)
        goodLines &= ~lines.flag
        goodLines &= (lines.status & ReferenceLineStatus.fromNames(*self.config.lineFlags)) == 0
        xSoftErrLines = np.hypot(xErrLines, self.config.linesSoften)
        ySoftErrLines = np.hypot(yErrLines, self.config.linesSoften)
        xInvErr2Lines = 1.0/(xErrLines**2 + self.config.linesSoften**2)
        yInvErr2Lines = 1.0/(yErrLines**2 + self.config.linesSoften**2)

        if self.debugInfo.plotBefore:
            self.plot(detectorMap, lines, goodLines, traces, goodTraces)

        numFitCoeff = DetectorDistortion.getNumDistortion(self.config.order)

        detectorMap = self.getDetectorMap(detectorMap)
        if isinstance(detectorMap, DistortedDetectorMap):
            oldDistortion = detectorMap.getDistortion()
            makeDistortion = partial(DetectorDistortion, distortionOrder=oldDistortion.getDistortionOrder(),
                                     range=oldDistortion.getRange(),
                                     rightCcd=oldDistortion.getRightCcdCoefficients())
        elif isinstance(detectorMap, DifferentialDetectorMap):
            oldDistortion = detectorMap.getModel()
            makeDistortion = partial(GlobalDetectorModel, distortionOrder=oldDistortion.getDistortionOrder(),
                                     scaling=oldDistortion.getScaling(),
                                     fiberCenter=oldDistortion.getFiberCenter(),
                                     highCcd=oldDistortion.getHighCcdCoefficients())
        else:
            raise AssertionError(f"Unrecognised detectorMap class: {type(detectorMap)}")

        xCoeffOld = oldDistortion.getXCoefficients()[numFitCoeff:]
        yCoeffOld = oldDistortion.getYCoefficients()[numFitCoeff:]

        def getNewDistortedDetectorMap(params):
            """Create a new detectorMap with the adjusted parameters

            Parameters
            ----------
            params : `numpy.ndarray` of `float`
                Adjustment parameters.

            Returns
            -------
            detectorMap : `pfs.drp.stella.DetectorMap`
                Adjusted detectorMap.
            """
            assert len(params) == 2*numFitCoeff
            xCoeffNew = np.concatenate((params[:numFitCoeff], xCoeffOld))
            yCoeffNew = np.concatenate((params[numFitCoeff:], yCoeffOld))
            newDistortion = makeDistortion(xDistortion=xCoeffNew, yDistortion=yCoeffNew)
            return type(detectorMap)(detectorMap.base, newDistortion, detectorMap.visitInfo,
                                     detectorMap.metadata)

        def calculateTraceResiduals(detectorMap):
            """Calculate residuals in trace measurements

            Parameters
            ----------
            detectorMap : `pfs.drp.stella.DetectorMap`
                Mapping of fiberId,wavelength --> x,y.

            Returns
            -------
            traceResid : `numpy.ndarray` of `float`
                Residuals in the trace measurements.
            """
            return xTraces - detectorMap.getXCenter(fiberIdTraces, yTraces)

        def calculateLineResiduals(detectorMap):
            """Calculate residuals in line measurements

            Parameters
            ----------
            detectorMap : `pfs.drp.stella.DetectorMap`
                Mapping of fiberId,wavelength --> x,y.

            Returns
            -------
            xResid, yResid : `numpy.ndarray` of `float`
                Residuals in line measurements.
            """
            model = detectorMap.findPoint(lines.fiberId, lines.wavelength)
            xResid = xLines - model[:, 0]
            yResid = yLines - model[:, 1]
            return xResid, yResid

        def calculateChi2(params, selectTraces, selectLines):
            """Calculate chi^2 given transformation parameters

            Parameters
            ----------
            params : `numpy.ndarray` of `float`
                Adjustment parameters.
            select : `numpy.ndarray` of `bool`
                Boolean array indicating which points to use.

            Returns
            -------
            chi2 : `float`
                Sum of the squared residuals. Not really chi^2, since we're not
                dividing by the errors, because we have no errors.
            """
            adjusted = getNewDistortedDetectorMap(params)
            traceResid = calculateTraceResiduals(adjusted)
            xResid, yResid = calculateLineResiduals(adjusted)
            useTraces = selectTraces & np.isfinite(traceResid)
            useLines = selectLines & np.isfinite(xResid) & np.isfinite(yResid)
            traceChi2 = np.sum(traceResid[useTraces]**2*xInvErr2Traces[useTraces])
            xChi2 = np.sum(xResid[useLines]**2*xInvErr2Lines[useLines])
            yChi2 = np.sum(yResid[useLines]**2*yInvErr2Lines[useLines])
            return traceChi2 + xChi2 + yChi2

        def calculateRobustRms(array):
            """Calculate a robust RMS from an array of values"""
            lower, upper = np.percentile(array, (25.0, 75.0))
            return 0.741*(upper - lower)

        def calculateWeightedRms(array, weights):
            """Calculate a weighted RMS"""
            return np.sqrt(np.sum(weights*array**2)/np.sum(weights))

        def fit(params, useTraces, useLines):
            """Fit adjustment parameters to our sample

            Parameters
            ----------
            params : `numpy.ndarray` of `float`
                Starting adjustment parameters.
            useTraces : `numpy.ndarray` of `bool`
                Boolean array indicating which trace measurements to use.
            useLines : `numpy.ndarray` of `bool`
                Boolean array indicating which line measurements to use.

            Returns
            -------
            params : `numpy.ndarray` of `float`
                Fit parameters.
            traceResid : `numpy.ndarray` of `float`
                Residual values for traces.
            traceRobustRms : `float`
                Measured robust RMS from ``traceResid``.
            traceWeightedRms : `float`
                Measured weighted RMS from ``traceResid``.
            traceChi2 : `float`
                chi2 value for traces.
            xResid : `numpy.ndarray` of `float`
                Residual values for line x positions.
            xRobustRms : `float`
                Measured robust RMS from ``xChi``.
            xWeightedRms : `float`
                Measured weighted RMS from ``xChi``.
            xChi2 : `float`
                chi2 value for line x positions.
            yResid : `numpy.ndarray` of `float`
                Residual values for line y positions.
            yRobustRms : `float`
                Measured robust RMS from ``yChi``.
            yWeightedRms : `float`
                Measured weighted RMS from ``yChi``.
            yChi2 : `float`
                chi2 value for line y positions.
            """
            result = scipy.optimize.minimize(partial(calculateChi2, selectTraces=useTraces,
                                                     selectLines=useLines), params, method="Powell")
            if not result.success:
                raise RuntimeError(f"Failed to fit detectorMap adjustment: {result}")
            params = result.x
            adjusted = getNewDistortedDetectorMap(params)
            traceResid = calculateTraceResiduals(adjusted)
            xResid, yResid = calculateLineResiduals(adjusted)
            if np.any(useTraces):
                traceRobustRms = calculateRobustRms(traceResid[useTraces])
                traceWeightedRms = calculateWeightedRms(traceResid[useTraces], xInvErr2Traces[useTraces])
                traceChi2 = np.sum(traceResid[useTraces]**2*xInvErr2Traces[useTraces])
            else:
                traceRobustRms = 0.0
                traceWeightedRms = 0.0
                traceChi2 = 0.0
            if np.any(useLines):
                xRobustRms = calculateRobustRms(xResid[useLines])
                yRobustRms = calculateRobustRms(yResid[useLines])
                xWeightedRms = calculateWeightedRms(xResid[useLines], xInvErr2Lines[useLines])
                yWeightedRms = calculateWeightedRms(yResid[useLines], yInvErr2Lines[useLines])
                xChi2 = np.sum(xResid[useLines]**2*xInvErr2Lines[useLines])
                yChi2 = np.sum(yResid[useLines]**2*yInvErr2Lines[useLines])
            else:
                xRobustRms, yRobustRms = 0.0, 0.0
                xWeightedRms, yWeightedRms = 0.0, 0.0
                xChi2, yChi2 = 0.0, 0.0

            self.log.debug("Adjustment parameters: %s", params)
            self.log.debug("Adjustment chi2: trace=%f x=%f y=%f total=%f",
                           traceChi2, xChi2, yChi2, traceChi2 + xChi2 + yChi2)
            self.log.debug("Adjustment robust RMS: trace=%f, x=%f, y=%f",
                           traceRobustRms, xRobustRms, yRobustRms)
            self.log.debug("Adjustment weighted RMS: trace=%f, x=%f, y=%f",
                           traceWeightedRms, xWeightedRms, yWeightedRms)
            return Struct(params=params, traceResid=traceResid, traceRobustRms=traceRobustRms,
                          traceWeightedRms=traceWeightedRms, traceChi2=traceChi2, xResid=xResid,
                          xRobustRms=xRobustRms, xWeightedRms=xWeightedRms, xChi2=xChi2, yResid=yResid,
                          yRobustRms=yRobustRms, yWeightedRms=yWeightedRms, yChi2=yChi2)

        params = np.concatenate((oldDistortion.getXCoefficients()[:numFitCoeff],
                                 oldDistortion.getYCoefficients()[:numFitCoeff]))
        oldParams = params.copy()
        keepTraces = np.ones(numTraces, dtype=bool)
        keepLines = np.ones(numLines, dtype=bool)

        for ii in range(self.config.iterations):
            selectTraces = goodTraces & keepTraces
            selectLines = goodLines & keepLines
            self.log.debug("Fit iteration %d: %d traces and %d lines",
                           ii, selectTraces.sum(), selectLines.sum())
            result = fit(params, selectTraces, selectLines)
            with np.errstate(invalid="ignore"):
                keepTraces = np.abs(result.traceResid) < self.config.rejection*xSoftErrTraces
                keepLines = ((np.abs(result.xResid) < self.config.rejection*xSoftErrLines) &
                             (np.abs(result.yResid) < self.config.rejection*ySoftErrLines))
            params = result.params

        # Final fit
        selectTraces = goodTraces & keepTraces
        selectLines = goodLines & keepLines
        self.log.debug("Final fit: %d traces and %d lines", selectTraces.sum(), selectLines.sum())
        result = fit(params, selectTraces, selectLines)
        self.log.info("detectorMap parameter adjustment: %s", params - oldParams)
        self.log.info("Adjustment chi2: trace=%f from %d/%d; x=%f y=%f from %d/%d; total=%f from %d",
                      result.traceChi2, selectTraces.sum(), goodTraces.sum(),
                      result.xChi2, result.yChi2, selectLines.sum(), goodLines.sum(),
                      result.traceChi2 + result.xChi2 + result.yChi2,
                      selectTraces.sum() + 2*selectLines.sum())
        self.log.info("Adjustment robust RMS (pixels): trace=%f; x=%f y=%f",
                      result.traceRobustRms, result.xRobustRms, result.yRobustRms)
        self.log.info("Adjustment weighted RMS (pixels): trace=%f; x=%f y=%f",
                      result.traceWeightedRms, result.xWeightedRms, result.yWeightedRms)

        result.detectorMap = getNewDistortedDetectorMap(result.params)
        if self.debugInfo.plotAfter:
            self.plot(result.detectorMap, lines, goodLines, traces, goodTraces)
        return result

    def plot(self, detectorMap, lines, selectLines, traces, selectTraces):
        """Plot lines and traces

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength --> x,y.
        lines : `pfs.drp.stella.ArcLineSet`
            Measured reference lines.
        selectLines : `numpy.ndarray` of `bool`
            Boolean array indicating which line measurements to use.
        traces : `dict` mapping `int` to `list` of `pfs.drp.stella.TracePeak`
            Measured peak positions for each row, indexed by (identified)
            fiberId.
        selectTraces : `numpy.ndarray` of `bool`
            Boolean array indicating which trace measurements to use.
        """
        import matplotlib.cm
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2)

        fiberId = sorted(set(traces.keys()) | set(lines.fiberId))
        numFibers = len(fiberId)
        colors = {ff: matplotlib.cm.rainbow(xx) for ff, xx in zip(fiberId, np.linspace(0, 1, numFibers))}
        start = 0
        selectTracesByFiber = {}
        for ff in traces:
            num = len(traces[ff])
            selectTracesByFiber[ff] = selectTraces[start:start + num]
            start += num
        xyModel = detectorMap.findPoint(lines.fiberId, lines.wavelength)

        # First plot: x,y positions
        ax = axes[0, 0]
        ax.scatter(lines.x[selectLines], lines.y[selectLines], marker="o", color="b")
        ax.scatter(xyModel[:, 0][selectLines], xyModel[:, 1][selectLines], marker="x", color="r")
        rows = np.arange(detectorMap.bbox.getMinY(), detectorMap.bbox.getMaxY(), dtype=float)
        for fiberId in traces:
            select = selectTracesByFiber[fiberId]
            ax.scatter(np.array([tt.peak for tt in traces[fiberId]])[select],
                       np.array([tt.row for tt in traces[fiberId]])[select],
                       marker="+", color="k", alpha=0.1)
            xCenter = detectorMap.getXCenter(fiberId, rows)
            ax.plot(xCenter, rows, linestyle=":", color=colors[fiberId])
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_title("Trace and line positions")

        # Second plot: trace residuals
        ax = axes[1, 0]
        for ff, color in zip(traces, matplotlib.cm.rainbow(np.linspace(0, 1, len(traces)))):
            xx = np.array([tt.peak for tt in traces[ff]], dtype=float)
            yy = np.array([tt.row for tt in traces[ff]], dtype=float)
            dx = xx - detectorMap.getXCenter(ff, yy)
            select = selectTracesByFiber[ff]
            ax.scatter(yy[select], dx[select], marker=".", color=color, alpha=0.1)
        ax.set_xlabel("y (pixels)")
        ax.set_ylabel("x residual (pixels)")
        ax.set_title("Trace residuals")

        # Third plot: x line residuals
        ax = axes[0, 1]
        ax.scatter(lines.x[selectLines], lines.x[selectLines] - xyModel[:, 0][selectLines],
                   marker=".", color=[colors[ff] for ff in lines.fiberId[selectLines]], alpha=0.1)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("x residual (pixels)")
        ax.set_title("Line x residuals")

        # Fourth plot: y line residuals
        ax = axes[1, 1]
        ax.scatter(lines.y[selectLines], lines.y[selectLines] - xyModel[:, 1][selectLines],
                   marker=".", color=[colors[ff] for ff in lines.fiberId[selectLines]], alpha=0.1)
        ax.set_xlabel("y (pixels)")
        ax.set_ylabel("y residual (pixels)")
        ax.set_title("Line y residuals")

        fig.tight_layout()
        plt.show()
