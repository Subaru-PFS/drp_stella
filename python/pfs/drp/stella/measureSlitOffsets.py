import numpy as np
import scipy.optimize

from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import Task, Struct

from pfs.datamodel.pfsConfig import FiberStatus
from .readLineList import ReadLineListTask
from .centroidLines import CentroidLinesTask

import lsstDebug

__all__ = ("MeasureSlitOffsetsConfig", "MeasureSlitOffsetsTask")


class MeasureSlitOffsetsConfig(Config):
    """Configuration for MeasureSlitOffsetsTask"""
    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read line list")
    centroidLines = ConfigurableField(target=CentroidLinesTask, doc="Centroid lines")
    rejIterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejThreshold = Field(dtype=float, default=3.0, doc="Rejection threshold (sigma)")


class MeasureSlitOffsetsTask(Task):
    """Measure consistent x,y offsets applicable to all fibers"""
    ConfigClass = MeasureSlitOffsetsConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")
        self.makeSubtask("centroidLines")
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, exposure, detectorMap, pfsConfig, apply=True):
        """Measure consistent x,y offsets applicable to all fibers

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure containing spectra.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.
        apply : `bool`
            Apply the measured offsets to the ``detectorMap``?

        Returns
        -------
        x, y : `float`
            Mean x and y shifts.
        chi2 : `float`
            chi^2 for the fit.
        dof : `int`
            Number of degrees of freedom.
        num : `int`
            Number of centroid measurements.
        """
        indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD)
        fiberId = pfsConfig.fiberId[indices]
        lines = self.readLineList.run(detectorMap=detectorMap, fiberId=fiberId,
                                      metadata=exposure.getMetadata())
        centroids = self.centroidLines.run(exposure, lines, detectorMap)

        good = ~centroids.flag
        offsets = None
        for ii in range(self.config.rejIterations):
            offsets = self.measureSlitOffsets(detectorMap, centroids, good, offsets)
            self.log.debug("Iteration %d slit offsets: spatial=%f spectral=%f chi2=%f soften=%f num=%d",
                           ii, offsets.x, offsets.y, offsets.chi2, offsets.soften, offsets.num)
            dx = np.abs((offsets.dx - offsets.x)/np.hypot(centroids.xErr[good], offsets.soften))
            dy = np.abs((offsets.dy - offsets.y)/np.hypot(centroids.yErr[good], offsets.soften))
            keep = (dx < self.config.rejThreshold) & (dy < self.config.rejThreshold)
            good[good] &= keep
            if np.all(keep):
                break
        else:
            # Final iteration with no rejection
            offsets = self.measureSlitOffsets(detectorMap, centroids, good, offsets)

        self.log.info("Measured slit offsets: spatial=%f spectral=%f chi2=%f soften=%f num=%d",
                      offsets.x, offsets.y, offsets.chi2, offsets.soften, offsets.num)
        self.displaySlitOffsets(exposure, centroids, detectorMap, fiberId, set(centroids.wavelength), offsets)
        if apply:
            self.applySlitOffsets(detectorMap, offsets)
        return offsets

    def measureSlitOffsets(self, detectorMap, centroids, good, offsets=None):
        """Measure slit offsets

        Simply measuring the mean x and y offsets is not sufficient because the
        detectorMap may include distortion (so the effects of a shift at the
        edge can be different from the effects of the same shift at the center).
        Instead, we fit for x,y slit offsets using the detectorMap.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        centroids : `pfs.drp.stella.ArcLineSet`
            Line centroids.
        good : `numpy.ndarray` of `float`
            Boolean array indicating which centroids are good.
        offsets : `lsst.pipe.base.Struct`
            Results from previous ``measureSlitOffsets`` invocations. Includes
            ``x`` and ``y`` members (`float`), that serve as the starting point
            for minimisation.

        Returns
        -------
        x, y : `float`
            Mean x and y shifts.
        chi2 : `float`
            chi^2 for the fit.
        dof : `int`
            Number of degrees of freedom.
        num : `int`
            Number of centroid measurements.
        dx, dy : `numpy.ndarray` of `float`
            Offsets in x and y for the good points.
        good : `numpy.ndarray` of `bool`
            Boolean array indicating which centroids are good.
        """
        fiberId = centroids.fiberId[good].astype(np.int32)
        wavelength = centroids.wavelength[good].astype(np.float32)
        xx = centroids.x[good]
        yy = centroids.y[good]
        xErr = centroids.xErr[good]
        yErr = centroids.yErr[good]

        def calculateOffsets(detMap):
            """Calculate the x,y offsets

            Offsets are in the sense of measured minus expected positions.

            Parameters
            ----------
            detMap : `pfs.drp.stella.DetectorMap`
                Mapping from fiberId,wavelength to x,y.

            Returns
            -------
            dx, dy : `numpy.ndarray` of `float`
                Offsets in x,y for each selected point.
            """
            points = detMap.findPoint(fiberId, wavelength)
            dx = xx - points[:, 0]
            dy = yy - points[:, 1]
            return dx, dy

        def offsetChi2(params):
            """Calculate chi^2 given dx,dy

            Parameters
            ----------
            params : `tuple` of 2 `float`s
                x and y offsets.

            Returns
            -------
            chi2 : `float`
                chi^2 for the offsets provided.
            """
            detMap = detectorMap.clone()
            detMap.applySlitOffset(*params)
            dx, dy = calculateOffsets(detMap)
            return ((dx/xErr)**2).sum() + ((dy/yErr)**2).sum()

        dx, dy = calculateOffsets(detectorMap)
        if offsets is not None:
            start = (offsets.x, offsets.y)
        else:
            start = (np.median(dx), np.median(dy))

        result = scipy.optimize.minimize(offsetChi2, start, method='Nelder-Mead')
        if not result.success:
            raise RuntimeError("Failed to fit slit offsets")

        dx2 = (dx - result.x[0])**2
        dy2 = (dy - result.x[1])**2
        xErr2 = xErr**2
        yErr2 = yErr**2

        def softenChi2(soften):
            """Return chi^2/dof (minus 1) with the softening applied

            This allows us to find the softening parameter that results in
            chi^2/dof = 1 by bisection.

            Parameters
            ----------
            soften : `float`
                Systematic error to add in quadrature to measured errors
                (pixels).

            Returns
            -------
            chi2 : `float`
                chi^2/dof - 1
            """
            xChi2 = np.sum(dx2/(soften**2 + xErr2))
            yChi2 = np.sum(dy2/(soften**2 + yErr2))
            return (xChi2 + yChi2)/dof - 1

        num = good.sum()
        dof = 2*num - 2
        if result.fun > dof:
            soften = scipy.optimize.bisect(softenChi2, 0.0, 1.0)
        else:
            soften = 0.0

        if self.debugInfo.plot:
            import itertools
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, )
            for ax, ((x, xErr, xText), (y, yText)) in zip(
                sum(axes.tolist(), []),
                itertools.product([(centroids.x, centroids.xErr, "x"), (centroids.y, centroids.yErr, "y")],
                                  [(dx - result.x[0], "dx"), (dy - result.x[1], "dy")])
            ):
                ax.scatter(x[good], y/xErr[good], marker=".", color="k")
                ax.axhline(0.0, linestyle=":", color="k")
                ax.set_xlabel(xText)
                ax.set_ylabel(yText)
            fig.suptitle(f"Offset: x={result.x[0]:.3f} y={result.x[1]:.3f} chi2={result.fun:.1f} "
                         f"soften={soften:.3f} num={num}")
            fig.tight_layout()
            plt.show()

        return Struct(x=result.x[0], y=result.x[1], chi2=result.fun, dof=2*num - 2, num=num,
                      dx=dx, dy=dy, good=good, soften=soften)

    def displaySlitOffsets(self, exposure, centroids, detectorMap, fiberId, wavelength, offsets):
        """Display the exposure with the before and after detectorMaps

        The measured line positions are shown in red, the 'before' detectorMap
        is shown in yellow, and the 'after' detectorMap is shown in green.

        The following debug parameters are used:
        - ``display`` (`bool`): display anything (defaults to ``False``)?
        - ``frame`` (`int`): display frame to use (defaults to ``1``).
        - ``backend`` (`str`): display backend to use (defaults to ``"ds9"``).

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image to display.
        centroids : `pfs.drp.stella.ArcLineSet`
            List of line centroids.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength and x,y.
        fiberId : iterable of `int`
            Fiber identifiers to plot.
        wavelength : iterable of `float`
            Wavelengths to mark.
        offsets : `lsst.pipe.base.Struct`
            Structure containing ``x`` and ``y`` (`float`) offsets.
        """
        if not self.debugInfo.display:
            return
        from lsst.afw.display import Display
        disp = Display(frame=self.debugInfo.frame or 1, backend=self.debugInfo.backend or "ds9")
        disp.mtv(exposure)
        with disp.Buffering():
            for line in centroids:
                disp.dot("x", line.x, line.y, size=5, ctype="red")
        detectorMap.display(disp, fiberId, wavelength, "yellow")
        fixed = detectorMap.clone()
        fixed.applySlitOffset(offsets.x, offsets.y)
        fixed.display(disp, fiberId, wavelength, "green")

    def applySlitOffsets(self, detectorMap, offsets):
        """Apply measured offsets to the detectorMap

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength and x,y. Will be modified.
        offsets : `lsst.pipe.base.Struct`
            Structure containing ``x`` and ``y`` (`float`) offsets.
        """
        self.log.info("Applying slit offsets: %f %f", offsets.x, offsets.y)
        detectorMap.applySlitOffset(offsets.x, offsets.y)
