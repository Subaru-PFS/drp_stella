import itertools
import numpy as np
import scipy.optimize

from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import Task, Struct

from pfs.datamodel.pfsConfig import FiberStatus
from .readLineList import ReadLineListTask
from .centroidLines import CentroidLinesTask
from .GlobalDetectorMapContinued import GlobalDetectorMap

import lsstDebug

__all__ = ("MeasureSlitOffsetsConfig", "MeasureSlitOffsetsTask")


class MeasureSlitOffsetsConfig(Config):
    """Configuration for MeasureSlitOffsetsTask"""
    readLineList = ConfigurableField(target=ReadLineListTask, doc="Read line list")
    centroidLines = ConfigurableField(target=CentroidLinesTask, doc="Centroid lines")
    rejIterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejThreshold = Field(dtype=float, default=3.0, doc="Rejection threshold (sigma)")
    soften = Field(dtype=float, default=0.01, doc="Softening to apply to centroid errors (pixels)")


class MeasureSlitOffsetsTask(Task):
    """Measure consistent x,y offsets applicable to all fibers"""
    ConfigClass = MeasureSlitOffsetsConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("readLineList")
        self.makeSubtask("centroidLines")
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, exposure, detectorMap, pfsConfig):
        """Measure consistent x,y offsets applicable to all fibers

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure containing spectra.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength and x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Top-end configuration.

        Returns
        -------
        spatial, spectral : `ndarray.Array` of `float`
            Spatial and spectral shifts for each fiber.
        chi2 : `float`
            chi^2 for the fit.
        dof : `int`
            Number of degrees of freedom.
        num : `int`
            Number of centroid measurements used.
        select : `numpy.ndarray` of `bool`
            Boolean array indicating which centroids were used.
        soften : `float`
            Systematic error in centroid (pixels) required to produce
            chi^2/dof = 1.
        """
        before = detectorMap.clone()
        indices = pfsConfig.selectByFiberStatus(FiberStatus.GOOD, detectorMap.fiberId)
        fiberId = detectorMap.fiberId[indices]
        lines = self.readLineList.run(detectorMap=detectorMap, fiberId=fiberId,
                                      metadata=exposure.getMetadata())
        centroids = self.centroidLines.run(exposure, lines, detectorMap)

        result = self.measureSlitOffsets(detectorMap, centroids)
        self.log.info("Mean spatial=%f spectral=%f; chi2/dof=%.1f/%d soften=%.3f num=%d",
                      result.spatial.mean(), result.spectral.mean(),
                      result.chi2, result.dof, result.soften, result.num)

        if self.debugInfo.plot:
            self.plotSlitOffsets(detectorMap, centroids, result)
        if self.debugInfo.display:
            self.displaySlitOffsets(exposure, centroids, before, detectorMap, fiberId,
                                    set(centroids.wavelength))
        return result

    def measureSlitOffsets(self, detectorMap, centroids):
        """Measure the slit offsets

        We iteratively measure the slit offsets and reject outliers.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping from fiberId,wavelength to x,y.
        centroids : `pfs.drp.stella.ArcLineSet`
            Line centroids.

        Returns
        -------
        spatial, spectral : `ndarray.Array` of `float`
            Spatial and spectral shifts for each fiber.
        chi2 : `float`
            chi^2 for the fit.
        dof : `int`
            Number of degrees of freedom.
        num : `int`
            Number of centroid measurements used.
        select : `numpy.ndarray` of `bool`
            Boolean array indicating which centroids were used.
        soften : `float`
            Systematic error in centroid (pixels) required to produce
            chi^2/dof = 1.
        """
        origSpatial = detectorMap.getSpatialOffsets().copy()
        origSpectral = detectorMap.getSpectralOffsets().copy()
        select = ~centroids.flag
        select &= np.isfinite(centroids.x) & np.isfinite(centroids.y)
        select &= np.isfinite(centroids.xErr) & np.isfinite(centroids.yErr)

        fiberId = centroids.fiberId.astype(np.int32)
        wavelength = centroids.wavelength
        xx = centroids.x
        yy = centroids.y
        xErr = np.hypot(centroids.xErr, self.config.soften)
        yErr = np.hypot(centroids.yErr, self.config.soften)

        for ii in range(self.config.rejIterations):
            detectorMap.measureSlitOffsets(fiberId[select], wavelength[select], xx[select], yy[select],
                                           xErr[select], yErr[select])

            points = detectorMap.findPoint(fiberId[select], wavelength[select])
            dx = (centroids.x[select] - points[:, 0])/xErr[select]
            dy = (centroids.y[select] - points[:, 1])/yErr[select]
            chi2 = np.sum(dx**2 + dy**2)
            self.log.debug("Iteration %d: chi2=%f num=%d", ii, chi2, select.sum())

            keep = (np.abs(dx) < self.config.rejThreshold) & (np.abs(dy) < self.config.rejThreshold)
            if np.all(keep):
                break
            select[select] &= keep
        else:
            # Final iteration with no rejection
            detectorMap.measureSlitOffsets(fiberId[select], wavelength[select], xx[select], yy[select],
                                           xErr[select], yErr[select])

        points = detectorMap.findPoint(fiberId[select], wavelength[select])
        dx2 = (centroids.x[select] - points[:, 0])**2
        dy2 = (centroids.y[select] - points[:, 1])**2
        xErr2 = centroids.xErr[select]**2
        yErr2 = centroids.yErr[select]**2
        numGood = select.sum()
        chi2 = np.sum(dx2/xErr2 + dy2/yErr2)
        dof = 2*numGood - (2 if isinstance(detectorMap, GlobalDetectorMap) else 2*len(set(fiberId[select])))
        self.log.debug("Final iteration: chi2/dof=%f/%d num=%d", chi2, dof, numGood)

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

        if chi2 > dof:
            try:
                soften = scipy.optimize.bisect(softenChi2, 0.0, 1.0)
            except Exception:
                soften = 1.0
        else:
            soften = 0.0

        newSpatial = detectorMap.getSpatialOffsets()
        newSpectral = detectorMap.getSpectralOffsets()

        return Struct(spatial=newSpatial - origSpatial, spectral=newSpectral - origSpectral,
                      chi2=chi2, dof=dof, num=numGood, select=select, soften=soften)

    def plotSlitOffsets(self, detectorMap, centroids, result):
        """Plot the slit offset measurements

        The good points are plotted in black, and the bad points in red.

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength and x,y after slit offsets were
            applied.
        centroids : `pfs.drp.stella.ArcLineSet`
            List of line centroids.
        """
        good = result.select
        bad = ~good
        points = detectorMap.findPoint(centroids.fiberId.astype(np.int32),
                                       centroids.wavelength)
        dx = centroids.x - points[:, 0]
        dy = centroids.y - points[:, 1]

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, )
        for ax, ((x, xText), (y, yText)) in zip(
            sum(axes.tolist(), []),
            itertools.product(
                [(centroids.x, "x"), (centroids.y, "y")],
                [(dx/centroids.xErr, "dx"), (dy/centroids.yErr, "dy")]
            )
        ):
            ax.scatter(x[good], y[good], marker=".", color="k")
            ax.scatter(x[bad], y[bad], marker=".", color="r")
            ax.axhline(0.0, linestyle=":", color="k")
            ax.set_xlabel(xText)
            ax.set_ylabel(yText + " (sigma)")
        fig.suptitle(f"Offset: chi2={result.chi2:.1f} soften={result.soften:.3f} "
                     f"num={result.num}/{len(centroids)}")
        fig.tight_layout()
        plt.show()

    def displaySlitOffsets(self, exposure, centroids, beforeDetectorMap, afterDetectorMap,
                           fiberId, wavelength):
        """Display the exposure with the before and after detectorMaps

        The measured line positions are shown in red, the 'before' detectorMap
        is shown in yellow, and the 'after' detectorMap is shown in green.

        The following debug parameters are used:
        - ``frame`` (`int`): display frame to use (defaults to ``1``).
        - ``backend`` (`str`): display backend to use (defaults to ``"ds9"``).

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image to display.
        centroids : `pfs.drp.stella.ArcLineSet`
            List of line centroids.
        beforeDetectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength and x,y before slit offsets were
            applied.
        afterDetectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength and x,y after slit offsets were
            applied.
        fiberId : iterable of `int`
            Fiber identifiers to plot.
        wavelength : iterable of `float`
            Wavelengths to mark.
        """
        from lsst.afw.display import Display
        disp = Display(frame=self.debugInfo.frame or 1, backend=self.debugInfo.backend or "ds9")
        disp.mtv(exposure)
        with disp.Buffering():
            for line in centroids:
                disp.dot("x", line.x, line.y, size=5, ctype="red")
        beforeDetectorMap.display(disp, fiberId, wavelength, "yellow")
        afterDetectorMap.display(disp, fiberId, wavelength, "green")
