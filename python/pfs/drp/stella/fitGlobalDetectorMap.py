from collections import defaultdict

import numpy as np
import scipy.optimize

import lsstDebug
from lsst.pex.config import Config, Field
from lsst.pipe.base import Task, Struct

from .GlobalDetectorMapContinued import GlobalDetectorMap
from .GlobalDetectorMap import GlobalDetectorModel
from .arcLine import ArcLineSet

__all__ = ("FitGlobalDetectorMapConfig", "FitGlobalDetectorMapTask")


def robustRms(array):
    """Calculate a robust RMS of the array using the inter-quartile range

    Uses the standard conversion of IQR to RMS for a Gaussian.

    Parameters
    ----------
    array : `numpy.ndarray`
        Array for which to calculate RMS.

    Returns
    -------
    rms : `float`
        Robust RMS.
    """
    lq, uq = np.percentile(array, (25.0, 75.0))
    return 0.741*(uq - lq)


def rmsPixelsToVelocity(rms, model):
    """Convert an RMS in pixels to velocity in km/s

    Parameters
    ----------
    rms : `float`
        RMS in the spectral dimension (pixels).
    model : `pfs.drp.stella.GlobalDetectorModel`
        Model of the focal plane.

    Returns
    -------
    rms : `float`
        Velocity RMS (km/s).
    """
    return 3.0e5*rms*model.getDispersion()/model.getWavelengthCenter()


def calculateFitStatistics(model, lines, selection, soften=0.0):
    """Calculate statistics of the model fit

    Parameters
    ----------
    model : `pfs.drp.stella.GlobalDetectorModel`
        Model that was fit to the data.
    lines : `pfs.drp.stella.ArcLineSet`
        Arc line measurements.
    selection : `numpy.ndarray` of `bool`
        Flags indicating which of the ``lines`` are to be used in the
        calculation.
    soften : `float`, optional
        Systematic error to add in quadrature to measured errors (pixels).

    Returns
    -------
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
    """
    fit = model(lines.fiberId.astype(np.int32), lines.wavelength)
    xResid = lines.x - fit[0]
    yResid = lines.y - fit[1]

    xRms = robustRms(xResid[selection])
    yRms = robustRms(yResid[selection])
    chi2 = np.sum((xResid[selection]**2/(lines.xErr[selection]**2 + soften**2)) +
                  (yResid[selection]**2/(lines.yErr[selection]**2 + soften**2)))
    return Struct(model=model, xResid=xResid, yResid=yResid, xRms=xRms, yRms=yRms,
                  chi2=chi2, soften=soften)


class FitGlobalDetectorMapConfig(Config):
    """Configuration for FitGlobablDetectorMapTask"""
    iterations = Field(dtype=int, default=3, doc="Number of rejection iterations")
    rejection = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    order = Field(dtype=int, default=7, doc="Distortion order")
    reserveFraction = Field(dtype=float, default=0.1, doc="Fraction of lines to reserve in the final fit")


class FitGlobalDetectorMapTask(Task):
    """Fit a GlobalDetectorMap to arc line measurements"""
    ConfigClass = FitGlobalDetectorMapConfig

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.debugInfo = lsstDebug.Info(__name__)

    def run(self, arm, bbox, lines, visitInfo, metadata=None):
        """Fit a GlobalDetectorMap to arc line measurements

        Parameters
        ----------
        arm : `str`
            Spectrograph arm (``b``, ``r``, ``n``, ``m``).
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        visitInfo : `lsst.afw.image.VisitInfo`
            Visit information for exposure.
        metadata : `lsst.daf.base.PropertyList`, optional
            DetectorMap metadata (FITS header).

        Returns
        -------
        detectorMap : `pfs.drp.stella.GlobalDetectorMap`
            Mapping of fiberId,wavelength to x,y.
        """
        lines = ArcLineSet([ll for ll in lines if not ll.flag])  # Only good measurements
        numLines = len(lines)

        good = np.ones(numLines, dtype=bool)
        rng = np.random.RandomState(visitInfo.getExposureId())
        numReserved = int(self.config.reserveFraction*numLines + 0.5)
        reservedIndices = rng.choice(np.arange(numLines, dtype=int), replace=False, size=numReserved)
        reserved = np.zeros_like(good)
        reserved[reservedIndices] = True

        result = None
        for ii in range(self.config.iterations):
            select = good & ~reserved
            result = self.fitModel(arm, bbox, lines, select, result.model if result is not None else None)
            self.log.debug("Fit iteration %d: chi2=%f xRMS=%f yRMS=%f (%f nm, %f km/s) from %d/%d lines",
                           ii, result.chi2, result.xRms, result.yRms,
                           result.yRms*result.model.getDispersion(),
                           rmsPixelsToVelocity(result.yRms, result.model), select.sum(),
                           numLines - numReserved)
            self.log.debug("Fit iteration %d: %s", ii, result.model)
            if self.debugInfo.plot:
                self.plotModel(lines, good, result)
            newGood = ((np.abs(result.xResid/lines.xErr) < self.config.rejection) &
                       (np.abs(result.yResid/lines.yErr) < self.config.rejection))
            if np.all(newGood == good):
                # Converged
                break
            good = newGood

        select = good & ~reserved
        result = self.fitModel(arm, bbox, lines, select, result.model if result is not None else None)
        self.log.info("Final fit: chi2=%f xRMS=%f yRMS=%f (%f nm, %f km/s) from %d/%d lines",
                      result.chi2, result.xRms, result.yRms, result.yRms*result.model.getDispersion(),
                      rmsPixelsToVelocity(result.yRms, result.model), select.sum(), numLines - numReserved)
        reservedStats = calculateFitStatistics(result.model, lines, reserved)
        self.log.info("Fit quality from reserved lines: "
                      "chi2=%f xRMS=%f yRMS=%f (%f nm, %f km/s) from %d lines (%.1f%%)",
                      reservedStats.chi2, reservedStats.xRms, reservedStats.yRms,
                      reservedStats.yRms*result.model.getDispersion(),
                      rmsPixelsToVelocity(reservedStats.yRms, result.model), reserved.sum(),
                      reserved.sum()/numLines*100)
        self.log.debug("    Final fit model: %s", result.model)

        result = self.fitSoftenedModel(arm, bbox, lines, select, reserved, result)
        if self.debugInfo.plot:
            self.plotModel(lines, good, result)

        detMap = GlobalDetectorMap(result.model, visitInfo, metadata)

        if self.debugInfo.lineQa:
            self.lineQa(lines, detMap)

        return detMap

    def fitModel(self, arm, bbox, lines, select, model=None, soften=0.0):
        """Fit a model to the arc lines

        Parameters
        ----------
        arm : `str`
            Spectrograph arm (``b``, ``r``, ``n``, ``m``).
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        select : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        model : `pfs.drp.stella.GlobalDetectorModel`, optional
            Model from previous fit iteration (provides a starting point for
            this iteration).
        soften : `float`
            Systematic error to add in quadrature to measured errors (pixels).

        Returns
        -------
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
        """
        dualDetector = (arm != "n")
        parameters = None
        if model is not None:
            parameters = model.getParameters().copy()

        fiberId = lines.fiberId.astype(np.int32)
        xErr = lines.xErr
        yErr = lines.yErr
        if soften > 0:
            xErr = np.hypot(soften, xErr)
            yErr = np.hypot(soften, yErr)

        try:
            model = GlobalDetectorModel.fit(
                bbox, self.config.order, dualDetector, fiberId, lines.wavelength,
                lines.x, lines.y, xErr, yErr, select, parameters
            )
        except RuntimeError:
            # Parameters from previous iteration may be corrupting this fit; start from scratch
            model = GlobalDetectorModel.fit(
                bbox, self.config.order, dualDetector, fiberId, lines.wavelength,
                lines.x, lines.y, xErr, yErr, select
            )

        return calculateFitStatistics(model, lines, select, soften)

    def fitSoftenedModel(self, arm, bbox, lines, select, reserved, result):
        """Fit with errors adjusted so that chi^2/dof <= 1

        This provides a measure of the systematic error.

        Parameters
        ----------
        arm : `str`
            Spectrograph arm (``b``, ``r``, ``n``, ``m``).
        bbox : `lsst.geom.Box2I`
            Bounding box for detector.
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        select : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        reserved : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` have been reserved for QA.
        result : `lsst.pipe.base.Struct`
            Results from ``fitModel``.

        Returns
        -------
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
        """
        dof = 2*select.sum()  # column and row for each point; ignoring the relatively small number of params
        if result.chi2/dof <= 1:
            self.log.info("No softening necessary")
            return result

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
            colChi2 = np.sum(result.xResid[select]**2/(soften**2 + lines.xErr[select]**2))
            rowChi2 = np.sum(result.yResid[select]**2/(soften**2 + lines.yErr[select]**2))
            return (colChi2 + rowChi2)/dof - 1

        soften = scipy.optimize.bisect(softenChi2, 0.0, 1.0)
        self.log.info("Softening errors by %f pixels (%f nm, %f km/s) to yield chi^2/dof=1", soften,
                      soften*result.model.getDispersion(),
                      rmsPixelsToVelocity(soften, result.model))

        result = self.fitModel(arm, bbox, lines, select, result.model, soften)
        self.log.info("Softened fit: chi2=%f xRMS=%f yRMS=%f (%f nm, %f km/s) from %d/%d lines",
                      result.chi2, result.xRms, result.yRms, result.yRms*result.model.getDispersion(),
                      rmsPixelsToVelocity(result.yRms, result.model), select.sum(), len(select))

        reservedStats = calculateFitStatistics(result.model, lines, reserved, soften)
        self.log.info("Softened fit quality from reserved lines: "
                      "chi2=%f xRMS=%f yRMS=%f (%f nm, %f km/s) from %d lines (%.1f%%)",
                      reservedStats.chi2, reservedStats.xRms, reservedStats.yRms,
                      reservedStats.yRms*result.model.getDispersion(),
                      rmsPixelsToVelocity(reservedStats.yRms, result.model), reserved.sum(),
                      reserved.sum()/len(reserved)*100)
        self.log.debug("    Softened fit model: %s", result.model)
        return result

    def lineQa(self, lines, detectorMap):
        """Check the quality of the model fit by looking at the lines

        We match the same lines from different fibers and compare the RMS with
        the measurement errors. The results are logged and plotted.

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        detectorMap : `pfs.drp.stella.BaseDetectorMap`
            Mapping of fiberId,wavelength to x,y.
        """
        matches = defaultdict(list)
        num = len(lines.fiberId)
        for ii in range(num):
            matches[lines.wavelength[ii]].append(ii)
        model = detectorMap.model
        stdev = []
        error = []
        for wl in sorted(matches.keys()):
            if len(matches[wl]) < 10:
                continue
            indices = np.array(matches[wl])
            fit = np.array([detectorMap.findWavelength(lines.fiberId[ii], lines.y[ii]) for ii in indices])
            resid = (fit - wl)/model.getDispersion()
            self.log.info("Line %f: rms residual=%f, mean error=%f num=%d",
                          wl, robustRms(resid), np.median(lines.yErr[indices]), len(indices))
            stdev.append(robustRms(resid))
            error.append(np.median(lines.yErr[indices]))

        import matplotlib.pyplot as plt
        plt.plot(error, stdev, 'ko')
        plt.xlabel("Median centroid error")
        plt.ylabel("Fit RMS")
        plt.show()

    def plotModel(self, lines, good, result):
        """Plot the model fit result

        Parameters
        ----------
        lines : `pfs.drp.stella.ArcLineSet`
            Arc line measurements.
        good : `numpy.ndarray` of `bool`
            Flags indicating which of the ``lines`` are to be fit.
        result : `lsst.pipe.base.Struct`
            Results from ``fitModel``.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from matplotlib.colors import Normalize
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        xErr = lines.xErr
        yErr = lines.yErr
        if result.soften > 0:
            xErr = np.hypot(result.soften, xErr)
            yErr = np.hypot(result.soften, yErr)

        xResid = result.xResid/xErr
        yResid = result.yResid/yErr

        fig, axes = plt.subplots(2, 3)
        axes[0, 0].scatter(lines.x[good], xResid[good], marker=".", color="b")
        axes[0, 0].scatter(lines.x[~good], xResid[~good], marker=".", color="r")
        axes[0, 0].set_xlabel("Spatial")
        axes[0, 0].set_ylabel(r"\Delta Spatial (\sigma)")
        axes[0, 0].set_title("Spatial residuals")

        axes[1, 0].scatter(lines.y[good], yResid[good], marker=".", color="b")
        axes[1, 0].scatter(lines.y[~good], yResid[~good], marker=".", color="r")
        axes[1, 0].set_xlabel("Spectral")
        axes[1, 0].set_ylabel(r"\Delta Spectral (\sigma)")
        axes[1, 0].set_title("Spectral residuals")

        numFibers = result.model.getNumFibers()
        numLines = min(10, numFibers)
        fiberId = result.model.getFiberId()[np.linspace(0, numFibers, numLines, False, dtype=int)]
        wavelength = np.linspace(lines.wavelength.min(), lines.wavelength.max(), numLines)
        ff, wl = np.meshgrid(fiberId, wavelength)
        ff = ff.flatten()
        wl = wl.flatten()
        xx, yy = result.model(ff, wl)

        for ii in range(numLines):
            select = ff == fiberId[ii]
            axes[0, 1].plot(xx[select], yy[select], 'k-')
            axes[1, 1].plot(xx[select], yy[select], 'k-')
            select = wl == wavelength[ii]
            axes[0, 1].plot(xx[select], yy[select], 'k-')
            axes[1, 1].plot(xx[select], yy[select], 'k-')
        axes[0, 1].scatter(lines.x[good], lines.y[good], marker=".", color="b")
        axes[0, 1].set_title("Good")
        axes[0, 1].set_xlabel("Spatial")
        axes[0, 1].set_ylabel("Spectral")
        axes[1, 1].scatter(lines.x[~good], lines.y[~good], marker=".", color="r")
        axes[1, 1].set_title("Rejected")
        axes[1, 1].set_xlabel("Spatial")
        axes[1, 1].set_ylabel("Spectral")

        cmap = matplotlib.cm.bwr

        norm = Normalize()
        norm.autoscale(xResid[good])
        ax = axes[0, 2]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax.scatter(lines.x[good], lines.y[good], marker=".", color=cmap(norm(xResid[good])))
        colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colors.set_array([])
        fig.colorbar(colors, cax=cax, orientation="vertical", label=r"\Delta Spatial (\sigma)")
        ax.set_title("Spatial offset")
        ax.set_xlabel("Spatial")
        ax.set_ylabel("Spectral")

        norm = Normalize()
        norm.autoscale(yResid[good])
        ax = axes[1, 2]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        ax.scatter(lines.x[good], lines.y[good], marker=".", color=cmap(norm(yResid[good])))
        colors = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colors.set_array([])
        fig.colorbar(colors, cax=cax, orientation="vertical", label=r"\Delta Spectral (\sigma)")
        ax.set_title("Spectral offset")
        ax.set_xlabel("Spatial")
        ax.set_ylabel("Spectral")

        fig.tight_layout()
        plt.show()
