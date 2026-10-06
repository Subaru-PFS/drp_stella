from typing import Optional

import numpy as np

from lsst.afw.image import Exposure
from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Struct, Task

from .arcLine import ArcLineSet
from .DetectorMapContinued import DetectorMap
from .fitPsfModel import fitHybridPsf
from .hybridPsfModel import HybridPsfModel
from .pfsPsf import PfsPsf
from .psfProfiles import ParametricPsfModel, PsfParams, WingParams
from .psfSpline import PsfSplineBasis, RegularizationConfig, SplineAxisConfig
from .referenceLine import ReferenceLineStatus

__all__ = ["FitPsfConfig", "FitPsfTask"]


class FitPsfConfig(Config):
    """Configuration for `FitPsfTask`"""

    order = Field(dtype=int, default=2, doc="Polynomial order (in fiberIndex, row) of the core parameters")
    wingOrder = Field(
        dtype=int,
        default=None,
        optional=True,
        doc="Polynomial order of the wing parameters; defaults to `order` if not set",
    )
    numWings = Field(dtype=int, default=1, doc="Number of Moffat wing components")
    halfSize = Field(dtype=int, default=12, doc="Half-size of the PSF stamp to fit and evaluate (pixels)")
    oversampling = Field(dtype=int, default=4, doc="Oversampling factor for exact pixel integration")

    splineExtent = Field(dtype=float, default=6.0, doc="Half-extent of the dP spline domain (pixels)")
    splineFineSpacing = Field(dtype=float, default=0.25, doc="Fine dP knot spacing near the center (pixels)")
    splineFineRadius = Field(dtype=float, default=2.0, doc="Radius of the fine dP knot region (pixels)")
    splineMediumSpacing = Field(dtype=float, default=0.5, doc="Medium dP knot spacing (pixels)")
    splineMediumRadius = Field(dtype=float, default=4.0, doc="Radius of the medium dP knot region (pixels)")
    splineCoarseSpacing = Field(dtype=float, default=1.5, doc="Coarse dP knot spacing near the edge (pixels)")

    regularizationSmoothness = Field(
        dtype=float, default=1.0, doc="Strength of the dP second-derivative smoothness penalty"
    )
    regularizationRidge = Field(
        dtype=float, default=1.0, doc="Strength of the dP shrink-to-zero ridge penalty"
    )
    regularizationRadiusScale = Field(
        dtype=float, default=8.0, doc="Radius (pixels) over which the dP regularization weight grows by e"
    )

    maxOuterIter = Field(dtype=int, default=6, doc="Maximum number of outer (alternating) fit iterations")
    thetaTol = Field(
        dtype=float,
        default=1.0e-6,
        doc="Convergence tolerance on the relative change in the parametric parameter vector",
    )
    fitCenters = Field(dtype=bool, default=True, doc="Update line centers each outer iteration?")
    centerStepSize = Field(
        dtype=float, default=0.01, doc="Finite-difference step size (pixels) for the center Jacobian"
    )
    maxCenterShift = Field(
        dtype=float, default=1.0, doc="Maximum allowed line center shift (pixels) per outer iteration"
    )

    gain = Field(dtype=float, default=1.0, doc="Detector gain (electrons/ADU)")
    readnoise = Field(dtype=float, default=0.0, doc="Detector read noise (ADU)")

    excludeStatus = ListField(
        dtype=str,
        default=["BAD"],
        doc="Reference line status flags indicating that a line should be excluded from the fit",
    )

    initialSigmaX = Field(dtype=float, default=1.0, doc="Initial guess for the core sigma (x, pixels)")
    initialSigmaY = Field(dtype=float, default=1.0, doc="Initial guess for the core sigma (y, pixels)")
    initialTophatWidth = Field(
        dtype=float, default=1.0, doc="Initial guess for the fiber top-hat width (pixels)"
    )
    initialWingScaleX = Field(
        dtype=float, default=4.0, doc="Initial guess for the wing scale length (x, pixels)"
    )
    initialWingScaleY = Field(
        dtype=float, default=4.0, doc="Initial guess for the wing scale length (y, pixels)"
    )
    initialWingBeta = Field(dtype=float, default=2.5, doc="Initial guess for the wing Moffat exponent")
    initialWingFraction = Field(dtype=float, default=0.05, doc="Initial guess for the wing flux fraction")

    def getSplineAxisConfig(self) -> SplineAxisConfig:
        """Build the (identical, for x and y) `SplineAxisConfig` from this config"""
        return SplineAxisConfig(
            extent=self.splineExtent,
            fineSpacing=self.splineFineSpacing,
            fineRadius=self.splineFineRadius,
            mediumSpacing=self.splineMediumSpacing,
            mediumRadius=self.splineMediumRadius,
            coarseSpacing=self.splineCoarseSpacing,
        )

    def getRegularizationConfig(self) -> RegularizationConfig:
        """Build the `RegularizationConfig` from this config"""
        return RegularizationConfig(
            smoothness=self.regularizationSmoothness,
            ridge=self.regularizationRidge,
            radiusScale=self.regularizationRadiusScale,
        )


class FitPsfTask(Task):
    """Fit a `PfsPsf` to a set of measured arc line positions on an exposure

    Implements the hybrid parametric + regularized-spline PSF model of
    PIPE2D-1823-psf.md: the PSF is decomposed as

    .. math::

        P(\\Delta x, \\Delta y; \\mathrm{fiberIndex}, \\mathrm{row}) =
            P_\\mathrm{param}(\\Delta x, \\Delta y; \\theta(\\mathrm{fiberIndex}, \\mathrm{row}))
            + \\delta P(\\Delta x, \\Delta y)

    where :math:`P_\\mathrm{param}` is a smoothly-varying parametric
    backbone (a fiber top-hat convolved with a Gaussian core, plus one or
    more Moffat wing components; see `pfs.drp.stella.psfProfiles`) whose
    parameters :math:`\\theta` vary as low-order polynomials of fiber index
    and detector row, and :math:`\\delta P` is a single, position-independent
    regularized cubic B-spline correction (see `pfs.drp.stella.psfSpline`)
    capturing whatever residual structure (e.g. an undersampled core, or
    non-Gaussian/non-Moffat features) the parametric form cannot represent.
    :math:`\\delta P` is constrained to have zero integral and zero first
    moments, so that it redistributes flux without competing with
    :math:`P_\\mathrm{param}` for the PSF's total flux or centroid.

    The two components are fit jointly to the measured line stamps by
    `pfs.drp.stella.fitPsfModel.fitHybridPsf`, alternating (per outer
    iteration) between: (1) a linear solve for each line's amplitude and a
    common background; (2) an optional local Gauss-Newton update of each
    line's center; (3) a nonlinear least-squares update of the parametric
    parameter vector :math:`\\theta`; and (4) a regularized, constrained
    linear solve for the :math:`\\delta P` spline coefficients. See that
    function's docstring for the full algorithm.

    Parameters
    ----------
    *args, **kwargs
        Passed to `lsst.pipe.base.Task`.
    """

    ConfigClass = FitPsfConfig
    _DefaultName = "fitPsf"

    def run(self, exposure: Exposure, detectorMap: DetectorMap, arcLines: ArcLineSet) -> Struct:
        """Fit a `PfsPsf` to arc lines measured on an exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure containing the arc line images.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping between fiberId,wavelength and detector position.
        arcLines : `pfs.drp.stella.ArcLineSet`
            Measured arc line positions. Lines with a status flag in
            ``self.config.excludeStatus`` are excluded from the fit.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Contains ``psf`` (the fitted `pfs.drp.stella.PfsPsf`) and
            ``fit`` (the `lsst.pipe.base.Struct` returned by
            `pfs.drp.stella.fitPsfModel.fitHybridPsf`, with elements
            ``model``, ``amplitudes``, ``xCenter``, ``yCenter``,
            ``numIter``).
        """
        badStatus = ReferenceLineStatus.fromNames(*self.config.excludeStatus)
        select = (arcLines.status & badStatus) == 0
        lines = arcLines[select]

        fiberIndex = np.searchsorted(detectorMap.fiberId, lines.fiberId).astype(float)
        row = np.array(lines.y, dtype=float)
        xCenter = np.array(lines.x, dtype=float)
        yCenter = row

        hybridModel = self._makeInitialModel(detectorMap)

        fit = fitHybridPsf(
            exposure.image,
            fiberIndex,
            row,
            xCenter,
            yCenter,
            hybridModel,
            self.config.halfSize,
            regularizationConfig=self.config.getRegularizationConfig(),
            oversampling=self.config.oversampling,
            gain=self.config.gain,
            readnoise=self.config.readnoise,
            maxOuterIter=self.config.maxOuterIter,
            thetaTol=self.config.thetaTol,
            fitCenters=self.config.fitCenters,
            centerStepSize=self.config.centerStepSize,
            maxCenterShift=self.config.maxCenterShift,
        )

        psf = PfsPsf(fit.model, detectorMap, self.config.halfSize, self.config.oversampling)
        return Struct(psf=psf, fit=fit)

    def _makeInitialModel(self, detectorMap: DetectorMap) -> HybridPsfModel:
        """Construct a `HybridPsfModel` with an initial guess, from the config

        Parameters
        ----------
        detectorMap : `pfs.drp.stella.DetectorMap`
            Used to set the parametric model's ``(fiberIndex, row)``
            polynomial domains.

        Returns
        -------
        hybridModel : `pfs.drp.stella.hybridPsfModel.HybridPsfModel`
            Model with an initial guess set on its ``parametricModel``, and
            zero ``dP`` coefficients.
        """
        numFibers = len(detectorMap.fiberId)
        bbox = detectorMap.bbox
        fiberDomain = (0.0, float(max(numFibers - 1, 1)))
        rowDomain = (float(bbox.minY), float(bbox.maxY))

        parametricModel = ParametricPsfModel(
            self.config.order,
            self.config.numWings,
            fiberDomain,
            rowDomain,
            wingOrder=self.config.wingOrder,
        )
        wings = [
            WingParams(
                scaleX=self.config.initialWingScaleX,
                scaleY=self.config.initialWingScaleY,
                beta=self.config.initialWingBeta,
                fraction=self.config.initialWingFraction,
            )
            for _ in range(self.config.numWings)
        ]
        parametricModel.setInitialGuess(
            self.config.initialSigmaX, self.config.initialSigmaY, self.config.initialTophatWidth, wings
        )

        axisConfig = self.config.getSplineAxisConfig()
        splineBasis = PsfSplineBasis(axisConfig, axisConfig)

        return HybridPsfModel(parametricModel, splineBasis)

    def plotDiagnostics(self, fit: Struct, arcLines: ArcLineSet, ax=None, show: bool = True):
        """Plot a basic per-line residual diagnostic from a fit result

        Parameters
        ----------
        fit : `lsst.pipe.base.Struct`
            The ``fit`` element of `run`'s result (or, equivalently, the
            direct return value of
            `pfs.drp.stella.fitPsfModel.fitHybridPsf`).
        arcLines : `pfs.drp.stella.ArcLineSet`
            The (unfiltered) arc lines passed to `run`; used only for its
            length, as a sanity check against ``fit``.
        ax : `matplotlib.axes.Axes`, optional
            Axes on which to plot; a new figure is created if not given.
        show : `bool`, optional
            Call ``matplotlib.pyplot.show()``?

        Returns
        -------
        figure : `matplotlib.figure.Figure`
            The figure containing the plot.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure
        amplitudes = fit.amplitudes[: len(fit.xCenter)]
        ax.plot(fit.yCenter, amplitudes, ".", alpha=0.5)
        ax.set_xlabel("row (pixels)")
        ax.set_ylabel("fitted amplitude")
        ax.set_title(f"FitPsfTask diagnostics: {fit.numIter} outer iterations, {len(fit.xCenter)} lines")
        figure.tight_layout()
        if show:
            plt.show()
        return figure
