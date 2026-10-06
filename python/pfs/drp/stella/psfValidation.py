import logging
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.stats import kstest

from lsst.pipe.base import Struct

from .fitPsfModel import (
    buildParametricDesignMatrix,
    buildSplineDesignMatrix,
    buildStampGeometry,
    fitHybridPsf,
    fitParametricPsf,
    solveAmplitudesAndBackground,
    solveSplineCoefficients,
)
from .hybridPsfModel import HybridPsfModel
from .psfPixelIntegration import integrateOverPixels
from .psfSpline import RegularizationConfig

__all__ = [
    "computeHeldOutChi2",
    "crossValidateByFiber",
    "compareModelVariants",
    "subPixelPhaseHistogram",
    "stackedResidualMap",
    "radialProfileWithBootstrap",
    "energyConservationCheck",
    "dpCoherenceCheck",
]

_LOG = logging.getLogger(__name__)


def computeHeldOutChi2(
    image,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    heldOutMask: np.ndarray,
    model,
    trainedAmplitudes: np.ndarray,
    halfSize: int,
    oversampling: int = 4,
    gain: float = 1.0,
    readnoise: float = 0.0,
) -> Struct:
    """Evaluate a fixed, already-fitted PSF model against held-out lines

    Solves only the held-out lines' amplitudes, holding the PSF shape
    (``model``) *and* the training lines' own amplitudes/background
    (``trainedAmplitudes``, as fit on the training lines alone) fixed, then
    reports the chi2 of the held-out lines' own stamp pixels. This is the
    core operation behind PIPE2D-1823-psf.md's cross-validation and
    model-variant-comparison diagnostics: the model's shape was fit on a
    disjoint set of lines, and this checks how well that shape predicts a
    different set.

    The stamp geometry is built from *every* line, not just the held-out
    ones, and the training lines' contribution is subtracted using their
    already-fitted (fixed) amplitudes rather than ignored: at the ~6.5 px
    fiber pitch, a held-out edge/gap fiber's stamp can still reach into its
    nearest-neighbor (training) fiber's own line, so solving for held-out
    amplitudes from the held-out lines' pixels alone would misattribute the
    neighbor's real flux as a spurious residual.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image containing the line stamps.
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and center position of *every* line
        (training and held-out together, matching ``heldOutMask`` and
        ``trainedAmplitudes``).
    heldOutMask : `numpy.ndarray`, `bool`
        True for lines to hold out and solve for here; False for the
        (already-fitted) training lines.
    model : `pfs.drp.stella.psfProfiles.ParametricPsfModel` or \
            `pfs.drp.stella.hybridPsfModel.HybridPsfModel`
        Already-fitted PSF model (shape only; not modified here).
    trainedAmplitudes : `numpy.ndarray`
        The training fit's amplitude vector (as returned by
        `~pfs.drp.stella.fitPsfModel.fitHybridPsf`/`fitParametricPsf`): one
        entry per training line, in the same relative order as
        ``fiberIndex[~heldOutMask]``, optionally followed by a background
        level.
    halfSize : `int`
        Half-size of each line's stamp, in pixels.
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    gain : `float`
        Detector gain (electrons/ADU).
    readnoise : `float`
        Detector read noise (ADU).

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``chi2`` (total chi2 of the held-out lines' own pixels),
        ``dof`` (those pixels minus the held-out amplitudes solved for),
        ``reducedChi2`` (``chi2/dof``, or ``chi2`` if ``dof <= 0``), and
        ``heldOutAmplitudes``.
    """
    heldOutMask = np.asarray(heldOutMask, dtype=bool)
    trainMask = ~heldOutMask
    numLines = len(fiberIndex)
    numTrain = int(trainMask.sum())

    geometry = buildStampGeometry(image, fiberIndex, row, xCenter, yCenter, halfSize, gain, readnoise)
    fullDesign = buildParametricDesignMatrix(
        model, fiberIndex, row, xCenter, yCenter, geometry, oversampling, includeBackground=False
    )
    backgroundLevel = float(trainedAmplitudes[numTrain]) if len(trainedAmplitudes) > numTrain else 0.0
    fixedAmplitudes = np.zeros(numLines)
    fixedAmplitudes[trainMask] = trainedAmplitudes[:numTrain]
    fixedContribution = fullDesign[:, trainMask] @ fixedAmplitudes[trainMask] + backgroundLevel

    heldOutDesign = fullDesign[:, heldOutMask]
    residualForHeldOut = geometry.dataVector - fixedContribution
    weights = 1.0 / np.sqrt(geometry.varianceVector)
    weightedDesign = scipy.sparse.diags(weights) @ heldOutDesign
    weightedResidual = weights * residualForHeldOut
    heldOutAmplitudes = scipy.sparse.linalg.lsqr(weightedDesign, weightedResidual)[0]

    predicted = fixedContribution + heldOutDesign @ heldOutAmplitudes
    variance = readnoise**2 + np.clip(predicted, 0.0, None) / gain
    residual = (geometry.dataVector - predicted) / np.sqrt(variance)

    isHeldOutPixel = np.zeros(geometry.numUsedPixels, dtype=bool)
    for ii in np.nonzero(heldOutMask)[0]:
        subIndex = geometry.pixelIndex[geometry.rowSlices[ii], geometry.colSlices[ii]]
        isHeldOutPixel[subIndex.ravel()] = True

    chi2 = float(np.sum(residual[isHeldOutPixel] ** 2))
    numHeldOutPixels = int(isHeldOutPixel.sum())
    dof = numHeldOutPixels - heldOutDesign.shape[1]
    reducedChi2 = chi2 / dof if dof > 0 else chi2
    return Struct(
        chi2=chi2,
        numUsedPixels=numHeldOutPixels,
        dof=dof,
        reducedChi2=reducedChi2,
        heldOutAmplitudes=heldOutAmplitudes,
    )


def crossValidateByFiber(
    image,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    heldOutMask: np.ndarray,
    hybridModelFactory: Callable[[], HybridPsfModel],
    halfSize: int,
    regularizationConfig: Optional[RegularizationConfig] = None,
    oversampling: int = 4,
    gain: float = 1.0,
    readnoise: float = 0.0,
    maxOuterIter: int = 6,
    thetaTol: float = 1.0e-6,
    fitCenters: bool = True,
) -> Struct:
    """Fit on a subset of fibers and check held-out prediction on the rest

    Implements PIPE2D-1823-psf.md's fiber cross-validation: fit the full
    hybrid model on the lines with ``heldOutMask`` false, then use
    `computeHeldOutChi2` to check how well that fitted shape predicts the
    lines with ``heldOutMask`` true. Call this twice -- once with the
    edge/gap-adjacent fibers held out, once with only those fibers used for
    training -- to cover both directions of the spec's cross-validation
    requirement.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image containing the line stamps.
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and initial center position of each line
        (training and held-out together).
    heldOutMask : `numpy.ndarray`, `bool`
        True for lines to hold out of the fit and use only for validation.
    hybridModelFactory : callable
        Zero-argument callable returning a fresh `HybridPsfModel` with an
        initial guess already set on its ``parametricModel``.
    halfSize : `int`
        Half-size of each line's stamp, in pixels.
    regularizationConfig : `pfs.drp.stella.psfSpline.RegularizationConfig`, optional
        Regularization strength recipe for the ``dP`` coefficient solve.
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    gain : `float`
        Detector gain (electrons/ADU).
    readnoise : `float`
        Detector read noise (ADU).
    maxOuterIter : `int`
        Maximum number of outer (alternating) fit iterations.
    thetaTol : `float`
        Convergence tolerance on the relative change in the parametric
        parameter vector.
    fitCenters : `bool`
        Update line centers each outer iteration during training?

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``trainFit`` (the `~pfs.drp.stella.fitPsfModel.fitHybridPsf`
        result on the training lines), ``heldOut`` (the
        `computeHeldOutChi2` result on the held-out lines), ``numTrain``,
        and ``numHeldOut``.
    """
    heldOutMask = np.asarray(heldOutMask, dtype=bool)
    trainMask = ~heldOutMask
    trainedModel = hybridModelFactory()
    trainFit = fitHybridPsf(
        image,
        fiberIndex[trainMask],
        row[trainMask],
        xCenter[trainMask],
        yCenter[trainMask],
        trainedModel,
        halfSize,
        regularizationConfig=regularizationConfig,
        oversampling=oversampling,
        gain=gain,
        readnoise=readnoise,
        maxOuterIter=maxOuterIter,
        thetaTol=thetaTol,
        fitCenters=fitCenters,
    )
    combinedXCenter = np.array(xCenter, dtype=float)
    combinedYCenter = np.array(yCenter, dtype=float)
    combinedXCenter[trainMask] = trainFit.xCenter
    combinedYCenter[trainMask] = trainFit.yCenter
    heldOut = computeHeldOutChi2(
        image,
        fiberIndex,
        row,
        combinedXCenter,
        combinedYCenter,
        heldOutMask,
        trainFit.model,
        trainFit.amplitudes,
        halfSize,
        oversampling,
        gain,
        readnoise,
    )
    return Struct(
        trainFit=trainFit, heldOut=heldOut, numTrain=int(trainMask.sum()), numHeldOut=int(heldOutMask.sum())
    )


def compareModelVariants(
    image,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    heldOutMask: np.ndarray,
    hybridModelFactory: Callable[[], HybridPsfModel],
    halfSize: int,
    regularizationConfig: Optional[RegularizationConfig] = None,
    oversampling: int = 4,
    gain: float = 1.0,
    readnoise: float = 0.0,
    maxOuterIter: int = 6,
    thetaTol: float = 1.0e-6,
    fitCenters: bool = True,
) -> Struct:
    """Compare P_param-only and hybrid models by held-out chi2

    Implements PIPE2D-1823-psf.md's "compare model variants by held-out
    chi2" diagnostic for variants (i) P_param only and (ii) hybrid; the
    optional third "purely regularized spline" variant is not implemented
    (deferred, per the spec's own "optionally").

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image containing the line stamps.
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and initial center position of each line
        (training and held-out together).
    heldOutMask : `numpy.ndarray`, `bool`
        True for lines to hold out of the fit and use only for validation.
    hybridModelFactory : callable
        Zero-argument callable returning a fresh `HybridPsfModel` with an
        initial guess already set on its ``parametricModel``; a second call
        to it provides the (fresh, independent) initial guess for the
        P_param-only variant too (its ``splineBasis`` is simply unused).
    halfSize : `int`
        Half-size of each line's stamp, in pixels.
    regularizationConfig : `pfs.drp.stella.psfSpline.RegularizationConfig`, optional
        Regularization strength recipe for the hybrid variant's ``dP``
        coefficient solve.
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    gain : `float`
        Detector gain (electrons/ADU).
    readnoise : `float`
        Detector read noise (ADU).
    maxOuterIter : `int`
        Maximum number of outer fit iterations, for both variants.
    thetaTol : `float`
        Convergence tolerance on the relative change in the parametric
        parameter vector.
    fitCenters : `bool`
        Update line centers each outer iteration during training?

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``paramOnly`` and ``hybrid``, each a `lsst.pipe.base.Struct`
        with ``fit`` (the variant's fit result) and ``heldOut`` (its
        `computeHeldOutChi2` result).
    """
    heldOutMask = np.asarray(heldOutMask, dtype=bool)
    trainMask = ~heldOutMask

    paramModel = hybridModelFactory().parametricModel
    paramFit = fitParametricPsf(
        image,
        fiberIndex[trainMask],
        row[trainMask],
        xCenter[trainMask],
        yCenter[trainMask],
        paramModel,
        halfSize,
        oversampling=oversampling,
        gain=gain,
        readnoise=readnoise,
        maxOuterIter=maxOuterIter,
        thetaTol=thetaTol,
    )
    paramHeldOut = computeHeldOutChi2(
        image,
        fiberIndex,
        row,
        xCenter,
        yCenter,
        heldOutMask,
        paramFit.model,
        paramFit.amplitudes,
        halfSize,
        oversampling,
        gain,
        readnoise,
    )

    hybridFit = fitHybridPsf(
        image,
        fiberIndex[trainMask],
        row[trainMask],
        xCenter[trainMask],
        yCenter[trainMask],
        hybridModelFactory(),
        halfSize,
        regularizationConfig=regularizationConfig,
        oversampling=oversampling,
        gain=gain,
        readnoise=readnoise,
        maxOuterIter=maxOuterIter,
        thetaTol=thetaTol,
        fitCenters=fitCenters,
    )
    combinedXCenter = np.array(xCenter, dtype=float)
    combinedYCenter = np.array(yCenter, dtype=float)
    combinedXCenter[trainMask] = hybridFit.xCenter
    combinedYCenter[trainMask] = hybridFit.yCenter
    hybridHeldOut = computeHeldOutChi2(
        image,
        fiberIndex,
        row,
        combinedXCenter,
        combinedYCenter,
        heldOutMask,
        hybridFit.model,
        hybridFit.amplitudes,
        halfSize,
        oversampling,
        gain,
        readnoise,
    )

    return Struct(
        paramOnly=Struct(fit=paramFit, heldOut=paramHeldOut),
        hybrid=Struct(fit=hybridFit, heldOut=hybridHeldOut),
    )


def subPixelPhaseHistogram(
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    clusterPValueThreshold: float = 0.01,
    axes: Optional[Tuple] = None,
    show: bool = True,
) -> Struct:
    """Plot sub-pixel phase histograms and flag clustering in either axis

    Per PIPE2D-1823-psf.md: an oversampled cross-dispersion ``dP`` model
    isn't supported by the data if cross-dispersion phases are clustered
    (rather than uniformly covering the pixel), so this must be checked and
    flagged rather than silently hidden by regularization.

    Parameters
    ----------
    xCenter, yCenter : `numpy.ndarray`
        Line center positions.
    clusterPValueThreshold : `float`
        Below this Kolmogorov-Smirnov p-value (against a uniform
        distribution on ``[0, 1)``), an axis is flagged as clustered.
    axes : `tuple` of two `matplotlib.axes.Axes`, optional
        Axes on which to plot (x phase, y phase); a new figure is created if
        not given.
    show : `bool`
        Call ``matplotlib.pyplot.show()``?

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``xPvalue``, ``yPvalue``, ``xClustered``, ``yClustered``,
        and ``figure``.
    """
    import matplotlib.pyplot as plt

    xPhase = np.mod(np.asarray(xCenter, dtype=float), 1.0)
    yPhase = np.mod(np.asarray(yCenter, dtype=float), 1.0)
    xResult = kstest(xPhase, "uniform")
    yResult = kstest(yPhase, "uniform")
    xClustered = bool(xResult.pvalue < clusterPValueThreshold)
    yClustered = bool(yResult.pvalue < clusterPValueThreshold)
    if xClustered:
        _LOG.warning(
            "Cross-dispersion (x) sub-pixel phases are significantly clustered (p=%.3g): an oversampled "
            "cross-dispersion dP model isn't supported by the data here. Consider a coarser cross-"
            "dispersion knot spacing.",
            xResult.pvalue,
        )
    if yClustered:
        _LOG.warning(
            "Along-dispersion (y) sub-pixel phases are significantly clustered (p=%.3g).", yResult.pvalue
        )

    if axes is None:
        figure, axes = plt.subplots(1, 2, figsize=(8, 3))
    else:
        figure = axes[0].figure
    axes[0].hist(xPhase, bins=20, range=(0.0, 1.0))
    axes[0].set_xlabel("x sub-pixel phase")
    axes[0].set_title(f"p={xResult.pvalue:.2g}" + (" (clustered)" if xClustered else ""))
    axes[1].hist(yPhase, bins=20, range=(0.0, 1.0))
    axes[1].set_xlabel("y sub-pixel phase")
    axes[1].set_title(f"p={yResult.pvalue:.2g}" + (" (clustered)" if yClustered else ""))
    figure.tight_layout()
    if show:
        plt.show()
    return Struct(
        xPvalue=float(xResult.pvalue),
        yPvalue=float(yResult.pvalue),
        xClustered=xClustered,
        yClustered=yClustered,
        figure=figure,
    )


def stackedResidualMap(
    image,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    model,
    halfSize: int,
    oversampling: int = 4,
    gain: float = 1.0,
    readnoise: float = 0.0,
    ax=None,
    show: bool = True,
) -> Struct:
    """Stack per-line (data - model)/sigma residual stamps

    Solves only amplitudes (and a background level) for the given lines,
    holding the PSF shape (``model``) fixed, then averages each line's
    residual stamp onto a common ``(2*halfSize + 1)``-pixel grid relative to
    its (rounded) center -- the same grid `~pfs.drp.stella.fitPsfModel.\
buildStampGeometry` already uses internally. Lines near the image edge
    contribute only their in-bounds pixels; out-of-bounds pixels are simply
    excluded from that line's contribution to the stack (`numpy.nanmean`).

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image containing the line stamps.
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and center position of each line.
    model : `pfs.drp.stella.psfProfiles.ParametricPsfModel` or \
            `pfs.drp.stella.hybridPsfModel.HybridPsfModel`
        Already-fitted PSF model (shape only; not modified here).
    halfSize : `int`
        Half-size of each line's stamp, in pixels.
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    gain : `float`
        Detector gain (electrons/ADU).
    readnoise : `float`
        Detector read noise (ADU).
    ax : `matplotlib.axes.Axes`, optional
        Axes on which to plot; a new figure is created if not given.
    show : `bool`
        Call ``matplotlib.pyplot.show()``?

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``stackedResidual`` (the ``(2*halfSize+1, 2*halfSize+1)``
        stacked map, `numpy.nan` where no line contributed),
        ``perLineResidual`` (the individual, pre-stack stamps), and
        ``figure``.
    """
    import matplotlib.pyplot as plt

    geometry = buildStampGeometry(image, fiberIndex, row, xCenter, yCenter, halfSize, gain, readnoise)
    designMatrix = buildParametricDesignMatrix(
        model, fiberIndex, row, xCenter, yCenter, geometry, oversampling
    )
    amplitudes = solveAmplitudesAndBackground(designMatrix, geometry)
    predicted = designMatrix @ amplitudes
    geometry.varianceVector = readnoise**2 + np.clip(predicted, 0.0, None) / gain
    residual = (geometry.dataVector - predicted) / np.sqrt(geometry.varianceVector)

    size = 2 * halfSize + 1
    numLines = len(fiberIndex)
    stack = np.full((numLines, size, size), np.nan)
    for ii in range(numLines):
        rowSlice = geometry.rowSlices[ii]
        colSlice = geometry.colSlices[ii]
        subIndex = geometry.pixelIndex[rowSlice, colSlice]
        stampResidual = residual[subIndex.ravel()].reshape(subIndex.shape)
        rowOffset = rowSlice.start - (int(np.round(yCenter[ii])) - halfSize)
        colOffset = colSlice.start - (int(np.round(xCenter[ii])) - halfSize)
        stack[
            ii,
            rowOffset : rowOffset + stampResidual.shape[0],
            colOffset : colOffset + stampResidual.shape[1],
        ] = stampResidual

    stackedResidual = np.nanmean(stack, axis=0)

    if ax is None:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    extent = (-halfSize - 0.5, halfSize + 0.5, -halfSize - 0.5, halfSize + 0.5)
    imagePlot = ax.imshow(stackedResidual, origin="lower", extent=extent, cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_title(f"Stacked (data-model)/sigma residual, {numLines} lines")
    figure.colorbar(imagePlot, ax=ax)
    if show:
        plt.show()

    return Struct(stackedResidual=stackedResidual, perLineResidual=stack, figure=figure)


def radialProfileWithBootstrap(
    image,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    hybridModelFactory: Callable[[], HybridPsfModel],
    halfSize: int,
    fiberIndexEval: float = 0.0,
    rowEval: float = 0.0,
    regularizationConfig: Optional[RegularizationConfig] = None,
    oversampling: int = 4,
    gain: float = 1.0,
    readnoise: float = 0.0,
    maxOuterIter: int = 6,
    thetaTol: float = 1.0e-6,
    fitCenters: bool = True,
    numBootstrap: int = 20,
    numRadialPoints: int = 30,
    numAngles: int = 16,
    rng: Optional[np.random.RandomState] = None,
    axes=None,
    show: bool = True,
) -> Struct:
    """Plot radial and directional PSF profiles, with bootstrap uncertainty

    Fits the full hybrid model once on all the given lines, then refits on
    ``numBootstrap`` resamples of the lines (sampled with replacement) to
    estimate the uncertainty on the total profile -- an approximate,
    line-level bootstrap (it resamples which lines contribute, not the
    underlying pixel data), matching PIPE2D-1823-psf.md's "uncertainties
    from bootstrapping over lines/fibers". ``P_param`` and ``dP`` are also
    plotted from the single full-data fit (not bootstrapped individually).

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image containing the line stamps.
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and initial center position of each line.
    hybridModelFactory : callable
        Zero-argument callable returning a fresh `HybridPsfModel` with an
        initial guess already set on its ``parametricModel``.
    halfSize : `int`
        Half-size of each line's stamp, in pixels.
    fiberIndexEval, rowEval : `float`
        Detector position at which to evaluate the profile.
    regularizationConfig : `pfs.drp.stella.psfSpline.RegularizationConfig`, optional
        Regularization strength recipe for the ``dP`` coefficient solve.
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    gain : `float`
        Detector gain (electrons/ADU).
    readnoise : `float`
        Detector read noise (ADU).
    maxOuterIter : `int`
        Maximum number of outer fit iterations, for every (re)fit.
    thetaTol : `float`
        Convergence tolerance on the relative change in the parametric
        parameter vector.
    fitCenters : `bool`
        Update line centers each outer iteration?
    numBootstrap : `int`
        Number of bootstrap resamples.
    numRadialPoints : `int`
        Number of radii (linearly spaced from 0 to ``halfSize``) at which to
        evaluate the radial profile.
    numAngles : `int`
        Number of angles averaged over at each radius, for the radial
        profile.
    rng : `numpy.random.RandomState`, optional
        Random number generator for the bootstrap resampling.
    axes : sequence of three `matplotlib.axes.Axes`, optional
        Axes for the (radial, x-directional, y-directional) plots; a new
        figure is created if not given.
    show : `bool`
        Call ``matplotlib.pyplot.show()``?

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``fullFit`` (the full-data
        `~pfs.drp.stella.fitPsfModel.fitHybridPsf` result), ``radii`` and,
        for each of ``radial``/``x``/``y``: ``<name>Total``, ``<name>Param``,
        ``<name>Dp`` (full-data profiles) and ``<name>Lower``/``<name>Upper``
        (bootstrap 16th/84th percentile band on the total profile), plus
        ``figure``.
    """
    import matplotlib.pyplot as plt

    if rng is None:
        rng = np.random.RandomState()
    radii = np.linspace(0.0, float(halfSize), numRadialPoints)
    angles = np.linspace(0.0, 2 * np.pi, numAngles, endpoint=False)
    cosAngles, sinAngles = np.cos(angles), np.sin(angles)

    def fit(fitFiberIndex, fitRow, fitXCenter, fitYCenter) -> Struct:
        model = hybridModelFactory()
        return fitHybridPsf(
            image,
            fitFiberIndex,
            fitRow,
            fitXCenter,
            fitYCenter,
            model,
            halfSize,
            regularizationConfig=regularizationConfig,
            oversampling=oversampling,
            gain=gain,
            readnoise=readnoise,
            maxOuterIter=maxOuterIter,
            thetaTol=thetaTol,
            fitCenters=fitCenters,
        )

    def profiles(model) -> Struct:
        radial = np.array(
            [np.mean(model.evaluate(rr * cosAngles, rr * sinAngles, fiberIndexEval, rowEval)) for rr in radii]
        )
        xDirectional = model.evaluate(radii, np.zeros_like(radii), fiberIndexEval, rowEval)
        yDirectional = model.evaluate(np.zeros_like(radii), radii, fiberIndexEval, rowEval)
        return Struct(radial=radial, x=xDirectional, y=yDirectional)

    fullFit = fit(fiberIndex, row, xCenter, yCenter)
    fullProfile = profiles(fullFit.model)
    paramProfile = profiles(fullFit.model.parametricModel)
    dpProfile = Struct(
        radial=fullProfile.radial - paramProfile.radial,
        x=fullProfile.x - paramProfile.x,
        y=fullProfile.y - paramProfile.y,
    )

    numLines = len(fiberIndex)
    bootstrapRadial = np.zeros((numBootstrap, numRadialPoints))
    bootstrapX = np.zeros((numBootstrap, numRadialPoints))
    bootstrapY = np.zeros((numBootstrap, numRadialPoints))
    for ii in range(numBootstrap):
        indices = rng.randint(0, numLines, size=numLines)
        resampleFit = fit(fiberIndex[indices], row[indices], xCenter[indices], yCenter[indices])
        resampleProfile = profiles(resampleFit.model)
        bootstrapRadial[ii] = resampleProfile.radial
        bootstrapX[ii] = resampleProfile.x
        bootstrapY[ii] = resampleProfile.y

    def band(bootstrapValues: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.percentile(bootstrapValues, 16, axis=0), np.percentile(bootstrapValues, 84, axis=0)

    radialLower, radialUpper = band(bootstrapRadial)
    xLower, xUpper = band(bootstrapX)
    yLower, yUpper = band(bootstrapY)

    if axes is None:
        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    else:
        figure = axes[0].figure
    for axis, name, total, paramValues, dpValues, lower, upper in (
        (
            axes[0],
            "radial",
            fullProfile.radial,
            paramProfile.radial,
            dpProfile.radial,
            radialLower,
            radialUpper,
        ),
        (axes[1], "x", fullProfile.x, paramProfile.x, dpProfile.x, xLower, xUpper),
        (axes[2], "y", fullProfile.y, paramProfile.y, dpProfile.y, yLower, yUpper),
    ):
        axis.semilogy(radii, np.clip(total, 1.0e-8, None), "-", label="total")
        axis.semilogy(radii, np.clip(paramValues, 1.0e-8, None), "--", label="P_param")
        axis.semilogy(radii, np.clip(np.abs(dpValues), 1.0e-8, None), ":", label="|dP|")
        axis.fill_between(radii, np.clip(lower, 1.0e-8, None), np.clip(upper, 1.0e-8, None), alpha=0.3)
        axis.set_xlabel(f"{name} offset (pixels)")
        axis.legend(fontsize="small")
    axes[0].set_ylabel("normalized flux")
    figure.tight_layout()
    if show:
        plt.show()

    return Struct(
        fullFit=fullFit,
        radii=radii,
        radialTotal=fullProfile.radial,
        radialParam=paramProfile.radial,
        radialDp=dpProfile.radial,
        radialLower=radialLower,
        radialUpper=radialUpper,
        xTotal=fullProfile.x,
        xParam=paramProfile.x,
        xDp=dpProfile.x,
        xLower=xLower,
        xUpper=xUpper,
        yTotal=fullProfile.y,
        yParam=paramProfile.y,
        yDp=dpProfile.y,
        yLower=yLower,
        yUpper=yUpper,
        figure=figure,
    )


def energyConservationCheck(
    hybridModel: HybridPsfModel,
    fiberIndex: float,
    row: float,
    halfExtent: Optional[int] = None,
    oversampling: int = 4,
) -> Struct:
    """Check how much of the fitted PSF's flux falls within its fit extent

    Parameters
    ----------
    hybridModel : `pfs.drp.stella.hybridPsfModel.HybridPsfModel`
        Fitted hybrid PSF model.
    fiberIndex : `float`
        Index of the fiber (not fiberId) within the detector.
    row : `float`
        Row (dispersion-direction pixel) on the detector.
    halfExtent : `int`, optional
        Half-size (pixels) of the region to integrate over; defaults to the
        ``dP`` spline's extent (rounded up), since ``dP`` is defined to be
        zero beyond it while ``P_param`` still extrapolates.
    oversampling : `int`
        Oversampling factor for exact pixel integration.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``totalFlux`` (the pixel-integrated flux within
        ``halfExtent``; should be close to 1 for a well-conserved PSF) and
        ``fractionOutside`` (``1 - totalFlux``).
    """
    if halfExtent is None:
        halfExtent = int(
            np.ceil(max(hybridModel.splineBasis.xConfig.extent, hybridModel.splineBasis.yConfig.extent))
        )
    xIndices = np.arange(-halfExtent, halfExtent + 1)
    yIndices = np.arange(-halfExtent, halfExtent + 1)

    def evaluate(dx, dy):
        return hybridModel.evaluate(dx, dy, fiberIndex, row)

    pixels = integrateOverPixels(evaluate, xIndices, yIndices, oversampling)
    totalFlux = float(pixels.sum())
    return Struct(totalFlux=totalFlux, fractionOutside=1.0 - totalFlux, halfExtent=halfExtent)


def dpCoherenceCheck(
    image,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    fullFit: Struct,
    halfSize: int,
    regularizationConfig: Optional[RegularizationConfig] = None,
    oversampling: int = 4,
    gain: float = 1.0,
    readnoise: float = 0.0,
    gridSpacing: float = 0.1,
) -> Struct:
    """Check whether ``dP`` is coherent (real) or noise-like

    Splits fibers into even/odd ``fiberIndex`` halves and, holding the
    already-fitted parametric ``theta``, centers, and amplitudes from
    ``fullFit`` fixed, independently re-solves for the ``dP`` coefficients
    on each half (the constrained, regularized solve from the last step of
    `~pfs.drp.stella.fitPsfModel.fitHybridPsf`). If ``dP`` reflects a real
    PSF feature, the two halves' fitted ``dP`` should be strongly
    correlated; if it is over-flexible (chasing noise), they should not be.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image containing the line stamps.
    fiberIndex, row : `numpy.ndarray`
        Fiber index and nominal row of each line (the same arrays passed to
        `~pfs.drp.stella.fitPsfModel.fitHybridPsf` to produce ``fullFit``).
    fullFit : `lsst.pipe.base.Struct`
        The full-data result of `~pfs.drp.stella.fitPsfModel.fitHybridPsf`.
    halfSize : `int`
        Half-size of each line's stamp, in pixels.
    regularizationConfig : `pfs.drp.stella.psfSpline.RegularizationConfig`, optional
        Regularization strength recipe for each half's ``dP`` solve.
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    gain : `float`
        Detector gain (electrons/ADU).
    readnoise : `float`
        Detector read noise (ADU).
    gridSpacing : `float`
        Spacing (pixels) of the grid used to evaluate the two halves' ``dP``
        for the correlation, over the spline's extent.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``correlation`` (Pearson correlation of the two halves'
        ``dP``, evaluated on a common grid), and ``evenCoefficients``,
        ``oddCoefficients`` (the two halves' fitted coefficient vectors).
    """
    if regularizationConfig is None:
        regularizationConfig = RegularizationConfig()
    fiberIndex = np.asarray(fiberIndex)
    xCenter = fullFit.xCenter
    yCenter = fullFit.yCenter
    splineBasis = fullFit.model.splineBasis
    parametricModel = fullFit.model.parametricModel
    numLines = len(fiberIndex)
    fullAmplitudes = fullFit.amplitudes[:numLines]
    backgroundLevel = float(fullFit.amplitudes[numLines]) if len(fullFit.amplitudes) > numLines else 0.0

    evenMask = (fiberIndex.astype(np.int64) % 2) == 0
    coefficients = {}
    for name, mask in (("even", evenMask), ("odd", ~evenMask)):
        geometry = buildStampGeometry(
            image, fiberIndex[mask], row[mask], xCenter[mask], yCenter[mask], halfSize, gain, readnoise
        )
        parametricDesign = buildParametricDesignMatrix(
            parametricModel,
            fiberIndex[mask],
            row[mask],
            xCenter[mask],
            yCenter[mask],
            geometry,
            oversampling,
            includeBackground=False,
        )
        lineAmplitudes = fullAmplitudes[mask]
        residual = geometry.dataVector - (parametricDesign @ lineAmplitudes) - backgroundLevel
        splineDesign = buildSplineDesignMatrix(
            splineBasis,
            lineAmplitudes,
            fiberIndex[mask],
            row[mask],
            xCenter[mask],
            yCenter[mask],
            geometry,
            oversampling,
        )
        coefficients[name] = solveSplineCoefficients(
            splineDesign, residual, geometry, splineBasis, regularizationConfig
        )

    extent = max(splineBasis.xConfig.extent, splineBasis.yConfig.extent)
    axisValues = np.arange(-extent, extent + gridSpacing, gridSpacing)
    xGrid, yGrid = np.meshgrid(axisValues, axisValues, indexing="xy")
    design = splineBasis.evaluateGrid(xGrid, yGrid)
    evenValues = design @ coefficients["even"]
    oddValues = design @ coefficients["odd"]
    correlation = float(np.corrcoef(evenValues, oddValues)[0, 1])

    return Struct(
        correlation=correlation, evenCoefficients=coefficients["even"], oddCoefficients=coefficients["odd"]
    )
