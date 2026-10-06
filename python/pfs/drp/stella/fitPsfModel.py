from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg

from lsst.pipe.base import Struct

from .psfPixelIntegration import integrateOverPixels
from .psfSpline import RegularizationConfig

__all__ = [
    "StampGeometry",
    "buildStampGeometry",
    "buildParametricDesignMatrix",
    "solveAmplitudesAndBackground",
    "fitParametricPsf",
    "updateLineCenters",
    "buildSplineDesignMatrix",
    "solveSplineCoefficients",
    "fitHybridPsf",
]


@dataclass
class StampGeometry:
    """Pixel-level bookkeeping for a joint fit of overlapping line stamps

    Parameters
    ----------
    pixelIndex : `numpy.ndarray`, shape ``image.shape``
        Compact index (``0..numUsedPixels - 1``) of each used pixel; ``-1``
        for pixels not used by any stamp.
    rowSlices, colSlices : `list` of `slice`
        Per-line stamp bounding box (clipped to the image), one pair per
        line.
    dataVector : `numpy.ndarray`, shape ``(numUsedPixels,)``
        Data values at the used pixels.
    varianceVector : `numpy.ndarray`, shape ``(numUsedPixels,)``
        Variance at the used pixels. This is mutated in place across outer
        iterations, as it is re-estimated from the current model.
    numUsedPixels : `int`
        Number of used pixels.
    """

    pixelIndex: np.ndarray
    rowSlices: List[slice]
    colSlices: List[slice]
    dataVector: np.ndarray
    varianceVector: np.ndarray
    numUsedPixels: int


def buildStampGeometry(
    image,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    halfSize: int,
    gain: float,
    readnoise: float,
) -> StampGeometry:
    """Build the shared pixel bookkeeping for a joint fit of many line stamps

    Pixels used by more than one stamp (neighboring fibers' wings overlap at
    the 6.5 px fiber pitch) are deduplicated via a compact index map, so the
    joint linear solve never touches a dense whole-detector array.

    The variance is bootstrapped directly from the data, since no model
    exists yet; it should be replaced with a model-based estimate (see
    `fitParametricPsf`) as soon as one is available, to avoid biasing the
    weighted least-squares solve.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image containing the line stamps.
    fiberIndex : `numpy.ndarray`
        Fiber index of each line (unused here; kept for a uniform call
        signature with `buildParametricDesignMatrix`).
    row, xCenter, yCenter : `numpy.ndarray`
        Nominal row and center position of each line.
    halfSize : `int`
        Half-size of each line's stamp, in pixels.
    gain : `float`
        Detector gain (electrons/ADU).
    readnoise : `float`
        Detector read noise (ADU).

    Returns
    -------
    geometry : `StampGeometry`
        Shared pixel bookkeeping.
    """
    height, width = image.array.shape
    pixelIndex = np.full((height, width), -1, dtype=np.int64)
    rowSlices = []
    colSlices = []
    for yy, xx in zip(yCenter, xCenter):
        centerRow = int(np.round(yy))
        centerCol = int(np.round(xx))
        rowSlice = slice(max(centerRow - halfSize, 0), min(centerRow + halfSize + 1, height))
        colSlice = slice(max(centerCol - halfSize, 0), min(centerCol + halfSize + 1, width))
        rowSlices.append(rowSlice)
        colSlices.append(colSlice)
        pixelIndex[rowSlice, colSlice] = 0

    used = pixelIndex >= 0
    numUsedPixels = int(used.sum())
    pixelIndex[used] = np.arange(numUsedPixels)
    dataVector = np.asarray(image.array)[used].astype(float)
    varianceVector = readnoise**2 + np.clip(dataVector, 0.0, None) / gain
    return StampGeometry(pixelIndex, rowSlices, colSlices, dataVector, varianceVector, numUsedPixels)


def buildParametricDesignMatrix(
    model,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    geometry: StampGeometry,
    oversampling: int,
    includeBackground: bool = True,
) -> scipy.sparse.csr_matrix:
    """Build the sparse unit-flux design matrix for the current parametric model

    Column ``j`` (for ``j < numLines``) holds the pixel-integrated,
    unit-flux ``P_param`` for line ``j`` evaluated at its nominal center; an
    optional final column of ones represents a constant background level.

    Parameters
    ----------
    model : `pfs.drp.stella.psfProfiles.ParametricPsfModel`
        Current parametric PSF model.
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and center position of each line.
    geometry : `StampGeometry`
        Shared pixel bookkeeping, as returned by `buildStampGeometry`.
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    includeBackground : `bool`
        Include a constant background column?

    Returns
    -------
    designMatrix : `scipy.sparse.csr_matrix`, shape ``(numUsedPixels, numColumns)``
        Sparse design matrix.
    """
    numLines = len(fiberIndex)
    rowsList = []
    colsList = []
    dataList = []
    for ii in range(numLines):
        rowSlice = geometry.rowSlices[ii]
        colSlice = geometry.colSlices[ii]
        xIndices = np.arange(colSlice.start, colSlice.stop) - xCenter[ii]
        yIndices = np.arange(rowSlice.start, rowSlice.stop) - yCenter[ii]

        def evaluate(dx, dy, fiberIndexValue=fiberIndex[ii], rowValue=row[ii]):
            return model.evaluate(dx, dy, fiberIndexValue, rowValue)

        stamp = integrateOverPixels(evaluate, xIndices, yIndices, oversampling)
        subIndex = geometry.pixelIndex[rowSlice, colSlice]
        rowsList.append(subIndex.ravel())
        colsList.append(np.full(subIndex.size, ii, dtype=np.int64))
        dataList.append(stamp.ravel())

    rows = np.concatenate(rowsList)
    cols = np.concatenate(colsList)
    data = np.concatenate(dataList)
    numColumns = numLines + (1 if includeBackground else 0)
    if includeBackground:
        backgroundRows = np.arange(geometry.numUsedPixels, dtype=np.int64)
        backgroundCols = np.full(geometry.numUsedPixels, numLines, dtype=np.int64)
        backgroundData = np.ones(geometry.numUsedPixels)
        rows = np.concatenate([rows, backgroundRows])
        cols = np.concatenate([cols, backgroundCols])
        data = np.concatenate([data, backgroundData])

    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(geometry.numUsedPixels, numColumns))


def solveAmplitudesAndBackground(
    designMatrix: scipy.sparse.csr_matrix, geometry: StampGeometry
) -> np.ndarray:
    """Solve for line amplitudes (and background) given a fixed PSF model

    Parameters
    ----------
    designMatrix : `scipy.sparse.csr_matrix`
        Sparse design matrix, as returned by `buildParametricDesignMatrix`.
    geometry : `StampGeometry`
        Shared pixel bookkeeping, providing the data and (inverse-variance)
        weights.

    Returns
    -------
    solution : `numpy.ndarray`
        Amplitude of each line, followed by the background level (if the
        design matrix includes a background column).
    """
    weights = 1.0 / np.sqrt(geometry.varianceVector)
    weightMatrix = scipy.sparse.diags(weights)
    weightedMatrix = weightMatrix @ designMatrix
    weightedData = weights * geometry.dataVector
    return scipy.sparse.linalg.lsqr(weightedMatrix, weightedData)[0]


def _residualsForTheta(
    thetaVector: np.ndarray,
    model,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    geometry: StampGeometry,
    amplitudes: np.ndarray,
    oversampling: int,
) -> np.ndarray:
    """Weighted residuals for a trial parametric-model parameter vector

    Used as the objective for `scipy.optimize.least_squares` in
    `fitParametricPsf`; amplitudes and variance are held fixed.
    """
    model.setParameterVector(thetaVector)
    designMatrix = buildParametricDesignMatrix(
        model, fiberIndex, row, xCenter, yCenter, geometry, oversampling
    )
    predicted = designMatrix @ amplitudes
    return (geometry.dataVector - predicted) / np.sqrt(geometry.varianceVector)


def fitParametricPsf(
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
    maxOuterIter: int = 6,
    thetaTol: float = 1.0e-6,
) -> Struct:
    """Fit only the parametric PSF backbone (``P_param``) to a set of line stamps

    This implements steps 1 and 3 of the full alternating optimizer (linear
    amplitude/background solve, then nonlinear parametric-``theta`` solve),
    holding line centers fixed and omitting the spline correction ``dP``.
    It is intended as the first checkpoint for validating recovery of the
    parametric shape, before adding centers and ``dP`` to the loop.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image containing the line stamps.
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and center position of each line.
    model : `pfs.drp.stella.psfProfiles.ParametricPsfModel`
        Parametric PSF model, with an initial guess already set (e.g. via
        `~pfs.drp.stella.psfProfiles.ParametricPsfModel.setInitialGuess`).
        Modified in place; also returned for convenience.
    halfSize : `int`
        Half-size of each line's stamp, in pixels.
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    gain : `float`
        Detector gain (electrons/ADU).
    readnoise : `float`
        Detector read noise (ADU).
    maxOuterIter : `int`
        Maximum number of outer (amplitude, theta) iterations.
    thetaTol : `float`
        Convergence tolerance on the relative change in the parameter
        vector between outer iterations.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``model`` (the fitted `ParametricPsfModel`), ``amplitudes``
        (final line amplitudes, with the background level as the last
        element), and ``numIter`` (number of outer iterations performed).
    """
    geometry = buildStampGeometry(image, fiberIndex, row, xCenter, yCenter, halfSize, gain, readnoise)
    theta = model.getParameterVector()
    bounds = model.getParameterBounds()
    amplitudes = None
    numIter = 0
    for numIter in range(1, maxOuterIter + 1):
        designMatrix = buildParametricDesignMatrix(
            model, fiberIndex, row, xCenter, yCenter, geometry, oversampling
        )
        amplitudes = solveAmplitudesAndBackground(designMatrix, geometry)
        predicted = designMatrix @ amplitudes
        geometry.varianceVector = readnoise**2 + np.clip(predicted, 0.0, None) / gain

        fit = scipy.optimize.least_squares(
            _residualsForTheta,
            theta,
            args=(model, fiberIndex, row, xCenter, yCenter, geometry, amplitudes, oversampling),
            bounds=bounds,
        )
        newTheta = fit.x
        model.setParameterVector(newTheta)
        change = np.max(np.abs(newTheta - theta)) / max(np.max(np.abs(theta)), 1.0e-12)
        theta = newTheta
        if change < thetaTol:
            break

    return Struct(model=model, amplitudes=amplitudes, numIter=numIter)


def updateLineCenters(
    model,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    geometry: StampGeometry,
    designMatrix: scipy.sparse.csr_matrix,
    amplitudes: np.ndarray,
    oversampling: int,
    stepSize: float = 0.01,
    maxShift: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update each line's center by one local Gauss-Newton step

    Each line's center is updated independently (holding every other line's
    contribution, the parametric shape, and ``dP`` fixed), using a
    finite-difference Jacobian of that line's own pixel-integrated stamp
    with respect to its center position. This is the analogue, for centers,
    of the finite-difference approach already used for the parametric
    ``theta`` solve.

    Parameters
    ----------
    model : `pfs.drp.stella.psfProfiles.ParametricPsfModel` or \
            `pfs.drp.stella.hybridPsfModel.HybridPsfModel`
        Current PSF model (``P_param`` alone, or ``P_param + dP``); only its
        ``evaluate(dx, dy, fiberIndex, row)`` method is used.
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and current center position of each line.
    geometry : `StampGeometry`
        Shared pixel bookkeeping, as returned by `buildStampGeometry`.
    designMatrix : `scipy.sparse.csr_matrix`
        Current design matrix, as returned by `buildParametricDesignMatrix`
        (evaluated at ``xCenter``, ``yCenter``).
    amplitudes : `numpy.ndarray`
        Current solution vector matching ``designMatrix``'s columns (line
        amplitudes, optionally followed by a background level).
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    stepSize : `float`
        Finite-difference step size (pixels) for the Jacobian.
    maxShift : `float`
        Maximum allowed center shift (pixels) per call, for stability.

    Returns
    -------
    newXCenter, newYCenter : `numpy.ndarray`
        Updated center positions.
    """
    predicted = designMatrix @ amplitudes
    numLines = len(fiberIndex)
    newXCenter = xCenter.copy()
    newYCenter = yCenter.copy()

    for ii in range(numLines):
        rowSlice = geometry.rowSlices[ii]
        colSlice = geometry.colSlices[ii]
        subIndex = geometry.pixelIndex[rowSlice, colSlice]
        flatIndex = subIndex.ravel()

        # ``predicted`` already includes this line's own contribution at its
        # current center, which is exactly what ``base`` below recomputes
        # directly -- so no need to add it back in and subtract it out again.
        residualLocal = geometry.dataVector[flatIndex] - predicted[flatIndex]
        weightLocal = 1.0 / geometry.varianceVector[flatIndex]

        xPixels = np.arange(colSlice.start, colSlice.stop)
        yPixels = np.arange(rowSlice.start, rowSlice.stop)

        def stampAt(xTrial, yTrial, fiberIndexValue=fiberIndex[ii], rowValue=row[ii]):
            def evaluate(dx, dy):
                return model.evaluate(dx, dy, fiberIndexValue, rowValue)

            return integrateOverPixels(evaluate, xPixels - xTrial, yPixels - yTrial, oversampling).ravel()

        gradX = (
            (stampAt(xCenter[ii] + stepSize, yCenter[ii]) - stampAt(xCenter[ii] - stepSize, yCenter[ii]))
            / (2 * stepSize)
            * amplitudes[ii]
        )
        gradY = (
            (stampAt(xCenter[ii], yCenter[ii] + stepSize) - stampAt(xCenter[ii], yCenter[ii] - stepSize))
            / (2 * stepSize)
            * amplitudes[ii]
        )

        sqrtWeight = np.sqrt(weightLocal)
        jacobian = np.column_stack([gradX, gradY]) * sqrtWeight[:, np.newaxis]
        residual = residualLocal * sqrtWeight

        delta, *_ = np.linalg.lstsq(jacobian, residual, rcond=None)
        delta = np.clip(delta, -maxShift, maxShift)
        newXCenter[ii] = xCenter[ii] + delta[0]
        newYCenter[ii] = yCenter[ii] + delta[1]

    return newXCenter, newYCenter


def buildSplineDesignMatrix(
    splineBasis,
    amplitudes: np.ndarray,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    geometry: StampGeometry,
    oversampling: int,
) -> scipy.sparse.csr_matrix:
    """Build the sparse design matrix for the ``dP`` spline coefficients

    Since ``dP`` is linear in its coefficients (unlike ``P_param``), each
    line's amplitude-scaled, pixel-integrated contribution
    (`pfs.drp.stella.psfSpline.PsfSplineBasis.stampDesignMatrix`, evaluated
    at that line's center) can be assembled directly into a single sparse
    design matrix, matching `buildParametricDesignMatrix`'s pixel-index
    conventions; overlapping lines' contributions to shared pixels are
    summed.

    Parameters
    ----------
    splineBasis : `pfs.drp.stella.psfSpline.PsfSplineBasis`
        The ``dP`` basis.
    amplitudes : `numpy.ndarray`, shape ``(len(fiberIndex),)``
        Current amplitude of each line (excluding any background term).
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and center position of each line.
    geometry : `StampGeometry`
        Shared pixel bookkeeping, as returned by `buildStampGeometry`.
    oversampling : `int`
        Oversampling factor for exact pixel integration.

    Returns
    -------
    designMatrix : `scipy.sparse.csr_matrix`, shape \
            ``(geometry.numUsedPixels, splineBasis.numCoefficients)``
        Sparse design matrix: ``designMatrix @ coefficients`` gives each
        used pixel's total (amplitude-weighted) ``dP`` contribution.
    """
    numLines = len(fiberIndex)
    rowsList = []
    colsList = []
    dataList = []
    for ii in range(numLines):
        rowSlice = geometry.rowSlices[ii]
        colSlice = geometry.colSlices[ii]
        xIndices = np.arange(colSlice.start, colSlice.stop) - xCenter[ii]
        yIndices = np.arange(rowSlice.start, rowSlice.stop) - yCenter[ii]
        stampDesign = splineBasis.stampDesignMatrix(xIndices, yIndices, oversampling).tocoo()
        subIndex = geometry.pixelIndex[rowSlice, colSlice]
        flatIndex = subIndex.ravel()
        rowsList.append(flatIndex[stampDesign.row])
        colsList.append(stampDesign.col)
        dataList.append(stampDesign.data * amplitudes[ii])

    rows = np.concatenate(rowsList)
    cols = np.concatenate(colsList)
    data = np.concatenate(dataList)
    shape = (geometry.numUsedPixels, splineBasis.numCoefficients)
    designMatrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=shape)
    designMatrix.sum_duplicates()
    return designMatrix


def solveSplineCoefficients(
    splineDesignMatrix: scipy.sparse.csr_matrix,
    residualVector: np.ndarray,
    geometry: StampGeometry,
    splineBasis,
    regularizationConfig: RegularizationConfig,
) -> np.ndarray:
    """Solve the regularized, hard-constrained linear system for the ``dP`` coefficients

    The hard integral/first-moment constraints
    (`pfs.drp.stella.psfSpline.PsfSplineBasis.constraintMatrix`) are enforced
    exactly by solving in the reduced null-space basis
    (`~pfs.drp.stella.psfSpline.PsfSplineBasis.nullSpaceBasis`) rather than
    by penalizing them; the regularization matrix
    (`~pfs.drp.stella.psfSpline.PsfSplineBasis.regularizationMatrix`) is
    projected into the same reduced basis and added to the normal
    equations (Tikhonov form).

    Parameters
    ----------
    splineDesignMatrix : `scipy.sparse.csr_matrix`
        Design matrix, as returned by `buildSplineDesignMatrix`.
    residualVector : `numpy.ndarray`, shape ``(geometry.numUsedPixels,)``
        Data residual (data minus the current ``P_param``-only, background-
        included prediction) at each used pixel, to be explained by ``dP``.
    geometry : `StampGeometry`
        Shared pixel bookkeeping, providing inverse-variance weights.
    splineBasis : `pfs.drp.stella.psfSpline.PsfSplineBasis`
        The ``dP`` basis.
    regularizationConfig : `pfs.drp.stella.psfSpline.RegularizationConfig`
        Regularization strength recipe.

    Returns
    -------
    coefficients : `numpy.ndarray`, shape ``(splineBasis.numCoefficients,)``
        Fitted ``dP`` coefficients, satisfying the hard constraints to
        numerical precision by construction.
    """
    nullSpace = splineBasis.nullSpaceBasis()
    regularization = splineBasis.regularizationMatrix(regularizationConfig)
    reducedRegularization = nullSpace.T @ (regularization @ nullSpace)

    sqrtWeights = 1.0 / np.sqrt(geometry.varianceVector)
    weightedDesign = scipy.sparse.diags(sqrtWeights) @ splineDesignMatrix
    weightedResidual = sqrtWeights * residualVector

    # Form the (small, numCoefficients x numCoefficients) full normal matrix
    # via sparse-sparse multiplication first, then project into the reduced
    # null-space basis -- (D @ N).T @ (D @ N) == N.T @ (D.T @ D) @ N -- so we
    # never materialize a dense (numUsedPixels, reducedDim) intermediate.
    fullNormalMatrix = (weightedDesign.T @ weightedDesign).toarray()
    fullNormalVector = weightedDesign.T @ weightedResidual

    normalMatrix = nullSpace.T @ fullNormalMatrix @ nullSpace + reducedRegularization
    normalVector = nullSpace.T @ fullNormalVector
    reducedCoefficients = scipy.linalg.solve(normalMatrix, normalVector, assume_a="pos")
    return nullSpace @ reducedCoefficients


def _residualsForHybridTheta(
    thetaVector: np.ndarray,
    parametricModel,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    geometry: StampGeometry,
    amplitudes: np.ndarray,
    oversampling: int,
    dPContribution: np.ndarray,
) -> np.ndarray:
    """Weighted residuals for a trial parametric-model parameter vector

    Like `_residualsForTheta`, but adds a precomputed, fixed ``dP``
    contribution to the prediction rather than re-evaluating the full
    hybrid model on every trial: ``dP``'s coefficients (and every other
    quantity but ``theta``) are held fixed during this fit, so its
    amplitude-weighted contribution to each pixel is constant throughout
    the `scipy.optimize.least_squares` call. Used as the objective for
    `scipy.optimize.least_squares` in `fitHybridPsf`.
    """
    parametricModel.setParameterVector(thetaVector)
    designMatrix = buildParametricDesignMatrix(
        parametricModel, fiberIndex, row, xCenter, yCenter, geometry, oversampling
    )
    predicted = designMatrix @ amplitudes + dPContribution
    return (geometry.dataVector - predicted) / np.sqrt(geometry.varianceVector)


def fitHybridPsf(
    image,
    fiberIndex: np.ndarray,
    row: np.ndarray,
    xCenter: np.ndarray,
    yCenter: np.ndarray,
    hybridModel,
    halfSize: int,
    regularizationConfig: RegularizationConfig = None,
    oversampling: int = 4,
    gain: float = 1.0,
    readnoise: float = 0.0,
    maxOuterIter: int = 6,
    thetaTol: float = 1.0e-6,
    fitCenters: bool = True,
    centerStepSize: float = 0.01,
    maxCenterShift: float = 1.0,
) -> Struct:
    """Fit the full hybrid PSF model (``P_param + dP``) to a set of line stamps

    Implements the full four-step alternating optimizer of
    PIPE2D-1823-psf.md: (1) amplitudes and background (linear), (2) line
    centers (local Gauss-Newton, optional), (3) parametric ``theta``
    (nonlinear), (4) ``dP`` coefficients (regularized, hard-constrained
    linear) -- repeated to convergence on ``theta``.

    Parameters
    ----------
    image : `lsst.afw.image.Image`
        Image containing the line stamps.
    fiberIndex, row, xCenter, yCenter : `numpy.ndarray`
        Fiber index, nominal row, and initial center position of each line.
    hybridModel : `pfs.drp.stella.hybridPsfModel.HybridPsfModel`
        Hybrid PSF model, with an initial guess already set on its
        ``parametricModel``. Modified in place; also returned for
        convenience.
    halfSize : `int`
        Half-size of each line's stamp, in pixels.
    regularizationConfig : `pfs.drp.stella.psfSpline.RegularizationConfig`
        Regularization strength recipe for the ``dP`` coefficient solve;
        defaults to `RegularizationConfig`'s defaults if not given.
    oversampling : `int`
        Oversampling factor for exact pixel integration.
    gain : `float`
        Detector gain (electrons/ADU).
    readnoise : `float`
        Detector read noise (ADU).
    maxOuterIter : `int`
        Maximum number of outer iterations.
    thetaTol : `float`
        Convergence tolerance on the relative change in the parametric
        parameter vector between outer iterations.
    fitCenters : `bool`
        Update line centers each outer iteration?
    centerStepSize : `float`
        Finite-difference step size (pixels) for the center Jacobian.
    maxCenterShift : `float`
        Maximum allowed center shift (pixels) per outer iteration.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        Contains ``model`` (the fitted `HybridPsfModel`), ``amplitudes``
        (line amplitudes followed by the background level), ``xCenter``,
        ``yCenter`` (final center positions), and ``numIter``.
    """
    if regularizationConfig is None:
        regularizationConfig = RegularizationConfig()
    numLines = len(fiberIndex)
    geometry = buildStampGeometry(image, fiberIndex, row, xCenter, yCenter, halfSize, gain, readnoise)
    theta = hybridModel.parametricModel.getParameterVector()
    bounds = hybridModel.parametricModel.getParameterBounds()
    currentXCenter = np.array(xCenter, dtype=float)
    currentYCenter = np.array(yCenter, dtype=float)
    amplitudes = None
    numIter = 0

    for numIter in range(1, maxOuterIter + 1):
        designMatrix = buildParametricDesignMatrix(
            hybridModel, fiberIndex, row, currentXCenter, currentYCenter, geometry, oversampling
        )
        amplitudes = solveAmplitudesAndBackground(designMatrix, geometry)
        predicted = designMatrix @ amplitudes
        geometry.varianceVector = readnoise**2 + np.clip(predicted, 0.0, None) / gain

        if fitCenters:
            currentXCenter, currentYCenter = updateLineCenters(
                hybridModel,
                fiberIndex,
                row,
                currentXCenter,
                currentYCenter,
                geometry,
                designMatrix,
                amplitudes,
                oversampling,
                stepSize=centerStepSize,
                maxShift=maxCenterShift,
            )
            designMatrix = buildParametricDesignMatrix(
                hybridModel, fiberIndex, row, currentXCenter, currentYCenter, geometry, oversampling
            )
            amplitudes = solveAmplitudesAndBackground(designMatrix, geometry)
            predicted = designMatrix @ amplitudes
            geometry.varianceVector = readnoise**2 + np.clip(predicted, 0.0, None) / gain

        dPContribution = (
            buildSplineDesignMatrix(
                hybridModel.splineBasis,
                amplitudes[:numLines],
                fiberIndex,
                row,
                currentXCenter,
                currentYCenter,
                geometry,
                oversampling,
            )
            @ hybridModel.splineCoefficients
        )
        fit = scipy.optimize.least_squares(
            _residualsForHybridTheta,
            theta,
            args=(
                hybridModel.parametricModel,
                fiberIndex,
                row,
                currentXCenter,
                currentYCenter,
                geometry,
                amplitudes,
                oversampling,
                dPContribution,
            ),
            bounds=bounds,
        )
        newTheta = fit.x
        hybridModel.parametricModel.setParameterVector(newTheta)

        lineAmplitudes = amplitudes[:numLines]
        backgroundLevel = amplitudes[numLines] if len(amplitudes) > numLines else 0.0
        parametricDesign = buildParametricDesignMatrix(
            hybridModel.parametricModel,
            fiberIndex,
            row,
            currentXCenter,
            currentYCenter,
            geometry,
            oversampling,
            includeBackground=False,
        )
        residualForSpline = geometry.dataVector - (parametricDesign @ lineAmplitudes) - backgroundLevel
        splineDesign = buildSplineDesignMatrix(
            hybridModel.splineBasis,
            lineAmplitudes,
            fiberIndex,
            row,
            currentXCenter,
            currentYCenter,
            geometry,
            oversampling,
        )
        newCoefficients = solveSplineCoefficients(
            splineDesign, residualForSpline, geometry, hybridModel.splineBasis, regularizationConfig
        )
        hybridModel.setSplineCoefficients(newCoefficients)

        change = np.max(np.abs(newTheta - theta)) / max(np.max(np.abs(theta)), 1.0e-12)
        theta = newTheta
        if change < thetaTol:
            break

    return Struct(
        model=hybridModel,
        amplitudes=amplitudes,
        xCenter=currentXCenter,
        yCenter=currentYCenter,
        numIter=numIter,
    )
