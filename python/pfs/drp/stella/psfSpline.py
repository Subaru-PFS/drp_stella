from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse
from scipy.interpolate import BSpline
from scipy.linalg import null_space

from .psfPixelIntegration import binningMatrix2D, makeOversampledGrid

__all__ = [
    "SplineAxisConfig",
    "buildNonuniformKnots",
    "BSplineAxis",
    "RegularizationConfig",
    "PsfSplineBasis",
]

_DEGREE = 3  # cubic B-splines throughout


@dataclass
class SplineAxisConfig:
    """Non-uniform knot spacing recipe for one axis of the ``dP`` spline

    Following PIPE2D-1823-psf.md: fine spacing near the PSF center (where the
    correction needs to resolve the undersampled core-to-wing transition),
    coarsening with radius, out to a hard extent beyond which ``dP`` is
    identically zero (`PsfSplineBasis.evaluateGrid` returns zero rows there).

    Parameters
    ----------
    extent : `float`
        Half-extent of the spline domain, in pixels; must reach past
        ``mediumRadius``. ``dP`` is defined on ``[-extent, extent]`` and is
        zero outside it.
    fineSpacing, fineRadius : `float`
        Knot spacing within ``[-fineRadius, fineRadius]``.
    mediumSpacing, mediumRadius : `float`
        Knot spacing between ``fineRadius`` and ``mediumRadius``.
    coarseSpacing : `float`
        Knot spacing between ``mediumRadius`` and ``extent``.
    """

    extent: float
    fineSpacing: float = 0.25
    fineRadius: float = 2.0
    mediumSpacing: float = 0.5
    mediumRadius: float = 4.0
    coarseSpacing: float = 1.5


def buildNonuniformKnots(config: SplineAxisConfig) -> np.ndarray:
    """Build symmetric, non-uniformly-spaced interior breakpoints for one axis

    Parameters
    ----------
    config : `SplineAxisConfig`
        Knot spacing recipe.

    Returns
    -------
    breakpoints : `numpy.ndarray`
        Strictly increasing breakpoints from ``-config.extent`` to
        ``config.extent``, symmetric about zero.
    """
    if config.fineRadius >= config.mediumRadius:
        raise ValueError("fineRadius must be less than mediumRadius")
    if config.mediumRadius >= config.extent:
        raise ValueError("mediumRadius must be less than extent")

    def arangeInclusive(start: float, stop: float, step: float) -> np.ndarray:
        # Snap the step so the segment lands exactly on `stop`, keeping
        # adjoining segments' breakpoints contiguous (each segment's `start`
        # is the previous segment's `stop`).
        num = max(int(np.round((stop - start) / step)), 1)
        adjustedStep = (stop - start) / num
        return start + adjustedStep * np.arange(num + 1)

    fine = arangeInclusive(0.0, config.fineRadius, config.fineSpacing)
    medium = arangeInclusive(config.fineRadius, config.mediumRadius, config.mediumSpacing)
    coarse = arangeInclusive(config.mediumRadius, config.extent, config.coarseSpacing)
    positive = np.unique(np.concatenate([fine, medium, coarse]))
    return np.unique(np.concatenate([-positive[::-1], positive]))


def _basisIntegralsAndMoments(knots: np.ndarray, degree: int, numBasis: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the integral and first moment of each basis function

    The integral of a normalized B-spline basis function follows the
    standard identity ``integral_i = (knots[i+degree+1] - knots[i]) /
    (degree+1)`` (e.g. de Boor, "A Practical Guide to Splines"), computed
    exactly. The first moment has no similarly simple closed form here, so
    it is computed by Gauss-Legendre quadrature on each knot span (exact for
    the piecewise-polynomial integrand ``x * B_i(x)``, since ``degree+2``
    quadrature points integrate polynomials up to degree ``2*(degree+2)-1``,
    well above the required ``degree+1``).

    Parameters
    ----------
    knots : `numpy.ndarray`
        Full (clamped) knot vector.
    degree : `int`
        Spline degree.
    numBasis : `int`
        Number of basis functions.

    Returns
    -------
    integral, moment : `numpy.ndarray`, shape ``(numBasis,)``
        Integral and first moment of each basis function.
    """
    integral = (knots[degree + 1 : degree + 1 + numBasis] - knots[:numBasis]) / (degree + 1)

    breakpoints = np.unique(knots)
    nodes, weights = np.polynomial.legendre.leggauss(degree + 2)
    moment = np.zeros(numBasis)
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
        half = 0.5 * (upper - lower)
        mid = 0.5 * (upper + lower)
        xx = mid + half * nodes
        ww = weights * half
        basisMatrix = BSpline.design_matrix(xx, knots, degree, extrapolate=False).toarray()
        moment += (ww * xx) @ basisMatrix
    return integral, moment


class BSplineAxis:
    """A 1D cubic B-spline basis with non-uniform, clamped knots

    Parameters
    ----------
    config : `SplineAxisConfig`
        Knot spacing recipe.
    """

    def __init__(self, config: SplineAxisConfig):
        self.degree = _DEGREE
        self.extent = config.extent
        self.breakpoints = buildNonuniformKnots(config)
        self.knots = np.concatenate(
            [
                np.full(self.degree, self.breakpoints[0]),
                self.breakpoints,
                np.full(self.degree, self.breakpoints[-1]),
            ]
        )
        self.numBasis = len(self.knots) - self.degree - 1
        self._integral, self._moment = _basisIntegralsAndMoments(self.knots, self.degree, self.numBasis)

    def integral(self) -> np.ndarray:
        """Return the integral of each basis function, shape ``(numBasis,)``"""
        return self._integral

    def firstMoment(self) -> np.ndarray:
        """Return the first moment (``integral of x*B_i``) of each basis function"""
        return self._moment

    def designMatrix(self, xx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate all basis functions at the given (flattened) positions

        Parameters
        ----------
        xx : `numpy.ndarray`, shape ``(numPoints,)``
            Positions at which to evaluate.

        Returns
        -------
        indices : `numpy.ndarray`, shape ``(numPoints, degree+1)``
            Column (basis function) index of each nonzero entry per row.
        data : `numpy.ndarray`, shape ``(numPoints, degree+1)``
            Basis function values for each nonzero entry per row.
        inside : `numpy.ndarray`, shape ``(numPoints,)``, `bool`
            Whether each position is within ``[-extent, extent]``; outside
            this range ``dP`` is defined to be zero, so callers should treat
            ``data`` for those rows as zero regardless of the raw
            (clamped-position) values returned here.
        """
        xx = np.asarray(xx, dtype=float)
        inside = (xx >= self.breakpoints[0]) & (xx <= self.breakpoints[-1])
        clipped = np.clip(xx, self.breakpoints[0], self.breakpoints[-1])
        sparseMatrix = BSpline.design_matrix(clipped, self.knots, self.degree, extrapolate=False).tocsr()
        nnzPerRow = np.diff(sparseMatrix.indptr)
        expected = self.degree + 1
        if not np.all(nnzPerRow == expected):
            raise ValueError(
                f"Unexpected B-spline design matrix sparsity (expected {expected} nonzeros per row)"
            )
        indices = sparseMatrix.indices.reshape(-1, expected)
        data = sparseMatrix.data.reshape(-1, expected)
        return indices, data, inside

    def greville(self) -> np.ndarray:
        """Return the Greville abscissa (nominal center) of each basis function

        Used only to weight the regularization strength by radius; not
        otherwise meaningful (it is not, e.g., the point of peak value).
        """
        return np.array([np.mean(self.knots[ii + 1 : ii + self.degree + 1]) for ii in range(self.numBasis)])


@dataclass
class RegularizationConfig:
    """Regularization strength recipe for the ``dP`` spline coefficients

    The penalty combines a discrete second-derivative smoothness term (along
    each tensor axis, on the coefficient grid) and a ridge (shrink-to-zero)
    term, both scaled by a weight that grows with radius from the PSF
    center, so that poorly-constrained large-radius coefficients are damped
    more strongly than the well-constrained core.

    Parameters
    ----------
    smoothness : `float`
        Overall strength of the second-derivative penalty.
    ridge : `float`
        Overall strength of the ridge (shrink-to-zero) penalty.
    radiusScale : `float`
        Radius (pixels) at which the radius-dependent weight has grown by a
        factor of ``e`` relative to the center; the weight is
        ``exp(radius / radiusScale)``.
    """

    smoothness: float = 1.0
    ridge: float = 1.0
    radiusScale: float = 8.0


def _secondDifferenceMatrix1D(numBasis: int) -> scipy.sparse.csr_matrix:
    """Discrete second-difference operator on a 1D coefficient sequence"""
    if numBasis < 3:
        return scipy.sparse.csr_matrix((0, numBasis))
    rows = np.repeat(np.arange(numBasis - 2), 3)
    cols = np.concatenate([np.arange(ii, ii + 3) for ii in range(numBasis - 2)])
    data = np.tile([1.0, -2.0, 1.0], numBasis - 2)
    return scipy.sparse.csr_matrix((data, (rows, cols)), shape=(numBasis - 2, numBasis))


class PsfSplineBasis:
    """Tensor-product cubic B-spline basis for the ``dP`` correction

    Coefficients are indexed by a single flat vector, with ``yIndex`` slow
    and ``xIndex`` fast (matching this codebase's row-major image
    convention): coefficient ``c[yIndex * numBasisX + xIndex]`` multiplies
    ``Bx_xIndex(dx) * By_yIndex(dy)``.

    Parameters
    ----------
    xConfig, yConfig : `SplineAxisConfig`
        Knot spacing recipes for the spatial (x) and dispersion (y) axes.
    """

    def __init__(self, xConfig: SplineAxisConfig, yConfig: SplineAxisConfig):
        self.xConfig = xConfig
        self.yConfig = yConfig
        self.xAxis = BSplineAxis(xConfig)
        self.yAxis = BSplineAxis(yConfig)
        self.numBasisX = self.xAxis.numBasis
        self.numBasisY = self.yAxis.numBasis
        self.numCoefficients = self.numBasisX * self.numBasisY
        self._constraintMatrix = None
        self._nullSpaceBasis = None
        self._regularizationMatrixCache = {}

    def evaluateGrid(self, xGrid: np.ndarray, yGrid: np.ndarray) -> scipy.sparse.csr_matrix:
        """Build the sparse design matrix mapping coefficients to grid values

        Parameters
        ----------
        xGrid, yGrid : `numpy.ndarray`
            Positions (offsets from the PSF center) at which to evaluate,
            matching shapes (e.g. as produced by
            `pfs.drp.stella.psfPixelIntegration.makeOversampledGrid`).

        Returns
        -------
        design : `scipy.sparse.csr_matrix`, shape ``(xGrid.size, numCoefficients)``
            Sparse design matrix: ``design @ coeffs`` evaluates ``dP`` on the
            (flattened) grid. Rows for positions outside either axis' extent
            are identically zero.
        """
        xFlat = np.asarray(xGrid, dtype=float).ravel()
        yFlat = np.asarray(yGrid, dtype=float).ravel()
        xIndices, xData, xInside = self.xAxis.designMatrix(xFlat)
        yIndices, yData, yInside = self.yAxis.designMatrix(yFlat)

        numPoints = xFlat.size
        numX = xIndices.shape[1]
        numY = yIndices.shape[1]
        cols = (np.repeat(yIndices, numX, axis=1) * self.numBasisX + np.tile(xIndices, (1, numY))).ravel()
        data = (np.repeat(yData, numX, axis=1) * np.tile(xData, (1, numY))).ravel()
        inside = xInside & yInside
        data = data * np.repeat(inside, numX * numY)
        rows = np.repeat(np.arange(numPoints), numX * numY)

        design = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(numPoints, self.numCoefficients))
        design.eliminate_zeros()
        return design

    def stampDesignMatrix(
        self, xIndices: np.ndarray, yIndices: np.ndarray, oversampling: int = 4
    ) -> scipy.sparse.csr_matrix:
        """Build the pixel-integrated design matrix for ``dP`` on a stamp

        Unlike `evaluateGrid` (which evaluates ``dP`` at continuous
        positions), this integrates each basis function exactly over
        detector pixels -- the same oversample-then-bin approach as
        `pfs.drp.stella.psfPixelIntegration.integrateOverPixels`, but kept as
        a linear operator (rather than pre-evaluated values) since ``dP`` is
        linear in its coefficients, unlike ``P_param``.

        Parameters
        ----------
        xIndices, yIndices : `numpy.ndarray`
            Pixel-center coordinates (offsets from the PSF center) along x
            and y, evenly spaced by 1.
        oversampling : `int`, optional
            Number of oversampled sub-pixels per pixel, per axis.

        Returns
        -------
        design : `scipy.sparse.csr_matrix`, shape ``(len(yIndices)*len(xIndices), numCoefficients)``
            Sparse design matrix, row-major (y slow, x fast) over pixels,
            matching `pfs.drp.stella.psfPixelIntegration.integrateOverPixels`'s
            pixel ordering: ``(design @ coeffs).reshape(len(yIndices),
            len(xIndices))`` gives the pixel-integrated ``dP`` stamp.
        """
        xGrid, yGrid = makeOversampledGrid(xIndices, yIndices, oversampling)
        oversampledDesign = self.evaluateGrid(xGrid, yGrid)
        binningOperator = binningMatrix2D(len(xIndices), len(yIndices), oversampling)
        return binningOperator @ oversampledDesign

    def constraintMatrix(self) -> np.ndarray:
        """Build the hard linear constraint matrix ``C`` (integral, first moments)

        Depends only on the (immutable) spline configuration, so the result
        is computed once and cached.

        Returns
        -------
        constraints : `numpy.ndarray`, shape ``(3, numCoefficients)``
            Rows are, in order: total integral, first moment in x, first
            moment in y. A coefficient vector ``c`` satisfies the
            constraints iff ``constraints @ c == 0``.
        """
        if self._constraintMatrix is None:
            integralX, momentX = self.xAxis.integral(), self.xAxis.firstMoment()
            integralY, momentY = self.yAxis.integral(), self.yAxis.firstMoment()
            constraints = np.zeros((3, self.numCoefficients))
            constraints[0] = np.outer(integralY, integralX).ravel()
            constraints[1] = np.outer(integralY, momentX).ravel()
            constraints[2] = np.outer(momentY, integralX).ravel()
            self._constraintMatrix = constraints
        return self._constraintMatrix

    def nullSpaceBasis(self) -> np.ndarray:
        """Build a basis for the null space of `constraintMatrix`

        Any coefficient vector parameterized as ``c = N @ z`` (for the
        returned ``N`` and arbitrary ``z``) satisfies the hard constraints
        (zero integral and zero first moments) by construction, to numerical
        precision -- rather than merely being penalized toward them. Depends
        only on the (immutable) spline configuration, so the result is
        computed once and cached.

        Returns
        -------
        basis : `numpy.ndarray`, shape ``(numCoefficients, numCoefficients - 3)``
            Orthonormal null-space basis.
        """
        if self._nullSpaceBasis is None:
            self._nullSpaceBasis = null_space(self.constraintMatrix())
        return self._nullSpaceBasis

    def regularizationMatrix(self, config: RegularizationConfig) -> scipy.sparse.csr_matrix:
        """Build the regularization matrix in the full coefficient basis

        The returned matrix ``R`` is such that the penalty term added to the
        (weighted) least-squares objective is ``coeffs.T @ R @ coeffs``;
        callers doing the constrained fit in the reduced ``z`` basis should
        project it via ``N.T @ R @ N`` (`nullSpaceBasis`). Results are
        cached by the (immutable) spline configuration together with
        ``config``'s values, since the same ``config`` is typically reused
        across many outer fit iterations.

        Parameters
        ----------
        config : `RegularizationConfig`
            Regularization strength recipe.

        Returns
        -------
        matrix : `scipy.sparse.csr_matrix`, shape ``(numCoefficients, numCoefficients)``
            Symmetric positive semi-definite regularization matrix.
        """
        key = (config.smoothness, config.ridge, config.radiusScale)
        if key not in self._regularizationMatrixCache:
            dx1 = _secondDifferenceMatrix1D(self.numBasisX)
            dy1 = _secondDifferenceMatrix1D(self.numBasisY)
            identityX = scipy.sparse.identity(self.numBasisX, format="csr")
            identityY = scipy.sparse.identity(self.numBasisY, format="csr")
            smoothnessX = scipy.sparse.kron(identityY, dx1.T @ dx1, format="csr")
            smoothnessY = scipy.sparse.kron(dy1.T @ dy1, identityX, format="csr")
            smoothness = config.smoothness * (smoothnessX + smoothnessY)

            xCenters, yCenters = self.xAxis.greville(), self.yAxis.greville()
            radius = np.sqrt(np.add.outer(yCenters**2, xCenters**2)).ravel()
            weight = np.exp(radius / config.radiusScale)
            ridge = config.ridge * scipy.sparse.diags(weight**2)

            self._regularizationMatrixCache[key] = (smoothness + ridge).tocsr()
        return self._regularizationMatrixCache[key]
