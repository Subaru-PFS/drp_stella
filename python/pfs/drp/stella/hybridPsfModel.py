import numpy as np

from lsst.pipe.base import Struct

from .psfProfiles import ParametricPsfModel
from .psfSpline import PsfSplineBasis
from .psfPixelIntegration import makeOversampledGrid

__all__ = ["HybridPsfModel"]


class HybridPsfModel:
    """The full hybrid PSF model, ``P = P_param + dP``

    Combines the parametric backbone (`ParametricPsfModel`, smoothly varying
    across the detector) with a spatially-constant regularized spline
    correction (`PsfSplineBasis`), whose coefficients are constrained (by
    construction, via its null-space parameterization) to carry zero total
    flux and zero first moments, so ``dP`` cannot trade off against line
    amplitudes or centers.

    Parameters
    ----------
    parametricModel : `ParametricPsfModel`
        The parametric PSF backbone, ``P_param``.
    splineBasis : `PsfSplineBasis`
        The tensor-product B-spline basis for the correction, ``dP``.
    """

    def __init__(self, parametricModel: ParametricPsfModel, splineBasis: PsfSplineBasis):
        self.parametricModel = parametricModel
        self.splineBasis = splineBasis
        self.splineCoefficients = np.zeros(splineBasis.numCoefficients)

    def setSplineCoefficients(self, coeffs: np.ndarray) -> None:
        """Set the (full, constraint-satisfying) ``dP`` coefficient vector

        Parameters
        ----------
        coeffs : `numpy.ndarray`, shape ``(splineBasis.numCoefficients,)``
            Coefficients, e.g. as produced by ``nullSpaceBasis() @ z`` for
            some reduced vector ``z`` (see `PsfSplineBasis.nullSpaceBasis`).
        """
        coeffs = np.asarray(coeffs, dtype=float)
        if len(coeffs) != self.splineBasis.numCoefficients:
            raise ValueError(f"Expected {self.splineBasis.numCoefficients} coefficients, got {len(coeffs)}")
        self.splineCoefficients = coeffs

    def evaluate(self, dx: np.ndarray, dy: np.ndarray, fiberIndex: float, row: float) -> np.ndarray:
        """Evaluate ``P = P_param + dP`` at continuous offsets from the PSF center

        Parameters
        ----------
        dx, dy : `numpy.ndarray`
            Positions at which to evaluate, relative to the PSF center.
        fiberIndex : `float`
            Index of the fiber (not fiberId) within the detector.
        row : `float`
            Row (dispersion-direction pixel) on the detector.

        Returns
        -------
        values : `numpy.ndarray`
            ``P`` evaluated at ``(dx, dy)``; matches the shape of ``dx``.
        """
        dx = np.asarray(dx, dtype=float)
        pParam = self.parametricModel.evaluate(dx, dy, fiberIndex, row)
        design = self.splineBasis.evaluateGrid(dx, dy)
        dP = (design @ self.splineCoefficients).reshape(dx.shape)
        return pParam + dP

    def checkNonNegativity(
        self,
        xIndices: np.ndarray,
        yIndices: np.ndarray,
        fiberIndex: float,
        row: float,
        oversampling: int = 4,
    ) -> Struct:
        """Diagnostic: check whether ``P = P_param + dP`` stays non-negative

        The spec requires reporting this rather than silently enforcing it;
        a bounded solver or soft positivity penalty is future work if
        violations turn out to be frequent or severe.

        Parameters
        ----------
        xIndices, yIndices : `numpy.ndarray`
            Pixel-center coordinates (offsets from the PSF center) along x
            and y, evenly spaced by 1, spanning the region to check.
        fiberIndex : `float`
            Index of the fiber (not fiberId) within the detector.
        row : `float`
            Row (dispersion-direction pixel) on the detector.
        oversampling : `int`, optional
            Oversampling factor for the grid on which to check.

        Returns
        -------
        result : `Struct`
            ``minValue``: the minimum value found (negative indicates a
            violation). ``fractionNegative``: fraction of oversampled grid
            points with a negative value.
        """
        xGrid, yGrid = makeOversampledGrid(xIndices, yIndices, oversampling)
        values = self.evaluate(xGrid, yGrid, fiberIndex, row)
        return Struct(minValue=float(values.min()), fractionNegative=float(np.mean(values < 0)))
