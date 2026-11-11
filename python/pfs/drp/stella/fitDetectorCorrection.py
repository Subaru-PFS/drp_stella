"""Fit an empirical correction for detector effects from trace centroids

Detector effects (e.g., the epoxy bumps around the edges of the CCDs) displace
the traces by up to ~0.1 pixel, on scales too small to be captured by the
polynomial distortion of the optics. These effects are fixed to the silicon, so
we model them as a `BSplineDistortion` in the pixel frame, which is applied by
the ``DetectorModel`` of the `OpticalModelDetectorMap`.

We currently fit only the distortion in x (from trace centroids, especially
those from quartz exposures); the distortion in y is left empty.
"""

from typing import List, Optional

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.interpolate import BSpline

from lsst.geom import Box2I
from lsst.pex.config import Config, Field
from lsst.pipe.base import Task

from .BSplineDistortionContinued import BSpline2D, BSplineDistortion

__all__ = ("FitDetectorCorrectionConfig", "FitDetectorCorrectionTask", "addBSplineDistortions")


class FitDetectorCorrectionConfig(Config):
    """Configuration for FitDetectorCorrectionTask"""

    order = Field(dtype=int, default=3, doc="Order of the B-splines")
    knotSpacing = Field(dtype=float, default=64.0, doc="Spacing of knots in the interior (pixels)")
    edgeKnotSpacing = Field(
        dtype=float,
        default=32.0,
        doc="Spacing of knots near the edges of the detector (and the chip gap) (pixels)",
    )
    edgeWidth = Field(
        dtype=float,
        default=150.0,
        doc="Width of the region near the edges of the detector (and the chip gap) that uses edgeKnotSpacing "
        "(pixels)",
    )
    binSize = Field(dtype=float, default=16.0, doc="Size of bins for averaging the residuals (pixels)")
    minCount = Field(dtype=int, default=5, doc="Minimum number of measurements in a bin for it to be used")
    maxResidual = Field(
        dtype=float,
        default=0.3,
        doc="Maximum absolute residual (pixels) for a measurement to be used",
    )
    smoothing = Field(
        dtype=float,
        default=0.01,
        doc="Weight of the penalty on the second differences of the spline coefficients (in x and y), "
        "relative to the median number of measurements in a bin. This keeps poorly-constrained "
        "coefficients smooth.",
    )
    damp = Field(
        dtype=float,
        default=0.03,
        doc="Weight of the penalty on the size of the spline coefficients, relative to the median number of "
        "measurements in a bin. Together with the smoothing, this takes the correction smoothly to zero "
        "where there are no measurements.",
    )

    def validate(self):
        super().validate()
        if self.edgeKnotSpacing > self.knotSpacing:
            raise ValueError(f"edgeKnotSpacing ({self.edgeKnotSpacing}) > knotSpacing ({self.knotSpacing})")


class FitDetectorCorrectionTask(Task):
    """Fit an empirical correction for detector effects from trace centroids

    The correction is a `BSplineDistortion` in the pixel frame, with separate
    regions for the two CCDs of a divided detector. The residuals of the
    trace centroids (measured minus model, in x) are averaged in bins, and
    the spline coefficients are fit to the bin averages by least squares.

    The median residual for each fiber is subtracted before fitting, since
    it is degenerate with the slit offsets.
    """

    ConfigClass = FitDetectorCorrectionConfig
    _DefaultName = "fitDetectorCorrection"

    def run(
        self,
        bbox: Box2I,
        isDivided: bool,
        fiberId: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        dx: np.ndarray,
    ) -> BSplineDistortion:
        """Fit the correction

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the detector.
        isDivided : `bool`
            Is the detector divided into two CCDs? If so, the correction is
            fit separately for each, divided at the center of the ``bbox``.
        fiberId : `numpy.ndarray` of `int`
            Fiber identifiers of the trace measurements.
        xx, yy : `numpy.ndarray` of `float`
            Positions of the trace measurements in the pixel frame.
        dx : `numpy.ndarray` of `float`
            Residuals in x of the trace measurements (measured minus model;
            pixels).

        Returns
        -------
        correction : `pfs.drp.stella.BSplineDistortion`
            Fit correction. The distortion in y is empty.
        """
        order = self.config.order
        select = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(dx)
        fiberId = fiberId[select]
        xx = xx[select]
        yy = yy[select]
        dx = subtractGroupMedians(fiberId, dx[select])
        keep = np.abs(dx) < self.config.maxResidual
        self.log.debug("Fitting detector correction to %d/%d trace measurements", keep.sum(), select.size)

        xBoundaries, regions = self.getRegions(bbox, isDivided)
        yLow = bbox.getMinY() - 0.5
        yHigh = bbox.getMaxY() + 0.5
        yKnots = self.makeKnots(yLow, yHigh)
        xSplines = []
        for xLow, xHigh in regions:
            xKnots = self.makeKnots(xLow, xHigh)
            inRegion = keep & (xx >= xLow) & (xx < xHigh)
            coeff = self.fitSpline(xKnots, yKnots, xx[inRegion], yy[inRegion], dx[inRegion])
            xSplines.append(BSpline2D(order, xKnots, yKnots, coeff))
        ySplines = [BSpline2D(order) for _ in regions]
        return BSplineDistortion(np.array(xBoundaries, dtype=float), xSplines, ySplines)

    def getRegions(self, bbox: Box2I, isDivided: bool):
        """Return the regions in x

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box of the detector.
        isDivided : `bool`
            Is the detector divided into two CCDs?

        Returns
        -------
        xBoundaries : `list` of `float`
            Boundaries between the regions.
        regions : `list` of `tuple` of `float`
            Lower and upper bound in x for each region.
        """
        xLow = bbox.getMinX() - 0.5
        xHigh = bbox.getMaxX() + 0.5
        if not isDivided:
            return [], [(xLow, xHigh)]
        xCenter = bbox.getCenterX()  # Must match DetectorModel
        return [xCenter], [(xLow, xCenter), (xCenter, xHigh)]

    def makeKnots(self, low: float, high: float) -> np.ndarray:
        """Make a full knot vector over a range

        The knots are spaced by ``edgeKnotSpacing`` within ``edgeWidth`` of the
        ends of the range, and by ``knotSpacing`` in between.

        Parameters
        ----------
        low, high : `float`
            Range of the knots.

        Returns
        -------
        knots : `numpy.ndarray` of `float`
            Full knot vector, including repeated boundary knots.
        """
        order = self.config.order
        width = min(self.config.edgeWidth, 0.5 * (high - low))

        def spaced(start: float, stop: float, spacing: float) -> np.ndarray:
            num = max(int(np.round((stop - start) / spacing)), 1)
            return np.linspace(start, stop, num + 1)

        breakpoints = np.unique(
            np.concatenate(
                [
                    spaced(low, low + width, self.config.edgeKnotSpacing),
                    (
                        spaced(low + width, high - width, self.config.knotSpacing)
                        if high - low > 2 * width
                        else []
                    ),
                    spaced(high - width, high, self.config.edgeKnotSpacing),
                ]
            )
        )
        return np.concatenate([[low] * order, breakpoints, [high] * order])

    def fitSpline(
        self, xKnots: np.ndarray, yKnots: np.ndarray, xx: np.ndarray, yy: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """Fit spline coefficients to binned values

        Parameters
        ----------
        xKnots, yKnots : `numpy.ndarray` of `float`
            Full knot vectors.
        xx, yy : `numpy.ndarray` of `float`
            Positions of the measurements.
        values : `numpy.ndarray` of `float`
            Measured values.

        Returns
        -------
        coeff : `numpy.ndarray` of `float`, shape ``(numX, numY)``
            Spline coefficients.
        """
        order = self.config.order
        numX = len(xKnots) - order - 1
        numY = len(yKnots) - order - 1
        if len(values) == 0:
            self.log.warning("No measurements for detector correction region; setting to zero")
            return np.zeros((numX, numY))

        # Average in bins, using the mean position of the measurements in each bin
        binSize = self.config.binSize
        xEdges = np.arange(xKnots[0], xKnots[-1] + binSize, binSize)
        yEdges = np.arange(yKnots[0], yKnots[-1] + binSize, binSize)
        count, _, _ = np.histogram2d(xx, yy, bins=(xEdges, yEdges))
        sumValues, _, _ = np.histogram2d(xx, yy, bins=(xEdges, yEdges), weights=values)
        sumX, _, _ = np.histogram2d(xx, yy, bins=(xEdges, yEdges), weights=xx)
        sumY, _, _ = np.histogram2d(xx, yy, bins=(xEdges, yEdges), weights=yy)
        use = count >= self.config.minCount
        num = count[use]
        mean = sumValues[use] / num
        xMean = sumX[use] / num
        yMean = sumY[use] / num
        weights = np.sqrt(num)

        design = makeDesignMatrix(xKnots, yKnots, order, xMean, yMean)
        weighted = scipy.sparse.diags(weights) @ design

        # Regularization: penalize second differences of the coefficients (smoothness) and the size of the
        # coefficients (damping). The penalties are scaled by the typical weight of a bin, so that they
        # dominate only where there are few or no measurements.
        scale = np.median(num)
        penalty = makePenaltyMatrix(numX, numY, self.config.smoothing * scale, self.config.damp * scale)
        matrix = scipy.sparse.vstack([weighted, penalty]).tocsr()
        rhs = np.concatenate([mean * weights, np.zeros(penalty.shape[0])])
        solution = scipy.sparse.linalg.lsqr(matrix, rhs, atol=1.0e-10, btol=1.0e-10, iter_lim=20000)
        coeff = solution[0]
        self.log.debug(
            "Fit detector correction with %d coefficients to %d bins: lsqr istop=%d after %d iterations",
            coeff.size,
            use.sum(),
            solution[1],
            solution[2],
        )
        return coeff.reshape(numX, numY)


def makeDesignMatrix(
    xKnots: np.ndarray, yKnots: np.ndarray, order: int, xx: np.ndarray, yy: np.ndarray
) -> scipy.sparse.csr_matrix:
    """Make the design matrix for a tensor-product B-spline

    Parameters
    ----------
    xKnots, yKnots : `numpy.ndarray` of `float`
        Full knot vectors.
    order : `int`
        Order of the B-spline.
    xx, yy : `numpy.ndarray` of `float`
        Positions at which to evaluate the basis functions.

    Returns
    -------
    design : `scipy.sparse.csr_matrix`
        Design matrix, with a row for each position and a column for each
        coefficient (ordered with y varying fastest).
    """
    numX = len(xKnots) - order - 1
    numY = len(yKnots) - order - 1
    xx = np.clip(xx, xKnots[order], xKnots[numX])
    yy = np.clip(yy, yKnots[order], yKnots[numY])
    xBasis = BSpline.design_matrix(xx, xKnots, order).tocsr()
    yBasis = BSpline.design_matrix(yy, yKnots, order).tocsr()
    numPoints = len(xx)
    width = order + 1
    xIndex = xBasis.indices.reshape(numPoints, width)
    xValue = xBasis.data.reshape(numPoints, width)
    yIndex = yBasis.indices.reshape(numPoints, width)
    yValue = yBasis.data.reshape(numPoints, width)
    columns = (xIndex[:, :, None] * numY + yIndex[:, None, :]).reshape(numPoints, -1)
    values = (xValue[:, :, None] * yValue[:, None, :]).reshape(numPoints, -1)
    rows = np.repeat(np.arange(numPoints), width * width)
    return scipy.sparse.csr_matrix((values.ravel(), (rows, columns.ravel())), shape=(numPoints, numX * numY))


def makePenaltyMatrix(numX: int, numY: int, smoothing: float, damp: float) -> scipy.sparse.csr_matrix:
    """Make the regularization rows for a tensor-product B-spline fit

    The penalty is ``smoothing * (|D_x c|^2 + |D_y c|^2) + damp * |c|^2``,
    where ``D_x`` and ``D_y`` are the second-difference operators along each
    axis of the coefficient grid, and ``c`` are the coefficients. In the
    absence of measurements, this takes the coefficients smoothly to zero
    over a scale of roughly ``(smoothing/damp)**0.25`` knots.

    Parameters
    ----------
    numX, numY : `int`
        Number of coefficients in x and y.
    smoothing : `float`
        Weight of the second-difference penalty.
    damp : `float`
        Weight of the coefficient-size penalty.

    Returns
    -------
    penalty : `scipy.sparse.csr_matrix`
        Rows to append to the (weighted) design matrix, with zero targets.
        Coefficients are ordered with y varying fastest.
    """

    def secondDifference(num: int) -> scipy.sparse.csr_matrix:
        """Second-difference operator for a vector of length ``num``"""
        if num < 3:
            return scipy.sparse.csr_matrix((0, num))
        return scipy.sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(num - 2, num))

    blocks = []
    if smoothing > 0:
        root = np.sqrt(smoothing)
        blocks.append(root * scipy.sparse.kron(secondDifference(numX), scipy.sparse.identity(numY)))
        blocks.append(root * scipy.sparse.kron(scipy.sparse.identity(numX), secondDifference(numY)))
    if damp > 0:
        blocks.append(np.sqrt(damp) * scipy.sparse.identity(numX * numY))
    if not blocks:
        return scipy.sparse.csr_matrix((0, numX * numY))
    return scipy.sparse.vstack(blocks).tocsr()


def subtractGroupMedians(groups: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Subtract the median of the values in each group

    Parameters
    ----------
    groups : `numpy.ndarray`
        Group identifier for each value.
    values : `numpy.ndarray` of `float`
        Values.

    Returns
    -------
    result : `numpy.ndarray` of `float`
        Values with the median of each group subtracted.
    """
    result = values.astype(float).copy()
    order = np.argsort(groups, kind="stable")
    _, start, counts = np.unique(groups[order], return_index=True, return_counts=True)
    for ss, nn in zip(start, counts):
        index = order[ss : ss + nn]
        result[index] -= np.median(result[index])
    return result


def addBSplineDistortions(lhs: Optional[BSplineDistortion], rhs: BSplineDistortion) -> BSplineDistortion:
    """Add two `BSplineDistortion`s with the same regions and knots

    Parameters
    ----------
    lhs : `pfs.drp.stella.BSplineDistortion` or `None`
        First distortion. If `None`, ``rhs`` is returned.
    rhs : `pfs.drp.stella.BSplineDistortion`
        Second distortion.

    Returns
    -------
    sum : `pfs.drp.stella.BSplineDistortion`
        Sum of the distortions.
    """
    if lhs is None:
        return rhs
    if not np.array_equal(lhs.getXBoundaries(), rhs.getXBoundaries()):
        raise RuntimeError("Regions of distortions don't match")

    def addSplines(left: List[BSpline2D], right: List[BSpline2D]) -> List[BSpline2D]:
        result = []
        for ll, rr in zip(left, right):
            if ll.isEmpty():
                result.append(rr.copy())
                continue
            if rr.isEmpty():
                result.append(ll.copy())
                continue
            if (
                ll.getOrder() != rr.getOrder()
                or not np.array_equal(ll.getXKnots(), rr.getXKnots())
                or not np.array_equal(ll.getYKnots(), rr.getYKnots())
            ):
                raise RuntimeError("Knots of splines don't match")
            result.append(
                BSpline2D(
                    ll.getOrder(), ll.getXKnots(), ll.getYKnots(), ll.getCoefficients() + rr.getCoefficients()
                )
            )
        return result

    return BSplineDistortion(
        lhs.getXBoundaries(),
        addSplines(lhs.getXSplines(), rhs.getXSplines()),
        addSplines(lhs.getYSplines(), rhs.getYSplines()),
    )
