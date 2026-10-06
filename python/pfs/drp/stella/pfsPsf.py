import pickle
from typing import Optional, Tuple

import numpy as np

import lsst.afw.detection
import lsst.afw.image
import lsst.geom
from lsst.afw.geom.ellipses import Quadrupole
from lsst.afw.typehandling import StorableHelperFactory

from .DetectorMapContinued import DetectorMap
from .hybridPsfModel import HybridPsfModel
from .psfProfiles import ParametricPsfModel
from .psfPixelIntegration import integrateOverPixels
from .psfSpline import PsfSplineBasis, SplineAxisConfig

__all__ = ["PfsPsf"]


def _positionToIndex(position: float) -> Tuple[int, float]:
    """Split a continuous position into an integer pixel index and residual

    Following this codebase's convention (nearest-pixel rounding with
    ties rounding up): ``index = floor(position + 0.5)``, and the residual
    is the (sub-pixel) offset of the true position from that index.

    Parameters
    ----------
    position : `float`
        Continuous position.

    Returns
    -------
    index : `int`
        Nearest integer pixel index.
    residual : `float`
        ``position - index``, in ``[-0.5, 0.5)``.
    """
    index = int(np.floor(position + 0.5))
    return index, position - index


class PfsPsf(lsst.afw.detection.Psf):
    """A PSF model for PFS spectrograph data

    Wraps a `pfs.drp.stella.hybridPsfModel.HybridPsfModel` (the hybrid
    parametric + regularized-spline PSF model of PIPE2D-1823-psf.md,
    ``P = P_param + dP``) as an `lsst.afw.detection.Psf`, so it can be
    attached to an `lsst.afw.image.Exposure` and used by code that expects
    the standard `Psf` interface (``computeImage``, ``computeKernelImage``,
    ``computeShape``, etc.).

    The wrapped model varies smoothly with detector position: evaluating it
    requires knowing which fiber (by index, not ``fiberId``) and detector
    row a given ``(x, y)`` position corresponds to, which this class
    determines from ``detectorMap``.

    Parameters
    ----------
    hybridModel : `pfs.drp.stella.hybridPsfModel.HybridPsfModel`
        Hybrid parametric + spline PSF model, already fit (or with an
        initial guess set).
    detectorMap : `pfs.drp.stella.DetectorMap`
        Mapping between ``(fiberId, wavelength)`` and detector ``(x, y)``,
        used to determine the fiber index and row for a given position, and
        to provide ``getAveragePosition``.
    halfSize : `int`
        Half-size of the PSF stamp to compute, in pixels: images are
        ``2*halfSize + 1`` pixels on a side.
    oversampling : `int`, optional
        Oversampling factor for exact pixel integration of the model (see
        `pfs.drp.stella.psfPixelIntegration.integrateOverPixels`).
    """

    _factory = StorableHelperFactory(__name__, "PfsPsf")

    def __init__(
        self,
        hybridModel: HybridPsfModel,
        detectorMap: DetectorMap,
        halfSize: int,
        oversampling: int = 4,
    ):
        lsst.afw.detection.Psf.__init__(self, isFixed=False)
        self.hybridModel = hybridModel
        self.detectorMap = detectorMap
        self.halfSize = halfSize
        self.oversampling = oversampling

    def __deepcopy__(self, memo=None) -> "PfsPsf":
        """Return an independent copy, with its own (non-aliased) model"""
        hybridModel = _hybridModelFromState(_hybridModelState(self.hybridModel))
        return PfsPsf(hybridModel, self.detectorMap.clone(), self.halfSize, self.oversampling)

    def resized(self, width: int, height: int) -> "PfsPsf":
        """Return a copy with a different stamp size

        The underlying PSF model itself has no notion of image size beyond
        the stamp half-size used to compute its kernel image, so this
        simply rebuilds the wrapper with a new ``halfSize``.

        Parameters
        ----------
        width, height : `int`
            New image dimensions; must be equal and odd (as for any
            symmetric PSF stamp).

        Returns
        -------
        resized : `PfsPsf`
            Copy of ``self`` with the new stamp size.
        """
        if width != height or width % 2 == 0:
            raise ValueError(f"width and height must be equal and odd (got {width}, {height})")
        hybridModel = _hybridModelFromState(_hybridModelState(self.hybridModel))
        return PfsPsf(hybridModel, self.detectorMap.clone(), width // 2, self.oversampling)

    def isPersistable(self) -> bool:
        return True

    def _getPersistenceName(self) -> str:
        return "PfsPsf"

    def _getPythonModule(self) -> str:
        return __name__

    def _write(self) -> bytes:
        state = dict(
            hybridModel=_hybridModelState(self.hybridModel),
            detectorMap=self.detectorMap.toBytes(),
            halfSize=self.halfSize,
            oversampling=self.oversampling,
        )
        return pickle.dumps(state)

    @staticmethod
    def _read(pkl: bytes) -> "PfsPsf":
        state = pickle.loads(pkl)
        hybridModel = _hybridModelFromState(state["hybridModel"])
        detectorMap = DetectorMap.fromBytes(state["detectorMap"])
        return PfsPsf(hybridModel, detectorMap, state["halfSize"], state["oversampling"])

    def getAveragePosition(self) -> lsst.geom.Point2D:
        """Return a representative position: the geometric center of the detector"""
        bbox = self.detectorMap.bbox
        return lsst.geom.Point2D(bbox.getCenterX(), bbox.getCenterY())

    def getPosition(self, fiberId: int, wavelength: float) -> lsst.geom.Point2D:
        """Return the detector position of a given fiber and wavelength

        Parameters
        ----------
        fiberId : `int`
            Fiber identifier.
        wavelength : `float`
            Wavelength (nm).

        Returns
        -------
        position : `lsst.geom.Point2D`
            Detector position.
        """
        return self.detectorMap.findPoint(fiberId, wavelength)

    def _resolvePosition(self, position: lsst.geom.Point2D) -> Tuple[float, float]:
        """Determine the (fiberIndex, row) corresponding to a detector position

        Parameters
        ----------
        position : `lsst.geom.Point2D`
            Detector position.

        Returns
        -------
        fiberIndex : `float`
            0-based index of the nearest fiber within ``self.detectorMap``.
        row : `float`
            Dispersion-direction (y) position, i.e. ``position.getY()``.
        """
        fiberId = self.detectorMap.findFiberId(position)
        fiberIndex = int(np.searchsorted(self.detectorMap.fiberId, fiberId))
        return float(fiberIndex), position.getY()

    def _computeStamp(self, position: lsst.geom.Point2D) -> Tuple[np.ndarray, int, int]:
        """Compute the normalized pixel-integrated PSF image at a position

        Shared by `_doComputeKernelImage` and `_doComputeImage`, which
        differ only in where the resulting stamp is placed (local vs.
        detector frame).

        Parameters
        ----------
        position : `lsst.geom.Point2D`
            Detector position at which to evaluate the PSF.

        Returns
        -------
        stamp : `numpy.ndarray`, shape ``(2*halfSize+1, 2*halfSize+1)``
            Pixel-integrated PSF image, normalized to sum to one.
        indexX, indexY : `int`
            Nearest-integer pixel indices of ``position``, on the detector.
        """
        fiberIndex, row = self._resolvePosition(position)
        indexX, residualX = _positionToIndex(position.getX())
        indexY, residualY = _positionToIndex(position.getY())
        xOffsets = np.arange(-self.halfSize, self.halfSize + 1) - residualX
        yOffsets = np.arange(-self.halfSize, self.halfSize + 1) - residualY

        def evaluate(dx, dy):
            return self.hybridModel.evaluate(dx, dy, fiberIndex, row)

        stamp = integrateOverPixels(evaluate, xOffsets, yOffsets, self.oversampling)
        stamp = stamp / np.sum(stamp)
        return stamp, indexX, indexY

    def _doComputeKernelImage(
        self, position: Optional[lsst.geom.Point2D] = None, color=None
    ) -> lsst.afw.image.Image:
        if position is None:
            position = self.getAveragePosition()
        stamp, _, _ = self._computeStamp(position)
        bbox = lsst.geom.Box2I(
            lsst.geom.Point2I(-self.halfSize, -self.halfSize),
            lsst.geom.Extent2I(2 * self.halfSize + 1, 2 * self.halfSize + 1),
        )
        image = lsst.afw.image.Image(bbox, dtype=np.float64)
        image.array[:] = stamp
        return image

    def _doComputeImage(
        self, position: Optional[lsst.geom.Point2D] = None, color=None
    ) -> lsst.afw.image.Image:
        if position is None:
            position = self.getAveragePosition()
        stamp, indexX, indexY = self._computeStamp(position)
        bbox = lsst.geom.Box2I(
            lsst.geom.Point2I(indexX - self.halfSize, indexY - self.halfSize),
            lsst.geom.Extent2I(2 * self.halfSize + 1, 2 * self.halfSize + 1),
        )
        image = lsst.afw.image.Image(bbox, dtype=np.float64)
        image.array[:] = stamp
        return image

    def _doComputeBBox(self, position: Optional[lsst.geom.Point2D] = None, color=None) -> lsst.geom.Box2I:
        return lsst.geom.Box2I(
            lsst.geom.Point2I(-self.halfSize, -self.halfSize),
            lsst.geom.Extent2I(2 * self.halfSize + 1, 2 * self.halfSize + 1),
        )

    def _doComputeShape(self, position: Optional[lsst.geom.Point2D] = None, color=None) -> Quadrupole:
        if position is None:
            position = self.getAveragePosition()
        image = self.computeKernelImage(position)
        bbox = image.getBBox()
        xGrid, yGrid = np.meshgrid(
            np.arange(bbox.minX, bbox.maxX + 1, dtype=float),
            np.arange(bbox.minY, bbox.maxY + 1, dtype=float),
            indexing="xy",
        )
        weight = image.array
        total = np.sum(weight)
        meanX = np.sum(weight * xGrid) / total
        meanY = np.sum(weight * yGrid) / total
        ixx = np.sum(weight * (xGrid - meanX) ** 2) / total
        iyy = np.sum(weight * (yGrid - meanY) ** 2) / total
        ixy = np.sum(weight * (xGrid - meanX) * (yGrid - meanY)) / total
        return Quadrupole(ixx, iyy, ixy)

    def _doComputeApertureFlux(
        self, radius: float, position: Optional[lsst.geom.Point2D] = None, color=None
    ) -> float:
        if position is None:
            position = self.getAveragePosition()
        image = self.computeKernelImage(position)
        bbox = image.getBBox()
        xGrid, yGrid = np.meshgrid(
            np.arange(bbox.minX, bbox.maxX + 1, dtype=float),
            np.arange(bbox.minY, bbox.maxY + 1, dtype=float),
            indexing="xy",
        )
        inside = np.hypot(xGrid, yGrid) <= radius
        return float(np.sum(image.array[inside]))

    def plotKernelImage(self, position: Optional[lsst.geom.Point2D] = None, ax=None, show: bool = True):
        """Plot the local (position-independent-frame) kernel image

        Parameters
        ----------
        position : `lsst.geom.Point2D`, optional
            Detector position at which to evaluate the PSF; defaults to
            `getAveragePosition`.
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

        if position is None:
            position = self.getAveragePosition()
        image = self.computeKernelImage(position)
        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure
        bbox = image.getBBox()
        imagePlot = ax.imshow(
            np.log10(np.clip(image.array, 1.0e-8, None)),
            origin="lower",
            extent=(bbox.minX - 0.5, bbox.maxX + 0.5, bbox.minY - 0.5, bbox.maxY + 0.5),
        )
        ax.set_title(f"PfsPsf kernel image at ({position.getX():.1f}, {position.getY():.1f})")
        figure.colorbar(imagePlot, ax=ax)
        if show:
            plt.show()
        return figure

    def plotProfile(self, position: Optional[lsst.geom.Point2D] = None, ax=None, show: bool = True):
        """Plot the radial profile of the kernel image at a position

        Parameters
        ----------
        position : `lsst.geom.Point2D`, optional
            Detector position at which to evaluate the PSF; defaults to
            `getAveragePosition`.
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

        if position is None:
            position = self.getAveragePosition()
        image = self.computeKernelImage(position)
        bbox = image.getBBox()
        xGrid, yGrid = np.meshgrid(
            np.arange(bbox.minX, bbox.maxX + 1, dtype=float),
            np.arange(bbox.minY, bbox.maxY + 1, dtype=float),
            indexing="xy",
        )
        radius = np.hypot(xGrid, yGrid)
        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure
        ax.semilogy(radius.ravel(), np.clip(image.array, 1.0e-8, None).ravel(), ".", alpha=0.4)
        ax.set_xlabel("radius (pixels)")
        ax.set_ylabel("normalized flux")
        ax.set_title(f"PfsPsf radial profile at ({position.getX():.1f}, {position.getY():.1f})")
        figure.tight_layout()
        if show:
            plt.show()
        return figure


def _parametricModelState(model: ParametricPsfModel) -> dict:
    """Serialize a `ParametricPsfModel` to a picklable state dict"""
    return dict(
        order=model.order,
        wingOrder=model.wingOrder,
        numWings=model.numWings,
        fiberDomain=model.fiberDomain,
        rowDomain=model.rowDomain,
        parameterVector=model.getParameterVector(),
    )


def _parametricModelFromState(state: dict) -> ParametricPsfModel:
    """Reconstruct a `ParametricPsfModel` from a state dict produced by `_parametricModelState`"""
    model = ParametricPsfModel(
        state["order"],
        state["numWings"],
        state["fiberDomain"],
        state["rowDomain"],
        wingOrder=state["wingOrder"],
    )
    model.setParameterVector(state["parameterVector"])
    return model


def _splineBasisState(basis: PsfSplineBasis) -> dict:
    """Serialize a `PsfSplineBasis` to a picklable state dict"""
    return dict(xConfig=basis.xConfig, yConfig=basis.yConfig)


def _splineBasisFromState(state: dict) -> PsfSplineBasis:
    """Reconstruct a `PsfSplineBasis` from a state dict produced by `_splineBasisState`"""
    return PsfSplineBasis(state["xConfig"], state["yConfig"])


def _hybridModelState(model: HybridPsfModel) -> dict:
    """Serialize a `HybridPsfModel` to a picklable state dict

    `NormalizedPolynomial2D` (backing `ChebyshevSurface2D`, wrapped inside
    ``model.parametricModel``) is not itself picklable or deepcopy-able, so
    the model is decomposed into plain parameter vectors and reconstruction
    metadata instead of being pickled directly.
    """
    return dict(
        parametricModel=_parametricModelState(model.parametricModel),
        splineBasis=_splineBasisState(model.splineBasis),
        splineCoefficients=model.splineCoefficients.copy(),
    )


def _hybridModelFromState(state: dict) -> HybridPsfModel:
    """Reconstruct a `HybridPsfModel` from a state dict produced by `_hybridModelState`"""
    parametricModel = _parametricModelFromState(state["parametricModel"])
    splineBasis = _splineBasisFromState(state["splineBasis"])
    model = HybridPsfModel(parametricModel, splineBasis)
    model.setSplineCoefficients(state["splineCoefficients"])
    return model
