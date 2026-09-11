import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import numpy as np

import lsst.afw.image as afwImage

__all__ = ("PsfMatchDiagnostic", "plotPsfMatchDiagnostic")

ImageLike = Union[np.ndarray, afwImage.Image, afwImage.MaskedImage, afwImage.Exposure]

# Backends known not to deliver mouse/key events to a running python process.
_STATIC_BACKENDS = frozenset({"agg", "pdf", "ps", "svg", "cairo", "pgf", "template"})


def _toArrayAndOrigin(image: ImageLike) -> Tuple[np.ndarray, float, float]:
    """Unwrap an image-like object to a plain array and its origin

    Parameters
    ----------
    image : array-like or `lsst.afw.image.Image` or `MaskedImage` or
        `Exposure`
        Image to unwrap.

    Returns
    -------
    array : `numpy.ndarray`
        Image pixel values.
    x0, y0 : `float`
        Origin of ``array`` (0, 0 for a plain array; otherwise the
        image's bounding box minimum).
    """
    if isinstance(image, np.ndarray):
        return image, 0.0, 0.0
    if isinstance(image, afwImage.Exposure):
        image = image.maskedImage
    if isinstance(image, afwImage.MaskedImage):
        image = image.image
    if not isinstance(image, afwImage.Image):
        raise TypeError(
            f"Unrecognized image type: {type(image)!r}; expected ndarray, " "Image, MaskedImage or Exposure"
        )
    bbox = image.getBBox()
    return np.asarray(image.array), float(bbox.getMinX()), float(bbox.getMinY())


def _autoStretch(values: np.ndarray, percentile: float, symmetric: bool) -> Tuple[float, float]:
    """Compute automatic colormap stretch limits

    Parameters
    ----------
    values : `numpy.ndarray`
        Values over which to compute the stretch.
    percentile : `float`
        Percentile used to set the limits.
    symmetric : `bool`
        Force the stretch to be symmetric about zero?

    Returns
    -------
    vmin, vmax : `float`
        Stretch limits.
    """
    finite = values[np.isfinite(values)]
    if symmetric:
        limit = np.percentile(np.abs(finite), percentile)
        return -float(limit), float(limit)
    return float(np.percentile(finite, 100 - percentile)), float(np.percentile(finite, percentile))


class PsfMatchDiagnostic:
    """Interactive 2x2 diagnostic view of a PSF-matching result

    Plots ``source``, ``target``, ``convolved`` (``source`` after
    PSF-matching) and their difference (``target - convolved``), sharing
    one colormap stretch between ``source``/``target``/``convolved`` and a
    separate stretch for the difference. Pan/zoom is linked across all
    four panels (via ``sharex``/``sharey``).

    Unlike `pfs.drp.stella.psfMatch.plotSpatialKernel`, this class returns
    ``self`` rather than a bare ``(fig, axes)`` tuple: it holds live state
    (marks, drag-in-progress bookkeeping) that a caller may want to keep a
    handle on. ``self.fig`` and ``self.axes`` (shape ``(2, 2)``) are
    available for anyone who only wants those.

    Key/mouse bindings (need an event-capable backend; see Notes):

    - Left click (no drag): add a mark, shown in all four panels.
    - Right click (no drag): remove the nearest mark to the click.
    - Right click and drag: adjust the stretch of the panel group under
      the cursor (dragging right brightens, left darkens; up increases
      contrast, down decreases it).
    - ``c``: clear all marks.
    - ``0``: reset the stretch to the initial, automatically computed
      values.
    - ``h``: print this help text.

    Any of the above is suppressed while the matplotlib toolbar's Pan or
    Zoom tool is engaged, so as not to conflict with it.

    Parameters
    ----------
    source, target, convolved : array-like or `lsst.afw.image.Image` or
        `MaskedImage` or `Exposure`
        Images to display; must all have the same shape. ``convolved``
        is typically a PSF-matching result's ``matchedExposure``.
    vmin, vmax : `float`, optional
        Stretch limits for ``source``/``target``/``convolved``. Either
        left as `None` (the default) is set automatically from
        ``percentile`` (and ``symmetric``).
    percentile : `float`
        Percentile used to set ``vmin``/``vmax`` automatically.
    symmetric : `bool`
        Force the automatic ``source``/``target``/``convolved`` stretch
        to be symmetric about zero?
    diffVmin, diffVmax : `float`, optional
        Stretch limits for the difference image; as ``vmin``/``vmax``
        but for the difference.
    diffPercentile : `float`
        As ``percentile``, but for the difference image.
    diffSymmetric : `bool`
        As ``symmetric``, but for the difference image. Defaults to
        `True`, since a residual is naturally zero-centered.
    cmap, diffCmap : `str` or `matplotlib.colors.Colormap`
        Colormaps for the shared and difference stretch groups.
    titles : `tuple` of `str`
        Panel titles, in the order (source, target, convolved,
        difference).
    markStyle : `dict`, optional
        Extra keyword arguments for `matplotlib.axes.Axes.plot`,
        overriding the default mark style.
    clickDragThreshold : `float`
        Cumulative screen-pixel displacement beyond which a mouse-down
        followed by mouse-up is treated as a drag rather than a click.
    hitRadiusPx : `float`
        Maximum screen-pixel distance for a right-click to remove a
        mark.
    figsize : `tuple` of `float`, optional
        Figure size, passed to ``matplotlib.pyplot.subplots``.
    fig : `matplotlib.figure.Figure`, optional
    axes : `numpy.ndarray` of `matplotlib.axes.Axes`, optional
        Existing figure and 2x2 grid of axes to plot into, instead of
        creating new ones. If either is provided, both must be.

    Notes
    -----
    Marking and stretch-dragging require an event-capable backend (e.g.,
    ``%matplotlib widget`` in Jupyter, or a native GUI backend such as
    ``macosx``/``qtagg``/``tkagg`` in a script). Under a static backend
    (e.g., ``inline`` or ``Agg``) the figure still renders correctly, but
    without live interaction.
    """

    def __init__(
        self,
        source: ImageLike,
        target: ImageLike,
        convolved: ImageLike,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        percentile: float = 99.5,
        symmetric: bool = False,
        diffVmin: Optional[float] = None,
        diffVmax: Optional[float] = None,
        diffPercentile: float = 99.5,
        diffSymmetric: bool = True,
        cmap: str = "viridis",
        diffCmap: str = "RdBu_r",
        titles: Tuple[str, str, str, str] = ("source", "target", "convolved", "target - convolved"),
        markStyle: Optional[dict] = None,
        clickDragThreshold: float = 4.0,
        hitRadiusPx: float = 15.0,
        figsize: Optional[Tuple[float, float]] = None,
        fig: Optional[Figure] = None,
        axes: Optional[np.ndarray] = None,
    ):
        sourceArr, x0, y0 = _toArrayAndOrigin(source)
        targetArr, _, _ = _toArrayAndOrigin(target)
        convolvedArr, _, _ = _toArrayAndOrigin(convolved)
        if sourceArr.shape != targetArr.shape or sourceArr.shape != convolvedArr.shape:
            raise ValueError(
                f"source ({sourceArr.shape}), target ({targetArr.shape}) and convolved "
                f"({convolvedArr.shape}) must have the same shape"
            )
        diffArr = targetArr - convolvedArr
        height, width = sourceArr.shape
        extent = (x0, x0 + width, y0, y0 + height)

        if vmin is None or vmax is None:
            autoVmin, autoVmax = _autoStretch(
                np.concatenate([sourceArr.ravel(), targetArr.ravel(), convolvedArr.ravel()]),
                percentile,
                symmetric,
            )
            vmin = autoVmin if vmin is None else vmin
            vmax = autoVmax if vmax is None else vmax
        if diffVmin is None or diffVmax is None:
            autoVmin, autoVmax = _autoStretch(diffArr.ravel(), diffPercentile, diffSymmetric)
            diffVmin = autoVmin if diffVmin is None else diffVmin
            diffVmax = autoVmax if diffVmax is None else diffVmax

        if (fig is None) != (axes is None):
            raise ValueError("fig and axes must be provided together")
        if fig is None:
            fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True, squeeze=False)
        elif axes.shape != (2, 2):
            raise ValueError(f"axes must have shape (2, 2); got {axes.shape}")

        self.fig = fig
        self.axes = axes
        axSource, axTarget = axes[0, 0], axes[0, 1]
        axConvolved, axDiff = axes[1, 0], axes[1, 1]

        sharedNorm = Normalize(vmin=vmin, vmax=vmax)
        diffNorm = Normalize(vmin=diffVmin, vmax=diffVmax)

        imSource = axSource.imshow(sourceArr, origin="lower", cmap=cmap, norm=sharedNorm, extent=extent)
        axTarget.imshow(targetArr, origin="lower", cmap=cmap, norm=sharedNorm, extent=extent)
        axConvolved.imshow(convolvedArr, origin="lower", cmap=cmap, norm=sharedNorm, extent=extent)
        imDiff = axDiff.imshow(diffArr, origin="lower", cmap=diffCmap, norm=diffNorm, extent=extent)

        for axis, title in zip((axSource, axTarget, axConvolved, axDiff), titles):
            axis.set_title(title)

        cbarShared = fig.colorbar(imSource, ax=[axSource, axTarget, axConvolved], shrink=0.8)
        cbarDiff = fig.colorbar(imDiff, ax=[axDiff], shrink=0.8)

        self._norms: Dict[str, Normalize] = {"shared": sharedNorm, "diff": diffNorm}
        self._colorbars: Dict[str, Colorbar] = {"shared": cbarShared, "diff": cbarDiff}
        self._colorbarMappables: Dict[str, AxesImage] = {"shared": imSource, "diff": imDiff}
        self._axisGroup: Dict[Axes, str] = {
            axSource: "shared",
            axTarget: "shared",
            axConvolved: "shared",
            axDiff: "diff",
        }
        self._initRanges: Dict[str, Tuple[float, float]] = {
            "shared": (vmin, vmax),
            "diff": (diffVmin, diffVmax),
        }

        self._markStyle = dict(marker="+", color="red", markersize=12, markeredgewidth=1.5, linestyle="None")
        if markStyle:
            self._markStyle.update(markStyle)
        self._clickDragThreshold = clickDragThreshold
        self._hitRadiusPx = hitRadiusPx
        self.marks: List[dict] = []
        self._press: Optional[dict] = None

        backend = matplotlib.get_backend().lower()
        if backend in _STATIC_BACKENDS or "inline" in backend:
            warnings.warn(
                f"Backend {matplotlib.get_backend()!r} does not deliver mouse/key events: "
                "marking and interactive stretch adjustment will not work.",
                stacklevel=2,
            )

        canvas = fig.canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._onPress),
            canvas.mpl_connect("motion_notify_event", self._onMotion),
            canvas.mpl_connect("button_release_event", self._onRelease),
            canvas.mpl_connect("key_press_event", self._onKey),
        ]

    def _toolbarActive(self) -> bool:
        """Is the matplotlib navigation toolbar's Pan or Zoom tool engaged?"""
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", ""))

    def _applyStretch(self, group: str, vmin: float, vmax: float) -> None:
        """Set a stretch group's colormap limits and redraw"""
        norm = self._norms[group]
        norm.vmin, norm.vmax = vmin, vmax
        self._colorbars[group].update_normal(self._colorbarMappables[group])
        self.fig.canvas.draw_idle()

    def resetStretch(self) -> None:
        """Reset both stretch groups to their initial, automatic values"""
        for group, (vmin, vmax) in self._initRanges.items():
            self._applyStretch(group, vmin, vmax)

    def addMark(self, x: float, y: float) -> None:
        """Add a mark at ``(x, y)`` (data coordinates) to every panel"""
        artists = [axis.plot([x], [y], **self._markStyle)[0] for axis in self.axes.ravel()]
        self.marks.append({"xy": (x, y), "artists": artists})
        self.fig.canvas.draw_idle()

    def removeNearestMark(self, xPixel: float, yPixel: float, axis: Axes) -> None:
        """Remove the mark nearest to a point, if within ``hitRadiusPx``

        Parameters
        ----------
        xPixel, yPixel : `float`
            Point to search from, in canvas display-pixel coordinates
            (e.g., a `matplotlib.backend_bases.MouseEvent`'s ``x``/``y``).
        axis : `matplotlib.axes.Axes`
            Axis whose data-to-display transform is used to locate the
            marks (any of the four panels gives the same answer, since
            they all map to the same canvas).
        """
        if not self.marks:
            return
        distances = [
            math.hypot(*(axis.transData.transform(mark["xy"]) - (xPixel, yPixel))) for mark in self.marks
        ]
        index = int(np.argmin(distances))
        if distances[index] > self._hitRadiusPx:
            return
        for artist in self.marks[index]["artists"]:
            artist.remove()
        del self.marks[index]
        self.fig.canvas.draw_idle()

    def clearMarks(self) -> None:
        """Remove all marks"""
        for mark in self.marks:
            for artist in mark["artists"]:
                artist.remove()
        self.marks = []
        self.fig.canvas.draw_idle()

    def close(self) -> None:
        """Disconnect all event handlers"""
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)
        self._cids = []

    def _onPress(self, event: MouseEvent) -> None:
        """Event handler for mouse button press"""
        if event.inaxes not in self._axisGroup or self._toolbarActive():
            return
        group = self._axisGroup[event.inaxes]
        initVmin, initVmax = self._initRanges[group]
        initHalfRange = 0.5 * (initVmax - initVmin)
        if initHalfRange <= 0:
            initHalfRange = 1.0
        refWidth, refHeight = self.fig.canvas.get_width_height()
        norm = self._norms[group]
        self._press = dict(
            x=event.x,
            y=event.y,
            button=event.button,
            axis=event.inaxes,
            group=group,
            startRange=(norm.vmin, norm.vmax),
            dragged=False,
            refSpan=(max(refWidth, 1), max(refHeight, 1)),
            clampRange=(initHalfRange / 1000.0, initHalfRange * 1000.0),
        )

    def _onMotion(self, event: MouseEvent) -> None:
        """Event handler for mouse motion: live stretch adjustment"""
        if self._press is None or self._toolbarActive() or event.x is None or event.y is None:
            return
        dx = event.x - self._press["x"]
        dy = event.y - self._press["y"]
        if not self._press["dragged"] and math.hypot(dx, dy) > self._clickDragThreshold:
            self._press["dragged"] = True
        if self._press["button"] != 3 or not self._press["dragged"]:
            return

        startVmin, startVmax = self._press["startRange"]
        startCenter = 0.5 * (startVmin + startVmax)
        startHalfRange = 0.5 * (startVmax - startVmin)
        refWidth, refHeight = self._press["refSpan"]
        loLimit, hiLimit = self._press["clampRange"]

        # Dragging right brightens (shifts the center down); dragging up increases contrast
        # (shrinks the half-range). Both are normalized by the figure's screen-pixel size.
        newCenter = startCenter - (dx / refWidth) * (startVmax - startVmin)
        newHalfRange = min(max(startHalfRange * 2.0 ** (-dy / refHeight), loLimit), hiLimit)
        self._applyStretch(self._press["group"], newCenter - newHalfRange, newCenter + newHalfRange)

    def _onRelease(self, event: MouseEvent) -> None:
        """Event handler for mouse button release: dispatch clicks"""
        if self._press is None:
            return
        press = self._press
        self._press = None
        if self._toolbarActive() or press["dragged"] or event.inaxes is not press["axis"]:
            return
        if press["button"] == 1 and event.xdata is not None and event.ydata is not None:
            self.addMark(event.xdata, event.ydata)
        elif press["button"] == 3:
            self.removeNearestMark(event.x, event.y, press["axis"])

    def _onKey(self, event: KeyEvent) -> None:
        """Event handler for key presses"""
        if event.key == "c":
            self.clearMarks()
        elif event.key == "0":
            self.resetStretch()
        elif event.key == "h":
            print(self.__class__.__doc__)


def plotPsfMatchDiagnostic(
    source: ImageLike, target: ImageLike, convolved: ImageLike, **kwargs
) -> PsfMatchDiagnostic:
    """Construct a `PsfMatchDiagnostic` for ``source``, ``target`` and
    ``convolved``

    Convenience wrapper: see `PsfMatchDiagnostic` for details and the full
    list of keyword arguments.

    Parameters
    ----------
    source, target, convolved : array-like or `lsst.afw.image.Image` or
        `MaskedImage` or `Exposure`
        Images to display; must all have the same shape.
    **kwargs
        Additional arguments for `PsfMatchDiagnostic`.

    Returns
    -------
    diagnostic : `PsfMatchDiagnostic`
        The interactive diagnostic plot.
    """
    return PsfMatchDiagnostic(source, target, convolved, **kwargs)
