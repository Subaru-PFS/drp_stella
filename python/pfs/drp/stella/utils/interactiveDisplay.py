import math
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import DrawEvent, KeyEvent, MouseEvent
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.transforms import Bbox
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

import lsst.afw.image as afwImage
import lsst.afw.display.rgb as afwRgb

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


def _zscaleStretch(array: np.ndarray, nSamples: int, contrast: float) -> Tuple[float, float]:
    """Compute ds9/IRAF-style zscale stretch limits for a 2-D array

    Delegates to `lsst.afw.display.rgb.getZScale`, the same algorithm used
    by ``lsst.display.matplotlib``'s own ``"zscale"`` stretch option.

    Parameters
    ----------
    array : `numpy.ndarray`
        Image to compute the stretch from.
    nSamples : `int`
        Number of pixels to sample when fitting the background.
    contrast : `float`
        Scaling applied to the fitted slope; lower values increase the
        stretch's contrast.

    Returns
    -------
    vmin, vmax : `float`
        Stretch limits.
    """
    image = afwImage.ImageF(np.ascontiguousarray(array, dtype=np.float32))
    z1, z2 = afwRgb.getZScale(image, nSamples, contrast)
    return float(z1), float(z2)


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
      contrast, down decreases it). Live updates during the drag are
      throttled to ``dragUpdateInterval`` and drawn by blitting only the
      affected panels (the colorbar is left showing the value from
      *before* the drag started until it ends, since redrawing it on
      every update is comparatively expensive) -- this keeps dragging
      responsive even over a slow connection (e.g., a remote display).
      The final position, and colorbar, is always applied in full once
      the mouse button is released, even if the last few updates during
      the drag itself were skipped by the throttle.
      **Currently disabled by default** (``dragStretchEnabled=False``)
      pending further work; pass ``dragStretchEnabled=True`` to turn it
      back on. A right click without a drag still removes the nearest
      mark as usual.
    - ``c``: clear all marks.
    - ``0``: reset the stretch to the initial, automatically computed
      values.
    - ``h``: print this help text.

    Any of the above is suppressed while the matplotlib toolbar's Pan or
    Zoom tool is engaged, so as not to conflict with it. That tool is
    normally "sticky" (it stays engaged, intercepting every subsequent
    right-drag for its own pan/zoom-out gesture, until the toolbar button is
    clicked again) -- to avoid a single pan or zoom permanently blocking
    marking/stretch-dragging, we automatically disengage it once its
    gesture completes, so the next right-drag goes back to adjusting the
    stretch. Click the toolbar button again for another pan/zoom.

    Parameters
    ----------
    source, target, convolved : array-like or `lsst.afw.image.Image` or
        `MaskedImage` or `Exposure`
        Images to display; must all have the same shape. ``convolved``
        is typically a PSF-matching result's ``matchedExposure``.
    vmin, vmax : `float`, optional
        Stretch limits for ``source``/``target``/``convolved``. Either
        left as `None` (the default) is set automatically according to
        ``stretchAlgorithm``.
    stretchAlgorithm : `str`
        Algorithm used to compute ``vmin``/``vmax`` automatically, if
        either is `None`: ``"zscale"`` (the default; the classic ds9/IRAF
        algorithm, computed per-image and then combined by taking the
        widest limits of the three) or ``"percentile"`` (see
        ``percentile``, ``symmetric``).
    percentile : `float`
        Percentile used to set ``vmin``/``vmax`` automatically, if
        ``stretchAlgorithm`` is ``"percentile"``.
    symmetric : `bool`
        If ``stretchAlgorithm`` is ``"percentile"``, force the automatic
        ``source``/``target``/``convolved`` stretch to be symmetric about
        zero?
    zscaleSamples : `int`
        Number of pixels to sample, if ``stretchAlgorithm`` is
        ``"zscale"``.
    zscaleContrast : `float`
        Contrast parameter, if ``stretchAlgorithm`` is ``"zscale"``.
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
    dragStretchEnabled : `bool`
        Enable right-click-and-drag live stretch adjustment. Currently
        defaults to `False` (disabled) pending further work; the
        underlying implementation is retained, so this can be re-enabled
        by passing `True`.
    dragUpdateInterval : `float`
        Minimum time (seconds) between live stretch redraws while
        dragging; higher values trade responsiveness-to-the-mouse for
        fewer redraws (helpful over a slow connection). The final
        position is always applied when the drag ends, regardless of
        this throttle.
    interactive : `bool`
        Connect mouse/key event handling at all (marking, drag-stretch,
        and the toolbar guard/auto-disengage logic)? Set `False` for a
        plain static plot -- e.g., if the matplotlib toolbar's own
        pan/zoom is misbehaving, since that is caused by our handlers'
        interaction with it (see Notes).
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

    With ``interactive=True`` (the default), the toolbar's own Pan/Zoom
    tools are auto-disengaged once their gesture completes (see
    ``_disengageToolbar``), so that a single pan or zoom doesn't
    permanently block our own marking/stretch handling -- but this
    means the toolbar button must be clicked again for *each* pan/zoom,
    which can read as "zoom only works once". Pass ``interactive=False``
    to skip connecting any of our handlers, leaving the toolbar's
    pan/zoom to behave exactly as it would on a plain matplotlib figure.
    """

    def __init__(
        self,
        source: ImageLike,
        target: ImageLike,
        convolved: ImageLike,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        stretchAlgorithm: str = "zscale",
        percentile: float = 99.5,
        symmetric: bool = False,
        zscaleSamples: int = 1000,
        zscaleContrast: float = 0.25,
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
        dragStretchEnabled: bool = False,
        dragUpdateInterval: float = 1.0 / 30.0,
        interactive: bool = True,
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
            if stretchAlgorithm == "zscale":
                z1s, z2s = zip(
                    *(
                        _zscaleStretch(arr, zscaleSamples, zscaleContrast)
                        for arr in (sourceArr, targetArr, convolvedArr)
                    )
                )
                autoVmin, autoVmax = min(z1s), max(z2s)
            elif stretchAlgorithm == "percentile":
                autoVmin, autoVmax = _autoStretch(
                    np.concatenate([sourceArr.ravel(), targetArr.ravel(), convolvedArr.ravel()]),
                    percentile,
                    symmetric,
                )
            else:
                raise ValueError(
                    f"Unrecognized stretchAlgorithm: {stretchAlgorithm!r}; "
                    "expected 'zscale' or 'percentile'"
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
        imTarget = axTarget.imshow(targetArr, origin="lower", cmap=cmap, norm=sharedNorm, extent=extent)
        imConvolved = axConvolved.imshow(
            convolvedArr, origin="lower", cmap=cmap, norm=sharedNorm, extent=extent
        )
        imDiff = axDiff.imshow(diffArr, origin="lower", cmap=diffCmap, norm=diffNorm, extent=extent)

        for axis, title in zip((axSource, axTarget, axConvolved, axDiff), titles):
            axis.set_title(title)

        # Attach each colorbar directly beside one specific Axes (via a divider), rather than
        # letting fig.colorbar(..., ax=[...]) steal room from the whole figure for a list of
        # Axes: with our 2x2 layout, the "shared" group spans both columns (it includes
        # axTarget, top-right), so a colorbar auto-placed to the right of that whole group
        # lands over on top of axDiff (bottom-right), which is not part of the group and so
        # never gets shrunk to make room for it.
        sharedCax = make_axes_locatable(axTarget).append_axes("right", size="5%", pad=0.1)
        cbarShared = fig.colorbar(imSource, cax=sharedCax)
        diffCax = make_axes_locatable(axDiff).append_axes("right", size="5%", pad=0.1)
        cbarDiff = fig.colorbar(imDiff, cax=diffCax)

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
        # Axes/images belonging to each stretch group, so a live drag update can redraw (by
        # blitting) only what actually changed, rather than the whole figure.
        self._groupAxes: Dict[str, List[Axes]] = {
            "shared": [axSource, axTarget, axConvolved],
            "diff": [axDiff],
        }
        self._groupImages: Dict[str, List[AxesImage]] = {
            "shared": [imSource, imTarget, imConvolved],
            "diff": [imDiff],
        }

        self._markStyle = dict(marker="+", color="red", markersize=12, markeredgewidth=1.5, linestyle="None")
        if markStyle:
            self._markStyle.update(markStyle)
        self._clickDragThreshold = clickDragThreshold
        self._hitRadiusPx = hitRadiusPx
        self._dragStretchEnabled = dragStretchEnabled
        self._dragUpdateInterval = dragUpdateInterval
        self.marks: List[dict] = []
        self._press: Optional[dict] = None
        self._toolbarGestureActive = False
        self._lastDragUpdate = 0.0
        # Cache of the canvas's rendered pixels, refreshed after every real (non-blitted) draw,
        # so a live drag update can cheaply restore it and paint just the changed panels on top.
        self._background = None

        self._cids: List[int] = []
        if not interactive:
            return

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
            canvas.mpl_connect("draw_event", self._onDraw),
        ]

    def _toolbarActive(self) -> bool:
        """Is the matplotlib navigation toolbar's Pan or Zoom tool engaged?"""
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", ""))

    def _disengageToolbar(self) -> None:
        """Toggle off the toolbar's Pan or Zoom tool, if engaged

        The toolbar's Pan/Zoom tools are "sticky": once toggled on, they stay
        engaged (and so keep intercepting every right-drag for their own
        pan/zoom-out gesture, per matplotlib's own button handling) until the
        user remembers to click the button again. We call this once a
        press/release we deferred to the toolbar has completed, so a single
        pan or zoom does not permanently block our own marking/stretch
        handling.
        """
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is None:
            return
        mode = str(getattr(toolbar, "mode", ""))
        if mode == "zoom rect" and callable(getattr(toolbar, "zoom", None)):
            toolbar.zoom()
        elif mode == "pan/zoom" and callable(getattr(toolbar, "pan", None)):
            toolbar.pan()

    def _onDraw(self, event: DrawEvent) -> None:
        """Cache the rendered canvas, for a live drag update to blit against"""
        self._background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _applyStretch(self, group: str, vmin: float, vmax: float) -> None:
        """Set a stretch group's colormap limits and fully redraw

        Includes the colorbar. Used whenever a redraw isn't performance
        sensitive (a reset, a mark change, or the final position once a
        drag ends) -- see `_blitGroup` for the cheaper alternative used
        for live updates during a drag.
        """
        norm = self._norms[group]
        norm.vmin, norm.vmax = vmin, vmax
        self._colorbars[group].update_normal(self._colorbarMappables[group])
        self.fig.canvas.draw_idle()

    def _blitGroup(self, group: str) -> None:
        """Cheaply redraw one stretch group's current norm by blitting

        Restores the last cached full render and repaints only this
        group's own images (and any marks over them) on top, leaving the
        colorbar showing its previous value -- much cheaper than a full
        redraw, which matters for live updates during a drag over a slow
        connection. Falls back to a full redraw if blitting isn't
        available (e.g., no cached background yet, or an unsupported
        backend).
        """
        canvas = self.fig.canvas
        if self._background is None or not canvas.supports_blit:
            canvas.draw_idle()
            return
        canvas.restore_region(self._background)
        for axis, image in zip(self._groupAxes[group], self._groupImages[group]):
            axis.draw_artist(image)
            for line in axis.lines:
                axis.draw_artist(line)
        canvas.blit(Bbox.union([axis.bbox for axis in self._groupAxes[group]]))

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
        if self._toolbarActive():
            # This press has gone to the toolbar's own Pan/Zoom handling instead of ours;
            # remember to disengage that tool once its gesture completes (see _onRelease).
            self._toolbarGestureActive = True
            return
        if event.inaxes not in self._axisGroup:
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
        if not self._dragStretchEnabled:
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
        group = self._press["group"]
        norm = self._norms[group]
        # Always keep the norm itself exactly in sync with the mouse -- this is essentially
        # free, and guarantees the final value is correct however the throttle below lands.
        norm.vmin, norm.vmax = newCenter - newHalfRange, newCenter + newHalfRange

        # Throttle the (comparatively expensive) redraw: doing one on every single motion
        # event can make dragging feel laggy rather than live, especially over a slow
        # connection (e.g., a remote display).
        now = time.monotonic()
        if now - self._lastDragUpdate < self._dragUpdateInterval:
            return
        self._lastDragUpdate = now
        self._blitGroup(group)

    def _onRelease(self, event: MouseEvent) -> None:
        """Event handler for mouse button release: dispatch clicks"""
        if self._toolbarGestureActive:
            self._toolbarGestureActive = False
            self._disengageToolbar()
            return
        if self._press is None:
            return
        press = self._press
        self._press = None
        if press["dragged"]:
            # norm.vmin/vmax are always kept current by _onMotion, even when the throttle
            # skipped a redraw; do one final full (non-blitted) redraw now so the colorbar
            # catches up to the final stretch.
            norm = self._norms[press["group"]]
            self._applyStretch(press["group"], norm.vmin, norm.vmax)
            return
        if self._toolbarActive() or event.inaxes is not press["axis"]:
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
