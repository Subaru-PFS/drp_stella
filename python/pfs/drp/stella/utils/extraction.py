from functools import partial
from textwrap import dedent
from typing import Callable, Dict, Optional, Tuple

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backend_bases import KeyEvent
import matplotlib.pyplot as plt
import numpy as np

from lsst.afw.image import Exposure
from lsst.afw.display import Display, RED
from lsst.geom import Box2D, Point2D

from .display import showAllSpectraAsImage
from .. import DetectorMap
from pfs.datamodel import PfsFiberArraySet

__all__ = ("ExtractionExplorer",)


class ExtractionExplorer:
    """Interactive exploration of extracted spectra and image

    We display the extracted spectra and the image, and allow the user to pan
    and zoom in each. Hitting <space> in the spectra display will pan the image
    to the corresponding point. Hitting <enter> in either display will pan that
    display to the corresponding point. The user can zoom in and out in each
    display using the "+" and "-" keys. Hitting "q" in either display will quit
    the interactivity.

    Parameters
    ----------
    pfsArm : `PfsFiberArraySet`
        Extracted spectra.
    exposure : `Exposure`
        Image to display.
    detectorMap : `DetectorMap`
        Mapping between fibers and image coordinates.
    title : `str`, optional
        Title for the figures.
    zoom : `float`, optional
        Initial zoom factor for the image.
    smin, smax : `float`, optional
        Minimum and maximum value for the spectra display.
    imin, imax : `float`, optional
        Minimum and maximum value for the image display.
    """
    def __init__(
        self,
        pfsArm: PfsFiberArraySet,
        exposure: Exposure,
        detectorMap: DetectorMap,
        title: Optional[str] = None,
        *,
        zoom: float = 100.0,
        smin: Optional[float] = None,
        smax: Optional[float] = None,
        imin: Optional[float] = None,
        imax: Optional[float] = None,
    ):
        self._pfsArm = pfsArm
        self._exposure = exposure
        self._detectorMap = detectorMap
        self._title = title
        self._zoom = zoom

        # Truncate the Reds colormap to avoid the white end and very dark end.
        # This will make all bad pixels some sort of red, which doesn't clash with the blue/green viridis.
        cmapBad = LinearSegmentedColormap.from_list(
            "truncated_reds", plt.get_cmap("Reds")(np.linspace(0.3, 0.8, 10))
        )
        showAllSpectraAsImage(pfsArm, detectorMap, vmin=smin, vmax=smax, cmap="viridis", cmap_bad=cmapBad)
        self._sFig = plt.gcf()
        self._sAx = self._sFig.gca()
        self._iFig = plt.figure()
        self._display = Display(frame=self._iFig)
        self._display.scale("linear", imin, imax)
        self._display.mtv(exposure)
        self._display.zoom(zoom)
        self._iAx = self._iFig.gca()

        if title:
            self._sFig.suptitle(title)
            self._iFig.suptitle(title)

        self._sConn = None
        self._iConn = None
        self._middle = len(pfsArm)//2
        self._length = pfsArm.length
        self._wl0 = pfsArm.wavelength[self._middle][0]
        self._wl1 = pfsArm.wavelength[self._middle][-1]
        self._dot = []

        self._pFig = plt.figure()
        self._pAx = self._pFig.gca()

        spectraHelp = dedent("""
        Spectra display key bindings:
            q: Quit
            space: Pan image to point and plot spectrum
            enter: Pan spectrum to point
            i: Inspect spectrum of fiber
            +: Zoom in
            -: Zoom out
        """)
        spectraMenu = {
            "q": "spectraQuit",
            " ": "spectraPanImage",
            "i": "spectraPlot",
            "enter": "spectraPan",
            "+": "spectraZoomIn",
            "-": "spectraZoomOut",
        }
        spectraKeyPress = partial(self._keyPress, menu=spectraMenu, helpText=spectraHelp)
        self._sConn = self._sFig.canvas.mpl_connect("key_press_event", spectraKeyPress)

        imageHelp = dedent("""
        Image display key bindings:
            q: Quit
            +: Zoom in
            -: Zoom out
            space: Pan image
        """)
        imageMenu = {
            "q": "imageQuit",
            "+": "imageZoomIn",
            "-": "imageZoomOut",
            " ": "imagePan",
        }
        imageKeyPress = partial(self._keyPress, menu=imageMenu, helpText=imageHelp)
        self._iConn = self._iFig.canvas.mpl_connect("key_press_event", imageKeyPress)

    def _keyPress(self, event: KeyEvent, *, menu: Dict[str, Callable[[KeyEvent], None]], helpText: str):
        """Handle key press events"""
        if event.key == "h" or (command := menu.get(event.key, None)) is None:
            print(helpText)
            return
        getattr(self, command)(event)

    def spectraQuit(self, event: KeyEvent):
        """Event handler to quit the spectra display"""
        self._sFig.canvas.mpl_disconnect(self._sConn)

    def spectraPanImage(self, event: KeyEvent):
        """Event handler to pan the image to the point clicked in the spectra"""
        # Remove any existing dot
        for dot in self._dot:
            dot.remove()
        self._dot = []
        yy = int(self._length*(event.xdata - self._wl0)/(self._wl1 - self._wl0))
        fiberIndex = int(event.ydata + 0.5)
        wavelength = np.interp(yy, np.arange(self._length), self._pfsArm.wavelength[fiberIndex])
        fiberId = self._pfsArm.fiberId[fiberIndex]
        xx, yy = self._detectorMap.findPoint(fiberId, wavelength)
        self._display.pan(xx, yy)
        self._display.dot("x", xx, yy, size=10, ctype=RED)
        self._dot += self._iAx.lines[-2:]  # The "x" is implemented as two lines
        self._iFig.suptitle(f"{self._title}\nfiberId={fiberId}, wavelength={wavelength:.1f}")
        self.spectraPlot(event)

    def getSpectraFiberId(self, y: float) -> int:
        """Return the fiberId corresponding to the y-coordinate in the spectra"""
        return self._pfsArm.fiberId[int(y + 0.5)]

    def getSpectraWavelength(self, x: float, y: float) -> float:
        """Return the wavelength corresponding to the x,y coordinates in the spectra"""
        return self.getSpectraFiberIdWavelength(x, y)[1]

    def getSpectraFiberIdWavelength(self, x: float, y: float) -> Tuple[int, float]:
        """Return the fiberId and wavelength corresponding to the x,y coordinates in the spectra"""
        row = int(self._length*(x - self._wl0)/(self._wl1 - self._wl0))
        fiberIndex = int(y + 0.5)
        wavelength = np.interp(row, np.arange(self._length), self._pfsArm.wavelength[fiberIndex])
        return self._pfsArm.fiberId[fiberIndex], wavelength

    def getSpectraBox(self) -> Box2D:
        """Return a Box2D representing the current view of the spectra"""
        x0, x1 = self._sAx.get_xlim()
        y0, y1 = self._sAx.get_ylim()
        return Box2D(Point2D(x0, y0), Point2D(x1, y1))

    def spectraPan(self, event: KeyEvent):
        """Event handler to pan the spectra"""
        box = self.getSpectraBox()
        self._sAx.set_xlim(event.xdata - 0.5*box.getWidth(), event.xdata + 0.5*box.getWidth())
        self._sAx.set_ylim(event.ydata - 0.5*box.getHeight(), event.ydata + 0.5*box.getHeight())

    def spectraZoom(self, factor: float):
        """Zoom the spectra"""
        box = self.getSpectraBox()
        halfWidth = 0.5*factor*box.getWidth()
        halfHeight = 0.5*factor*box.getHeight()
        self._sAx.set_xlim(box.getCenterX() - halfWidth, box.getCenterX() + halfWidth)
        self._sAx.set_ylim(box.getCenterY() - halfHeight, box.getCenterY() + halfHeight)

    def spectraZoomIn(self, event: KeyEvent):
        """Event handler to zoom in on the spectra"""
        self.spectraZoom(0.5)

    def spectraZoomOut(self, event: KeyEvent):
        """Event handler to zoom out on the spectra"""
        self.spectraZoom(2.0)

    def spectraPlot(self, event: KeyEvent):
        """Event handler to plot the spectrum of the fiber"""
        fiberIndex = int(event.ydata + 0.5)
        fiberId = self.getSpectraFiberId(event.ydata)
        self._pFig.clear()
        self._pAx = self._pFig.gca()
        self._pAx.plot(self._pfsArm.wavelength[fiberIndex], self._pfsArm.flux[fiberIndex], "k-", label="Flux")
        self._pAx.plot(
            self._pfsArm.wavelength[fiberIndex], self._pfsArm.sky[fiberIndex], "b-", alpha=0.5, label="Sky"
        )
        self._pAx.plot(
            self._pfsArm.wavelength[fiberIndex],
            np.sqrt(self._pfsArm.variance[fiberIndex]),
            "r-",
            alpha=0.5,
            label="Error",
        )
        self._pFig.suptitle(f"{self._title}\nfiberId={fiberId}")
        self._pAx.set_xlabel("Wavelength (nm)")
        self._pAx.set_ylabel("Flux")
        box = self.getSpectraBox()
        wlMin = self.getSpectraWavelength(box.getMinX(), event.ydata)
        wlMax = self.getSpectraWavelength(box.getMaxX(), event.ydata)
        select = (self._pfsArm.wavelength[fiberIndex] >= wlMin)
        select &= (self._pfsArm.wavelength[fiberIndex] <= wlMax)
        fluxMin = np.nanmin(self._pfsArm.flux[fiberIndex][select])
        fluxMax = np.nanmax(self._pfsArm.flux[fiberIndex][select])
        fluxRange = fluxMax - fluxMin
        self._pAx.set_xlim(wlMin, wlMax)
        self._pAx.set_ylim(fluxMin - 0.05*fluxRange, fluxMax + 0.15*fluxRange)

        # Show masks
        mask = self._pfsArm.mask[fiberIndex]
        maskPlanes = self._pfsArm.flags.interpret(np.bitwise_or.reduce(mask[select]))
        for ii, plane in enumerate(maskPlanes):
            choose = select & ((mask & self._pfsArm.flags.get(plane)) != 0)
            yMask = 0.99 - 0.02*ii
            fluxMask = self._pAx.transData.inverted().transform(
                self._pAx.transAxes.transform((0.5, yMask))
            )[1]
            wavelength = self._pfsArm.wavelength[fiberIndex][choose]
            self._pAx.scatter(wavelength, np.full_like(wavelength, fluxMask), marker="o", label=plane)

        self._pAx.legend()

    def imageQuit(self, event: KeyEvent):
        """Event handler to quit the image display"""
        self._iFig.canvas.mpl_disconnect(self._iConn)

    def imageZoom(self, factor: float):
        """Zoom the image"""
        self._zoom *= factor
        self._display.zoom(self._zoom)

    def imageZoomIn(self, event: KeyEvent):
        """Event handler to zoom in on the image"""
        self.imageZoom(2.0)

    def imageZoomOut(self, event: KeyEvent):
        """Event handler to zoom out on the image"""
        self.imageZoom(0.5)

    def imagePan(self, event: KeyEvent):
        """Event handler to pan the image"""
        self._display.pan(event.xdata, event.ydata)
