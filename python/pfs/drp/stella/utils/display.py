import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplColors

import lsst.geom as geom
from pfs.drp.stella.datamodel.drp import PfsArm
from pfs.datamodel.pfsConfig import FiberStatus, TargetType

__all__ = ["addPfsCursor", "showAllSpectraAsImage", "showDetectorMap"]


def get_norm(image, algorithm, minval, maxval, **kwargs):
    if minval == "minmax":
        minval = np.nanmin(image)
        maxval = np.nanmax(image)

    importError = None
    try:
        from lsst.display.matplotlib import AsinhNormalize, AsinhZScaleNormalize, ZScaleNormalize
    except ImportError as exc:
        importError = exc

    if algorithm == "asinh":
        if importError:
            raise NotImplementedError("asinh stretches require display_matplotlib") from importError

        if minval == "zscale":
            norm = AsinhZScaleNormalize(image=image, Q=kwargs.get("Q", 8.0))
        else:
            norm = AsinhNormalize(minimum=minval,
                                  dataRange=maxval - minval, Q=kwargs.get("Q", 8.0))
    elif algorithm == "linear":
        if minval == "zscale":
            if importError:
                raise NotImplementedError("zscale stretches require display_matplotlib") from importError

            norm = ZScaleNormalize(image=image,
                                   nSamples=kwargs.get("nSamples", 1000),
                                   contrast=kwargs.get("contrast", 0.25))
        else:
            norm = mplColors.Normalize(minval, maxval)
    else:
        raise RuntimeError("Unsupported stretch algorithm \"%s\"" % algorithm)

    return norm


def showAllSpectraAsImage(spec, vmin=None, vmax=None, lines=None, labelLines=True,
                          fiberIndex=None, **kwargs):
    """Plot all the spectra in a pfsArm or pfsMerged object

    spec : `pfsArm` or `pfsMerged` or `pfsObject`
       set of spectra
    vmin : `float` or None
       minimum value to display
    vmax : `float` or None
       maximum value to display
    lines : `list` of `pfs.drp.stella.ReferenceLine`
       list of lines to display, as returned by `pfs.drp.stella.readLineListFile`
    labelLines : `bool`
       Draw a panel identifying the species in lines
    fiberIndex : `list` of `int`
       Only show this set of fibres; these are indices into spec, _not_ fiberId
    """

    if kwargs:
        kwargs0 = kwargs
        kwargs = kwargs.copy()
        for k in ["algorithm", "minval", "maxval"]:
            if k in kwargs:
                del kwargs[k]

        norm = get_norm(spec.flux, kwargs0.get("algorithm", "linear"),
                        kwargs0.get("minval", vmin), kwargs0.get("maxval", vmax), **kwargs)
    else:
        norm = None

    if norm is None:
        kwargs = dict(vmin=vmin, vmax=vmax)
    else:
        kwargs = dict(norm=norm)

    if lines is None:
        lines = []

    if lines:
        fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 10], hspace=0.025))
        plt.axes(axs[1])
    else:
        fig, axs = plt.subplots(1, 1)
        axs = [axs]

    ibar = len(spec)//2
    lam0, lam1 = spec.wavelength[ibar][0], spec.wavelength[ibar][-1]

    flux = spec.flux
    fiberId = spec.fiberId
    wavelength = spec.wavelength

    if fiberIndex is not None and len(fiberIndex) != 0:
        flux = spec.flux[fiberIndex]
        fiberId = spec.fiberId[fiberIndex]
        wavelength = spec.wavelength[fiberIndex]

    imshown = plt.imshow(flux, aspect='auto', origin='lower',
                         extent=(lam0, lam1, 0, flux.shape[0] - 1), **kwargs)

    plt.colorbar(imshown)

    def format_coord(x, y):
        col = int(len(spec.wavelength[len(spec)//2])*(x - lam0)/(lam1 - lam0) + 0.5)
        row = int(y + 0.5)

        # \u03BB is $\lambda$
        return f"fiberId: {fiberId[row]}  \u03BB: {wavelength[row][col]:8.3f}nm"

    ax = plt.gca()
    ax.format_coord = format_coord
    ax.get_cursor_data = lambda ev: None  # disabled

    if not isinstance(spec, PfsArm):
        xlabel = "wavelength (nm)"
        # Only show wavelengths for which we have data; especially interesting
        # if we only merged e.g. b and r
        have_data = np.sum((spec.mask & spec.flags["NO_DATA"]) == 0, axis=0)
        ll = np.where(have_data > 0, spec.wavelength[0], np.NaN)
        plt.xlim(np.nanmin(ll), np.nanmax(ll))
    else:
        xlabel = f"approximate wavelength for fiber {spec.fiberId[ibar]} (INDEX {ibar}) (nm)"

    axs[-1].set_xlabel(xlabel)
    plt.ylabel("fiber INDEX")

    if lines:
        plt.axes(axs[0])
        plt.colorbar().remove()  # resize window to match image by adding an invisible colorbar
        plt.yticks(ticks=[], labels=[])

        colors = dict(ArI="cyan", HgI="blue", KrI="peachpuff", NeI="red", XeI="silver",
                      OI="green", NaI="darkorange", OH="magenta")
        labels = {}
        for l in lines:
            lam = l.wavelength
            if lam0 < lam < lam1:
                lab = l.description
                color = colors.get(lab, f"C{len(colors)}" if labelLines else 'black')
                plt.axvline(lam, color=color, label=None if lab in labels else lab, alpha=1)
                colors[lab] = color
                labels[lab] = True

        if labelLines:
            plt.legend(fontsize=8, loc=(1.01, 0.0), ncol=2)


try:
    from lsst.display.matplotlib import DisplayImpl
except ImportError:
    DisplayImpl = None

if not hasattr(DisplayImpl, "set_format_coord"):  # old version of display_matplotlib
    def addPfsCursor(disp, detectorMap=None):
        """Add PFS specific information to an afwDisplay.Display

        Requires that the detectorMap be provided, and must be
        called _after_ mtv (so that the display is actually created).

        Assumes matplotlib.  N.b. this will be easier in the next
        release of display_matplotlib
        """
        disp._impl._detMap = detectorMap

        axes = disp._impl.display.frame.axes
        if len(axes) < 1:
            print("addPfsCursor must be called after display.mtv()")
            return

        ax = axes[0]

        if ax.format_coord is None or \
           ax.format_coord.__doc__ is None or "PFS" not in ax.format_coord.__doc__:

            def pfs_format_coord(x, y, disp_impl=disp._impl,
                                 old_format_coord=ax.format_coord):
                "PFS addition to display_matplotlib's cursor display"
                msg = ""

                detMap = disp._impl._detMap
                if detMap is not None:
                    fid = detMap.findFiberId(geom.PointD(x, y))
                    msg += f"FiberId {fid:3}    {detMap.findWavelength(fid, y):8.3f}nm" + " "

                return msg + old_format_coord(x, y)

            ax.format_coord = pfs_format_coord
else:
    def addPfsCursor(display, detectorMap=None):
        """Add PFS specific information to an afwDisplay.Display display

        Returns the callback function

        display may be None to only return the callback

        You should call this function again if the detectorMap
        changes (e.g. the arm that you're looking at), e.g.
               addPfsCursor(display, butler.get("detectorMap", dataId))
        """

        def pfs_format_coord(x, y, detectorMap=detectorMap):
            "PFS addition to display_matplotlib's cursor display"

            if detectorMap is None:
                return ""
            else:
                fid = detectorMap.findFiberId(geom.PointD(x, y))
                return f"FiberId {fid:3}    {detectorMap.findWavelength(fid, y):8.3f}nm"

        if display is not None:
            display.set_format_coord(pfs_format_coord)

        return pfs_format_coord


def showDetectorMap(display, pfsConfig, detMap, width=100, zoom=0, xc=None, fiberIds=None, lines=None):
    """Plot the detectorMap on a display"""

    plt.sca(display.frame.axes[0])

    height = detMap.getBBox().getHeight()
    y = np.arange(0, height)

    SuNSS = TargetType.SUNSS_IMAGING in pfsConfig.targetType

    showAll = False
    if xc is None:
        if fiberIds is None:
            fiberIds = pfsConfig.fiberId
            showAll = True
        else:
            try:
                fiberIds[0]
            except TypeError:
                fiberIds = [fiberIds]

            xc = detMap.getXCenter(fiberIds[len(fiberIds)//2], height/2)
    else:
        pass  # xc is already set

    nFiberShown = 0
    for fid in detMap.fiberId:
        alpha = 0.5 if showAll else 1.0
        ls = '-'
        if fid in pfsConfig.fiberId:
            ind = pfsConfig.selectFiber(fid)[0]
            imagingFiber = pfsConfig.targetType[ind] == TargetType.SUNSS_IMAGING
            if pfsConfig.fiberStatus[ind] == FiberStatus.BROKENFIBER:
                ls = ':'
                color = 'cyan' if SuNSS and imagingFiber else 'magenta'
            else:
                color = 'green' if SuNSS and imagingFiber else 'red'
        else:
            if SuNSS:
                continue

            color = 'blue'
            ls = '--'

        fiberX = detMap.getXCenter(fid, height//2)
        if showAll or np.abs(fiberX - xc) < width/2:
            fiberX = detMap.getXCenter(fid)
            plt.plot(fiberX[::20], y[::20], ls=ls, alpha=alpha, label=f"{fid}",
                     color=color if showAll else None)
            nFiberShown += 1
    #
    # Plot the position of a set of lines
    #
    if lines:
        stride = 10

        for l in lines:
            if fiberIds is None:
                fiberIds = pfsConfig.fiberId

            xy = np.empty((2, len(fiberIds)))
            for i, fid in enumerate(fiberIds):
                if i%stride != 0 and i != len(pfsConfig) - 1:
                    continue

                xc, yc = detMap.findPoint(fid, l.wavelength)
                ctype = {0: 'GREEN', 1: 'BLACK', 2: 'RED'}.get(l.status, 'BLUE')

                if len(fiberIds) == 1:
                    display.dot('o', xc, yc, ctype=ctype)
                else:
                    xy[0, i] = xc
                    xy[1, i] = yc

            if len(fiberIds) > 1:
                plt.plot(xy[0], xy[1], color=ctype)

    if not showAll:
        if nFiberShown > 0:
            plt.legend()
        if zoom > 0:
            display.zoom(zoom, xc, np.mean(y))
