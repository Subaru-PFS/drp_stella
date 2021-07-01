import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplColors

import lsst.geom as geom
import lsst.afw.detection as afwDetect
import lsst.afw.display.utils as afwDisplayUtils
from pfs.drp.stella.datamodel.drp import PfsArm
from pfs.datamodel.pfsConfig import FiberStatus, TargetType
from pfs.drp.stella.referenceLine import ReferenceLineStatus


__all__ = ["addPfsCursor", "makeCRMosaic", "showAllSpectraAsImage", "showDetectorMap"]


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
                          fiberIndex=None, fig=None, **kwargs):
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
    fig : `matplotlib.Figure`
       The figure to use; or ``None``
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

    if fig is None:
        fig = plt.figure()
    else:
        fig.clf()

    axs = []
    if lines:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 10], hspace=0.025)
        axs.append(fig.add_subplot(gs[0, 0]))
        axs.append(fig.add_subplot(gs[1, 0], sharex=axs[0]))

        plt.axes(axs[1])
    else:
        gs = fig.add_gridspec(1, 1)
        axs.append(fig.add_subplot(gs[0, 0]))

    ibar = len(spec)//2
    lam0, lam1 = spec.wavelength[ibar][0], spec.wavelength[ibar][-1]

    flux = spec.flux
    fiberId = spec.fiberId
    mask = spec.mask
    wavelength = spec.wavelength

    if fiberIndex is not None and len(fiberIndex) != 0:
        flux = flux[fiberIndex]
        fiberId = fiberId[fiberIndex]
        mask = mask[fiberIndex]
        wavelength = wavelength[fiberIndex]

    imshown = plt.imshow(flux, aspect='auto', origin='lower',
                         extent=(lam0, lam1, -0.5, flux.shape[0] - 1 + 0.5), **kwargs)

    plt.colorbar(imshown)

    def format_coord(x, y):
        col = int(len(spec.wavelength[len(spec)//2])*(x - lam0)/(lam1 - lam0) + 0.5)
        row = int(y + 0.5)

        # \u03BB is $\lambda$
        maskVal = mask[row][col]
        maskDescrip = f"[{' '.join(spec.flags.interpret(maskVal))}]" if maskVal != 0 else ""
        return f"fiberId: {fiberId[row]}  \u03BB: {wavelength[row][col]:8.3f}nm {maskDescrip}"

    ax = plt.gca()
    ax.format_coord = format_coord
    ax.get_cursor_data = lambda ev: None  # disabled

    if not isinstance(spec, PfsArm):
        xlabel = "wavelength (nm)"
        # Only show wavelengths for which we have data; especially interesting
        # if we only merged e.g. b and r
        have_data = np.sum((mask & spec.flags["NO_DATA"]) == 0, axis=0)
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
        for ll in lines:
            lam = ll.wavelength
            if lam0 < lam < lam1:
                lab = ll.description
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

        axes = disp._impl._figure.axes
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


def getCtypeFromReferenceLineDefault(line):
    """Return a colour name given a pfs.drp.stella.ReferenceLine

    line.status == NOT_VISIBLE: GRAY
                   BLEND: BLUE
                   SUSPECT: YELLOW
                   REJECTED: RED
                   BROAD: CYAN
    else: GREEN

    N.b. returning "IGNORE" causes the line to be skipped
    """

    status = line.status
    if status & ReferenceLineStatus.NOT_VISIBLE:
        ctype = 'GRAY'
    elif status & ReferenceLineStatus.BLEND:
        ctype = 'BLUE'
    elif status & ReferenceLineStatus.SUSPECT:
        ctype = 'YELLOW'
    elif status & ReferenceLineStatus.REJECTED:
        ctype = 'RED'
    elif status & ReferenceLineStatus.BROAD:
        ctype = 'CYAN'
    else:
        ctype = 'GREEN'

    return ctype


def showDetectorMap(display, pfsConfig, detMap, width=100, zoom=0, xcen=None, fiberIds=None, lines=None,
                    alpha=1.0, getCtypeFromReferenceLine=getCtypeFromReferenceLineDefault):
    """Plot the detectorMap on a display

    Parameters:
      display: `lsst.afw.display.Display`
         the Display to use
      pfsConfig: `pfs.drp.stella.datamodel.PfsConfig`
         describe the fibers in use
      detMap: `pfs.drp.stella.DetectorMap`
         The layout and wavelength solutions
      width: `int`
         The width (in pixels) of the fibres to label, centered on fiberLines
      zoom: `int`
         Zoom the display by this amount about fiberLines
      xcen: `int`
         Label fibres near this x-value
      fiberIds: `list` of `int`
         Label fibres near this set of fiberIds
      lines: `pfs.drp.stella.referenceLine.ReferenceLineSet`
         Lines to show on the display
      alpha: `float`
         The transparency to use when plotting traces/lines
      getCtypeFromStatus: function returning `str`
        Function called to return the name of a colour, given a
        `pfs.drp.stella.referenceLine.ReferenceLineStatus`.  Return "IGNORE" to ignore the line;
        default pfs.drp.utils.display.getCtypeFromStatusDefault

    If xcen and fiberId are omitted, show all fibres in the pfsConfig
    """

    plt.sca(display._impl._figure.axes[0])
    height = detMap.getBBox().getHeight()
    y = np.arange(0, height)

    SuNSS = TargetType.SUNSS_IMAGING in pfsConfig.targetType

    showAll = False
    if xcen is None:
        if fiberIds is None:
            fiberIds = pfsConfig.fiberId
            showAll = True
        else:
            try:
                fiberIds[0]
            except TypeError:
                fiberIds = [fiberIds]

            xcen = detMap.getXCenter(fiberIds[len(fiberIds)//2], height/2)
    else:
        pass  # xcen is already set

    nFiberShown = 0
    for fid in detMap.fiberId:
        ls = '-'
        if fid in pfsConfig.fiberId:
            ind = pfsConfig.selectFiber(fid)
            imagingFiber = pfsConfig.targetType[ind] == TargetType.SUNSS_IMAGING
            if pfsConfig.fiberStatus[ind] == FiberStatus.BROKENFIBER:
                ls = ':'
                color = 'cyan' if SuNSS and imagingFiber else 'magenta'
            else:
                color = 'green' if SuNSS and imagingFiber else 'red'
        else:
            if SuNSS:
                continue

        ind = pfsConfig.selectFiber(fid)
        imagingFiber = pfsConfig.targetType[ind] == TargetType.SUNSS_IMAGING
        if pfsConfig.fiberStatus[ind] == FiberStatus.BROKENFIBER:
            ls = ':'
            color = 'cyan' if SuNSS and imagingFiber else 'magenta'
        else:
            color = 'green' if SuNSS and imagingFiber else 'red'

        fiberX = detMap.getXCenter(fid, height//2)
        if showAll or np.abs(fiberX - xcen) < width/2:
            fiberX = detMap.getXCenter(fid)
            plt.plot(fiberX[::20], y[::20], ls=ls, alpha=alpha, label=f"{fid}",
                     color=color if showAll else None)
            nFiberShown += 1
    #
    # Plot the position of a set of lines
    #
    if lines:
        stride = len(pfsConfig)//25 + 1 if fiberIds is None else 1

        for ll in lines:
            if fiberIds is None:
                fiberIds = pfsConfig.fiberId

            xy = np.zeros((2, len(fiberIds))) + np.NaN
            for i, fid in enumerate(fiberIds):
                if i%stride != 0 and i != len(pfsConfig) - 1:
                    continue

                xc, yc = detMap.findPoint(fid, ll.wavelength)

                ctype = getCtypeFromReferenceLine(ll)
                if ctype == "IGNORE":
                    continue

                if len(fiberIds) == 1:
                    display.dot('o', xc, yc, ctype=ctype)
                else:
                    xy[0, i] = xc
                    xy[1, i] = yc

            if len(fiberIds) > 1:
                good = np.isfinite(xy[0])
                if sum(good) > 0:
                    plt.plot(xy[0][good], xy[1][good], color=ctype, alpha=alpha)

    if not showAll:
        if nFiberShown > 0:
            plt.legend()
        if zoom > 0:
            display.zoom(zoom, xcen, np.mean(y))


def getIndex(mos, x, y):                # should be a method of lsst.afw.display.utils.Mosaic
    """Get the index for a panel

    Parameters
    ----------
    x : `float`
        The x coordinate of a point in the mosaic
    y : `float`
        The y coordinate of a point in the mosaic
    """

    ix = int(x + 0.5)//(mos.xsize + mos.gutter)
    iy = int(y + 0.5)//(mos.ysize + mos.gutter)

    return ix + iy*mos.nx


def makeCRMosaic(exposure, raw=None, size=31, rGrow=3, maskPlaneName="CR", callback=None, display=None):
    """Return a mosaic of all the cosmic rays found in exposure

    This may be converted to an `lsst.afw.image.MaskedImage` with
       mos.makeImage(display=display)

    This is done for you if you provide ``display``, in which case
    the cursor will also readback the

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
       Exposure containing cosmic rays
    raw : `afw.image.Exposure` (optional)
       If provided, also show the raw (but presumably bias-subtracted)
       image of the same part of the frame
    size : `int`
       Size of the cutouts around each CR
    rGrow : `int`
       How much to grow each CR's Footprint, used to merge fragments
    maskPlaneName : `str` or `list` of `str`
       Mask plane[s] to search for objects (default: "CR")
    callback : ``callable``
       A function to call on each `lsst.afw.detection.Footprint`;
       only include footprints for which ``callback`` returns True
    display : `lsst.afw.display.Display` (optional)
        Display to use

    Returns
    -------
        mosaic : `lsst.afw.display.utils.Mosaic`

    N.b. sets mosaic.xy0[] to the XY0 values for each cutout
    """
    fs = afwDetect.FootprintSet(exposure.mask,
                                afwDetect.Threshold(exposure.mask.getPlaneBitMask(maskPlaneName),
                                                    afwDetect.Threshold.BITMASK))
    isotropic = True
    fs = afwDetect.FootprintSet(fs, rGrow, isotropic)
    footprints = fs.getFootprints()

    mos = afwDisplayUtils.Mosaic(gutter=1 if raw is None else 2, background=np.NaN)
    mos.xy0 = []
    rawGutter = 1
    for foot in footprints:
        if callback is not None and not callback(foot):
            continue

        xc, yc = [int(z) for z in foot.getCentroid()]
        crBBox = geom.BoxI(geom.PointI(xc - size//2, yc - size//2), geom.ExtentI(size, size))
        crBBox.clip(exposure.getBBox())

        if raw:
            rmos = afwDisplayUtils.Mosaic(gutter=rawGutter, background=np.NaN, mode='x')
            rmos.append(raw.maskedImage[crBBox])
            rmos.append(exposure.maskedImage[crBBox])

            mos.append(rmos.makeMosaic(display=None))
        else:
            mos.append(exposure.maskedImage[crBBox])

        mos.xy0.append(crBBox.getBegin())

    def cr_format_coord(x, y):
        "Return coordinates in exposure of CR cutouts"

        x, y = int(x + 0.5), int(y + 0.5)
        ind = getIndex(mos, x, y)
        xy0 = mos.getBBox(ind).getBegin()

        x, y = geom.PointI(x, y) - xy0

        x %= size + rawGutter  # in case raw was provided and we're in the right-hand subpanel

        x += mos.xy0[ind][0]
        y += mos.xy0[ind][1]

        return f"({x:6.1f} {y:6.1f})"

    if display is not None:
        if mos.nImage > 0:
            mos.makeMosaic(display=display)
            display.set_format_coord(cr_format_coord)
        else:
            if callback is None:
                msg = "No cosmic rays were detected"
            else:
                msg = "No CRs satisfy your criteria"

            print(msg)

    return mos
