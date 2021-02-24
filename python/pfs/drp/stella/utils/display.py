import unicodedata
import matplotlib.pyplot as plt

import lsst.geom as geom
from pfs.drp.stella.datamodel.drp import PfsArm

try:
    from lsst.display.matplotlib import AsinhNormalize, AsinhZScaleNormalize, ZScaleNormalize
except ImportError:
    print("Unable to use display_matplotlib stretch functionality")

    AsinhNormalize = None

    
__all__ = ["addPfsCursor", "showAllSpectraAsImage"]

def get_norm(image, algorithm, minval, maxval, **kwargs):
    if minval == "minmax":
        minval = np.nanmin(image)
        maxval = np.nanmax(image)

    if algorithm == "asinh":
        if AsinhNormalize is None:
            raise NotImplementedError("asinh stretches use display_matplotlib")

        if minval == "zscale":
            norm = AsinhZScaleNormalize(image=image, Q=kwargs.get("Q", 8.0))
        else:
            norm = AsinhNormalize(minimum=minval,
                                  dataRange=maxval - minval, Q=kwargs.get("Q", 8.0))
    elif algorithm == "linear":
        if minval == "zscale":
            if AsinhNormalize is None:
                raise NotImplementedError("asinh stretches use display_matplotlib")

            norm = ZScaleNormalize(image=image,
                                   nSamples=kwargs.get("nSamples", 1000),
                                   contrast=kwargs.get("contrast", 0.25))
        else:
            norm = plt.colors.Normalize(minval, maxval)
    else:
        raise RuntimeError("Unsupported stretch algorithm \"%s\"" % algorithm)
        
    return norm


def showAllSpectraAsImage(spec, vmin=None, vmax=None, **kwargs):
    """Plot all the spectra in a pfsArm or pfsMerged object"""
    
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
        
    ibar = len(spec)//2
    lam0, lam1 = spec.wavelength[ibar][0], spec.wavelength[ibar][-1]
        
    imshown = plt.imshow(spec.flux, aspect='auto', origin='lower',
                         extent=(lam0, lam1, 0, len(spec) - 1), **kwargs)
    plt.colorbar(imshown)

    lambda_str = "\u03BB"  # used in cursor display string

    def format_coord(x, y):
        col = int(len(spec.wavelength[len(spec)//2])*(x - lam0)/(lam1 - lam0) + 0.5)
        row = int(y + 0.5)

        return f"fiberId: {spec.fiberId[row]}  {lambda_str}: {spec.wavelength[row][col]:8.3f}nm"

    ax = plt.gca()
    ax.format_coord = format_coord
    ax.get_cursor_data = lambda ev: None  # disabled
    
    if not isinstance(spec, PfsArm):
        xlabel = "wavelength (nm)"
        plt.xlim(360, 1000)
    else:
        xlabel = f"wavelength for fiber INDEX {ibar} (nm)"

    plt.xlabel(xlabel)
    plt.ylabel("fiber INDEX")

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
           ax.format_coord.__doc__ is None or \
            "PFS" not in ax.format_coord.__doc__:

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
    def addPfsCursor(disp, detectorMap=None):
        """Add PFS specific information to an afwDisplay.Display

        You should call this function again if the detectorMap 
        changes (e.g. the arm that you're looking at), e.g.
               addPfsCursor(disp, butler.get("detectorMap", dataId))
        """

        def pfs_format_coord(x, y, detectorMap=detectorMap):
            "PFS addition to display_matplotlib's cursor display"

            if detectorMap is None:
                return ""
            else:
                fid = detectorMap.findFiberId(geom.PointD(x, y))
                return f"FiberId {fid:3}    {detectorMap.findWavelength(fid, y):8.3f}nm"

        disp.set_format_coord(pfs_format_coord)
