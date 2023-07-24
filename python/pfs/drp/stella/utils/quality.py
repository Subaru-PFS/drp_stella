import matplotlib.pyplot as plt
import numpy as np
import lsst.daf.persistence as dafPersist

from .stability import addTraceLambdaToArclines

__all__ = ["momentsToABT", "showImageQuality"]


def momentsToABT(ixx, ixy, iyy):
    """Return a, b, theta given an ellipses second central moments
    a: semi-major axis
    b: semi-minor axis
    theta: angle of major axis clockwise from +x axis (radians)
    """
    xx_p_yy = ixx + iyy
    xx_m_yy = ixx - iyy
    tmp = np.sqrt(xx_m_yy*xx_m_yy + 4*ixy*ixy)

    a = np.sqrt(0.5*(xx_p_yy + tmp))
    with np.testing.suppress_warnings() as suppress:
        suppress.filter(RuntimeWarning, "invalid value encountered in sqrt")
        b = np.sqrt(0.5*(xx_p_yy - tmp))
    theta = 0.5*np.arctan2(2*ixy, xx_m_yy)

    return a, b, theta


def showImageQuality(dataIds, showWhisker=False, showFWHM=False,
                     showFWHMHistogram=False, showFluxHistogram=False,
                     assembleSpectrograph=True,
                     logScale=True, gridsize=100,
                     butler=None, alsCache=None, figure=None):
    """
    Make QA plots for image quality

    dataIds: list of dataIds to analyze
    showWhisker: Show a whisker plot [default]
    showFWHM:    Show a 2-D image of the FWHM
    showFWHMHistogram:    Show a histogram of the FWHM
    showFluxHistogram:    Show a histogram of line fluxes
    assembleSpectrograph: If true, and if more than one arm is present, arrange plots as brn columns
    logScale:    Show log histograme [default]
    gridsize     Passed to hexbin (default: 100, the same as matplotlib)
    butler       A butler to read data that isn't in the alsCache
    alsCache     A dict to cache line shape data; returned by this function
    figure       The figure to use; or None

    Typical usage would be something like:

    %----
    import pfs.drp.stella.utils.quality as qa

    try:
        alsCache
    except NameError:
        alsCache = {}

    dataIds = []
    for spectrograph in [1, 3]:
        dataIds.append(dict(visit=97484, arm='n', spectrograph=spectrograph))

    alsCache = qa.showImageQuality(dataIds, alsCache=alsCache, butler=butler)
    %----

    Return:
      alsCache
    """
    if not (showWhisker or showFWHM or showFWHMHistogram or showFluxHistogram):
        showWhisker = True

    visits = sorted(set([dataId["visit"] for dataId in dataIds]))
    spectrographs = sorted(set([dataId["spectrograph"] for dataId in dataIds]))
    arms = set([dataId["arm"] for dataId in dataIds])
    if assembleSpectrograph and len(arms) > 1:

        ny = len(arms)
        nx = len(spectrographs)*len(visits)

        dataIds = []
        for a in "nmrb":
            if a in arms:
                for v in visits:
                    for s in spectrographs:
                        dataIds.append(dict(visit=v, spectrograph=s, arm=a))
        n = len(dataIds)
    else:
        n = len(dataIds)
        ny = int(np.sqrt(n))
        nx = n//ny
        if nx*ny < n:
            ny += 1

    fig, axs = plt.subplots(ny, nx, num=figure, sharex=True, sharey=True, squeeze=False, layout="constrained")
    axs = axs.flatten()

    if alsCache is None:
        alsCache = {}

    for dataId in dataIds:
        dataIdStr = '%(visit)d %(arm)s%(spectrograph)d' % dataId
        if dataIdStr not in alsCache:
            if butler is None:
                raise RuntimeError(f"I'm unable to read data for {dataIdStr} without a butler")

            try:
                detMap = butler.get("detectorMap_used", dataId, visit=0)
            except dafPersist.NoResults:
                detMap = butler.get("detectorMap", dataId)

            alsCache[dataIdStr] = addTraceLambdaToArclines(butler.get('arcLines', dataId), detMap)

    for i, (ax, dataId) in enumerate(zip(axs, dataIds)):
        plt.sca(ax)

        dataIdStr = '%(visit)d %(arm)s%(spectrograph)d' % dataId
        als = alsCache[dataIdStr]

        ll = np.isfinite(als.xx + als.xy + als.yy)

        a, b, theta = momentsToABT(als.xx, als.xy, als.yy)
        r = np.sqrt(a*b)
        fwhm = 2*np.sqrt(2*np.log(2))*r

        if showWhisker or showFWHM:
            q10 = np.nanpercentile(als.flux, [10])[0]

            ll = np.isfinite(als.xx + als.xy + als.yy)
            ll &= fwhm < 8
            ll &= als.flux > q10
            ll &= (als.fiberId % 10) == 0

            if showWhisker:
                imageSize = 4096            # used in estimating scale

                arrowSize = 4
                norm = plt.Normalize(2, 4)
                cmap = plt.colormaps["viridis"]
                C = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

                Q = plt.quiver(als.x[ll], als.y[ll], (fwhm*np.cos(theta))[ll], (fwhm*np.sin(theta))[ll],
                               fwhm[ll], cmap=cmap, norm=norm,
                               headwidth=0, pivot="middle",
                               angles='xy', scale_units='xy', scale=arrowSize*30/imageSize)

                plt.quiverkey(Q, 0.1, 1.025, arrowSize, label=f"{arrowSize:.2g} pixels")
            elif showFWHM:
                norm = plt.Normalize(2.5, 3.5)
                C = plt.hexbin(als.x[ll], als.y[ll], fwhm[ll], norm=norm, gridsize=gridsize)
            else:
                raise RuntimeError("You can't get here")

            # We'll use C when we add a colorbar to the entire figure
            plt.xlim(plt.ylim(-1, 4096))
            plt.xlabel("x (pixels)")
            plt.ylabel("y (pixels)")

            ax.set_aspect(1)
            ax.label_outer()
        else:
            if showFWHMHistogram:
                plt.hist(fwhm, bins=np.linspace(0, 10, 100))
                plt.xlabel(r"FWHM (pix)")
            elif showFluxHistogram:
                q99 = np.nanpercentile(als.flux, [99])[0]
                plt.hist(als.flux, bins=np.linspace(0, q99, 100))
                plt.xlabel("flux")
            else:
                raise RuntimeError("You must want *something* plotted")

            if logScale:
                plt.yscale("log")

        plt.text(0.9, 1.02, dataIdStr, transform=ax.transAxes, ha='right')

    for i in range(n, nx*ny):
        axs[i].set_axis_off()

    if showWhisker or showFWHM:
        # Setting shrink is a black art
        if ny == 1:
            shrink = 0.99 if nx == 1 else 0.95/nx
        elif ny == 2:
            shrink = 0.99 if nx <= 4 else 0.72
        else:
            shrink = 1 if nx <= 2 else 0.93 if nx <= 4 else 0.85

        fig.colorbar(C, shrink=shrink, label="FWHM (pixels)", ax=axs)

    kwargs = {}
    if showWhisker and ny < nx:
        kwargs.update(y=0.76)

    plt.suptitle(f"visit{'s' if len(visits) > 1 else ''} {' '.join([str(v) for v in visits])}", **kwargs)

    return alsCache
