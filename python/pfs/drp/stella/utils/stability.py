import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplColors
from matplotlib.animation import FuncAnimation

from lsst.daf.butler import DatasetNotFoundError

from pfs.drp.stella.utils import addPfsCursor

__all__ = ["addTraceLambdaToArclines", ]


def addTraceLambdaToArclines(als, detectorMap):
    """

    als: `pfs.drp.stella.arcLine.ArcLineSet`
       The object returned by ``butler.get("arcLines", dataId)``
    detectorMap
    """

    als.data["lam"] = detectorMap.findWavelength(als.fiberId, als.y)
    with np.testing.suppress_warnings() as suppress:
        suppress.filter(RuntimeWarning, "divide by zero encountered in true_divide")
        als.data["lamErr"] = als.yErr*als.data.lam/als.y
    als.data["tracePos"] = detectorMap.getXCenter(als.fiberId, als.y)

    als.tracePos = als.data["tracePos"]
    als.lam = als.data["lam"]
    als.lamErr = als.data["lamErr"]

    return als


def modelSlit(fiberId, y, f):
    one = np.ones_like(fiberId)
    A = np.vstack([one, fiberId, y]).T
    c0, c1, c2 = np.linalg.lstsq(A, f, rcond=None)[0]

    return c0, c1, c2


def fitResiduals(fiberId, y, dz, fitType="linear", genericName=False):
    """fitType: "none", None, "linear", "mean", "median" "per fiber"
    """
    fiberIds = sorted(set(fiberId))

    if fitType in (None, "none"):
        fit = 0
        name = "none"
    elif fitType in ("mean", "median"):
        if len(dz) == 0:
            fit = 0
        else:
            fit = np.nanmean(dz) if fitType == "mean" else np.nanmedian(dz)
        if genericName:
            name = fitType
        else:
            name = f"{fit:.3f} (%s)" % fitType
    elif fitType == "linear":
        c0, c1, c2 = modelSlit(fiberId, y, dz)
        c1 *= max(fiberIds)
        c2 *= 4000

        fit = c0 + c1*fiberId/np.max(fiberIds) + c2*y/4000

        if genericName:
            name = "c0 + c1 fiberId/max(fiberIds) + c2 y/4000"
        else:
            name = f"{c0:.3f} + {c1:.3f} fiberId/max(fiberIds) + {c2:.3f} y/4000"
    elif fitType == "per fiber":
        fit = np.empty_like(dz)
        for i, fid in enumerate(fiberIds):
            ind = np.where(fiberId == fid)[0]
            fit[ind] = np.nanmedian(dz[ind])

        name = r"$median_{fiber}$"
    else:
        raise RuntimeError(f"Unknown fit algorithm: {fitType}")

    return fit, name


def assembleTitle(title0, fitName, nFiber, nLine, lamErrMax, minSN):
    title = f"{title0}" + (f"  correction: {fitName}" if fitName else "")

    title += f"  {nFiber} fibers,"
    if lamErrMax > 0:
        title += f" %s < {lamErrMax}" % (r'$\sigma_\lambda$',)
    if minSN > 0:
        title += f" S/N > {minSN}"

    title += f", {nLine} lines"

    return title


def plotArcResiduals(als,
                     detectorMap=None,
                     title="",
                     fitType="mean",
                     plotWavelength=True,
                     byRow=None,        # plot dx and dy against row, not column.  True if byFiberId isn't set
                     byFiberId=None,      # plot dx and dy against fiberId, not column (similar to row)
                     usePixels=True,    # report wavelength residuals in pixels, not nm
                     showChi=False,
                     vmin=-0.3, vmax=0.3,
                     soften=0,          # softening used for chi plots
                     lamErrMax=0,
                     minSN=0,
                     hexBin=False,
                     gridsize=100,
                     linewidths=None,
                     nsigma=0,
                     figure=None):

    """

    fitType: "linear", "mean", "median" "per fiber"
    usePixels: report wavelength residuals in pixels, not nm
    """
    if plotWavelength and usePixels:
        if detectorMap is None:
            raise RuntimeError("You must provide a DetectorMap if usePixels is True")

        indices = len(als.fiberId)//2
        dispersion = detectorMap.getDispersion(als.fiberId[indices], als.wavelength[indices])

    if byRow is None:
        if byFiberId is None:
            byRow = True
        else:
            byRow = False
            byFiberId = True
    elif byRow and byFiberId:
        print("Warning: ignoring byFiberId as byRow is True")

    isQuartz = len(als.data[als.description != 'Trace']) == 0

    ll = (als.flag == False)   # centroider succeeded # noqa E712: als.flag is a numpy array
    if not isQuartz:
        with np.testing.suppress_warnings() as suppress:
            if lamErrMax > 0:
                suppress.filter(RuntimeWarning, "invalid value encountered in less")
                ll = np.logical_and(ll, als.lamErr < lamErrMax)
            else:
                # for some reason als.lam == als.wavelength for some lines with "good" fits.  Traces??
                ll &= (als.lam != als.wavelength) & (als.lamErr > 1e-3)

            if minSN > 0:
                ll = np.logical_and(ll, als.intensityErr < als.intensity/minSN)

    fiberIds = np.array(sorted(set(als.fiberId)))
    nFiber = len(fiberIds)

    fig, axs = plt.subplots(1 if isQuartz else 2, 1,
                            num=figure, squeeze=False, sharex=False, gridspec_kw=dict(hspace=0.2))
    axs = axs.flatten()

    for iax, plotWavelength in enumerate([False] if isQuartz else [True, False]):
        plt.sca(axs[iax])

        if plotWavelength:
            x = als.wavelength
            y = als.lam - als.wavelength
            dy = als.lamErr
            yUnit = "nm"
        else:
            x = als.y if byRow else als.fiberId if byFiberId else als.x
            y = als.tracePos - als.x
            dy = als.xErr
            yUnit = "pixel"

        fit, fitName = fitResiduals(als.fiberId[ll], als.y[ll], y[ll].to_numpy(), fitType=fitType)
        y[ll] -= fit

        if showChi:
            y /= np.hypot(dy, soften)
        elif plotWavelength and usePixels:
            y /= dispersion
            dy /= dispersion
            yUnit = "pixel"

        if sum(ll) == 0:
            xlim = (np.nan, np.nan)
        elif byFiberId:
            xlim = plt.xlim(np.min(als.fiberId) - 1, np.max(als.fiberId) + 1)
        else:
            xlim = plt.xlim(0.95*np.nanmin(x[ll]), 1.05*np.nanmax(x[ll]))

        if usePixels and not showChi:
            ylim = plt.ylim(np.array([vmin, vmax]) + np.nanmedian(y))
        elif plotWavelength:
            ylim = plt.ylim((4 if showChi else 0.03)*(np.array([-1, 1])) + np.nanmedian(y))
        else:
            ylim = plt.ylim(np.nanmin(y), np.nanmax(y))

        if hexBin:
            cmap = plt.matplotlib.cm.get_cmap().copy()
            cmap.set_bad(color='black')
            plt.hexbin(x[ll], y[ll], extent=(xlim[0], xlim[1], ylim[0], ylim[1]), bins='log',
                       gridsize=gridsize, linewidths=linewidths, cmap=cmap)
            plt.axhline(0, color="white", alpha=0.5)
        else:
            if plotWavelength:
                colorvec = als.fiberId
            else:
                colorvec = als.x if byRow else als.y
                colorvec = 255*(colorvec - np.nanmin(colorvec))/(np.nanmax(colorvec) - np.nanmin(colorvec))
                colorvec = np.where(np.isnan(colorvec), 0, colorvec.astype(int))

            color = plt.cm.Spectral(np.linspace(0.0, 1, max(colorvec) + 1))
            plt.scatter(x[ll], y[ll], c=color[colorvec[ll]], marker='o',
                        alpha=max(0.1, 0.9 - 0.35*nFiber/250), edgecolors='none')
            plt.axhline(0, color="black", zorder=-1)

        if sum(ll) == 0:
            rms = np.nan
        else:
            clippedL = ll
            if nsigma > 0:
                for i in range(2):
                    q1, q3 = np.nanpercentile(y[clippedL], [25, 75])
                    rms = 0.741*(q3 - q1)
                    clippedL = np.logical_and(ll, y < nsigma*rms)

            q1, q3 = np.nanpercentile(y[clippedL], [25, 75])
            rms = 0.741*(q3 - q1)

        plt.xlabel("wavelength (nm)" if plotWavelength else ("fiberId" if byFiberId else
                                                             ("row" if byRow else "column") + " (pixel)"))
        plt.ylabel(r"$\chi$" if showChi else
                   f"(measured - true) {f'wavelength ({yUnit})' if plotWavelength else 'position (pixels)'}")

        plt.title(f"rms = {rms:.3f}" +
                  (f" (soften = {soften}{yUnit})" if showChi else yUnit) +
                  (f" (clipped {nsigma} sigma)" if nsigma > 0 else ""), color="red", y=0.90)

    nFiber = len(set(als.fiberId))
    nLine = sum(ll)//nFiber
    title = assembleTitle(title, fitName, nFiber, nLine, lamErrMax, minSN)

    if isQuartz:
        plt.title(f"{axs[0].get_title()}\n{title}", color='black')
    else:
        plt.suptitle(title)

    return fig

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def plotArcResiduals2D(als, detectorMap, title="", fitType="mean",
                       maxCentroidErr=0.1, maxDetectorMapError=1, minSN=0, lamErrMax=0,
                       drawQuiver=True, arrowSize=0.1,
                       vmin=None, vmax=None, percentiles=[25, 75],
                       hexBin=False, gridsize=100, linewidths=None, figure=None):
    """
    fitType: "linear", "mean", "median" "per fiber"
    arrowSize: characteristic arrow length in pixels
    """
    isQuartz = len(als.data[als.description != 'Trace']) == 0  # It's a quartz exposure, not an arc

    indices = len(als.fiberId)//2
    dx = als.tracePos - als.x
    if isQuartz:
        dy = np.zeros_like(dx)
    else:
        dy = (als.lam - als.wavelength)/detectorMap.getDispersion(als.fiberId[indices],
                                                                  als.wavelength[indices])

    with np.testing.suppress_warnings() as suppress:
        suppress.filter(RuntimeWarning, "invalid value encountered in less")
        ll = (als.flag == False)   # centroider succeeded # noqa E712: als.flag is a numpy array
        ll = np.logical_and(ll, als.xErr if isQuartz else np.hypot(als.xErr, als.yErr) < maxCentroidErr)
        ll = np.logical_and(ll, np.hypot(dx, dy) < maxDetectorMapError)
        if minSN > 0:
            ll = np.logical_and(ll, als.intensityErr < als.intensity/minSN)
        if lamErrMax > 0:
            ll = np.logical_and(ll, als.lamErr < lamErrMax)

    indices = len(als.fiberId)//2
    dx = als.tracePos - als.x
    if isQuartz:
        dy = np.zeros_like(dx)
    else:
        dy = (als.lam - als.wavelength)/detectorMap.getDispersion(als.fiberId[indices],
                                                                  als.wavelength[indices])

    for dz in [dx] if isQuartz else [dx, dy]:
        fit, fitName = fitResiduals(als.fiberId[ll], als.y[ll], dz.to_numpy()[ll],
                                    fitType=fitType, genericName=True)
        dz[ll] -= fit

    if drawQuiver:
        fig, ax = plt.subplots(1, 1, num=figure)
        plt.sca(ax)

        Q = plt.quiver(als.x[ll], als.y[ll], dx[ll], dy[ll], alpha=0.5,
                       angles='xy', scale_units='xy', scale=arrowSize*20/detectorMap.getBBox().getHeight())
        plt.quiverkey(Q, 0.1, 1.075, arrowSize, label=f"{arrowSize:.2g} pixels")

        plt.xlabel("x")
        plt.ylabel("y")

        bbox = detectorMap.getBBox()
        plt.xlim(bbox.getMinX(), bbox.getMaxX())
        plt.ylim(bbox.getMinY(), bbox.getMaxY())

        plt.gca().format_coord = addPfsCursor(None, detectorMap)

        plt.gca().set_aspect('equal')
    else:
        fig, axs = plt.subplots(1, 1 if isQuartz else 2,
                                num=figure, sharex=True, sharey=True, squeeze=False,
                                gridspec_kw=dict(wspace=0.0))
        axs = axs.flatten()

        v0, v1 = np.percentile(np.concatenate((dx, dy)), percentiles)
        if vmin is None:
            vmin = v0
        if vmax is None:
            vmax = v1

        pfs_format_coord = addPfsCursor(None, detectorMap)

        for ax, dz in zip(axs, [dx, dy]):
            plt.sca(ax)

            norm = mplColors.Normalize(vmin, vmax)

            if hexBin:
                plt.hexbin(als.x[ll], als.y[ll], dz[ll], norm=norm, gridsize=gridsize)
            else:
                plt.scatter(als.x[ll], als.y[ll], c=dz[ll], s=3, vmin=vmin, vmax=vmax)

            bbox = detectorMap.getBBox()
            plt.xlim(bbox.getMinX(), bbox.getMaxX())
            plt.ylim(bbox.getMinY(), bbox.getMaxY())

            ax.set_aspect('equal')
            ax.set_title(f"{'dx' if ax == axs[0] else 'dy'}, pixels")

            plt.xlabel("x")
            if ax == axs[0]:
                plt.ylabel("y")

            ax.format_coord = pfs_format_coord

        cax = None if isQuartz else fig.add_axes([0.915, 0.243, 0.02, 0.505])
        plt.colorbar(cax=cax)

    nFiber = len(set(als.fiberId))
    nLine = sum(ll)//nFiber
    title = assembleTitle(title, fitName, nFiber, nLine, lamErrMax, minSN)

    if isQuartz:
        plt.title(f"{axs[0].get_title()}\n{title}", color='black')
    else:
        plt.suptitle(title, y=0.92 if drawQuiver else 0.81)

    return fig


class PlotArcLines:
    """Make animations of arcLines data using `matplotlib.animation.FuncAnimation`

    Within a notebook you need _two_ cells to use this:
    # ---
    %%capture

    matplotlib.pyplot.rcParams["animation.html"] = "jshtml"

    pal = PlotArcLines(butler, dataId, visits)
    # ---
    anim = pal.animate()
    anim.save(fileName, writer="ffmpeg", dpi=300)

    anim
    # ---

    You can also render a particular visit with e.g.
    # ---
    pal = PlotArcLines(butler, dataId)
    pal(45865)    # if you passed visits to PlotArcLines you can use an index instead if a visit
    #pal.fig      # needed if you initialised pal in a different, %%captured, cell
    # ---
    """
    def __init__(self, butler, dataId0, visits=[], maxCentroidErr=0.1, maxDetectorMapError=1,
                 fitType=None, getDetectorMap=None, arrowSize=0.1, drawQuiver=True,
                 hexBin=False, gridsize=100, linewidths=None, vmin=None, vmax=None, percentiles=[25, 75]):
        """
        maxCentroidErr : `float`
           maximum allowed error in a line centroid
        maxDetectorMapError :`float`
           maximum allowed error relative to detectorMap.  Applied after any model fitting

        """
        self.butler = butler
        self.dataId0 = dataId0
        self.visits = visits
        self.fitType = fitType
        self.getDetectorMap = getDetectorMap

        self.maxCentroidErr = maxCentroidErr
        self.maxDetectorMapError = maxDetectorMapError

        self.drawQuiver = drawQuiver
        self.arrowSize = arrowSize

        self.hexBin = hexBin
        self.gridsize = gridsize
        self.linewidths = linewidths
        self.vmin = vmin
        self.vmax = vmax
        self.percentiles = percentiles

        if drawQuiver:
            fig, axs = plt.subplots(1, 1, figsize=(4.5, 4.5), squeeze=False)
            axs = axs.flatten()
        else:
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, squeeze=False,
                                    gridspec_kw=dict(wspace=0.0))
            axs = list(axs.flatten()) + [fig.add_subplot(1, 3, 3, position=(0.915, 0.243, 0.02, 0.505))]

        self.fig = fig
        self.axs = axs
        self._pobj = None   # The plotting object (or list of objects) to update
        self._label = None  # the label to update

    def animate(self, repeat=False):
        return FuncAnimation(self.fig, self, init_func=self.init, frames=len(self.visits), repeat=repeat)

    def __call__(self, i):
        """Update the plot.

        i : `int`
           Index into self.visits

           if i < 0 this is the init function, and it should process frame 0;
           if it is >= len(self.visits) it's taken to be a visit number instead
        """

        init = i < 0 or self._pobj is None

        if i < len(self.visits):
            visit = self.visits[i]
        else:
            visit = i

        dataId = self.dataId0.copy()
        dataId.update(visit=visit)

        detectorMap = self.butler.get("detectorMap", dataId) if self.getDetectorMap is None else \
            self.getDetectorMap(self.butler, dataId)
        try:
            als = addTraceLambdaToArclines(self.butler.get('lines', dataId), detectorMap)
        except DatasetNotFoundError as e:
            print(e)
            return self._label, self._pobj
        if len(als.fiberId) == 0:
            return

        md = self.butler.get('raw_md', dataId)

        def figLabel(dataId, fitName):
            return f"{'%(visit)d %(arm)s%(spectrograph)d' % dataId}   " \
                f"{md['DATE-OBS']}T{md['UT'][:-4]}   {fitName}"

        indices = len(als.fiberId)//2
        dy = (als.lam - als.wavelength)/detectorMap.getDispersion(als.fiberId[indices],
                                                                  als.wavelength[indices])
        dx = als.tracePos - als.x

        for i in range(2):
            with np.testing.suppress_warnings() as suppress:
                suppress.filter(RuntimeWarning, "invalid value encountered in less")
                ll = (als.flag == False)   # centroider succeeded # noqa E712: als.flag is a numpy array
                ll = np.logical_and(ll, np.hypot(als.xErr, als.yErr) < self.maxCentroidErr)

        for dz in [dx, dy]:
            fit, fitName = fitResiduals(als.fiberId[ll], als.y[ll], dz[ll],
                                        fitType=self.fitType, genericName=True)
            dz -= fit

        if self.maxDetectorMapError > 0:
            with np.testing.suppress_warnings() as suppress:
                suppress.filter(RuntimeWarning, "invalid value encountered in less")
                ll = np.logical_and(ll, np.hypot(dx, dy) < self.maxDetectorMapError)
        #
        # Ready to start plotting
        #
        if init:
            if self.drawQuiver:
                for ax in self.axs:
                    ax.cla()

            if self.drawQuiver:
                Q = ax.quiver(als.x[ll], als.y[ll], dx[ll], dy[ll], alpha=0.5,
                              angles='xy', scale_units='xy',
                              scale=self.arrowSize*100/detectorMap.getBBox().getHeight())
                self._pobj = [Q]

                ax.quiverkey(Q, 0.9, 1.075 - 0.1, self.arrowSize, label=f"{self.arrowSize:.2g} pixels")

                ax.set_xlabel("x")
                ax.set_ylabel("y")

                bbox = detectorMap.getBBox()
                ax.set_xlim(bbox.getMinX(), bbox.getMaxX())
                ax.set_ylim(bbox.getMinY(), bbox.getMaxY())
                ax.set_aspect('equal')
            else:
                if self.vmin is None or self.vmax is None:
                    v0, v1 = np.percentile(np.concatenate((dx, dy)), self.percentiles)
                if self.vmin is None:
                    self.vmin = v0
                if self.vmax is None:
                    self.vmax = v1

                self._pobj = []
                for ax, dz in zip(self.axs, [dx, dy]):
                    if self.hexBin:
                        norm = mplColors.Normalize(self.vmin, self.vmax)

                        self.norm = norm
                        self._pobj.append(ax.hexbin(als.x[ll], als.y[ll], dz[ll], norm=self.norm,
                                                    gridsize=self.gridsize, linewidths=0.2))
                    else:
                        self._pobj.append(ax.scatter(als.x[ll], als.y[ll], c=dz[ll], s=3,
                                                     vmin=self.vmin, vmax=self.vmax))

                    bbox = detectorMap.getBBox()
                    ax.set_xlim(bbox.getMinX(), bbox.getMaxX())
                    ax.set_ylim(bbox.getMinY(), bbox.getMaxY())

                    ax.set_aspect('equal')
                    ax.set_title(f"{'dx' if ax == self.axs[0] else 'dy'}, pixels")

                    ax.set_xlabel("x")
                    if ax == self.axs[0]:
                        ax.set_ylabel("y")

                ax = self.axs[-1]
                self.fig.colorbar(self._pobj[0], cax=ax)

            for ax in self.axs:
                ax.format_coord = addPfsCursor(None, detectorMap)

            self._label = self.fig.suptitle(figLabel(dataId, fitName), y=0.95 if self.drawQuiver else 0.82)
        else:
            if self.drawQuiver:
                Q = self._pobj[0]
                Q.set_offsets(np.c_[als.x[ll], als.y[ll]])

                def set_XY(Q, x, y):
                    Q.X = x
                    Q.Y = y
                    Q.XY = np.column_stack((x, y))
                    Q.N = len(x)

                set_XY(Q, als.x[ll], als.y[ll])
                Q.set_UVC(dx[ll], dy[ll])
            elif self.hexBin:
                for ax, pc, dz in zip(self.axs, self._pobj, [dx, dy]):
                    nhexbin = ax.hexbin(als.x[ll], als.y[ll], dz[ll], norm=self.norm,
                                        gridsize=self.gridsize, linewidths=self.linewidths)
                    pc.update_from(nhexbin)
            else:
                for scat, dz in zip(self._pobj, [dx[ll], dy[ll]]):
                    scat.set_offsets(np.c_[als.x[ll], als.y[ll]])
                    scat.set_color(scat.cmap(scat.norm(dz)))

            self._label.set_text(figLabel(dataId, fitName))

        return [_ for _ in [self._label] + self._pobj]

    def init(self):
        self.__call__(-1)
