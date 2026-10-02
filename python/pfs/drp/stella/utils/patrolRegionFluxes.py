import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from pfs.datamodel import FiberStatus, TargetType
from pfs.utils import OneCobra
from .sysUtils import calculateNxNy
from .display import addFiberCursor
from .quality import opaqueColorbar
from pfs.drp.stella.telecentricity import twoCircleOverlapOnSphere, getFiberAngle

__all__ = ["plotFluxAsXY", "drawMtpBoundaries", "addPatrolRegionCursor", "plotFluxAsThetaPhi",
           "plotFluxAsThetaPlusPhi", "plotFluxHistogramsByCobra", "plotFluxHistogramsByExposure",
           "plotSkyNormsAsXY"]

penumbra = 1.65                         # size of penumbra as a multiple of black spot


def readSkyNorms(butler, visits, pfi, gfm, readSpectra=False, arm='r', rawFluxes={}, verbose=True):
    nVisit = len(visits)

    for i, visit in enumerate(visits):
        if verbose:
            print(f"Reading {visit}  ({i+1}/{len(visits)})", end='\r', flush=True)

        sn = butler.get("skyNorms", visit=visit)
        pfsConfig = butler.get("pfsConfig", visit=visit).select(targetType=~TargetType.ENGINEERING)
        spec = butler.get("pfsMerged", visit=visit) if readSpectra else None

        if i == 0:
            nFiber = len(pfsConfig)

            norms = np.empty((len(visits), nFiber))

            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            pfsConfigs = []

            allCobras = pfi.getAllDefinedCobras()
            zc = pfi.calibModel.centers
            #
            # Sort into fiberId order to match pfsConfig
            #
            cids = gfm.fiberIdToCobraId(pfsConfig.fiberId)
            allCobras = np.array(allCobras)[cids - 1]
            zc = zc[cids - 1]

            rawFluxes[arm] = np.empty((nVisit, nFiber))
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        sns = sn.eval(pfsConfig.fiberId)
        norms[i] = np.where(sns.masks, np.nan, sns.values).T

        if spec is not None:
            lam0, lamn = (433, 508) if arm == 'b' else (707.1, 790.0)
            lam0 += 20  # avoid problems in the serial correction
            lamn -= 20  # due to windowed data
            rawFluxes[arm][i] = np.nanmean(np.where((spec.wavelength > lam0) &
                                                    (spec.wavelength < lamn) & (spec.mask == 0x0),
                                                    spec.flux, np.nan), axis=1)

        allPos = pfsConfig.pfiCenter.T[0] + 1j*pfsConfig.pfiCenter.T[1]
        theta, phi, flags = pfi.positionsToAngles(allCobras, allPos)
        # Only keep first solution for (theta, phi)
        pfsConfig.cobraTheta = np.rad2deg(np.where((flags & pfi.SOLUTION_OK) == 0x0, np.nan, theta))[:, 0]
        pfsConfig.cobraPhi = np.rad2deg(np.where((flags & pfi.SOLUTION_OK) == 0x0, np.nan, phi))[:, 0]
        del allPos
        del theta
        del phi

        pfsConfigs.append(pfsConfig)

    if verbose:
        print()

    return pfsConfigs, norms


def addPatrolRegionCursor(ax, fid, pfsConfigs, gfm=None, showMTP=False, values=[], dither=(0, 0),
                          thetaPhi=False):
    """Add a cursor to a plot of a PFS fiber positions

    Parameters
    ----------
    ax: `matplotlib.axes.Axes`
        Axes to which to add the cursor.
    fid: `int`
        Desired fiberId
    pfsConfigs : `list(pfs.datamodel.PfsConfig)`
        PFS fiber configurations for each visit
    dither: (`float`, `float`)
       Adjust pfiCenter by uniform distribution U(-dither[0]/2, dither[1]/2) to show multiple exposures
       of same pfsConfig.  Default: 0.0
    """
    from scipy.spatial import KDTree

    nVisit = len(pfsConfigs)
    xy = np.empty((nVisit, 2))
    for i, pfsConfig in enumerate(pfsConfigs):
        if i == 0:
            ind = np.where(pfsConfig.fiberId == fid)[0][0]

        if thetaPhi:
            xy[i][0] = pfsConfig.cobraTheta[ind]
            xy[i][1] = pfsConfig.cobraPhi[ind]
        else:
            xy[i] = pfsConfig.pfiCenter[ind].T

    if np.sum(dither) > 0:
        np.random.seed(10000 + fid)     # we need to be consistent with plotFluxAsXY()

        xy[:, 0] += np.random.uniform(-dither[0]/2, dither[0]/2, nVisit)
        xy[:, 1] += np.random.uniform(-dither[1]/2, dither[1]/2, nVisit)

    xy = np.where(np.isfinite(xy), xy, -1000).reshape(xy.shape)
    tree = KDTree(xy)

    def format_coord(x, y):
        """Format the cursor position

        Parameters
        ----------
        x : `float`
            X-coordinate of the cursor.
        y : `float`
            Y-coordinate of the cursor.

        Returns
        -------
        coord : `str`
            Formatted cursor position.
        """
        _, kd_index = tree.query([x, y])
        pfsConfig = pfsConfigs[kd_index]
        ind = np.where(pfsConfig.fiberId == fid)[0][0]

        string = f"x={x:.3f}, y={y:.3f}: visits[{kd_index}]={pfsConfig.visit} "

        fiberStatus = pfsConfig.fiberStatus[ind]
        if fiberStatus > 0:
            string += f" {FiberStatus(fiberStatus)}"
        if gfm is not None and showMTP:
            string += f" {gfm.fiberIdToMTP([fid], pfsConfig)[0][0]}"
        for v in values:
            string += f" {v[kd_index][ind]:.2f}"

        return string

    ax.format_coord = format_coord


def findNeighboringBlackSpots(fid, blackSpots):
    ll = blackSpots.fiberId == fid
    x0 = blackSpots[ll].x.iloc[0]
    y0 = blackSpots[ll].y.iloc[0]

    r = np.hypot(blackSpots.x - x0, blackSpots.y - y0)
    ir = np.argsort(r)

    return blackSpots.fiberId[ir][r[ir] < 10].to_numpy()


def plotFluxAsXY(fids, fluxes, pfi, gfm, pfsConfigs,
                 blackSpots=None, showConstantThetaPlusPhi=True, subtractMedian=False,
                 showPatrolRegion=True, showMTP=False,
                 markerstyle='o', badFiberStatus=(FiberStatus.NOTCONVERGED), dither=0,
                 showInputScans=False, title=None, showRMS=False,
                 fluxLabel=None,
                 vmin=None, vmax=None, cmap="seismic",
                 figure=None, overplotFigure=False):
    r"""Plot the per-fibre flux as a function of (x, y) within the patrol region

    dither:
       Adjust pfiCenter by uniform distribution U(-dither/2, dither/2) to show multiple exposures of same
       pfsConfig
    showConstantThetaPlusPhi: (`bool`)
       Show lines of constant theta + phi;
       if only the phi arm is bent, the fluxes will be constant on these lines
    subtractMedian: (`bool`)
       Subtract the median flux from each patrol region.  N.b. _not_ applied to value of point under mouse
    markerstyle: (`str`)
       Desired MarkerStyle (default: 'o')
    vmin, vmax:
       Minimum and maximum values of percentage flux deviations
       N.b. one or both may be None, in which case the other is used (or a default of +- 3)
    fluxLabel: (`str`)
       Label for colorbar (default if None: r"$\Delta$ flux (%)")
    blackSpots: `pandas.DataFrame`
       As returned by pfs.utils.butler.get('black_dots').  Draw spot and its penumbra
    """

    title = [] if title is None else [title]

    if fluxLabel is None:
        fluxLabel = r"$\Delta$ flux (%)"

    # fluxes can be an ndarray of per-fibre fluxes for each visit (shape (nVisit, nFiber)),
    # or a dict of two such ndarrays.
    if isinstance(fluxes, (dict)) and len(fluxes) == 2:
        fluxDict = fluxes
        names = list(fluxes)
        title.append(f"Symbols: {names[0]}:{names[1]}")
    else:
        fluxDict = dict(only=fluxes)

    dither = (dither, dither)

    zc = pfi.calibModel.centers
    tht0 = pfi.calibModel.tht0

    cids = gfm.fiberIdToCobraId(pfsConfigs[0].fiberId)
    tht0 = tht0[cids - 1]
    zc = zc[cids - 1]

    n = len(fids)
    nx, ny = calculateNxNy(n)

    if overplotFigure and figure is None:
        overplotFigure = False

    if overplotFigure:
        fig = figure
        axs = fig.axes
    else:
        fig, axs = plt.subplots(ny, nx, num=figure, squeeze=False, sharex=False, sharey=False)
        axs = axs.flatten()

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for fid, ax in zip(sorted(fids), axs):
        plt.sca(ax)

        vals = np.array(list(fluxDict.values())).copy()
        if subtractMedian:
            for i, v in enumerate(vals):
                vals[i] = vals[i] - np.nanmedian(v, axis=0)

        addPatrolRegionCursor(ax, fid, pfsConfigs, values=100*vals,
                              gfm=gfm, showMTP=showMTP, dither=dither)

        cids = gfm.fiberIdToCobraId(fid)
        L1 = np.array(pfi.calibModel.L1)[cids - 1]
        L2 = np.array(pfi.calibModel.L2)[cids - 1]

        x, y = [], []
        s = []
        ind = -1
        for i in range(len(pfsConfigs)):
            pfsConfig = pfsConfigs[i]

            if ind < 0:
                ind = np.where(pfsConfig.fiberId == fid)[0][0]
                cen = np.stack([zc.real, zc.imag])[:, ind]

            xy = pfsConfig.pfiCenter[ind]
            x.append(xy[0])
            y.append(xy[1])

            psize = 400/n
            if pfsConfig.fiberStatus[ind] in (badFiberStatus):
                psize /= 2
            s.append(psize)

        s = np.array(s)
        s *= {'h': 1.2, '*': 1.5}.get(markerstyle, 1)

        x = np.array(x)
        y = np.array(y)

        if np.sum(dither) > 0:
            np.random.seed(10000 + fid)  # we need to be consistent with addPatrolRegionCursor()
            x += np.random.uniform(-dither[0]/2, dither[0]/2, len(x))
            y += np.random.uniform(-dither[1]/2, dither[1]/2, len(y))

        for i, fluxes in enumerate(fluxDict.values()):
            from matplotlib.markers import MarkerStyle

            c = 100*(fluxes[:, ind] - (np.nanmedian(fluxes[:, ind], axis=0) if subtractMedian else 0))
            c = c[:len(x)]   # in case we messed with x

            vmin, vmax = setVminVmax(vmin, vmax)

            S = plt.scatter(x, y, c=c, s=s,
                            marker=MarkerStyle(markerstyle, fillstyle='full' if i == 0 else 'right'),
                            edgecolors='black', cmap=cmap, vmin=vmin, vmax=vmax)

        plt.plot(*cen, '+', color='red', zorder=-1)

        rad = 5.60   # some cobras have larger L1/L2
        plt.xlim(cen[0] + rad*np.array([-1, 1]))
        plt.ylim(cen[1] + rad*np.array([-1, 1]))
        plt.gca().set_aspect(1)

        if n > 4:
            plt.tick_params(axis='x', labelbottom=False)
            plt.tick_params(axis='y', labelleft=False)

        plt.text(0.01, 0.99, ("fiberId " if n < 20 else "") + f"{fid}",
                 color='green' if cmap == "seismic" else 'red',
                 ha='left', va='top', transform=ax.transAxes)

        if showRMS:
            q25, q50, q75 = np.nanpercentile(c, [25, 50, 75])
            rms = 0.741*(q75 - q25)

            plt.text(0.99, 0.99, f"RMS {rms:.2f}%", color='green',
                     ha='right', va='top', transform=ax.transAxes)

        # draw black spot and its penumbra
        if blackSpots is not None:
            for _fid in findNeighboringBlackSpots(fid, blackSpots):
                spot = blackSpots[blackSpots.fiberId == _fid].iloc[0]
                c = plt.Circle((spot.x, spot.y), spot.r, color='black', alpha=0.2)
                ax.add_patch(c)
                c = plt.Circle((spot.x, spot.y), penumbra*spot.r, color='black', alpha=0.2)
                ax.add_patch(c)

        if showInputScans:   # show where we think we moved the cobras while taking these data
            a = np.pi*np.linspace(0, 1, 101)

            alpha = 0.1
            lab = "Cobra tracks"
            for r in [1, 3.5]:
                plt.plot(cen[0] + r*np.cos(2*a), cen[1] + r*np.sin(2*a), '-', color='black', alpha=alpha,
                         label=lab, zorder=-1)
                lab = None

            if False:  # show the phi scans too
                r = 2.4
                for a0 in np.deg2rad(60*np.arange(6)) - tht0[ind]:
                    plt.plot(cen[0] + r*np.cos(a0) - r*np.cos(a + a0),
                             cen[1] + r*np.sin(a0) - r*np.sin(a + a0), '-',
                             color='black', alpha=alpha, zorder=-1)

            if ax == axs[0]:
                ax.legend(loc=(0.01, 1.05))

        # plot lines of constant theta + phi == tht0 + n pi/6)
        if showConstantThetaPlusPhi:
            a = np.pi*np.linspace(0, 1, 101)

            alpha = 0.2

            L1, L2 = 2*[np.mean([L1, L2])]
            for i, theta0 in enumerate(np.deg2rad(60*np.arange(6)) + tht0[ind]):
                t = theta0 - a
                plt.plot(cen[0] + L1*np.cos(t) - L2*np.cos(t + a),
                         cen[1] + L1*np.sin(t) - L2*np.sin(t + a), '-', color='green',
                         alpha=alpha*(3 if i == 0 else 1), zorder=-1,
                         label=(r"$\theta + \phi = \theta_0 + n\,\pi/6$" if
                                i == 0 and not overplotFigure else None))

            if ax == axs[0]:
                ax.legend(loc=(0.01, 1.05))

        if showPatrolRegion:
            c = plt.Circle(cen, L1 + L2, color='blue', alpha=0.05, zorder=-1)
            ax.add_patch(c)

    for ax in axs[n:]:
        ax.set_visible(False)

    with opaqueColorbar(S):
        plt.colorbar(S, ax=axs, label=fluxLabel)

    fig.supxlabel("PFI x (mm)")
    fig.supylabel("PFI y (mm)")

    if title:
        plt.suptitle(" ".join(title))


def plotFluxAsThetaPhi(fids, fluxes, theta, phi, pfsConfigs,
                       showConstantThetaPlusPhi=True, subtractMedian=False, blackSpots=None, gfm=None,
                       markerstyle='o', title=None, dither=0, showMTP=False,
                       fluxLabel=None, showRMS=False,
                       vmin=None, vmax=None, cmap="seismic",
                       figure=None, overplotFigure=False):
    r"""Plot the per-fibre flux as a function of (x, y) within the patrol region

    showConstantThetaPlusPhi: (`bool`)
       Show lines of constant theta + phi;
       if only the phi arm is bent, the fluxes will be constant on these lines
    subtractMedian: (`bool`)
       Subtract the median flux from each patrol region
    vmin, vmax:
       Minimum and maximum values of percentage flux deviations
       N.b. one or both may be None, in which case the other is used (or a default of +- 3)
    fluxLabel: (`str`)
       Label for colorbar (default if None: r"$\Delta$ flux (%)")
    markerstyle: (`str`)
       Desired MarkerStyle (default: 'o')
    blackSpots: `pandas.DataFrame`
       As returned by pfs.utils.butler.get('black_dots').  If present and a cobra is
       behind the spot or its penumbra, draw point as half the radius of other points
    dither:
       Adjust (theta, phi) by uniform distribution U(-dither/2, 0.5*dither/2) (degrees)
       to show multiple exposures of same pfsConfig
    """

    if fluxLabel is None:
        fluxLabel = r"$\Delta$ flux (%)"

    dither = (dither, dither/2)         # range in theta is twice that in phi

    n = len(fids)
    nx, ny = calculateNxNy(n)

    if overplotFigure:
        fig = figure
        axs = fig.axes
    else:
        fig, axs = plt.subplots(ny, nx, num=figure, squeeze=False, sharex=True, sharey=True)
        axs = axs.flatten()

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for fid, ax in zip(sorted(fids), axs):
        plt.sca(ax)

        pfsConfig = pfsConfigs[0]
        ind = np.where(pfsConfig.fiberId == fid)[0][0]

        vals = fluxes - (np.nanmedian(fluxes, axis=0) if subtractMedian else 0)

        addPatrolRegionCursor(ax, fid, pfsConfigs, values=[100*vals],
                              gfm=gfm, showMTP=showMTP, dither=dither, thetaPhi=True)

        t = np.fmod(theta[:, ind], 360)
        p = phi[:, ind]

        c = 100*vals[:, ind]
        c = c[:len(t)]

        if False:
            ll = np.isfinite(c)
            if not ll.all():
                t = t[ll]
                p = p[ll]
                c = c[ll]

        s = np.array(len(c)*[400/n])
        s *= {'h': 1.2, '*': 1.5}.get(markerstyle, 1)

        if blackSpots is not None:
            cobra = None if gfm is None else OneCobra(gfm.fiberIdToCobraId(fid) - 1)
            for _fid in findNeighboringBlackSpots(fid, blackSpots):

                spot = blackSpots[blackSpots.fiberId == _fid].iloc[0]

                if cobra is not None:
                    a = np.linspace(0, 2*np.pi, 100)
                    for fac in [1, penumbra]:
                        x, y = spot.x + fac*spot.r*np.cos(a), spot.y + fac*spot.r*np.sin(a)
                        tt, pp, flags = cobra.positionsToAngles(x + 1j*y)
                        tt, pp = tt[:, 0], pp[:, 0]
                        #
                        # Stop part of the black spot wrapping from near 2 pi to near 0 or vice versa
                        if np.mean(tt) > np.pi:
                            tt += np.where(tt > np.pi/2, 0, 2*np.pi)
                        else:
                            tt -= np.where(tt < 3*np.pi/2, 0, 2*np.pi)
                        pp = np.where(pp > 0.999*np.pi, 1.03*np.pi, pp)  # top of spot is weird, move it up

                        plt.fill(np.rad2deg(tt), np.rad2deg(pp), color='black', alpha=0.2)
                        plt.fill(np.rad2deg(tt) - 360, np.rad2deg(pp), color='black', alpha=0.2)
                        plt.fill(np.rad2deg(tt) + 360, np.rad2deg(pp), color='black', alpha=0.2)

            for i in range(len(pfsConfigs)):
                x, y = pfsConfigs[i].select(fiberId=fid).pfiCenter.T
                if np.hypot(x - spot.x, y - spot.y) < penumbra*spot.r:
                    s[i] /= 4

        if np.sum(dither) > 0:
            np.random.seed(10000 + fid)  # we need to be consistent with addPatrolRegionCursor()
            t = t + np.random.uniform(-dither[0]/2, dither[0]/2, len(t))  # not +=; don't make change in place
            p = p + np.random.uniform(-dither[1]/2, dither[1]/2, len(p))

        vmin, vmax = setVminVmax(vmin, vmax)

        if False:
            cmap = plt.get_cmap(cmap).copy()
            cmap.set_bad('green', alpha=0.5)
        S = plt.scatter(t, p, c=c, s=s, edgecolors='black',
                        cmap=cmap, marker=markerstyle, vmin=vmin, vmax=vmax)

        # draw lines of constant theta + phi?
        if showConstantThetaPlusPhi:
            for a in 60*np.arange(-1, 10):
                ax.axline((a, 0), slope=-1, color="black", alpha=0.25, zorder=-1,
                          label=r"$\theta + \phi = n\pi/6$" if a < 0 else None)
            if False:
                plt.legend(loc="upper right")

        if n > 4:
            plt.tick_params(axis='x', labelbottom=False)
            plt.tick_params(axis='y', labelleft=False)

        ypos, va = (1.01, "bottom") if n <= 4 else (0.99, "top")
        plt.text(0.01, ypos, ("fiberId " if n < 20 else "") + f"{fid}", color='green',
                 ha='left', va=va, transform=ax.transAxes)

        if showRMS:
            q25, q50, q75 = np.nanpercentile(c, [25, 50, 75])
            rms = 0.741*(q75 - q25)

            plt.text(0.99, 1.01, f"RMS {rms:.2f}%", color='green',
                     ha='right', va='bottom', transform=ax.transAxes)

    plt.xlim(-5, 365)
    plt.ylim(-5, 185)

    for ax in axs[n:]:
        ax.set_visible(False)

    with opaqueColorbar(S):
        plt.colorbar(S, ax=axs, label=fluxLabel)

    fig.supxlabel(r"$\theta$")
    fig.supylabel(r"$\phi$")

    if title:
        plt.suptitle(title)


def plotFluxAsThetaPlusPhi(fids, fluxes, theta, phi, pfi, gfm, pfsConfigs,
                           blackSpots=False,
                           title=None,
                           scaleSize=1.0,
                           ymin=None, ymax=None,
                           rmin=0, rmax=4,
                           figure=None):
    """Plot the per-fibre flux as a function of theta + phi

    rmin, rmax:
       Minimum and maximum values of radius for the scatter plot (defaults: 0, 4)
    ymin, ymax:
       Minimum and maximum values of percentage flux deviations
       N.b. one or both may be None, in which case the other is used (or a default of +- 3)
    blackSpots: `pandas.DataFrame`
       As returned by pfs.utils.butler.get('black_dots').  If present and a cobra is
       behind the spot or it's penumbra, draw point as half the radius of other points
    """

    zc = pfi.calibModel.centers

    cids = gfm.fiberIdToCobraId(pfsConfigs[0].fiberId)
    zc = zc[cids - 1]

    ii = pfsConfigs[0].selectFiber(fids)  # indices of desired fibres

    n = len(ii)
    nx, ny = calculateNxNy(n)

    fig, axs = plt.subplots(ny, nx, num=figure, squeeze=False, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    axs = axs.flatten()

    for i, ax in zip(ii, axs):
        plt.sca(ax)

        s = len(pfsConfigs)*[scaleSize*200/n]

        r = np.empty(len(s))
        for j, pfsConfig in enumerate(pfsConfigs):
            px, py = pfsConfig.pfiCenter[i].T - np.stack([zc[i].real, zc[i].imag])
            r[j] = np.hypot(px, py)

            if blackSpots is not None:
                spot = blackSpots[blackSpots.fiberId == pfsConfig.fiberId[i]].iloc[0]
                px, py = pfsConfig.pfiCenter[i].T

                if np.hypot(px - spot.x, py - spot.y) < penumbra*spot.r:
                    s[j] /= 4

        S = plt.scatter(theta[:, i] + phi[:, i], 100*(fluxes[:, i] - 1), s=s, c=r, vmin=rmin, vmax=rmax)

        plt.axhline(0, color="black", alpha=0.2, zorder=-1)

        plt.text(0.1, 0.8, ("fiberId " if n < 20 else "") + f"{pfsConfig.fiberId[i]}", transform=ax.transAxes)

    plt.colorbar(S, ax=axs, label="R (mm)")

    fig.supxlabel(r"$\theta + \phi$")
    fig.supylabel(r"$\Delta$ flux (%)")

    plt.xlim(-30, 570)
    ymin, ymax = setVminVmax(ymin, ymax)
    plt.ylim(ymin, ymax)

    if title:
        plt.suptitle(title)


def formatDesignName(designName):
    name = []
    ang = "A"
    for cpt in designName.split('_'):
        if cpt == "scan":
            continue

        if cpt in ['theta', 'phi']:
            ang = cpt
            continue

        mat = re.search(r"^theta(.*)", cpt)
        if mat:
            name.append(rf"$\theta$={mat.group(1)}")
            continue

        mat = re.search(r"^radius(.*)$", cpt)
        if mat:
            name.append(f"R={mat.group(1)}")
            continue

        mat = re.search(r"^angle(.*)$", cpt)
        if mat:
            name.append(rf"$\{ang}$={mat.group(1)}")
            continue

        name.append(cpt)

    return " ".join(name)


def plotFluxHistogramsByCobra(fids, fluxes, pfsConfigs,
                              subtractMedian=False,
                              title=None,
                              modifier=None,
                              vmin=None, vmax=None, nbin=101,
                              density=True,
                              figure=None, overplotFigure=False):
    r"""Plot the per-fibre flux as a function of (x, y) within the patrol region

    showConstantThetaPlusPhi: (`bool`)
       Show lines of constant theta + phi;
       if only the phi arm is bent, the fluxes will be constant on these lines
    subtractMedian: (`bool`)
       Subtract the median flux from each patrol region.  N.b. _not_ applied to value of point under mouse
    vmin, vmax:
       Minimum and maximum values of percentage flux deviations
       N.b. one or both may be None, in which case the other is used (or a default of +- 3)
    modifier: (`str`)
       Modifier for Delta flux (default: "")
    blackSpots: `pandas.DataFrame`
       As returned by pfs.utils.butler.get('black_dots').  Draw spot and its penumbra
    """

    title = [] if title is None else [title]

    n = len(fids)
    nx, ny = calculateNxNy(n)

    if overplotFigure and figure is None:
        overplotFigure = False

    if overplotFigure:
        fig = figure
        axs = fig.axes
    else:
        fig, axs = plt.subplots(ny, nx, num=figure, squeeze=False, sharex=True, sharey=True)
        axs = axs.flatten()

    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    for fid, ax in zip(sorted(fids), axs):
        plt.sca(ax)

        ind = np.where(pfsConfigs[0].fiberId == fid)[0][0]
        y = 100*(fluxes[:, ind] - (np.nanmedian(fluxes[:, ind], axis=0) if subtractMedian else 0))
        vmin, vmax = setVminVmax(vmin, vmax)

        bins = np.linspace(vmin, vmax, nbin)

        q25, q50, q75 = np.nanpercentile(y, [25, 50, 75])
        rms = 0.741*(q75 - q25)

        plt.hist(y - q50, bins, density=density, alpha=0.5, label=f"RMS {rms:.2f}% {modifier}")
        plt.legend(loc="upper right")

        plt.text(0.01, 0.99, ("fiberId " if n < 20 else "") + f"{fid}",
                 ha='left', va='top', transform=ax.transAxes)

    for ax in axs[n:]:
        ax.set_visible(False)

    fig.supxlabel(r"$\Delta$ flux (%)")
    fig.supylabel("N")

    if title:
        plt.suptitle(" ".join(title))


def plotFluxHistogramsByExposure(fluxes, pfsConfigs, bins=None, stacked=False,
                                 radii=[0, 60, 120, 180], alpha=-1, figure=None):
    """Plot histograms of flux variation for a set of exposures
    radii: `list` of `float`
       If not None, list of bin boundaries in PFI mm to split the histograms
    """
    if radii is not None:
        if len(radii) == 0 or (len(radii) == 1 and radii[0] == 0):
            raise RuntimeError(f"Please specify some radii; saw {radii}")

        if radii[0] != 0:
            radii = [0] + list(radii)

    if alpha <= 0:
        alpha = 1 if stacked else np.min([0.5, 3/(1 + len(radii))])

    n = len(pfsConfigs)
    nx, ny = calculateNxNy(n)

    fig, axs = plt.subplots(ny, nx, num=figure, squeeze=False, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=0)
    axs = axs.flatten()

    if bins is None:
        bins = 3*np.linspace(-1, 1, 41)

    for i, ax in zip(range(n), axs):
        plt.sca(ax)

        pfsConfig = pfsConfigs[i]
        px, py = pfsConfig.pfiCenter.T

        dfluxes = 100*(fluxes[i] - 1)
        if radii is None:
            plt.hist(dfluxes, bins=bins, alpha=1)
        else:
            bottom = 0
            for i in range(len(radii)):
                r = np.hypot(px, py)
                if i == 0:
                    ll = r < radii[i+1]
                    lab = f"r < {radii[i+1]}"
                elif i == len(radii) - 1:
                    ll = r >= radii[i]
                    lab = f"r >= {radii[i]}"
                else:
                    ll = (r >= radii[i]) & (r < radii[i + 1])
                    lab = f"{radii[i]} <= r < {radii[i + 1]}"
                bottom = plt.hist(dfluxes[ll], bins=bins, bottom=bottom,
                                  alpha=alpha, density=True, label=lab, zorder=len(radii) - i)[0]

                if not stacked:
                    bottom = 0

            if ax == axs[0]:
                ax.legend(ncol=min([4, len(radii)]), loc=(0.01, 1.05))

        plt.text(0.01, 0.99, f"{formatDesignName(pfsConfig.designName)}",
                 ha='left', va='top', transform=plt.gca().transAxes, zorder=10)

        plt.axvline(0, color='black', zorder=-1, alpha=0.1)

    fig.supxlabel(r"$\Delta$ flux (%)")

    for ax in axs[n:]:
        ax.set_visible(False)


def setVminVmax(vmin, vmax, default=3):
    if vmin is None:
        vmin = -default if vmax is None else -vmax
    if vmax is None:
        vmax = -vmin

    return vmin, vmax


def plotSkyNormsAsXY(fiberId, data, x, y, gfm, pfi,
                     fluxLabel=None,
                     showPatrolRegion=True,
                     blackSpots=None, showBlackSpots=True,
                     scalePatrolRegion=1.0, title=None,
                     vmin=None, vmax=None, cmap="seismic", markerSize=None, alpha=0.5,
                     pfsConfig=None,
                     figure=None):
    r""" XXX

    scalePatrolRegion : float
       Scale down the cobras, so that they don't overlap (0.7 is a useful value)
    vmin, vmax: float, float
       Minimum and maximum values of percentage flux deviations
       N.b. one or both may be None, in which case the other is used (or a default of +- 3)
    markerSize: int
       Value of s for plt.scatter();
       if None (default) guessed as 0.5 or 3 if data is probably skyNorms or quartz
    pfsConfig:
       Used to add a cobra-aware cursor to the plot
    fluxLabel: (`str`)
       Label for colorbar (default if None: r"$\Delta$ flux (%)")
    blackSpots: (`pandas.DataFrame`)
       If non-None, drop data affected by the black spots.
       As read by `pfs.utils.butler.Butler().get('black_dots')`
    showBlackSpots: (`bool`)
       Show the black spots and penumbra if blackSpots is not None; default True)
    """

    title = [] if title is None else [title]

    if fluxLabel is None:
        fluxLabel = r"$\Delta$ flux (%)"

    if scalePatrolRegion == 1.0:
        xx, yy = x, y
    else:
        cids = gfm.fiberIdToCobraId(fiberId)
        zc = pfi.calibModel.centers[cids - 1]
        xc, yc = zc.real, zc.imag

        xx = xc + scalePatrolRegion*(x - xc)
        yy = yc + scalePatrolRegion*(y - yc)

    norms = data
    c = 100*(norms - np.nanmedian(norms))
    vmin, vmax = setVminVmax(vmin, vmax, 4)

    if blackSpots is not None:
        for bs in blackSpots.itertuples():
            c[np.hypot(x - bs.x, y - bs.y) < penumbra*bs.r] = np.nan

    ii = np.random.shuffle(np.arange(xx.size))
    if markerSize is None:
        markerSize = (0.5 if data.shape[0] > 200 else 3)
    S = plt.scatter(xx.flatten()[ii], yy.flatten()[ii], c=c.flatten()[ii],
                    alpha=alpha, vmin=vmin, vmax=vmax, s=markerSize, cmap=cmap)

    with opaqueColorbar(S):
        plt.colorbar(S, label=fluxLabel)

    ax = plt.gca()
    ax.set_aspect(1)

    if showBlackSpots and blackSpots is not None:
        for bs in blackSpots.itertuples():

            x, y = bs.x, bs.y
            r = scalePatrolRegion*bs.r

            if scalePatrolRegion != 1.0:
                zc = pfi.calibModel.centers[gfm.fiberIdToCobraId(bs.fiberId) - 1]
                xc, yc = zc.real, zc.imag

                x = xc + scalePatrolRegion*(x - xc)
                y = yc + scalePatrolRegion*(y - yc)

            ax.add_patch(plt.Circle((x, y), r, color='black', alpha=0.2, zorder=2))
            ax.add_patch(plt.Circle((x, y), penumbra*r, color='black', alpha=0.2, zorder=2))

    plt.xlim(210*np.array([-1, 1]))
    plt.ylim(230*np.array([-1, 1]))

    plt.xlabel("PFI x (mm)")
    plt.ylabel("PFI y (mm)")

    if pfsConfig:
        addFiberCursor(plt.gca(), pfsConfig)

    if scalePatrolRegion != 1.0:
        title.append(f"Cobras scaled by {scalePatrolRegion}")

    if title:
        plt.suptitle(" ".join(title))


def drawMtpBoundaries(pfsConfig, gfm, pfi=None, fids=None, npt=20, scalePatrolRegion=0.65,
                      color=None, showSpectrograph=False,
                      alpha=0.5, zorder=None):
    """Draw the boundaries of the MTPs for the specified fiberIds
    """
    if fids is None:
        fids = pfsConfig.fiberId

    spectrograph = pfsConfig.select(fiberId=fids).spectrograph

    if color is None:
        color = {1: 'green', 2: 'orchid', 3: 'orange', 4: 'magenta'} if showSpectrograph else "black"

    if isinstance(color, str):
        colorDict = {}
        for s in set(spectrograph):
            colorDict[s] = color
    else:
        colorDict = color
        color = list(colorDict.values())[0]

    MTPs = np.array([_[0] for _ in gfm.fiberIdToMTP(fids)])
    cobraId = np.array([int(_[2]) for _ in gfm.fiberIdToMTP(pfsConfig.fiberId)])
    xy = np.array([(x, y) for x, y in gfm.xyForCobraIds(cobraId)])
    patrolR = np.full(len(cobraId), 4.75) if pfi is None else pfi.calibModel.L1 + pfi.calibModel.L2
    a = np.linspace(0, 2*np.pi, npt)

    for s, MTP in set(zip(spectrograph, MTPs)):
        ll = MTPs == MTP
        points = np.reshape(np.array([np.stack([x + R*np.cos(a), y + R*np.sin(a)]).T for
                                      (x, y), R in zip(xy[ll], scalePatrolRegion*patrolR[ll])]).flatten(),
                            (len(a)*np.sum(ll), 2))

        hull = ConvexHull(points)

        x, y = points[hull.vertices, 0], points[hull.vertices, 1]
        plt.plot(np.append(x, x[0]), np.append(y, y[0]), color=colorDict.get(s, color),
                 alpha=alpha, zorder=zorder, label=f"{MTP}")


def plotPatrolRegionModel(theta_axis, phi_axis, principal_ray, rho, fiberId=-1, gfm=None,
                          what="throughput",
                          blackSpots=None, vmin=None, vmax=None, title="", size=5000):
    """Plot a model of the flux variation over a patrol region, given the bend geometry and principal ray

    theta_axis:  unit vector specifying the axis of the cobra
    phi_axis:  unit vector specifying the axis of the fibre,
    what: `str`
       Desired quantity. Options: "throughput", "delta", "phi", "theta", "theta +/- phi"
    vmin, vmax:
       Minimum and maximum values of percentage flux deviations
       N.b. one or both may be None, in which case the other is used (or a default of +- 3)
    """
    if fiberId > 0:
        if gfm is None:
            raise RuntimeError("You must provide gfm if you specify a fiberId")

        cobra = OneCobra(gfm.fiberIdToCobraId(fiberId) - 1)
    else:
        cobra = OneCobra(-1)

    zc = cobra.center
    xc, yc = zc.real, zc.imag
    R = cobra.L1 + cobra.L2

    r = R*np.sqrt(np.random.uniform(size=size))
    psi = np.random.uniform(0, 2*np.pi, size=len(r))

    x, y = r*np.cos(psi) + xc, r*np.sin(psi) + yc

    theta, phi, flags = cobra.positionsToAngles(x + 1j*y)
    theta, phi = theta[:, 0], phi[:, 0]

    fibers = getFiberAngle(theta_axis, phi_axis, theta, phi)
    delta = np.arccos([np.dot(fiber, principal_ray) for fiber in fibers])

    c = None
    if what == "throughput":
        c = twoCircleOverlapOnSphere(rho, delta)/(2*np.pi*(1 - np.cos(rho)))

        c = 100*(c - np.mean(c))
        what = "flux decrement (%)"
        vmin, vmax = setVminVmax(vmin, vmax, 4)
    else:  # an angle
        if what == "delta":
            c = delta
            what = r"$\delta$"
            vmin = 0
        elif what == "phi":
            c = phi
            what = r"$\phi$"
            vmin, vmax = 0, 180
        elif what == "theta":
            c = theta
            what = r"$\theta$"
            vmin, vmax = 0, 360
        elif what == "phi":
            c = phi
            what = r"$\phi$"
            vmin, vmax = 0, 180
        elif what == "theta + phi":
            c = theta + phi
            what = r"$\theta + \phi$"
            vmin, vmax = 0, 540
        elif what == "theta - phi":
            c = theta - phi
            what = r"$\theta - \phi$"
            vmin, vmax = -180, 360

        if c is not None:
            c = np.rad2deg(c)
            what += " (degree)"

    if c is None:
        raise RuntimeError(f'Please specify a valid value of what (saw {what})\n'
                           'Valid options:  '
                           '"throughput", "delta", "phi", "theta", "theta + phi", "theta - phi"')

    y = np.array(y)

    # Draw black spot
    if fiberId > 0 and blackSpots is not None:
        i = blackSpots[blackSpots.fiberId == fiberId].spotId.iloc[0] - 1
        sid, sx, sy, spot_r, sfiberId = blackSpots.iloc[i]
        spot_x, spot_y = sx, sy

        penumbra_r = penumbra*spot_r

        ax = plt.gca()
        ax.add_patch(plt.Circle((spot_x, spot_y), penumbra_r, color='white', zorder=2))
        ax.add_patch(plt.Circle((spot_x, spot_y), spot_r, color='black', alpha=0.2, zorder=2))
        ax.add_patch(plt.Circle((spot_x, spot_y), penumbra_r, color='black', alpha=0.2, zorder=2))

    S = plt.scatter(x, y, c=c, s=6**2, vmin=vmin, vmax=vmax, cmap="seismic", edgecolor="black")
    plt.colorbar(S, label=what)

    if fiberId > 0:
        plt.text(0.01, 0.99, f"fiberId {fiberId}", color='green', ha='left', va='top', transform=ax.transAxes)

    plt.axvline(xc, color='green', alpha=0.5, zorder=1)

    zlim = 5.35*np.array([-1, 1])
    plt.xlim(xc + zlim)
    plt.ylim(yc + zlim)
    plt.gca().set_aspect(1)

    plt.title(title)
