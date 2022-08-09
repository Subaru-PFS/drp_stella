import numpy as np
import matplotlib.pyplot as plt

from pfs.datamodel.pfsConfig import FiberStatus, TargetType

import pfs.utils.fibpacking as fibpacking

__all__ = ["findSuNSSId", "plotSuNSSFluxes", "Bphoton"]


def findSuNSSId(pfsDesign, fiberId):
    """Return the SuNSS ferrule fiber ID given a pfsDesign and fiberId"""
    x0, y0 = pfsDesign.pfiNominal[pfsDesign.selectFiber(fiberId)]

    x = np.empty(127)
    y = np.empty_like(x)
    for i in range(len(x)):
        x[i], y[i] = fibpacking.fibxy(i + 1)

    r = np.hypot(x0 - x, y0 - y)

    return np.arange(len(r), dtype=int)[r < 1e-3][0] + 1


def findNeigboringFiberIds(pfsConfig, fiberId):
    """Find the fiberIds for the neighbours of a SuNSS fibre

    In the middle of the ferrule you'll get 6 fibres returned,
    but fewer at the edge.
    """
    targetType = pfsConfig.targetType[pfsConfig.fiberId == fiberId]
    xy = pfsConfig.pfiNominal
    xc, yc = xy[pfsConfig.fiberId == fiberId][0]

    d = np.hypot(xy[:, 0] - xc, xy[:, 1] - yc)
    ii = np.where(np.logical_and(d < 200, pfsConfig.targetType == targetType))[0]

    return sorted(pfsConfig.fiberId[ii[np.argsort(d[ii])]][1:])


def guessConnectors():
    pass


def makeMappingToConnector(pfsConfig):
    """Return a dict mapping fiberId to connector number along the slit"""
    conns = guessConnectors()

    lookupConnector = {}
    for DI in "DI":
        connId = np.empty(fibpacking.N_PER_FERRULE, dtype=int)
        i0, j = 0, 0
        for tt, cset in conns:
            if tt != DI:
                continue
            for c in cset:
                n = dict(IM1=13, IM2=12, US1=10, US2=27)[c]
                connId[i0:i0 + n] = j
                j += 1
                i0 += n

        lookupConnector.update(
            zip(pfsConfig.fiberId[pfsConfig.targetType == (TargetType.SUNSS_DIFFUSE if DI == "D" else
                                                           TargetType.SUNSS_IMAGING)], connId))

    return lookupConnector


def plotFerrule():
    """Plot the layout of fibres in a SuNSS ferrule"""

    for i in range(1, fibpacking.N_PER_FERRULE + 1):
        xc, yc = fibpacking.fibxy(i)
        plt.gcf().gca().add_artist(plt.Circle((xc, yc), fibpacking.FIBER_RADIUS,
                                              color='blue', alpha=0.25))
        plt.text(xc, yc, str(i), horizontalalignment='center', verticalalignment='center')

    plt.xlim(1300*np.array([-1, 1]))
    plt.ylim(plt.xlim())

    plt.gca().set_aspect('equal')

    plt.xlabel("x (microns)")
    plt.ylabel("y (microns)")

    plt.title("SuNSS fibre packing")


def plotSuNSSFluxes(pfsConfig, pfsSpec, lam0=None, lam1=None, statsOp=np.median, subtractSky=True,
                    fluxMax=None, starFibers=[], printFlux=False, md={}, showConnectors=False, fig=None):
    """Plot images of the SuNSS ferrule based on fluxes in extracted spectra
    pfsConfig : `pfsConfig`
    pfsSpec: pfsArm or pfsMerged

    Perform photometry for the fibreIds listed in starFibers; if it's
    "brightest" or "brightest+6" use the brightest fibre and its neighbours,
    usually 6.
    """

    if fig is None:
        fig = plt.figure()

    gs = fig.add_gridspec(1, 2, wspace=0)
    axs = []
    axs.append(fig.add_subplot(gs[0, 0]))
    axs.append(fig.add_subplot(gs[0, 1], sharey=axs[0]))
    axs[1].tick_params(axis='y', which='both', left=False, labelleft=False)

    if showConnectors:
        lookupConnector = makeMappingToConnector(pfsConfig)
        colors = [f"C{i%10}" for i in range(12)]

        printFlux = False
    else:
        i = len(pfsSpec)//2
        lam = pfsSpec.wavelength[i]
        # Is there  data in at least one fibre?
        have_data = np.sum((pfsSpec.mask & pfsSpec.flags["NO_DATA"]) == 0, axis=0)
        lam = np.where(have_data, lam, np.NaN)

        if lam0 is None:
            lam0 = np.nanmin(lam)
        if lam1 is None:
            lam1 = np.nanmax(lam)

        nanStatsOp = {
            np.mean: np.nanmean,
            np.median: np.nanmedian,
        }.get(statsOp)

        if nanStatsOp is None:
            print(f"I don't know how to make {statsOp} handle NaN; caveat emptor")
            nanStatsOp = statsOp

        with np.testing.suppress_warnings() as suppress:
            suppress.filter(RuntimeWarning, "All-NaN slice encountered")  # e.g. broken fibres
            suppress.filter(RuntimeWarning, "invalid value encountered in less_equal")
            suppress.filter(RuntimeWarning, "invalid value encountered in greater_equal")
            sky = {}
            if subtractSky:
                pfsFlux = pfsSpec.flux.copy()
            else:
                pfsFlux = pfsSpec.flux

            for DI in [TargetType.SUNSS_DIFFUSE, TargetType.SUNSS_IMAGING]:
                if subtractSky:
                    ll = pfsConfig.selectByTargetType(DI)
                    sky[DI] = np.nanmedian(np.where(pfsSpec.mask[ll] == 0, pfsFlux[ll], np.NaN), axis=0)
                    pfsFlux[ll] -= sky[DI]  # median per spectral element
                    sky[DI] = np.nanmean(sky[DI])
                else:
                    sky[DI] = 0

            windowed = np.where(np.logical_and(pfsSpec.wavelength >= lam0, pfsSpec.wavelength <= lam1),
                                pfsFlux, np.NaN)

            med = nanStatsOp(windowed, axis=1)

    if starFibers in ("brightest", "brightest+6"):
        brightestFiberId = pfsConfig.fiberId[np.nanargmax(med)]
        if starFibers == "brightest":
            starFibers = [brightestFiberId]
        else:
            starFibers = [brightestFiberId] + findNeigboringFiberIds(pfsConfig, brightestFiberId)

    visit = md.get('W_VISIT', "[unknown]")

    estimateFluxMax = fluxMax is None
    for i, DI in enumerate([TargetType.SUNSS_DIFFUSE, TargetType.SUNSS_IMAGING]):
        ax = axs[i]
        ax.text(0, 1200, str(DI), horizontalalignment='center')

        if estimateFluxMax:
            fluxMax = np.nanmax(med[pfsConfig.targetType == DI])

        color = 'red' if DI == TargetType.SUNSS_DIFFUSE else 'green'
        for (x, y), fid, tt, fs in zip(pfsConfig.pfiNominal, pfsConfig.fiberId,
                                       pfsConfig.targetType, pfsConfig.fiberStatus):
            if tt != DI:
                continue

            ind = pfsConfig.selectFiber(fid)

            if showConnectors:
                color = colors[lookupConnector[fid]]
                alpha = 0.5
            else:
                alpha = med[ind]/fluxMax
                if alpha < 0 or not np.isfinite(alpha):
                    alpha = 0.0
                elif alpha > 1:
                    alpha = 1
                else:
                    alpha *= 0.9

            ax.add_artist(plt.Circle((x, y), fibpacking.FIBER_RADIUS,
                                     color=color, alpha=alpha))
            ax.add_artist(plt.Circle((x, y), fibpacking.CORE_RADIUS,
                                     color='black', alpha=0.25, fill=False, zorder=-1))

            if not showConnectors and fid in starFibers:
                ax.add_artist(plt.Circle((x, y), fibpacking.FIBER_RADIUS, color='black', fill=False))

                if printFlux:
                    x, y = pfsConfig.pfiNominal[ind]
                    print(f"{visit} {x:8.1f} {y:8.1f}  {fid:3}"
                          f" {findSuNSSId(pfsConfig, fid):3} {med[ind]:6.3f}")

            broken_color = color
            textcolor = 'black'
            if fs == FiberStatus.BROKENFIBER:
                if showConnectors:
                    textcolor = 'red'
                else:
                    broken_color = 'gray'

                ax.add_artist(plt.Circle((x, y), fibpacking.FIBER_RADIUS, color=broken_color, alpha=0.5))

            ax.text(x, y, fid, color=textcolor,
                    horizontalalignment='center', verticalalignment='center', fontsize=8)

        ax.set_xlim(1300*np.array([-1, 1]))
        ax.set_ylim(plt.xlim())

        ax.set_aspect('equal')

        ax.set_xlabel("x (microns)")
        if i == 0:
            ax.set_ylabel("y (microns)")

        if showConnectors:
            plt.suptitle("Mapping to tower connectors", y=0.83)
        else:
            if i == 0:
                title = f"{visit} {'?brnm'[md['W_ARM']]}{md['W_SPMOD']}  {md['EXPTIME']:.1f}s" if md else ""
            else:
                title = r"$%.1f < \lambda < %.1f$" % (lam0, lam1)
            ax.set_title(title)

            plt.text(0.03, 0.03, f"sky = {sky[DI]:.3f}  fluxMax = sky + {fluxMax:.3f}",
                     transform=ax.transAxes)

        if md:
            plt.suptitle(f"{md['DATE-OBS']}Z{md['UT'][:-4]}", y=0.85)

    plt.tight_layout()


def Bphoton(lam, T):
    """Calculate a black-body spectrum in photons/m^2/sr/pixel

    lam : `np.array`
       Wavelengths where you want the spectrum evaluated, nm
    T : `float`
       Desired temperature, in K
    """
    h = 6.62559e-34
    c = 2.997e8
    k = 1.38e-23

    B = 2*c/lam**4/(np.exp(h*c/(lam*1e-9*k*T)) - 1)  # photons/s/m^2/m

    dlam = np.empty_like(lam)   # pixel widths
    dlam[1:] = lam[1:] - lam[:-1]
    dlam[0] = dlam[1]

    return B
