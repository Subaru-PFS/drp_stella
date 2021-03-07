import numpy as np
import matplotlib.pyplot as plt

from pfs.datamodel.pfsConfig import FiberStatus, TargetType

import pfs.utils.fibpacking as fibpacking

__all__ = ["findSuNSSId", "plotSuNSSFluxes"]


def findSuNSSId(pfsDesign, fiberId):
    """Return the SuNSS ferrule fiber ID given a pfsDesign and fiberId"""
    x0, y0 = pfsDesign.pfiNominal[pfsDesign.selectFiber(fiberId)][0]

    x = np.empty(127)
    y = np.empty_like(x)
    for i in range(len(x)):
        x[i], y[i] = fibpacking.fibxy(i + 1)

    r = np.hypot(x0 - x, y0 - y)

    return np.arange(len(r), dtype=int)[r < 1e-3][0] + 1


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
                    fluxMax=None, starFibers=[], printFlux=False, md={}, showConnectors=False):
    """Plot images of the SuNSS ferrule based on fluxes in extracted spectra
    pfsConfig : `pfsConfig`
    pfsSpec: pfsArm or pfsMerged
    """
    fig, axs = plt.subplots(1, 2, sharey='row', gridspec_kw=dict(wspace=0))

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
            if subtractSky:
                pfsFlux = pfsSpec.flux.copy()
                pfsFlux -= np.nanmedian(np.where(pfsSpec.mask == 0, pfsFlux, np.NaN), axis=0)
            else:
                pfsFlux = pfsSpec.flux

            windowed = np.where(np.logical_and(pfsSpec.wavelength >= lam0, pfsSpec.wavelength <= lam1),
                                pfsFlux, np.NaN)

            med = nanStatsOp(windowed, axis=1)

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

            ind = pfsConfig.selectFiber(fid)[0]

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
                          f"{findSuNSSId(pfsConfig, fid):3} {med[ind]:.3f}")

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
                title = f"{visit} {'brnm'[md['W_ARM']]}{md['W_SPMOD']}  {md['EXPTIME']:.1f}s" if md else ""
            else:
                title = r"$%.1f < \lambda < %.1f$" % (lam0, lam1)
            ax.set_title(title)

            plt.text(0.03, 0.03, f"fluxMax = {fluxMax:.2f}", transform=ax.transAxes)

        if md:
            plt.suptitle(f"{md['DATE-OBS']}Z{md['UT'][:-4]}", y=0.85)

    plt.tight_layout()
