import os
import numpy as np
import matplotlib.pyplot as plt

from pfs.datamodel.pfsConfig import PfsConfig, PfsDesign, FiberStatus, TargetType

import pfs.datamodel.fibpacking as fibpacking


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


def makeMappingToConnector():
    """Return a dict mapping fiberId to connector number along the slit"""
    conns = guessConnectors()

    lookupConnector = {}
    for DI in "DI":
        connId = np.empty(len(x), dtype=int)
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
            zip(pfsDesign.fiberId[pfsDesign.targetType == (TargetType.SUNSS_DIFFUSE if DI == "D" else
                                                     TargetType.SUNSS_IMAGING)], connId))
        
    return lookupConnector


def plotSuNSSFluxes(pfsConfig, pfsArm, lam0=None, lam1=None, statsOp=np.median, subtractSky=True, fluxMax=None,
                    starFibers=[], printFlux=False, md={}, showConnectors=False):

    fig, axs =  plt.subplots(1, 2, sharey='row', gridspec_kw=dict(wspace=0))

    if showConnectors:
        lookupConnector = makeMappingToConnector()
        colors = [f"C{i%10}" for i in range(12)]

        printFlux = False
    else:
        i = len(pfsArm)//2
        lam = pfsArm.wavelength[i]
        if lam0 is None:
            lam0 = np.min(lam)
        if lam1 is None:
            lam1 = np.max(lam)

        nanStatsOp = {
            np.mean : np.nanmean,
            np.median : np.nanmedian,
        }.get(statsOp)
        if nanStatsOp is None:
            print(f"I don't know how to make {statsOp} handle NaN; caveat emptor")
            nanStatsOp = statsOp
            
        if subtractSky:
            pfsFlux = pfsArm.flux + 0
            pfsFlux -= np.nanmedian(np.where(pfsArm.mask == 0, pfsFlux, np.NaN), axis=0)
        else:
            pfsFlux = pfsArm.flux
            
        windowed = np.where(np.logical_and(pfsArm.wavelength >= lam0, pfsArm.wavelength <= lam1),
                            pfsFlux, np.NaN)

        amed = nanStatsOp(windowed, axis=1)
        if False:
            sky = np.nanpercentile(windowed, [50])[0]
            pfsArm.flux -= sky

        amean = np.mean(amed)
        print(f"Mean flux = {amean:.3f}")

        if fluxMax:
            pass
            #amed /= amean
            #sky /= amean

        med = np.empty(len(pfsConfig))
        for i, fid in enumerate(pfsConfig.fiberId):
            if fid in pfsArm.fiberId:
                med[i] = amed[np.where(pfsArm.fiberId == fid)[0][0]]
            else:
                med[i] = np.NaN
        ##

        if False:
            print(f"sky = {sky}")
            if subtractSky:
                med -= sky 

    visit = md.get('W_VISIT', "[unknown]")

    for i, DI in enumerate([TargetType.SUNSS_DIFFUSE, TargetType.SUNSS_IMAGING]):
        ax = axs[i]
        ax.text(0, 1200, str(DI), horizontalalignment='center')
        
        if fluxMax is None:
            fluxMax = np.nanmax(med[pfsConfig.targetType == DI])

        color = 'red' if DI == TargetType.SUNSS_DIFFUSE else 'green'
        for (x, y), fid, tt, fs in zip(pfsConfig.pfiNominal, pfsConfig.fiberId,
                                       pfsConfig.targetType, pfsConfig.fiberStatus):
            if tt != DI:
                continue

            ind = pfsConfig.selectFiber(fid)[0]

            if showConnectors:
                color = colors[lookupConnector[fid]]
                alpha=0.5
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
                    print(f"{visit} {x:8.1f} {y:8.1f}  {fid:3} {findSuNSSId(fid):3} {med[ind]:.3f}")

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

        ax.set_aspect('equal');

        ax.set_xlabel("x (microns)")
        if i == 0:
            ax.set_ylabel("y (microns)")
            
        if showConnectors:
            plt.suptitle("Mapping to tower connectors", y=0.83);
        else:
            title = f"{visit} {'brnm'[md['W_ARM']]}{md['W_SPMOD']}  {md['EXPTIME']:.1f}s" if i == 0 else \
                       r"$%.1f < \lambda < %.1f$" % (lam0, lam1)
            ax.set_title(title)
            
            plt.text(0.03, 0.03, f"fluxMax = {fluxMax:.2f}", transform=ax.transAxes)
            
        if md:
            plt.suptitle(f"{md['DATE-OBS']}Z{md['UT'][:-4]}", y=0.85)

    plt.tight_layout();
