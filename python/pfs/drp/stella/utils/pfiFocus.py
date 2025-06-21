import numpy as np

import matplotlib.pyplot as plt
from pfs.datamodel.pfsConfig import TargetType


# Wavelength ranges for photometry

windowDefinitions = dict(
    b=(390, 610),
    bw=(408, 473),
    b1=(390, 473),
    b2=(473, 610),
    r=(665, 930),
    rw=(665, 747),
    r1=(665, 747),
    r2=(747, 930),
)

bandToColor = dict(
    b='blue',
    bw='blue',
    b1='blue',
    b2='green',
    r='red',
    rw='red',
    r1='red',
    r2='magenta',
)  # not arm as we will have sub-arm bands


def getXY0XYD(pfsConfig0, pfsConfig):
    x0, y0 = pfsConfig0.pfiNominal.T
    x, y = pfsConfig.pfiNominal.T
    d = 1e3*np.hypot(x - x0, y - y0)

    return (x0, y0), (x, y), d


def estimateFiberFluxes(butler, visits, windows=["b", "r"], cache={},
                        missingCamera=lambda arm, s: False,
                        ):
    """
    Parameters
    ----------
    butler: `lsst.daf.persistence.Butler`
        A butler to read data
    visits: `list` of `int`
        List of visits to process (but see processList)
    windows: `list` of `str`
       The windows to process; e.g. ["b1", "b2", "r1", "r2"]
    missingCamera: func(arm, spectrograph)
        Return True iff a camera is known to be missing
    cache: `dict`
        Dictionary used to save this routine's calculations; pass in previous version if you don't
        want to recalculate
    verbose: `int`
        Be chatty if verbose > 0
    """
    visit0 = visits[0]

    fluxes = cache.get("fluxes", {})
    pfsConfigs = cache.get("pfsConfigs", {})
    isWindowed = cache.get("isWindowed", {})
    focusz = cache.get("focusz", {})

    # setup cache
    cache["fluxes"] = fluxes
    cache["pfsConfigs"] = pfsConfigs
    cache["isWindowed"] = isWindowed
    cache["visits"] = visits
    cache["focusz"] = focusz

    dataId = dict()
    for window in windows:
        arm = window[:1]

        try:
            fluxes[window]
        except KeyError:
            fluxes[window] = {}

        dataId["arm"] = arm
        for i, v in enumerate(visits[1:] + [visit0]):  # we need to set visits[1] early
            del i
            if v in fluxes[window]:
                continue

            print(f"Processing {v} {arm} ({window})")

            dataId["visit"] = v
            pfsConfigs[v] = butler.get("pfsConfig", dataId).select(targetType=~TargetType.ENGINEERING)
            if v not in isWindowed or v not in focusz:
                md = butler.get("raw_md", visit=v, arm=arm, spectrograph=1)  # SM1 arms all present in March

                isWindowed[v] = md["W_CDROW0"] != 0
                focusz[v] = md["W_M2OFF3"]

            if isWindowed[v] and len(window) > 1:
                break

            lam0, lam1 = windowDefinitions[window + ("w" if isWindowed[visits[1]] else "")]

            fluxes[window][v] = []
            for s in range(1, 5):
                if missingCamera(arm, s):
                    continue

                dataId["spectrograph"] = s
                # filter out not-GOOD and UNASSIGNED fibres
                spec = butler.get("pfsArm", dataId)
                spec.flux[(spec.mask & ~spec.flags["REFLINE"]) != 0] = np.nan
                with np.testing.suppress_warnings() as suppress:
                    suppress.filter(RuntimeWarning, "All-NaN slice encountered")
                    fluxes[window][v].append(list(np.nanmedian(
                        np.where((spec.wavelength > lam0) & (spec.wavelength < lam1), spec.flux, np.nan),
                        axis=1)))

            pfsConfigs[v] = pfsConfigs[v]
            fluxes[window][v] = np.array(sum(fluxes[window][v], []))

    cache["focusz"] = focusz

    return cache


def plotVariantOffsets(cache, showHist=True, title="", figure=None):

    pfsConfigs = cache.get("pfsConfigs", {})
    visits = cache.get("visits", {})
    visit0 = visits[0]

    pfsConfig0 = pfsConfigs[visit0]
    pfsConfig = pfsConfigs[visits[1]]

    (x0, y0), (x, y), d = getXY0XYD(pfsConfig0, pfsConfig)

    if showHist:
        figure, axs = plt.subplots(1, 2, num=figure, sharex=False, sharey=False, squeeze=False)
        axs = axs.flatten()
    else:
        axs = [plt.gca()]

    plt.sca(axs[0])

    arrowSize = 200
    if True:
        ll = d >= 0   # i.e. everything
        fac = 1  # don't grow the arrow
    else:
        ll = d < 10
        fac = 1e-2

    Q = plt.quiver(x0[ll], y0[ll], (x - x0)[ll], (y - y0)[ll], scale=fac*1e4*100e-3/200)
    arrowSize *= fac
    plt.quiverkey(Q, 0.2, 0.925, 1e-3*arrowSize, label=f"{arrowSize:.0f} micron", color='red')

    plt.gca().set_aspect(1)

    if showHist:
        plt.sca(axs[1])

        plt.hist(d, bins=100)
        plt.xlabel("Variant offset (micron)")
        axs[1].set_aspect(1.24*axs[0].get_aspect())  # 1.24 adjusts size to match quiverplot

    plt.suptitle(title, y=0.8 if showHist else 0.95)


def plotRadialProfiles(windows=["b1", "b2", "r1", "r2"],
                       nbin=10, rmax=200, mag=None, filterName=None, magMin=None, magMax=None, cache={},
                       normPercentile=100, title="", figure=None):
    """
    Parameters
    ----------
    rmax: `float`
        Maximum radius for profiles; microns
    """
    pfsConfigs = cache.get("pfsConfigs", {})
    visits = cache.get("visits", {})
    focuszDict = cache.get("focusz", {})

    visit0 = visits[0]

    pfsConfig0 = pfsConfigs[visit0]
    pfsConfig = pfsConfigs[visits[1]]

    (x0, y0), (x, y), d = getXY0XYD(pfsConfig0, pfsConfig)

    rbin = np.arange(nbin + 1)*rmax/(nbin - 1)   # N.b. rmax == rbin[nbin - 1]

    if magMin is None:
        magMin = -20
    if magMax is None:
        magMax = -20

    if magMin > 0 or magMax > 0:
        if mag is None:
            raise RuntimeError("You must provide a mag array if you set magMin or magMax")
        elif filterName is None:
            raise RuntimeError("filterName is not set")
    fluxes = cache.get("fluxes")
    if not fluxes:
        raise RuntimeError("You must pass a cache with \"fluxes\" set (returned by estimateFiberFluxes())")

    isWindowed = cache.get("isWindowed", {})

    binnedDs = {}
    binnedDErrors = {}
    binnedFluxes = {}
    binnedFluxErrors = {}
    rms = {}
    meanCenteredFlux = {}
    meanCenteredFlux["normPercentile"] = normPercentile

    for window in windows:
        arm = window[:1]

        binnedDs[window] = np.full((len(visits) - 1, nbin), np.nan)
        binnedDErrors[window] = np.full_like(binnedDs[window], np.nan)
        binnedFluxes[window] = np.full_like(binnedDs[window], np.nan)
        binnedFluxErrors[window] = np.full_like(binnedFluxes[window], np.nan)
        rms[window] = {}
        meanCenteredFlux[window] = {}

        for quadrant in ["NW", "NE", "SE", "SW", None]:
            rms[window][quadrant] = np.full(len(visits) - 1, np.nan)
            meanCenteredFlux[window][quadrant] = np.full_like(rms[window][quadrant], np.nan)

        flux0 = fluxes[window][visit0]
        isSky = pfsConfig0.getSelection(targetType=TargetType.SKY)

        if arm == 'b':
            assert len(flux0) == 1794, f"b3 is missing, saw {len(flux0)} for {visit0} {window}"
            ll = pfsConfig.spectrograph != 3
        else:
            ll = pfsConfig.spectrograph != 666

        for iv, v in enumerate(visits[1:]):
            if v not in fluxes[window]:
                continue

            flux = fluxes[window][v].copy()
            config = pfsConfigs[v]
            if arm == 'b':
                assert len(flux0) == 1794, f"b3 is missing, saw {len(flux0)} for {v} {window}"

            isSky = config.getSelection(targetType=TargetType.SKY)

            keep = config.getSelection(targetType=[TargetType.FLUXSTD, TargetType.SCIENCE])
            sky0 = np.nanmedian(flux0[isSky[ll]])

            keep &= ~np.isin(config.fiberId, (206, 214, 391))
            if magMin > 0:
                keep &= (mag > magMin)
            if magMax > 0:
                keep &= (mag < magMax)

            keep &= d > 1

            sky = np.nanmedian(flux[isSky[ll]])
            II = (flux - sky)
            I0 = (flux0 - sky0)

            quadrants = ["NW", "NE", "SE", "SW", None]  # None must come last as we set e.g. binnedDs
            _keep = keep
            for quadrant in quadrants:
                if quadrant is None:
                    keep = _keep
                else:
                    N = x > 0
                    W = y > 0

                    if quadrant == "NW":
                        keep = _keep & N & W
                    elif quadrant == "NE":
                        keep = _keep & N & ~W
                    elif quadrant == "SE":
                        keep = _keep & ~N & ~W
                    elif quadrant == "SW":
                        keep = _keep & ~N & W

                for i in range(nbin):
                    isWindow = np.where(keep[ll] & (d[ll] > rbin[i]) & (d[ll] <= rbin[i + 1]))[0]
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        binnedDs[window][iv][i] = np.mean(d[ll][isWindow])
                        binnedDErrors[window][iv][i] = np.std(d[ll][isWindow])
                        norm = 1/np.mean(I0[isWindow])   # flux in this annulus in visit0 (all fibres centred)
                        binnedFluxes[window][iv][i] = norm*np.mean(II[isWindow])
                        binnedFluxErrors[window][iv][i] = norm*np.std(II[isWindow])/np.sqrt(len(isWindow))

                annularFlux = binnedFluxes[window][iv]*2*np.pi*binnedDs[window][iv]

                good = np.isfinite(annularFlux + binnedDs[window][iv])
                flux = np.trapezoid(annularFlux[good], binnedDs[window][iv][good])
                rms[window][quadrant][iv] = \
                    np.sqrt(np.trapezoid(annularFlux[good]*binnedDs[window][iv][good]**2,
                                         binnedDs[window][iv][good])/flux)

                meanCenteredFlux[window][quadrant][iv] = \
                    np.nanmean(II[keep[ll] & (d[ll] < 10)])  # < 1 micron offset

            keep = _keep                # restore saved value

            if False:
                norm = 1/np.nanmax(binnedFluxes[window][iv])
            else:
                norm = 1/flux

            binnedFluxes[window][iv] *= norm
            binnedFluxErrors[window][iv] *= norm

        # normalise the arms
        norm = 1/np.nansum(binnedFluxes[window])
        for iv, v in enumerate(visits[1:]):
            binnedFluxes[window][iv] *= norm
            binnedFluxErrors[window][iv] *= norm

        # normalise the quadrants
        for quadrant in quadrants:
            meanCenteredFlux[window][quadrant] /= np.nanpercentile(meanCenteredFlux[window][quadrant],
                                                                   [normPercentile])[0]

    for window in windows:
        if window in binnedFluxes:
            max0 = np.nanmax(binnedFluxes[window][:, 0])
            binnedFluxes[window] /= max0
            binnedFluxErrors[window] /= max0

    nDither = len(visits[1:])
    ny = int(np.sqrt(nDither))
    nx = nDither//ny
    if nx*ny < nDither:
        ny += 1

    figure, axs = plt.subplots(ny, nx, num=figure, sharex=True, sharey=True, squeeze=False)
    axs = axs.flatten()

    for window in windows:
        if window not in binnedFluxes:
            print(f"binnedFluxes are not available for window {window}")

            continue

        for ax, (iv, v) in zip(axs, enumerate(visits[1:])):
            plt.sca(ax)
            color = bandToColor.get(window)
            plt.plot(binnedDs[window][iv], binnedFluxes[window][iv], '-', alpha=0.25, zorder=-1, color=color)
            plt.errorbar(binnedDs[window][iv], binnedFluxes[window][iv],
                         xerr=binnedDErrors[window][iv], yerr=binnedFluxErrors[window][iv],
                         fmt='none', color=color)
            plt.text(0.4, 0.95, f"{v}\n{focuszDict[v]:5.3f}mm", transform=ax.transAxes, ha='left', va='top')

            if window == windows[0]:
                plt.axhline(0, color="black", alpha=0.2, zorder=-1)
                plt.axvline(50, color="black", alpha=0.2)

            plt.ylim(-0.1, 1.1)

    for i in range(nDither, nx*ny):
        plt.sca(axs[i])
        if i == nDither:
            for window in windows:
                plt.plot([0], [0], color=bandToColor.get(window), label=f"{window} "
                         f"{windowDefinitions[window + ('w' if isWindowed[visits[1]] else '')]}nm")
            plt.legend()
        axs[i].set_axis_off()

    title = []
    title.append(f"Arms: [{' '.join([w for w in windows])}]")
    title.append(f" {np.sum(keep)} objects")
    if magMin > 0 or magMax > 0:
        if magMin > 0:
            if magMax > 0:
                title.append(f"{magMin} < {filterName} < {magMax}")
            else:
                title.append(f"{magMin} < {filterName}")
        else:
            title.append(f"{filterName} < {magMax}")
    title.append('\n')
    title.append(f"{pfsConfigs[visits[-1]].designName}")
    title.append(f"{visit0}^{visits[1]}..{visits[-1]}")

    title.append('\n')
    windowsDescrip = [windowDefinitions[window + ('w' if isWindowed[visits[1]] else '')]
                      for window in windows]
    title.append(f"Flux in {windowsDescrip}nm")

    figure.supxlabel("Variant offset (micron)")
    figure.supylabel("Relative flux (total flux==1 for each profile)")

    title = " ".join(title)
    plt.suptitle(title)

    return rms, meanCenteredFlux


def plotRms(rms, windows=None, byQuadrant=True, title="", figure=None, cache={}):
    visits = cache.get("visits", [])
    focuszDict = cache.get("focusz", {})
    focusz = np.full(len(visits), np.nan)
    for iv, v in enumerate(visits):
        focusz[iv] = focuszDict[v]

    if byQuadrant:
        quadrants = ["NW", "NE", "SW", "SE"]
        figure, axs = plt.subplots(2, 2, num=figure, sharex=True, sharey=True, squeeze=False)
        axs = axs.flatten()
    else:
        quadrants = [None]
        axs = [plt.gca()]

    if not windows:
        windows = list(rms)

    for ax, quadrant in zip(axs, quadrants):
        plt.sca(ax)

        for window in windows:
            ifz = np.argsort(focusz[1:])
            plt.plot(focusz[1:][ifz], rms[window][quadrant][ifz], '.', color=bandToColor.get(window),
                     label=f"{window} {windowDefinitions[window]}nm")
            plt.plot(focusz[1:][ifz], rms[window][quadrant][ifz], '-', alpha=0.1, zorder=-1,
                     color=bandToColor.get(window))

            if ax == axs[-1]:
                plt.legend(loc=None if quadrant is None else (0.7, 0.02))

            if quadrant is not None:
                plt.text(0.1, 0.15, f"{quadrant}", transform=ax.transAxes, ha='left', va='top')

            plt.axhline(60, color='black', alpha=0.1)

    figure.supxlabel("W_M2OFF3")
    figure.supylabel("RMS (micron)")
    plt.suptitle(title)


def rmsFromCenteredFibers(meanCenteredFlux, windows=None, byQuadrant=True, cache={}, title="", figure=None):
    """
    """
    visits = cache.get("visits", [])
    focuszDict = cache.get("focusz", {})
    focusz = np.full(len(visits), np.nan)
    for iv, v in enumerate(visits):
        focusz[iv] = focuszDict[v]

    if not windows:
        windows = [el for el in meanCenteredFlux if el != "normPercentile"]

    if byQuadrant:
        quadrants = ["NW", "NE", "SW", "SE"]
        figure, axs = plt.subplots(2, 2, num=figure, sharex=True, sharey=True, squeeze=False)
        axs = axs.flatten()
    else:
        quadrants = [None]
        axs = [plt.gca()]

    normPercentile = meanCenteredFlux["normPercentile"]

    for ax, quadrant in zip(axs, quadrants):
        plt.sca(ax)

        for window in windows:
            if window not in meanCenteredFlux:
                if quadrant == quadrants[0]:
                    print(f"meanCenteredFlux is not available for window {window}")
                continue

            meanFlux = meanCenteredFlux[window][quadrant]

            ifz = np.argsort(focusz[1:])
            plt.plot(focusz[1:][ifz], meanFlux[ifz], '-', alpha=0.1, zorder=-1, color=bandToColor.get(window))
            plt.plot(focusz[1:][ifz], meanFlux[ifz], '.', color=bandToColor.get(window),
                     label=f"{window} {windowDefinitions[window]}nm")

            if ax == axs[-1]:
                plt.legend(loc=None if quadrant is None else (0.7, 0.6))

            if quadrant is not None:
                plt.text(0.1, 0.8, f"{quadrant}", transform=ax.transAxes, ha='left', va='top')

    figure.supxlabel("W_M2OFF3")
    figure.supylabel(f"Mean flux in centred fibres normalized to {normPercentile}%")
    plt.suptitle(title)
