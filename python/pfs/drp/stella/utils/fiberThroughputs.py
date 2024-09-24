import numpy as np
import matplotlib.pyplot as plt

from pfs.datamodel.pfsConfig import TargetType
from pfs.drp.stella.utils.raster import addCobraIdCallback


def clearCache(cache, visit, arm=None):
    """Clear the cache for a visit, and optionally an arm"""
    for what in cache.values():
        for k in what.values():
            arms = list(k)
            if arm is not None and arm in arms:
                arms = [arm]

            for a in arms:
                if visit in k[a]:
                    del k[a][visit]


def selectWavelengthInterval(arm):
    """
    Select an emission line to measure a flux.
    Each interval is (lamc, lam0, lam1, bkgd0, bkgd1)

    We'll measure the flux for the line at lamc in [lam0, lam1];
    if bkgd0 and bkgd1 are not None, measure the background in [bkgd0, lam0] + [lam1, bkgd1]

    """
    intervals = []
    if arm == 'b':
        lamc = 557.88870
        (lam0, lam1), (bkgd0, bkgd1) = (lamc - 0.5, lamc + 0.5), (lamc - 1.5, lamc + 1.5)

        intervals.append((lamc, lam0, lam1, bkgd0, bkgd1))
    elif arm == 'r':
        intervals.append((931.56795, 930.5, 933.1, None, None))
        intervals.append((947.94298, 947.25, 949, None, None))
    elif arm == 'n':
        if False:
            # (5-2)pP1f(3.5)|(5-2)pP1e(3.5)
            intervals.append((1097.53290, 1096, 1099, None, None))

            # (6-3)pP1f(2.5)|(6-3)pP1e(2.5)
            intervals.append((1153.87886, 1152.2, 1155.2, None, None))

        if True:
            # (6-3)pP1f(3.5)|(6-3)pP1e(3.5)
            intervals.append((1159.16913, 1157.4, 1161.3, None, None))

        if False:
            # (7-4)pP2f(2.5)|(7-4)pP2e(2.5) + (7-4)pP1f(3.5)|(7-4)pP1e(3.5)
            intervals.append(((1225.77314 + 1228.69878)/2, 1227.5, 1230, None, None))
    else:
        raise RuntimeError(f"selectWavelengthInterval doesn't know about arm={arm}")

    return intervals


def getWavelengthLabel(arm, what="flux"):
    """Generate a label for the wavelength intervals used in arm"""
    labels = []
    for lamc, lam0, lam1, b0, b1 in selectWavelengthInterval(arm):
        labels.append(f"({lam0:.1f} < $\\lambda$ < {lam1:.1f})" if what == "continuum" else f"{lamc:.2f}")

    if len(labels) == 1:
        if what == "continuum":
            return labels[0]
        else:
            return labels[0] + "nm line"
    else:
        if what == "continuum":
            return "(" + "), (".join(labels) + ")"
        else:
            return "(" + "), (".join(labels) + ") nm lines"


def showWavelengthInterval(arm):
    for lamc, lam0, lam1, bkgd0, bkgd1 in selectWavelengthInterval(arm):
        if bkgd0:
            plt.axvspan(bkgd0, bkgd1, color='black', alpha=0.05, zorder=-2)

        plt.axvspan(lam0, lam1, color='black', alpha=0.05 if bkgd0 else 0.1, zorder=-1)


def measureFlux(y, lam, arm, std=None, fitLine=True):
    from scipy.optimize import curve_fit

    nspec = y.shape[0]
    totflux = np.zeros(nspec)
    flux = np.zeros_like(totflux)

    for (lamc, lam0, lam1, bkgd0, bkgd1) in selectWavelengthInterval(arm):

        if not fitLine:
            flux = np.nanmean(np.where((lam > lam0) & (lam < lam1), y, np.NaN), axis=1)
            totflux += flux
            continue

        if std is None:
            std = np.ones_like(y)

        class Func:
            def __init__(self, lineCenter, alpha=0.1):
                self._lineCenter = lineCenter
                self._alpha = alpha
                self._C = 1/np.sqrt(2*np.pi*alpha**2)

            def __call__(self, x, flux, bkgd, alpha=0.09, dx=0):
                return bkgd + \
                    flux/np.sqrt(2*np.pi*alpha**2)*np.exp(-0.5*((x - self._lineCenter + dx)/alpha)**2)

        func = Func(lamc, 0.09)

        for i in range(nspec):
            if bkgd0 is None:
                bkgd0, bkgd1 = lam0, lam1

            ll = (lam[i] > bkgd0) & (lam[i] < bkgd1)

            try:
                popt, pcov = curve_fit(func, lam[i][ll], y[i][ll], p0=[0, 10000, 0.09, 0][:3],
                                       sigma=std[i][ll], check_finite=False)
            except RuntimeError:
                flux[i] = np.NaN
            else:
                flux[i] = popt[0]

            if True:
                if i == 200:
                    color = plt.errorbar(lam[i][ll], y[i][ll], std[i][ll], )[0].get_color()
                    plt.plot(lam[i][ll], func(lam[i][ll], *popt), ':', color=color)

        totflux += flux

    return totflux


def estimateFiberThroughputs(butler, visits, arms="brn", what="flux",
                             missingCamera=lambda arm, s: False, medianPerArm={},
                             showHome=True, cache={}, verbose=0,
                             processList="all", reprocessList=[],
                             ):
    """
    Parameters
    ----------
    butler: `lsst.daf.persistence.Butler`
        A butler to read data
    visits: `list` of `int`
        List of visits to process (but see processList)
    arms: `str` or `list` of `str`
       The arms to process; e.g. "brn" or ["b", "r", "n"]; your choice
    what: `str`
        A list of desired parameters; your choices are:
            "flux"        calculate the flux in a wavelength band lam0 < lam < lam1
            "continuum"  show the flux in a wavelength band lam0 < lam < lam1 (aperture)
            "normedFlux"  calculate the flux in a wavelength band lam0 < lam < lam1, divided by spec.norm
            "norm"        show the fibre normalisation
    missingCamera: func(arm, spectrograph)
        Return True iff a camera is known to be missing
    medianPerArm: `dict` indexed by arm (`str`) then spectrograph (`int`)
        Normalise the arms/spectrographs by medianPerArm[arm][spectrograph]
    showHome: `bool`
        Show cobras at home, not pfiNominal, positions (default: True)
    cache: `dict`
        Dictionary used to save this routine's calculations; pass in previous version if you don't
        want to recalculate (see also processList and reprocessList)
    verbose: `int`
        Be chatty if verbose > 0
    processList: `list` of `int`
        List of visits to process, if not in cache.  If set to "all", process any visits in visits
        which are not in the cache; if [] skip any checking.
    reprocessList: `list` of `int`
        List of visits which should be reprocessed even if in the cache
    """
    visitC = cache.get("visitC", {})
    visitConfig = cache.get("visitConfig", {})
    visitMetadata = cache.get("visitMetadata", {})

    # setup cache
    cache["visitC"] = visitC
    cache["visitConfig"] = visitConfig
    cache["visitMetadata"] = visitMetadata

    for w in ["flux", "continuum", "normedFlux", "norm"]:
        if w not in visitC:
            visitC[w] = {}
        if w not in visitConfig:
            visitConfig[w] = {}
        if w not in visitMetadata:
            visitMetadata[w] = {}

        for a in "brnm":
            if a not in visitC[w]:
                visitC[w][a] = {}
            if a not in visitConfig[w]:
                visitConfig[w][a] = {}
            if a not in visitMetadata[w]:
                visitMetadata[w][a] = {}

    if processList == "all":
        processList = visits.copy()

    if reprocessList:
        processList = set(processList + reprocessList)

    if isinstance(what, (list, tuple)):
        whatList = what
    else:
        whatList = [what]

    for i, visit in enumerate(visits):
        if visit not in processList:
            continue

        for arm in arms:
            dataId = dict(visit=visit, arm=arm)

            if visit in reprocessList:
                for what in whatList:
                    if visit in visitC[what][dataId["arm"]]:
                        del visitC[what][dataId["arm"]][visit]

            bad = []                    # see if any data is missing
            for s in [1, 2, 3, 4]:
                if missingCamera(arm, s):
                    continue

                if not butler.datasetExists("pfsArm", dataId, spectrograph=s):
                    bad.append(s)
            if bad:
                print(f"DataId visit={visit} arm={dataId['arm']} "
                      f"spectrograph=[{','.join([str(s) for s in bad])}] not found")
                continue

            inCache = True
            for what in whatList:
                if not (dataId["arm"] in visitC[what] and dataId["visit"] in visitC[what][dataId["arm"]]):
                    if verbose > 0:
                        frac = 1.0 if len(visits) == 1 else i/(len(visits) - 1)
                        print(f"{visit} {arm} {int(100*frac + 0.5)}%\r", end='', flush=True)

                    inCache = False
                    break

            if inCache:
                continue

            if verbose > 1:
                print(f"Reading {dataId}")

            pfsConfig = butler.get("pfsConfig", dataId, spectrograph=2)
            pfsConfig = pfsConfig[pfsConfig.targetType != TargetType.ENGINEERING]
            rawMd = butler.get("raw_md", dataId, spectrograph=2)

            ll = np.ones(len(pfsConfig), dtype=bool)
            c = {}
            for what in whatList:
                c[what] = []

            for spectrograph in range(1, 5):
                dataId.update(spectrograph=spectrograph)
                try:
                    spec = butler.get("pfsArm", dataId)
                except Exception as e:
                    if not missingCamera(arm, spectrograph):
                        print(f"Error reading pfsArm for {dataId}: {e}")

                    ll &= pfsConfig.spectrograph != spectrograph
                    continue

                for what in whatList:
                    fitLine = True
                    if what == "flux":
                        y = spec.flux
                    elif what == "continuum":
                        y = spec.flux
                        fitLine = False
                    elif what == "normedFlux":
                        y = spec.flux/spec.norm
                    elif what == "norm":
                        y = spec.norm
                    else:
                        raise RuntimeError(f"Unknown quantity to plot: {what}")

                    std = None if True else np.where((spec.mask & ~spec.flags["REFLINE"]) == 0,
                                                     np.sqrt(spec.variance), np.inf)
                    y = list(measureFlux(y, spec.wavelength, dataId["arm"], std, fitLine=fitLine))

                    c[what].append(y)

            for what in whatList:
                visitC[what][dataId["arm"]][dataId["visit"]] = np.array(sum(c[what], []))
                visitConfig[what][dataId["arm"]][dataId["visit"]] = pfsConfig[ll]
                visitMetadata[what][dataId["arm"]][dataId["visit"]] = rawMd

            assert arm == dataId['arm'] and visit == dataId['visit']

    return cache


def plotThroughputs(cache, visits, arms, what="flux", refVisits=[], refWhat=None, showHome=False,
                    fitModel=False, normalizeArms=True,
                    stretchToQlQh=True, low=5, high=95, vmin=None, vmax=None, s=None,
                    pfi=None, gfm=None, extraTitle=None,
                    figure=None):
    """
    Parameters
    ----------
    cache: `dict`
        Dictionary returned by estimateFiberThroughputs
    visits: `list` of `int`
        List of visits to process
    arms: `str` or `list` of `str`
       The arms to process; e.g. "brn" or ["b", "r", "n"]; your choice
    what: `str`
        Desired parameter; your choices are:
            "flux"        show the flux in a wavelength band lam0 < lam < lam1
            "continuum"  show the flux in a wavelength band lam0 < lam < lam1 (aperture)
            "normedFlux"  show the flux in a wavelength band lam0 < lam < lam1, divided by spec.norm
            "norm"        show the fibre normalisation
    refVisits: `list` of `int`
       if provided, throughputs will be measured relative to these visits
    refWhat: `str`
       The value of `what` used for the reference visit;  use `what` if `None`
    showHome: `bool`
       Show cobras at their home positions;  requires pfi and gfm
    fitModel: `bool`
       Fit and subtract a spatial model to the data (currently quadratic)
    normalizeArms: `bool`
       Divide the values by the per-arm median
    stretchToQlQh: `bool`
        Stretch the data to show the data between a high and low quantile (see arguments low and high),
        after normalising to the median; if false, stretch in (0, 1) after normalising to the maximum
    low:  `float`
        Lower percentile used by stretchToQlQh (default: 5)
    high:  `float`
        Higher percentile used by stretchToQlQh (default: 95)
    vmin: override any other vmin value (e.g. stretchToQlQh + (low, high))
    vmax: override any other vmax value (e.g. stretchToQlQh + (low, high))
    s: size "s" passed to plt.scatter
    pfi: `ics.cobraCharmer.pfiDesign.PFIDesign`
        PFI design, as returned by
            from pfs.utils.butler import Butler as pButler
            pbutler = pButler(configRoot=os.path.join(os.environ["PFS_INSTDATA_DIR"], "data"))
            pfi = pbutler.get('moduleXml', moduleName='ALL', version='')
        Needed if showHome is True
    gfm: `pfs.utils.fiberids.FiberIds`
        Grand Fiber Map, as returned by
            from pfs.utils.fiberids import FiberIds
            gfm = FiberIds()
    extraTitle: `bool`
        Extra informations to append to the title
"""
    visitC = cache["visitC"]
    visitConfig = cache["visitConfig"]
    visitMetadata = cache["visitMetadata"]

    if showHome:  # draw the cobras at their home positions
        if gfm is None or pfi is None:
            print("You must provide gfm and pfi in order to use showHome; ignoring")
            showHome = False

    if refWhat is None:
        refWhat = what
    if len(refVisits) == 0:
        refVisit = -1

    whatStr = dict(flux="",
                   continuum=" (aperture)",
                   normedFlux=" (divided by norm)",
                   norm=" (norm)")

    if figure:
        plt.figure(figure)
    else:
        figure = plt.gcf()

    plt.clf()

    for arm in arms:
        c = None
        cref = None

        for i, visit in enumerate(visits):
            dataId = dict(visit=visit, arm=arm)

            if dataId["arm"] in visitC[what] and dataId["visit"] in visitC[what][dataId["arm"]]:
                _c = visitC[what][dataId["arm"]][dataId["visit"]]
                if c is None:
                    c = np.empty((len(visits), _c.size))

                c[i] = _c/np.nanmedian(_c)
                pfsConfig = visitConfig[what][dataId["arm"]][dataId["visit"]]
                md = visitMetadata[what][dataId["arm"]][dataId["visit"]]
            else:
                raise RuntimeError(f"Unable to read value of {what} from cache for {dataId}")

        c = (np.nanmin if len(visits) < 3 else np.nanmedian)(c, axis=0)

        for i, refVisit in enumerate(refVisits):
            dataId = dict(visit=visit, arm=arm)

            if dataId["arm"] in visitC[what] and dataId["visit"] in visitC[what][dataId["arm"]]:
                _cref = visitC[refWhat][dataId["arm"]][refVisit]

                if cref is None:
                    cref = np.empty((len(refVisits), _c.size))

                cref[i] = _cref/np.nanmedian(_cref)
            else:
                raise RuntimeError(f"Unable to read value of {what} from cache for {dataId}")

        # divide by reference visit[s]
        if cref is not None:
            cref = np.nanmedian(cref, axis=0)
            with np.testing.suppress_warnings() as suppress:
                suppress.filter(RuntimeWarning, "invalid value encountered in true_divide")
                c /= cref
        #
        # Fit a spatial model to c?
        #
        if fitModel:
            from scipy.optimize import least_squares

            def spatialModel(p, x, y):
                if len(p) == 3:
                    a, mx, my = p
                    return a + mx*x + my*y
                else:
                    a, mx, my, mxx, mxy, myy = p
                    return a + mx*x + my*y + mxx*x**2 + mxy*x*y + myy*y**2

            def getResiduals(p, x, y, c):
                residuals = c - spatialModel(p, x, y)
                residuals = residuals[np.isfinite(residuals)]

                return residuals

            p0 = (1, 0, 0, 0, 0, 0)[:3]
            x, y = pfsConfig.pfiCenter.T
            ll = np.isfinite(x + y + c)
            res = least_squares(getResiduals, p0, args=(x[ll], y[ll], c[ll]), loss='soft_l1', f_scale=0.03)

            c -= spatialModel(res.x, x, y)
            c += 1 - np.nanmedian(c)
        #
        # Plot
        #
        title = [f"{','.join(str(v) for v in visits)}{whatStr[what]} {dataId['arm']}"]
        title.append(f"{md['DATE-OBS']}T{md['UT-STR'][:5]}Z")
        title.append("\n")
        lam0, lam1 = selectWavelengthInterval(dataId["arm"])[0][1:3]  # XXX just the first pair
        colorbarlabel = [f"{'relative ' if refVisit > 0 else ''}flux in "
                         + getWavelengthLabel(arm, what)]
        title.append(f"[{pfsConfig.designName}]")
        if fitModel:
            title.append("spatial model")
        if normalizeArms:
            title.append("arm normalized")
        if extraTitle:
            title.append(extraTitle)

        if refVisit > 0:
            title.append(f"\ndivided by {','.join(str(v) for v in refVisits)}{whatStr[refWhat]} "
                         f"[{visitConfig[refWhat][dataId['arm']][refVisit].designName}]")

        if False and showHome:
            title.append(" at home")

        if normalizeArms:
            for spec in set(pfsConfig.spectrograph):
                c[pfsConfig.spectrograph == spec] /= np.nanmedian(c[pfsConfig.spectrograph == spec])

        if stretchToQlQh:
            c /= np.nanmedian(c)
            colorbarlabel[0] = colorbarlabel[0]
            _vmin, _vmax = np.nanpercentile(c, [low, high])
            title.append(f"stretched {low}% .. {high}%")
        else:
            fac = 1.5
            _vmin, _vmax = 0, fac
            colorbarlabel.append(f"divided by {fac}*median")

        if not (vmin is None and vmax is None):
            title.pop()

        vmin = _vmin if vmin is None else vmin
        vmax = _vmax if vmax is None else vmax

        if showHome:  # draw the cobras at their home positions
            x, y = [], []
            for fid in pfsConfig.fiberId:
                cid = gfm.cobraId[gfm.fiberId == fid][0]
                zc = pfi.centers[cid - 1]
                x.append(zc.real)
                y.append(zc.imag)
        else:
            x, y = pfsConfig.pfiCenter.T

        S = plt.scatter(x, y, c=c, vmin=vmin, vmax=vmax, s=s)

        plt.colorbar(S, label="\n".join(colorbarlabel))

        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.title(" ".join(title))

        plt.xlim(plt.ylim(-240, 240))

        plt.gca().set_aspect(1)

        if True:
            addCobraIdCallback(figure, pfi, gfm, showMTP=True, textcolor='red')


def throughputByVisit(cache, arms, what="flux", medianPerArm={}, figure=None, title=""):
    """
    Parameters
    ----------
    cache: `dict`
        Dictionary returned by estimateFiberThroughputs
    arms: `str` or `list` of `str`
       The arms to process; e.g. "brn" or ["b", "r", "n"]; your choice
    what: `str`
        Desired parameter; your choices are:
            "flux"        show the flux in a wavelength band lam0 < lam < lam1
            "continuum"  show the flux in a wavelength band lam0 < lam < lam1 (aperture)
            "normedFlux"  show the flux in a wavelength band lam0 < lam < lam1, divided by spec.norm
            "norm"        show the fibre normalisation
    medianPerArm: `dict` indexed by arm (`str`) then spectrograph (`int`)
        Normalise the arms/spectrographs by medianPerArm[arm][spectrograph]
    figure: `matplotlib.figure.Figure`
        Figure to use; or None

    Return
    ------
    figure: `matplotlib.figure.Figure`
        The figure used
    medianPerArm: `dict` indexed by [`str`][`int`]
        Median flux per arm/spectrograph == medianPerArm[arm][spectrograph]
    """
    visitC = cache["visitC"]
    visitConfig = cache["visitConfig"]

    perSpec = {}
    perSpecVisit = {}
    for arm in arms:
        perSpec[arm] = np.full((len(visitC[what][arm]), 1 + 4), np.NaN)  # there's no spectrograph 0
        perSpecVisit[arm] = np.empty(perSpec[arm].shape[0])
        for iv, visit in enumerate(sorted(visitC[what][arm])):
            c = visitC[what][arm][visit].copy()
            pfsConfig = visitConfig[what][arm][visit]

            if np.sum(pfsConfig.targetType == TargetType.SKY) > 0:
                c[pfsConfig.targetType != TargetType.SKY] = np.NaN

            c /= np.nanmedian(c[pfsConfig.spectrograph == 1])
            perSpecVisit[arm][iv] = visit
            for s in set(pfsConfig.spectrograph):
                perSpec[arm][iv][s] = np.nanmedian(c[pfsConfig.spectrograph == s])

    figure, axs = plt.subplots(len(arms), 1, num=figure, sharex=True, sharey=True, squeeze=False,
                               layout="constrained")
    axs = axs.flatten()

    def format_coord(x, y):
        visit = int(perSpecVisit[arm][np.argmin(np.abs(perSpecVisit[arm] - x))])
        return f"visit {visit} {y:.3f}"

    for ax, arm in zip(axs, arms):
        plt.sca(ax)
        ax.format_coord = format_coord

        if arm not in medianPerArm:
            medianPerArm[arm] = {}

        for s in range(1, 4 + 1):
            medianPerArm[arm][s] = np.median(perSpec[arm][:, s])

            if np.isfinite(perSpec[arm][:, s]).any():
                color = f"C{s-1}"
                plt.plot(perSpecVisit[arm], perSpec[arm][:, s], '.-', label=f"{arm}{s}", color=color)
                plt.axhline(medianPerArm[arm][s], color=color, zorder=-1, alpha=0.25)

        plt.legend()

    plt.xlabel("pfs_visit")
    plt.ylabel("Relative flux in each spectrograph")
    plt.suptitle(title)

    return figure, medianPerArm


def throughputPerSpectrograph(cache, visit, arm, what="flux", title="",
                              refVisit=-1, refWhat=None, vmin=None, vmax=None):
    """
    Parameters
    ----------
    cache: `dict`
        Dictionary returned by estimateFiberThroughputs
    visit: `int`
        Visit to process
    arm: `str`
       The arm to show
    what: `str`
        Desired parameter; your choices are:
            "flux"        show the flux in a line fit in band lam0 < lam < lam1
            "continuum"   show the flux in a wavelength band lam0 < lam < lam1 (aperture)
            "normedFlux"  show the flux in a line fit in band lam0 < lam < lam1, divided by spec.norm
            "norm"        show the fibre normalisation
    refVisit: `int`
       if non-negative, throughputs will be measured relative to this visit
    refWhat: `str`
       The value of `what` used for the reference visit;  use `what` if `None`
    vmin: override any other vmin value (e.g. stretchToQlQh + (low, high))
    vmax: override any other vmax value (e.g. stretchToQlQh + (low, high))
    """
    if refWhat is None:
        refWhat = what

    visitC = cache["visitC"]
    visitConfig = cache["visitConfig"]

    c = visitC[what][arm][visit].copy()
    pfsConfig = visitConfig[what][arm][visit]

    if refVisit > 0 and visit != refVisit:
        cref = visitC[refWhat][arm][refVisit]
        with np.testing.suppress_warnings() as suppress:
            suppress.filter(RuntimeWarning, "invalid value encountered in true_divide")
            c /= cref

    x = pfsConfig.fiberHole
    y = pfsConfig.spectrograph

    im = np.zeros((len(set(y)), np.max(x)))
    im *= np.NaN
    im[y-1, x-1] = c
    im /= np.nanmedian(im)

    fac = 1.5
    _vmin, _vmax = 0, fac*np.nanmedian(c)

    vmin = _vmin if vmin is None else vmin
    vmax = _vmax if vmax is None else vmax

    II = plt.imshow(im, origin="lower", aspect="auto", interpolation='none', vmin=vmin, vmax=vmax,
                    extent=(0.5, im.shape[1] + 0.5, 0.5, 4.5))

    plt.colorbar(II, label=(f"{'relative' if refVisit > 0 else ''}flux in ") + getWavelengthLabel(arm, what))
    plt.gca().set_facecolor("black")

    plt.xlabel("Fibre Hole")
    plt.ylabel("spectrograph")

    title += f"{visit} {what} {visitConfig[what][arm][visit].designName}"
    if refVisit > 0:
        title += (f"\ndivided by {refVisit} {refWhat} "
                  f"[{visitConfig[refWhat][arm][refVisit].designName}]")

    plt.title(title)
