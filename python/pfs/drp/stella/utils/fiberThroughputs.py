import numpy as np
import matplotlib.pyplot as plt

from pfs.datamodel.pfsConfig import TargetType
from pfs.drp.stella.utils.raster import addCobraIdCallback


def selectWavelengthInterval(arm):
    """We'll measure the flux in [lam0, lam1]; if bkgd0 and bkgd1 are not None, measure the background
    The background will be measured in [bkgd0, lam0] + [lam1, bkgd1]
    """
    intervals = []
    if arm == 'b':
        lamc = 557.9
        (lam0, lam1), (bkgd0, bkgd1) = (lamc - 0.5, lamc + 0.5), (lamc - 1.5, lamc + 1.5)

        intervals.append((lam0, lam1, bkgd0, bkgd1))
    elif arm == 'r':
        intervals.append((930.5, 933.1, None, None))
        intervals.append((947.25, 949, None, None))
    elif arm == 'n':
        intervals.append((1142, 1148, None, None))
    else:
        raise RuntimeError(f"selectWavelengthInterval doesn't know about arm={arm}")

    return intervals


def getWavelengthLabel(arm):
    """Generate a label for the wavelength intervals used in arm"""
    labels = []
    for lam0, lam1, b0, b1 in selectWavelengthInterval(arm):
        labels.append(f"{lam0:.1f} < $\\lambda$ < {lam1:.1f}")

    if len(labels) == 1:
        return labels[0]
    else:
        return "(" + "), (".join(labels) + ")"


def showWavelengthInterval(arm):
    for lam0, lam1, bkgd0, bkgd1 in selectWavelengthInterval(arm):
        if bkgd0:
            plt.axvspan(bkgd0, bkgd1, color='black', alpha=0.05, zorder=-2)

        plt.axvspan(lam0, lam1, color='black', alpha=0.05 if bkgd0 else 0.1, zorder=-1)


def extractFlux(y, lam, l0, l1):
    return np.nansum(np.where((lam > l0) & (lam < l1), y, np.NaN), axis=1)


def measureBkgd(y, lam, arm):
    bkgds = []
    for lam0, lam1, bkgd0, bkgd1 in selectWavelengthInterval(arm):
        if bkgd0:
            bkgd = extractFlux(y, lam, bkgd0, lam0) + extractFlux(y, lam, lam1, bkgd1)
            bkgd /= (bkgd1 - bkgd0 - (lam1 - lam0))
        else:
            bkgd = 0*extractFlux(y, lam, lam0, lam1)

        bkgds.append(bkgd)

    return bkgds


def measureFlux(y, lam, arm):
    fluxes = []
    for (lam0, lam1, bkgd0, bkgd1), bkgd in zip(selectWavelengthInterval(arm), measureBkgd(y, lam, arm)):
        flux = extractFlux(y, lam, lam0, lam1)
        flux -= (lam1 - lam0)*bkgd

        fluxes.append(flux)

    return np.sum(fluxes, axis=0)


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
        Desired parameter; your choices are:
            "flux"        calculate the flux in a wavelength band lam0 < lam < lam1
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

    # setup cache
    cache["visitC"] = visitC
    cache["visitConfig"] = visitConfig

    for w in ["flux", "normedFlux", "norm"]:
        if w not in visitC:
            visitC[w] = {}
        if w not in visitConfig:
            visitConfig[w] = {}

        for a in "brnm":
            if a not in visitC[w]:
                visitC[w][a] = {}
            if a not in visitConfig[w]:
                visitConfig[w][a] = {}

    if processList == "all":
        processList = visits.copy()

    if reprocessList:
        processList = set(processList + reprocessList)

    for i, visit in enumerate(visits):
        if visit not in processList:
            continue

        for arm in arms:
            dataId = dict(visit=visit, arm=arm)

            if visit in reprocessList:
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

            if not (dataId["arm"] in visitC[what] and dataId["visit"] in visitC[what][dataId["arm"]]):
                if verbose > 0:
                    frac = 1.0 if len(visits) == 1 else i/(len(visits) - 1)
                    print(f"{visit} {arm} {int(100*frac + 0.5)}%\r", end='', flush=True)

                pfsConfig = butler.get("pfsConfig", dataId, spectrograph=2)
                pfsConfig = pfsConfig[pfsConfig.targetType != TargetType.ENGINEERING]

                ll = np.ones(len(pfsConfig), dtype=bool)
                c = []
                for spectrograph in range(1, 5):
                    dataId.update(spectrograph=spectrograph)
                    try:
                        spec = butler.get("pfsArm", dataId)
                    except Exception as e:
                        if not missingCamera(arm, spectrograph):
                            print(f"Error reading pfsArm for {dataId}: {e}")

                        ll &= pfsConfig.spectrograph != spectrograph
                        continue

                    if what == "flux":
                        y = spec.flux
                    elif what == "normedFlux":
                        y = spec.flux/spec.norm
                    elif what == "norm":
                        y = spec.norm
                    else:
                        raise RuntimeError(f"Unknown quantity to plot: {what}")

                    y = list(measureFlux(y, spec.wavelength, dataId["arm"]))

                    if False:
                        #
                        # Deal with fibres missing in spec;  PIPE2D-1401
                        # We insert NaNs into the measurements in the proper places
                        #
                        config = pfsConfig.select(spectrograph=spectrograph)
                        for fid in sorted(set(config.fiberId) - set(spec.fiberId)):
                            j = np.where(config.fiberId == fid)[0][0]
                            y[j:j] = [np.NaN]

                    c.append(y)

                visitC[what][dataId["arm"]][dataId["visit"]] = np.array(sum(c, []))
                visitConfig[what][dataId["arm"]][dataId["visit"]] = pfsConfig[ll]

                assert arm == dataId['arm'] and visit == dataId['visit']

    return cache


def plotThroughputs(cache, visits, arms, what="flux", refVisit=-1, showHome=False,
                    normalizeArms=True, stretchToQlQh=True, low=5, high=95,
                    medianPerArm={}, pfi=None, gfm=None, extraTitle=None, saveFig=True):
    """
    Parameters
    ----------
    cache: `dict`
        Dictionary returned by estimateFiberThroughputs
    visits: `list` of `int`
        List of visits to process (but see processList)
    arms: `str` or `list` of `str`
       The arms to process; e.g. "brn" or ["b", "r", "n"]; your choice
    what: `str`
        Desired parameter; your choices are:
            "flux"        show the flux in a wavelength band lam0 < lam < lam1
            "normedFlux"  show the flux in a wavelength band lam0 < lam < lam1, divided by spec.norm
            "norm"        show the fibre normalisation
    refVisit: `int`
       if non-negative, throughputs will be measured relative to this visit
    showHome: `bool`
       Show cobras at their home positions;  requires pfi and gfm
    stretchToQlQh: `bool`
        Stretch the data to show the data between a high and low quantile (see arguments low and high),
        after normalising to the median; if false, stretch in (0, 1) after normalising to the maximum
    low:  `float`
        Lower percentile used by stretchToQlQh (default: 5)
    high:  `float`
        Higher percentile used by stretchToQlQh (default: 95)
    medianPerArm: `dict` indexed by arm (`str`) then spectrograph (`int`)
        Normalise the arms/spectrographs by medianPerArm[arm][spectrograph]
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

    if showHome:  # draw the cobras at their home positions
        if gfm is None or pfi is None:
            print("You must provide gfm and pfi in order to use showHome; ignoring")
            showHome = False

    for visit in visits:
        for arm in arms:
            dataId = dict(visit=visit, arm=arm)

            plt.clf()

            if dataId["arm"] in visitC[what] and dataId["visit"] in visitC[what][dataId["arm"]]:
                c = visitC[what][dataId["arm"]][dataId["visit"]].copy()
                pfsConfig = visitConfig[what][dataId["arm"]][dataId["visit"]]
            else:
                raise RuntimeError(f"Unable to read value of {what} from cache for {dataId}")

            title = [f"{dataId['visit']}  {dataId['arm']}"]
            lam0, lam1 = selectWavelengthInterval(dataId["arm"])[0][:2]  # XXX just the first pair
            colorbarlabel = [f"{'relative' if refVisit > 0 else ''}flux in "
                             + getWavelengthLabel(arm)]
            title.append(f"[{pfsConfig.designName}]")
            title.append("\n")
            title.append(dict(flux="",
                              normedFlux=" (divided by norm)",
                              norm=" (norm)").get(what, what))
            if extraTitle:
                title.append(extraTitle)

            if refVisit > 0:
                title.append(f"\ndivided by {refVisit}  "
                             f"[{visitConfig[what][dataId['arm']][refVisit].designName}]")

            if False and showHome:
                title.append(" at home")

            if refVisit > 0 and dataId["visit"] != refVisit:
                cref = visitC[what][dataId["arm"]][refVisit]
                with np.testing.suppress_warnings() as suppress:
                    suppress.filter(RuntimeWarning, "invalid value encountered in true_divide")
                    c /= cref

            if normalizeArms:
                if arm in medianPerArm:
                    for s in medianPerArm[arm]:
                        c[pfsConfig.spectrograph == s] /= medianPerArm[arm][s]

            if stretchToQlQh:
                c /= np.nanmedian(c)
                colorbarlabel[0] = "relative " + colorbarlabel[0]
                vmin, vmax = np.nanpercentile(c, [low, high])
                title.append(f"stretched {low}% .. {high}%")
            else:
                fac = 1.5
                vmin, vmax = 0, fac*np.nanmedian(c)
                colorbarlabel.append(f"divided by {fac}*median")

            if showHome:  # draw the cobras at their home positions
                x, y = [], []
                for fid in pfsConfig.fiberId:
                    cid = gfm.cobraId[gfm.fiberId == fid][0]
                    zc = pfi.centers[cid - 1]
                    x.append(zc.real)
                    y.append(zc.imag)
            else:
                x, y = pfsConfig.pfiCenter.T

            S = plt.scatter(x, y, c=c, vmin=vmin, vmax=vmax, s=10)

            plt.colorbar(S, label="\n".join(colorbarlabel))

            plt.xlabel("x (mm)")
            plt.ylabel("y (mm)")
            plt.title(" ".join(title))

            plt.xlim(plt.ylim(-240, 240))

            plt.gca().set_aspect(1)

            if saveFig:
                if what == "flux" and refVisit > 0:
                    baseName = "relativeThroughput"
                else:
                    baseName = what

                plt.savefig(f"{baseName}-{visit}-{arm}.pdf", dpi=300)

            if False:
                addCobraIdCallback(plt.gcf(), pfi, gfm, textColor='black')


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


def throughputPerSpectrograph(cache, visit, arm, what="flux", title=""):
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
            "flux"        show the flux in a wavelength band lam0 < lam < lam1
            "normedFlux"  show the flux in a wavelength band lam0 < lam < lam1, divided by spec.norm
            "norm"        show the fibre normalisation
    """
    visitC = cache["visitC"]
    visitConfig = cache["visitConfig"]

    c = visitC[what][arm][visit]
    pfsConfig = visitConfig[what][arm][visit]

    x = pfsConfig.fiberHole
    y = pfsConfig.spectrograph

    im = np.zeros((len(set(y)), np.max(x)))
    im[y-1, x-1] = c

    fac = 1.5
    vmin, vmax = 0, fac*np.nanmedian(c)

    II = plt.imshow(im, origin="lower", aspect="auto", interpolation='none', vmin=vmin, vmax=vmax,
                    extent=(0.5, im.shape[1] + 0.5, 0.5, 4.5))

    plt.colorbar(II, label=f"flux in {getWavelengthLabel(arm)}")

    plt.xlabel("Fibre Hole")
    plt.ylabel("spectrograph")
    plt.title(title)
