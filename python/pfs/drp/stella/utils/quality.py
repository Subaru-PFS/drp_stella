import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lsst.daf.butler import DatasetNotFoundError

from .sysUtils import pd_read_sql
from .stability import addTraceLambdaToArclines


__all__ = ["momentsToABT", "getFWHM", "computeImageQuality", "plotImageQuality",
           "loadImageQualityData", "showImageQuality",
           "showCobraConvergence", "opaqueColorbar"]


from contextlib import contextmanager


@contextmanager
def opaqueColorbar(S):
    """A contextmanager to make a colorbar opaque (alpha=1)
    E.g.
       with opaqueColorbar(S):
          plt.colorbar(S, label="Wavelength (nm)")
    """
    a = S.get_alpha()
    try:
        S.set_alpha(1)
    except AttributeError:
        a = None
    try:
        yield S
    finally:
        if a:
            S.set_alpha(a)


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


def getFWHM(als):
    """Return the Gaussian-equivalent FWHM and position angle from an arcLines object"""
    a, b, theta = momentsToABT(als.xx, als.xy, als.yy)
    r = np.sqrt(a*b)
    return 2*np.sqrt(2*np.log(2))*r, theta


def computeImageQuality(als):
    """Compute image quality metrics from arc line measurements.

    Parameters
    ----------
    als : `ArcLineSet` or `pandas.DataFrame`
        Arc line measurements, enriched with ``lam``, ``lamErr``, and
        ``tracePos`` columns by `addTraceLambdaToArclines`.

    Returns
    -------
    data : `pandas.DataFrame`
        Arc line data with additional columns:

        ``fwhm``
            Gaussian-equivalent FWHM in pixels.  When full second moments are
            not available, populated from the trace width (``xx``).  All
            ``NaN`` when no finite shape measurements are available.
        ``theta``
            Position angle of the PSF major axis in radians.  ``NaN`` when
            only trace widths are available, or when no finite measurements
            exist.
        ``traceOnly``
            ``True`` when only trace widths (``xx``) were available, not the
            full second-moment tensor (``xx``, ``xy``, ``yy``).
    """
    data = als.data.copy() if hasattr(als, 'data') else als.copy()

    ll_full = np.isfinite(data.xx + data.xy + data.yy)
    if ll_full.sum() > 0:
        traceOnly = False
        fwhm, theta = getFWHM(data)
    elif np.isfinite(data.xx).sum() > 0:
        traceOnly = True
        fwhm = data["xx"].copy()
        theta = pd.Series(np.full(len(data), np.nan), index=data.index)
    else:
        traceOnly = False
        fwhm = pd.Series(np.full(len(data), np.nan), index=data.index)
        theta = pd.Series(np.full(len(data), np.nan), index=data.index)

    data["fwhm"] = fwhm
    data["theta"] = theta
    data["traceOnly"] = traceOnly

    return data


def plotImageQuality(
    ax,
    data,
    *,
    showWhisker=False,
    showFWHM=False,
    showFWHMAgainstLambda=False,
    showFWHMHistogram=False,
    showFluxHistogram=False,
    minFluxPercentile=10,
    vmin=2.5,
    vmax=3.5,
    maxFwhm=8,
    logScale=True,
    gridsize=100,
    stride=1,
    useSN=False,
):
    """Draw a single image-quality panel on *ax*.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to draw on.
    data : `pandas.DataFrame`
        Output of `computeImageQuality`.  Must contain ``fwhm``, ``theta``,
        ``x``, ``y``, ``flux``, ``fluxErr``, ``flag``, ``fiberId``, and
        ``lam`` columns.
    showWhisker : `bool`
        Draw FWHM as a whisker (quiver) plot coloured by FWHM magnitude.
    showFWHM : `bool`
        Draw a 2D spatial hexbin / scatter map of FWHM.
    showFWHMAgainstLambda : `bool`
        Scatter FWHM vs log(flux) or S/N, coloured by wavelength.
    showFWHMHistogram : `bool`
        Histogram of FWHM values.
    showFluxHistogram : `bool`
        Histogram of line fluxes.
    minFluxPercentile : `float`
        Minimum flux percentile for line selection in spatial plots.
    vmin, vmax : `float`
        FWHM color-scale range (pixels).
    maxFwhm : `float`
        Upper FWHM cutoff for line selection and histogram binning (pixels).
    logScale : `bool`
        Log y-axis for histograms.
    gridsize : `int`
        hexbin grid size; use ``<=0`` for scatter plot instead.
    stride : `int`
        Fiber-ID stride for downsampling in spatial plots.
    useSN : `bool`
        Use S/N instead of log10(flux) on the x-axis of FWHM-vs-λ.

    Returns
    -------
    C : `matplotlib.cm.ScalarMappable` or ``None``
        Colorable artist suitable for passing to ``fig.colorbar()``,
        or ``None`` when no colorbar is applicable.
    colorbarLabel : `str` or ``None``
        Colorbar label string, or ``None``.

    Raises
    ------
    RuntimeError
        If all provided fluxes are NaN (spatial/flux plots only), or no plot
        mode is enabled.
    """
    fwhm = data["fwhm"]
    theta = data["theta"]
    C = None
    colorbarLabel = None

    if showWhisker or showFWHM or showFWHMAgainstLambda:
        if np.sum(np.isfinite(data["flux"])) == 0:
            raise RuntimeError("All the provided fluxes are NaN")

        q10_arr = np.nanpercentile(data["flux"], [minFluxPercentile])
        if np.isnan(q10_arr).any():
            q10_arr = [np.nan]
        q10 = q10_arr[0]

        ll = np.isfinite(data["fwhm"])
        ll &= ~data["flag"]
        ll &= fwhm < maxFwhm
        ll &= data["flux"] > q10
        if not useSN:
            ll &= data["flux"] > 0      # guard against log10(<=0)
        else:
            ll &= data["fluxErr"] > 0   # guard against division by zero
        if stride > 1:
            ll &= (data["fiberId"] % stride) == 0

        norm = plt.Normalize(vmin, vmax)
        colorbarLabel = "FWHM (pixels)"

        if showWhisker:
            imageSize = 4096            # used in estimating scale
            arrowSize = 4
            cmap = plt.colormaps["viridis"]
            C = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            Q = ax.quiver(
                data["x"][ll], data["y"][ll],
                (fwhm * np.cos(theta))[ll], (fwhm * np.sin(theta))[ll],
                fwhm[ll], cmap=cmap, norm=norm,
                headwidth=0, pivot="middle",
                angles="xy", scale_units="xy", scale=arrowSize * 30 / imageSize,
            )
            ax.quiverkey(Q, 0.1, 1.025, arrowSize, label=f"{arrowSize:.2g} pixels")
        elif showFWHM:
            if gridsize <= 0:
                C = ax.scatter(data["x"][ll], data["y"][ll], c=fwhm[ll], s=5, norm=norm)
            else:
                C = ax.hexbin(data["x"][ll], data["y"][ll], fwhm[ll], norm=norm, gridsize=gridsize)
        elif showFWHMAgainstLambda:
            xarr = data["flux"] / data["fluxErr"] if useSN else np.log10(data["flux"])
            C = ax.scatter(xarr[ll], fwhm[ll], c=data["lam"][ll], marker=".", alpha=0.75)
            colorbarLabel = "Wavelength (nm)"
            ax.set_xlabel("Signal/Noise" if useSN else "lg(flux)")
            ax.set_ylabel("FWHM (pixels)")

        if not showFWHMAgainstLambda:
            ax.set_ylim(-1, 4096)
            ax.set_xlim(-1, 4096)
            ax.set_xlabel("x (pixels)")
            ax.set_ylabel("y (pixels)")
            ax.set_aspect(1)
    else:
        ll_hist = np.isfinite(fwhm) & ~data["flag"]

        if showFWHMHistogram:
            ax.hist(fwhm[ll_hist], bins=np.linspace(0, maxFwhm, 100))
            ax.set_xlabel("FWHM (pix)")
        elif showFluxHistogram:
            finite_flux = data["flux"][ll_hist]
            if np.sum(np.isfinite(finite_flux)) == 0:
                raise RuntimeError("All the provided fluxes are NaN")
            q99 = np.nanpercentile(finite_flux, [99])[0]
            ax.hist(finite_flux, bins=np.linspace(0, q99, 100))
            ax.set_xlabel("flux")
        else:
            raise RuntimeError(
                "No plot mode enabled; set one of showWhisker, showFWHM, "
                "showFWHMAgainstLambda, showFWHMHistogram, or showFluxHistogram"
            )

        if logScale:
            ax.set_yscale("log")

    if showWhisker:
        ax.set_title("FWHM Whisker")
    elif showFWHM:
        ax.set_title("FWHM Map")
    elif showFWHMAgainstLambda:
        ax.set_title("FWHM vs S/N" if useSN else "FWHM vs lg(flux)")
    elif showFWHMHistogram:
        ax.set_title("FWHM Histogram")
    elif showFluxHistogram:
        ax.set_title("Flux Histogram")

    return C, colorbarLabel


def loadImageQualityData(dataIds, butler, alsCache=None):
    """Load arc line measurements into a cache dict using a butler.

    Fetches ``lines`` and ``detectorMap`` datasets for any ``dataId`` not
    already present in *alsCache*, enriches the arc lines with wavelength
    and trace-position information via `addTraceLambdaToArclines`, and
    stores the result in the cache.

    Parameters
    ----------
    dataIds : iterable of `dict`
        Each dict must contain ``visit``, ``arm``, and ``spectrograph`` keys.
    butler : `lsst.daf.butler.Butler`
        Butler used to read ``lines`` and ``detectorMap`` datasets.
    alsCache : `dict`, optional
        Existing cache to update; entries already present are not re-fetched.
        If ``None``, a new empty cache is created.

    Returns
    -------
    alsCache : `dict`
        Mapping from ``"%(visit)d %(arm)s%(spectrograph)d"`` strings to
        enriched `ArcLineSet` objects (or ``None`` for datasets not found).
    """
    if alsCache is None:
        alsCache = {}

    for dataId in dataIds:
        if dataId is None:
            continue
        dataIdStr = '%(visit)d %(arm)s%(spectrograph)d' % dataId
        if dataIdStr not in alsCache or alsCache[dataIdStr] is None:
            detMap = butler.get("detectorMap", dataId)
            try:
                alsCache[dataIdStr] = addTraceLambdaToArclines(butler.get('lines', dataId), detMap)
            except DatasetNotFoundError:
                alsCache[dataIdStr] = None

    return alsCache


def showImageQuality(dataIds, showWhisker=False, showFWHM=False, showFWHMAgainstLambda=False,
                     showFWHMHistogram=False, showFluxHistogram=False,
                     useSN=False,
                     assembleSpectrograph=True,
                     minFluxPercentile=10,
                     vmin=2.5, vmax=3.5, maxFwhm=8,
                     logScale=True, gridsize=100, stride=1,
                     butler=None, alsCache=None, title="", figure=None):
    """
    Make QA plots for image quality

    dataIds: list of dataIds to analyze
    showWhisker: Show a whisker plot [default]
    showFWHM:    Show a 2-D image of the FWHM
    showFWHMAgainstLambda: Plot the FWHM against the log of the line flux (or S/N if useSN is True)
    showFWHMHistogram:    Show a histogram of the FWHM
    showFluxHistogram:    Show a histogram of line fluxes
    assembleSpectrograph: If true, merge visits and arrange plots as b[r,m]n columns
    vmin, vmax:   Range for norm of FWHM plots (default: 2.0, 3.5)
    maxFwhm:     Upper FWHM cutoff for line selection and histogram binning (default: 8)
    minFluxPercentile: Minimum percentile of flux to include lines (per detector; default 10)
    logScale:    Show log histograme [default]
    gridsize:    Passed to hexbin (default: 100, the same as matplotlib)
                 If <= 0, use plt.scatter() instead
    stride:      Stride when traversing fiberId (default: 1)
    useSN:       Use signal/noise rather than lg(flux) in showFWHMAgainstLambda plots
    butler:      A butler to read data not already in alsCache.  Either butler
                 or alsCache must be provided.
    alsCache:    A pre-populated cache of arc line data (from a previous call or
                 from `loadImageQualityData`).  Either butler or alsCache must be
                 provided.  Returned by this function so it can be reused.
    figure:      The figure to use; or None

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
    if butler is not None:
        alsCache = loadImageQualityData(dataIds, butler, alsCache)
    elif alsCache is None:
        raise RuntimeError("Please provide either a butler or a pre-populated alsCache")

    if not (showWhisker or showFWHM or showFWHMHistogram or showFluxHistogram or showFWHMAgainstLambda):
        showWhisker = True

    if len(dataIds) == 0:
        raise RuntimeError("Please provide some data for me to analyze")

    visits = sorted(set([dataId["visit"] for dataId in dataIds]))
    spectrographs = sorted(set([dataId["spectrograph"] for dataId in dataIds]))
    _arms = set([dataId["arm"] for dataId in dataIds])
    arms = []
    for a in "brmn":    # show the arms in this order
        if a in _arms:
            arms[0:0] = [a]

    if assembleSpectrograph:
        ny = len(arms)
        nx = len(spectrographs)

        SMs = {}
        for a in arms:
            for s in spectrographs:
                k = (a, s)
                for v in visits:
                    if k not in SMs:
                        SMs[k] = []

                    if dict(visit=v, arm=a, spectrograph=s) in dataIds:
                        SMs[k].append(v)

        n = len(SMs)
    else:
        n = len(dataIds)
        ny = int(np.sqrt(n))
        nx = n//ny
        if nx*ny < n:
            ny += 1

        SMs = {}
        for dataId in dataIds:
            v, s, a = dataId["visit"], dataId["spectrograph"], dataId["arm"]
            k = (a, s)
            if k not in SMs:
                SMs[k] = []
            SMs[k].append(v)

    fig, axs = plt.subplots(ny, nx, num=figure, sharex=True, sharey=True, squeeze=False, layout="constrained")
    axs = axs.flatten()

    # We need to fake SMs if we've been given a list of dataIds, but don't want to assemble
    # them into a set of spectrographs
    if assembleSpectrograph:
        pass
    else:
        SMs = {}
        for dataId in dataIds:
            SMs[(dataId["arm"], dataId["spectrograph"], dataId["visit"])] = [dataId["visit"]]

    C = None
    for i, (ax, SM) in enumerate(zip(axs, SMs)):
        plt.sca(ax)

        a, s = SM[:2]
        visits = SMs[SM]
        if not visits:
            ax.set_axis_off()
            continue

        v = visits[0]
        dataIdStr = f"{v} {a}{s}"
        als = alsCache[dataIdStr]
        if als is None:
            continue

        if len(visits) > 1:             # concatenate the sets of arcLines
            als = als.data.copy()
            for v in visits[1:]:
                dataIdStr = f"{v} {a}{s}"
                assert dataIdStr in alsCache

                if alsCache[dataIdStr] is not None:
                    als = pd.concat([als, alsCache[dataIdStr].data])

        try:
            als = computeImageQuality(als)
        except RuntimeError:
            ax.set_axis_off()
            continue
        if not np.isfinite(als["fwhm"]).any():
            ax.set_axis_off()
            continue

        try:
            C, colorbarLabel = plotImageQuality(
                ax, als,
                showWhisker=showWhisker,
                showFWHM=showFWHM,
                showFWHMAgainstLambda=showFWHMAgainstLambda,
                showFWHMHistogram=showFWHMHistogram,
                showFluxHistogram=showFluxHistogram,
                minFluxPercentile=minFluxPercentile,
                vmin=vmin,
                vmax=vmax,
                maxFwhm=maxFwhm,
                logScale=logScale,
                gridsize=gridsize,
                stride=stride,
                useSN=useSN,
            )
        except RuntimeError:
            ax.set_axis_off()
            continue

        ax.label_outer()
        txt = dataIdStr[-2:] if assembleSpectrograph else dataIdStr
        plt.text(0.9, 1.02, txt, transform=ax.transAxes, ha='right')

    for i in range(n, nx*ny):
        axs[i].set_axis_off()

    if showWhisker or showFWHM or showFWHMAgainstLambda:
        # Setting shrink is a black art
        if ny == 1:
            if showFWHMAgainstLambda:
                shrink = 1
            else:
                shrink = 0.99 if nx == 1 else 0.95/nx
        elif ny == 2:
            shrink = 0.99 if nx <= 4 else 0.72
        else:
            shrink = 1 if nx <= 2 else 0.93 if nx <= 4 else 0.85

        if C:
            with opaqueColorbar(C):
                fig.colorbar(C, shrink=shrink, label=colorbarLabel, ax=axs)

    kwargs = {}

    if assembleSpectrograph:
        kwargs.update(y={1: 1.0, 2: 0.8, 3: 1.0}.get(ny, 1))
    elif showWhisker and ny < nx:
        kwargs.update(y=0.76)

    if not title:
        title = f"visit{'s' if len(visits) > 1 else ''} {' '.join([str(v) for v in visits])}"
    plt.suptitle(title, **kwargs)

    return alsCache


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Look at cobra convergence
#
def getCobraDesignForVisit(opdb, pfs_visit_id):
    with opdb:
        tmp = pd_read_sql(f'''
            SELECT DISTINCT
                cobra_target.pfs_visit_id,
                cobra_target.cobra_id,
                pfi_nominal_x_mm,
                pfi_nominal_y_mm,
                center_x_mm as cobra_center_x_mm,
                center_y_mm as cobra_center_y_mm,
                motor_theta_length_mm,
                motor_phi_length_mm
            FROM cobra_target
            JOIN cobra_geometry on cobra_geometry.cobra_id = cobra_target.cobra_id
            WHERE
              cobra_target.pfs_visit_id = {pfs_visit_id} OR
              cobra_target.pfs_visit_id = (SELECT visit0 FROM pfs_config_sps WHERE
                                           pfs_visit_id = {pfs_visit_id})
            ''', opdb)
    return tmp


def getConvergenceForVisit(opdb, pfs_visit_id, calculateMean=True):
    with opdb:
        tmp = pd_read_sql(f'''
            SELECT DISTINCT
                mcs_exposure.mcs_frame_id, mcs_data.spot_id, cobra_match.cobra_id, cobra_match.iteration,
                pfi_center_x_mm, pfi_center_y_mm, pfi_target_x_mm, pfi_target_y_mm
            FROM mcs_exposure
            JOIN mcs_data ON mcs_data.mcs_frame_id = mcs_exposure.mcs_frame_id
            JOIN cobra_match ON cobra_match.spot_id = mcs_data.spot_id AND
                                cobra_match.mcs_frame_id = mcs_exposure.mcs_frame_id
            JOIN cobra_target ON cobra_target.pfs_visit_id = cobra_match.pfs_visit_id AND
                                 cobra_target.cobra_id = cobra_match.cobra_id AND
                                 cobra_target.iteration = cobra_match.iteration
            WHERE
              mcs_exposure.pfs_visit_id = {pfs_visit_id} OR
              mcs_exposure.pfs_visit_id = (SELECT visit0 FROM pfs_config_sps WHERE
                                           pfs_visit_id = {pfs_visit_id})
            ''', opdb)

    if calculateMean:
        grouped = tmp.groupby("cobra_id")
        tmp = grouped.agg(
            pfi_center_x_mm_mean=("pfi_center_x_mm", "mean"),
            pfi_center_y_mm_mean=("pfi_center_y_mm", "mean"),
        ).merge(tmp, on="cobra_id")

    return tmp


def getFiducialConvergenceForVisit(opdb, pfs_visit_id, calculateMean=True):
    with opdb:
        tmp = pd_read_sql(f'''
            SELECT DISTINCT
                mcs_exposure.mcs_frame_id,
                mcs_data.spot_id, fiducial_fiber_match.fiducial_fiber_id, fiducial_fiber_match.iteration,
                mcs_exposure.taken_at,
                mcs_center_x_pix, mcs_center_y_pix, mcs_exposure.altitude, mcs_exposure.insrot
                --
                -- replace previous line with this one when mm centres are available
                --   pfi_center_x_mm, pfi_center_y_mm
            FROM mcs_exposure
            JOIN mcs_data ON mcs_data.mcs_frame_id = mcs_exposure.mcs_frame_id
            JOIN fiducial_fiber_match ON fiducial_fiber_match.spot_id = mcs_data.spot_id AND
                                fiducial_fiber_match.mcs_frame_id = mcs_exposure.mcs_frame_id
            WHERE
                mcs_exposure.pfs_visit_id = {pfs_visit_id} OR
                mcs_exposure.pfs_visit_id = (SELECT visit0 FROM pfs_config_sps WHERE
                                             pfs_visit_id = {pfs_visit_id})
            ''', opdb)

    if "pfi_center_x_mm" in tmp:
        print("No need to calculate pfi_center_x_mm in getFiducialConvergenceForVisit")
    elif tmp.mcs_frame_id.nunique() == 0:
        raise RuntimeError(f"No mcs_frame_ids are associated with pfs_visit_id {pfs_visit_id}")
    else:
        from pfs.utils.coordinates.transform import makePfiTransform
        with opdb:
            tmp2 = pd_read_sql(f'''
                SELECT -- DISTINCT
                    *
                FROM mcs_pfi_transformation
                WHERE
                   mcs_frame_id IN ({", ".join([str(frame_id) for frame_id in tmp.mcs_frame_id.unique()])})
                ''', opdb)

        tmp["pfi_center_x_mm"] = np.empty(len(tmp))
        tmp["pfi_center_y_mm"] = np.empty(len(tmp))

        for mcs_frame_id in tmp.mcs_frame_id.unique():
            ai = tmp[tmp.mcs_frame_id == mcs_frame_id]
            mpt = makePfiTransform(tmp2.camera_name.iloc[0], altitude=ai.altitude.unique()[0],
                                   insrot=ai.insrot.unique()[0])
            assert ai.altitude.nunique() == 1

            mcs_pfi_transformation = tmp2[tmp2.mcs_frame_id == mcs_frame_id]

            args = []
            for p in ["x0", "y0", "theta", "dscale", "scale2"]:   # order must match mcsDistort.setArgs order
                args.append(mcs_pfi_transformation[p].iloc[0])

            mpt.mcsDistort.setArgs(args)
            mcs_center_x_mm, mcs_center_y_mm = mpt.mcsToPfi(tmp.mcs_center_x_pix, tmp.mcs_center_y_pix)

            ll = tmp.mcs_frame_id == mcs_frame_id
            for xy in ["x", "y"]:
                tmp[f"pfi_center_{xy}_mm"] = np.where(ll, (mcs_center_x_mm if xy == "x" else mcs_center_y_mm),
                                                      tmp[f"pfi_center_{xy}_mm"])

        del tmp["mcs_center_x_pix"]
        del tmp["mcs_center_y_pix"]

    if calculateMean:
        grouped = tmp.groupby("fiducial_fiber_id")
        tmp = grouped.agg(
            pfi_center_x_mm_mean=("pfi_center_x_mm", "mean"),
            pfi_center_y_mm_mean=("pfi_center_y_mm", "mean"),
        ).merge(tmp, on="fiducial_fiber_id")

    return tmp


#
# Done with code to read opdb.  On to work
#
def prepConvergenceData(cobraData, cobraIds, maxError, maxFinalError, relativeToTarget,
                        nIteration=5):
    if cobraIds is None:
        tmp = cobraData
    else:
        tmp = cobraData[cobraData.isin(dict(cobra_id=cobraIds)).cobra_id]

    tmp = tmp.copy()

    if relativeToTarget:
        tmp["dx_mm"] = tmp.pfi_center_x_mm - tmp.pfi_target_x_mm
        tmp["dy_mm"] = tmp.pfi_center_y_mm - tmp.pfi_target_y_mm
    else:
        tmp["dx_mm"] = tmp.pfi_center_x_mm - tmp.pfi_center_x_mm_mean
        tmp["dy_mm"] = tmp.pfi_center_y_mm - tmp.pfi_center_y_mm_mean

    tmp["dr_mm"] = np.hypot(tmp.dx_mm, tmp.dy_mm)

    if maxError > 0:
        grouped = tmp.groupby("cobra_id")
        tmp = grouped.agg(
            dr_mm_mean=("dr_mm", "mean"),
            dr_mm_max=("dr_mm", "max"),
        ).join(tmp.set_index('cobra_id'), on="cobra_id")

        tmp = tmp[tmp.dr_mm_max < maxError*1e-3]

    if maxFinalError > 0:
        ll = tmp.iteration == nIteration - 1
        tmp2 = pd.DataFrame(dict(cobra_id=tmp.cobra_id, dr_mm_final=tmp.dr_mm[ll]))
        tmp = tmp.join(tmp2.set_index('cobra_id'), on="cobra_id")
        tmp = tmp[tmp.dr_mm_final < maxFinalError*1e-3]

    return tmp


def showCobraConvergence(cobraData, cobraIds, fiducialData=None, fiducialIds=None, relativeToTarget=False,
                         iteration0=0,
                         keyLength=10,
                         maxError=25,
                         maxFinalError=0,
                         fiducialColor="#c0c0c0",
                         title="",
                         figure=None):
    """
    pandas dataframe as returned by getConvergenceForVisit()
    relativeToTarget: show position relative to target, rather than mean, position
    keyLength:  Length of arrow in the key, in microns
    maxError:  Ignore cobras with a maximum error larger than this (microns)
    maxFinalError:  Ignore cobras with a final error larger than this (microns)
    title: Extra string to add to title
    figure: matplotlib.Figure to use; or None
    """
    allCobraData = cobraData

    iterations = cobraData.iteration.unique()[iteration0:]
    nIteration = len(iterations)

    cobraData = prepConvergenceData(cobraData, cobraIds, maxError, maxFinalError, relativeToTarget,
                                    nIteration=nIteration)
    if fiducialData is not None:
        fiducialData = prepConvergenceData(fiducialData, fiducialIds, maxError, maxFinalError,
                                           relativeToTarget=False, nIteration=nIteration)

    ny = int(np.sqrt(nIteration))
    nx = nIteration//ny
    if nx*ny < nIteration:
        ny += 1

    fig, axs = plt.subplots(ny, nx, num=figure, sharex=True, sharey=True, squeeze=False)
    axs = axs.flatten()

    for ax, iteration in zip(axs, iterations):
        plt.sca(ax)

        ll = cobraData.iteration == iteration

        qlen = 1e-3*keyLength  # in mm
        kwargs = dict(scale_units='xy', scale=qlen*1e-2)
        Q = plt.quiver(cobraData.pfi_center_x_mm_mean[ll], cobraData.pfi_center_y_mm_mean[ll],
                       cobraData.dx_mm[ll], cobraData.dy_mm[ll], **kwargs)
        plt.quiverkey(Q, 0.2, 0.95, qlen, f"{keyLength} micron", color='red')

        if fiducialData is not None:
            ll = fiducialData.iteration == iteration
            plt.quiver(fiducialData.pfi_center_x_mm_mean[ll], fiducialData.pfi_center_y_mm_mean[ll],
                       fiducialData.dx_mm[ll], fiducialData.dy_mm[ll], color=fiducialColor, **kwargs)

        plt.plot([0], [0], '+', color='red')

        ll = allCobraData.iteration == iteration
        plt.plot(allCobraData.pfi_center_x_mm_mean[ll], allCobraData.pfi_center_y_mm_mean[ll], '.',
                 alpha=0.05, color='gray', zorder=-1)

        plt.text(0.9, 0.9, f"{iteration}", transform=ax.transAxes, ha="center")

        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")

        ax.set_aspect(1)
        ax.label_outer()

    for i in range(nIteration, nx*ny):
        axs[i].set_axis_off()

    plt.suptitle(title)
