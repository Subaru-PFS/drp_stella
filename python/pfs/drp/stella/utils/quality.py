import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lsst.daf.persistence as dafPersist

from .guiders import pd_read_sql
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