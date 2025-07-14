import numpy as np
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, RegularPolygon
import pandas as pd

from pfs.utils.datamodel.ag import AutoGuiderStarMask
from pfs.utils.coordinates.transform import MeasureDistortion
import pfs.utils.coordinates.CoordTransp as ct
from .sysUtils import pd_read_sql
from .quality import opaqueColorbar

__all__ = ["GuiderConfig", "agcCameraCenters", "showAgcErrorsForVisits",
           "readAgcDataFromOpdb",
           "readAGCStarsForVisitByPfsVisitId", "readAGCStarsForVisitSetByPfsVisitId",
           "readAGCPositionsForVisitByAgcExposureId", "showAgcErrorsForVisitsByCamera"]

# Approximate centers of AG camera (indexed by agc_camera_id)
agcCameraCenters = {
    0: ( 237.58,   -0.50),              # noqa E201, E241
    1: ( 120.19,  212.49),              # noqa E201, E241
    2: (-120.02,  212.10),              # noqa E201, E241
    3: (-242.08,    2.00),              # noqa E201, E241
    4: (-122.58, -211.67),              # noqa E201, E241
    5: ( 119.23, -209.79),              # noqa E201, E241
}

#
# Convert guider focus offset in microns to M2_OFF3 in mm
#
# Should come from e.g. pfs_utils
#
guiderFocus_to_M2_OFF3 = 1/800


def get_guiderFocus_to_M2_OFF3():
    return guiderFocus_to_M2_OFF3

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class MeasureXYRot(MeasureDistortion):
    def __init__(self, x, y, xoff, yoff, nsigma=5, alphaRot=0.0):
        """x, y: measured positions in pfi coordinates (mm)
        xoff, yoff: measured offsets (microns)
        nsigma: clip the fitting at this many standard deviations (None => 5).  No clipping if <= 0
        alphaRot: coefficient for the dtheta^2 term in the penalty function
        """
        self.nsigma = 5 if nsigma is None else nsigma
        self.alphaRot = alphaRot

        good = np.isfinite(x + y + xoff + yoff)

        self.x = x[good]
        self.y = y[good]

        self.xtrue = (x + 1e-3*xoff)[good]
        self.ytrue = (y + 1e-3*yoff)[good]
        #
        # The correct number of initial values; must match code in __call__()
        #
        x0, y0, dscale, theta, scale2 = np.array([0, 0, 0, 0, 0], dtype=float)
        self._args = np.array([x0, y0, dscale, theta, scale2])
        self.frozen = np.zeros(len(self._args), dtype=bool)

    def __call__(self, args):
        tx, ty = self.distort(self.x, self.y, *args)

        d = np.hypot(tx - self.xtrue, ty - self.ytrue)

        if self.nsigma > 0:
            d = d[self.clip(d, self.nsigma)]

        penalty = np.sum(d**2)
        penalty += self.alphaRot*(args[2] - 0.0)**2  # include a prior on the rotation, args[2]

        return penalty

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Communicate with the DB


def readSpSInfo(opdb, taken_after=None, min_exptime=0, exp_type='object', limit=0, windowed=False,
                showQuery=False):
    """Query the database for information about SpS visits

    windowed: `bool` only return windowed reads

    N.b. limit applies _before_ we deal with slight time_exp_start scatter,
    so you may get fewer visits than you hoped for
    """
    where = []
    if taken_after is not None:
        where.append(f"sps_exposure.time_exp_start > '{taken_after}'")
    if min_exptime > 0:
        where.append(f"sps_exposure.exptime > {min_exptime}")
    if exp_type is not None:
        where.append(f"sps_visit.exp_type = '{exp_type}'")
    if windowed:
        where.append("sequence_type LIKE '%windowed'")

    WHERE = "" if where == [] else f"WHERE {' AND '.join(where)}"
    LIMIT = "" if limit <= 0 else f"LIMIT {limit}"

    with opdb:
        tmp = pd_read_sql(f'''
           SELECT DISTINCT
               sps_exposure.pfs_visit_id AS pfs_visit_id, sps_exposure.time_exp_start as taken_at, exptime,
               exp_type, altitude, azimuth, insrot, design_name
               , sequence_group.group_id, sequence_group.group_name
            FROM
               sps_exposure
            JOIN pfs_visit ON pfs_visit.pfs_visit_id = sps_exposure.pfs_visit_id
            JOIN pfs_design ON pfs_design.pfs_design_id = pfs_visit.pfs_design_id
            JOIN sps_visit ON sps_visit.pfs_visit_id = sps_exposure.pfs_visit_id
            LEFT JOIN tel_status ON tel_status.pfs_visit_id = sps_exposure.pfs_visit_id
            LEFT JOIN visit_set ON visit_set.pfs_visit_id = sps_exposure.pfs_visit_id
            JOIN iic_sequence ON iic_sequence.iic_sequence_id = visit_set.iic_sequence_id
            LEFT JOIN sequence_group ON sequence_group.group_id = iic_sequence.group_id
            {WHERE}
            {LIMIT}
           ''', opdb, showQuery=showQuery)

    # The taken_at times don't quite match between the arms, so group the data by pfs_visit_id
    grouped = tmp.groupby("pfs_visit_id", as_index=False)
    tmp = grouped.agg(
        taken_at=("taken_at", "mean"),
        exptime=("exptime", "mean"),
        exp_type=("exp_type", "first"),
        altitude=("altitude", "mean"),
        azimuth=("azimuth", "mean"),
        insrot=("insrot", "mean"),
        group_id=("group_id", "first"),
        group_name=("group_name", "first"),
        design_name=("design_name", "first"),
    )

    return tmp


def readPfsDesign(opdb, pfs_visit_id):
    with opdb:
        tmp = pd_read_sql(f'''
           SELECT
               pfs_visit.pfs_design_id,
               pfs_design.design_name
           FROM pfs_visit
           JOIN pfs_design on pfs_design.pfs_design_id = pfs_visit.pfs_design_id
           WHERE
               pfs_visit_id = {pfs_visit_id}
           ''', opdb)

    return tmp


def readAGCPositionsForVisitByAgcExposureId(opdb, pfs_visit_id, flipToHardwareCoords):
    """Query the database for useful AG related things

    N.b. shutter_open is 0/1 when the spectrographs are in use, otherwise 2
    """
    with opdb:
        tmp = pd_read_sql(f'''
           SELECT
               min(agc_exposure.pfs_visit_id) AS pfs_visit_id,
               agc_exposure.agc_exposure_id,
               agc_match.guide_star_id,
               min(agc_exposure.taken_at) AS taken_at,
               avg(agc_exposure.altitude) AS altitude,
               avg(agc_exposure.azimuth) AS azimuth,
               avg(agc_exposure.insrot) AS insrot,
               avg(sps_exposure.exptime) AS exptime,  -- SpS
               min(m2_pos3) AS m2_pos3,
               avg(agc_match.agc_camera_id) AS agc_camera_id,
               avg(agc_nominal_x_mm) AS agc_nominal_x_mm,
               avg(agc_center_x_mm) AS agc_center_x_mm,
               avg(agc_nominal_y_mm) AS agc_nominal_y_mm,
               avg(agc_center_y_mm) AS agc_center_y_mm,
               min(agc_match.flags) AS agc_match_flags,
               CASE
                   WHEN min(sps_exposure.pfs_visit_id) IS NULL THEN 2
                   WHEN (min(agc_exposure.taken_at) BETWEEN min(sps_exposure.time_exp_start) AND
                                                            min(sps_exposure.time_exp_end)) THEN 1
                   ELSE 0
               END AS shutter_open,
               min(guide_delta_insrot) as guide_delta_insrot,
               min(guide_delta_az) as guide_delta_azimuth,
               min(guide_delta_el) as guide_delta_altitude
           FROM agc_exposure
           JOIN agc_data ON agc_data.agc_exposure_id = agc_exposure.agc_exposure_id
           JOIN agc_match ON agc_match.agc_exposure_id = agc_data.agc_exposure_id AND
                             agc_match.agc_camera_id = agc_data.agc_camera_id AND
                             agc_match.spot_id = agc_data.spot_id
           JOIN agc_guide_offset ON agc_guide_offset.agc_exposure_id = agc_exposure.agc_exposure_id
           LEFT JOIN sps_exposure ON sps_exposure.pfs_visit_id = agc_exposure.pfs_visit_id
           WHERE
               agc_exposure.pfs_visit_id = {pfs_visit_id}
           GROUP BY guide_star_id, agc_exposure.agc_exposure_id
           ''', opdb)

    if flipToHardwareCoords:
        tmp.agc_nominal_y_mm *= -1
        tmp.agc_center_y_mm *= -1

    return tmp


def readAGCStars(opdb, pfs_design_id, pfs_visit_id=0):
    """Return the AG guide stars for the specified design and (optionally) pfs_visit_id
    """

    if pfs_visit_id <= 0:   # Easy!
        with opdb:
            tmp = pd_read_sql(f'''
               SELECT
                   pfs_design.pfs_design_id, pfs_design_agc.agc_camera_id,
                   pfs_design_agc.guide_star_id, guide_star_ra, guide_star_dec,
                   guide_star_pm_ra, guide_star_pm_dec, guide_star_parallax
                   ra_center_design, dec_center_design, pa_design
               FROM pfs_design
               JOIN pfs_design_agc ON pfs_design_agc.pfs_design_id = pfs_design.pfs_design_id
               WHERE
                   pfs_design.pfs_design_id = {pfs_design_id}
               ''', opdb)
    else:
        with opdb:
            tmp = pd_read_sql(f'''
               SELECT DISTINCT
                  pfs_design.pfs_design_id, pfs_config_sps.pfs_visit_id, pfs_design_agc.agc_camera_id,
                  pfs_design_agc.guide_star_id, guide_star_ra, guide_star_dec,
                  guide_star_pm_ra, guide_star_pm_dec, guide_star_parallax,
                  ra_center_config, dec_center_config, pa_config,
                  agc_final_x_pix, agc_final_y_pix
               FROM pfs_design
               JOIN pfs_design_agc ON pfs_design_agc.pfs_design_id = pfs_design.pfs_design_id
               JOIN pfs_config_sps ON pfs_config_sps.pfs_visit_id = {pfs_visit_id}
               JOIN pfs_config ON pfs_config_sps.visit0 >= pfs_config.visit0 AND
                                  pfs_config.pfs_design_id = {pfs_design_id}
               JOIN pfs_config_agc ON pfs_config_agc.pfs_design_id = pfs_design.pfs_design_id AND
                                      pfs_config_agc.guide_star_id = pfs_design_agc.guide_star_id AND
                                      pfs_config_agc.visit0 = pfs_config_sps.visit0
               WHERE
                   pfs_design.pfs_design_id = {pfs_design_id}
                   AND converg_num_iter IS NOT NULL
                   -- AND pfs_design_agc.guide_star_id = 1154055066736045824
               ''', opdb)

    return tmp


def readAgcDataFromOpdb(opdb, visits, butler=None, dataId=None):
    """Read a useful set of data about the AGs from the opdb
    opdb: connection to the opdb
    visits: list of the desired visits
    butler: a butler to read INST-PA; or None
    """
    dd = []
    for v in visits:
        pos = readAGCPositionsForVisitByAgcExposureId(opdb, v, flipToHardwareCoords=True)
        if len(pos) > 0:
            pos.exptime = np.where(pos.exptime.isna(), 0, pos.exptime)
            dd.append(pos)

    agcData = pd.concat(dd)
    del dd
    #
    # Read the INST-PA from the image headers, as it's not yet available in the opdb (INSTRM-1955)
    #
    if butler is not None:
        if dataId is None:
            dataId = dict(spectrograph=1, arm="r")  # only used for metadata lookup

        inst_pa = np.empty(len(agcData))
        for pfs_visit_id in set(agcData.pfs_visit_id):
            sel = agcData.pfs_visit_id == pfs_visit_id

            md = butler.get("raw_md", dataId, visit=pfs_visit_id)
            inst_pa[sel] = md["INST-PA"]
        agcData["inst_pa"] = inst_pa

    return agcData


def readAGCStarsForVisitSetByPfsVisitId(opdb, visits, flipToHardwareCoords=True, useTraceRadius=True,
                                        butler=None):
    """Query the database for the properties of stars measured by the AGC code

    Parameters:
    opdb:
       Connection to the OPDB database
    visits: `list` of `int`
       List of desired values of pfs_visit_id
    flipToHardwareCoords: `bool`
       See readAGCStarsForVisitByPfsVisitId
    useTraceRadius: `bool`
       See readAGCStarsForVisitByPfsVisitId
    butler: `lsst.daf.Butler`
       Needed to read M2_OFF3 from metadata pre-2025-03-21
    """

    dd = []
    for v in visits:
        stars = readAGCStarsForVisitByPfsVisitId(opdb, v, flipToHardwareCoords,
                                                 useTraceRadius=useTraceRadius, butler=butler)
        if len(stars) > 0:
            if False:
                for f in stars:
                    if np.sum(stars[f].isna()) == len(stars):
                        stars[f] = 0

            dd.append(stars)

    if len(dd) == 0:
        raise RuntimeError("I was unable to find any AGC stars in visits "
                           f"{', '.join(str(v) for v in visits)}")

    return pd.concat(dd)


def readAGCStarsForVisitByPfsVisitId(opdb, pfs_visit_id, flipToHardwareCoords=True,
                                     useTraceRadius=True, butler=None):
    """Query the database for the properties of stars measured by the AGC code

    Parameters:
    opdb:
       Connection to the OPDB database
    visits: `int`
       Desired pfs_visit_id
    flipToHardwareCoords: `bool`
       Flip sign of agc_nominal_y_mm to convert to "hardware" coordinate system
    useTraceRadius: `bool`
       Use the "trace" definition of rms; sqrt(0.5*(Mxx + Myy))
       rather than "determinant" (Mxx*Myy - Mxy**2)**(1/4)
    butler: `lsst.daf.Butler`
       Needed to read M2_OFF3 from metadata pre-2025-03-21

    See readAGCStarsForVisitSetByPfsVisitId() to read a set of visits

    N.b. shutter_open is 0/1 when the spectrographs are in use, otherwise 2
    """
    extra = "agc_exposure.azimuth, agc_exposure.adc_pa, "
    with opdb:
        tmp = pd_read_sql(f'''
           SELECT
               agc_exposure.pfs_visit_id, agc_exposure.agc_exposure_id,
               agc_exposure.agc_exptime, agc_exposure.m2_pos3,
               {extra}
               agc_exposure.altitude, agc_exposure.insrot,
               agc_data.flags as agc_data_flags,
               agc_match.guide_star_id, agc_data.agc_camera_id, pfs_design_agc.guide_star_flag,
               agc_match.flags as agc_match_flags,
               agc_center_x_mm, agc_center_y_mm,
               agc_nominal_x_mm, agc_nominal_y_mm,
               CASE
                   WHEN sps_exposure.pfs_visit_id IS NULL THEN 2
                   WHEN (agc_exposure.taken_at BETWEEN sps_exposure.time_exp_start AND
                                                       sps_exposure.time_exp_end) THEN 1
                   ELSE 0
               END AS shutter_open,
               image_moment_00_pix, centroid_x_pix, centroid_y_pix,
               -- central_image_moment_02_pix as mxx,   -- note: 02 not 20
               -- central_image_moment_11_pix as mxy,
               -- central_image_moment_20_pix as myy,   -- note: 20 not 01
               central_image_moment_20_pix as mxx,   -- fixed for May 2025 run
               central_image_moment_11_pix as mxy,
               central_image_moment_02_pix as myy,
               peak_pixel_x_pix, peak_pixel_y_pix,
               peak_intensity, background,
               estimated_magnitude,
               agc_data.flags,
               guide_delta_z,
               guide_delta_z1, guide_delta_z2, guide_delta_z3, guide_delta_z4, guide_delta_z5, guide_delta_z6
           FROM agc_exposure
           JOIN agc_data ON agc_data.agc_exposure_id = agc_exposure.agc_exposure_id
           LEFT JOIN agc_match ON agc_match.agc_exposure_id = agc_data.agc_exposure_id AND
                             agc_match.agc_camera_id = agc_data.agc_camera_id AND
                             agc_match.spot_id = agc_data.spot_id
           LEFT JOIN agc_guide_offset ON agc_guide_offset.agc_exposure_id = agc_exposure.agc_exposure_id
           JOIN pfs_visit ON pfs_visit.pfs_visit_id = agc_exposure.pfs_visit_id
           LEFT JOIN pfs_design_agc ON pfs_design_agc.guide_star_id = agc_match.guide_star_id AND
                                  pfs_design_agc.pfs_design_id = pfs_visit.pfs_design_id
           LEFT JOIN sps_exposure ON sps_exposure.pfs_visit_id = agc_exposure.pfs_visit_id
           WHERE
               agc_exposure.pfs_visit_id = {pfs_visit_id}
           ''', opdb)

    tmp.drop_duplicates(inplace=True)   # the LEFT JOINs can generate duplicate rows

    try:
        tmp.guide_star_flag = np.where(np.isfinite(tmp.guide_star_flag), tmp.guide_star_flag, 0).astype(int)
    except TypeError:
        pass

    # m2_off3, tel_ra, tel_dec, aren't available from the agc_exposure table
    #
    # We rely on the fact that both agc_exposure_id and status_sequence_id are monotonic increasing,
    # and the the latter is incremented every time that the former is
    with opdb:
        tmp1 = pd_read_sql(f'''
           SELECT
               agc_exposure_id
           FROM
               agc_exposure
           WHERE
               pfs_visit_id = {pfs_visit_id}
           ORDER BY agc_exposure_id
           ''', opdb)

        tmp2 = pd_read_sql(f'''
           SELECT
               m2_off3, tel_ra, tel_dec
           FROM
               tel_status
           WHERE
               pfs_visit_id = {pfs_visit_id} AND caller = 'agcc' -- 'ag'
           ORDER BY status_sequence_id
           ''', opdb)

        doConcat = True                 # OK to merge tmp1 and tmp2
        if len(tmp1) != len(tmp2):      # should be impossible, but ...
            l1, l2 = len(tmp1), len(tmp2)
            if True:
                action = "dropping last "
                if l1 < l2:
                    tmp2 = tmp2[:l1 - l2]
                    action += ("row" if l2 - l1 == 1 else f"{l2 - l1} rows") + " of tel_status"
                else:
                    tmp1 = tmp1[:l2 - l1]
                    action += ("row" if l1 - l2 == 1 else f"{l1 - l2} rows") + " of agc_exposure"
            else:
                if l1 == l2 - 1:
                    tmp2 = tmp2[:-1]
                    action = "dropping last row of tel_status"
                else:
                    tmp["m2_off3"] = np.full(len(tmp), np.nan)
                    action = "m2_off3 set to NaN"
                    doConcat = False

            print(f"pfs_visit_id {pfs_visit_id}: mismatch in number of agc_exposures ({l1}) and "
                  f"number of AG entries in tel_status ({l2}); {action}")

        if doConcat:
            if np.sum(~pd.isnull(tmp2.m2_off3)) == 0:
                tmp2.fillna(np.nan, inplace=True)  # generates a pandas 3.0 warning which I don't understand

            tmp1 = pd.concat([tmp1, tmp2], axis=1)

            tmp = pd.merge(tmp, tmp1, on='agc_exposure_id')

    if flipToHardwareCoords:
        tmp.agc_nominal_y_mm *= -1

    setImageSizes(tmp, useTraceRadius)

    if len(tmp) > 0 and \
       np.sum(np.isfinite(tmp.m2_off3)) == 0:   # Not available from tel_status until 2025-03-21.
        if butler is not None:
            #  Fake it from m2_pos3
            W_M2OFF3 = np.zeros(len(tmp))
            for v in tmp.pfs_visit_id.unique():
                ll = tmp.pfs_visit_id == v
                W_M2OFF3[ll] = find_W_M2OFF3(butler, v) + (tmp.m2_pos3[ll] - tmp.m2_pos3[ll].iloc[0])
                tmp["m2_off3"] = W_M2OFF3

    return tmp


def readTelStatus(opdb, pfs_visit_id):
    """Return the telescope status for a given pfs_visit_id
    """

    with opdb:
        tmp = pd_read_sql(f'''
            SELECT
                *
            FROM tel_status
            WHERE
                pfs_visit_id = {pfs_visit_id}
            ''', opdb)

    return tmp


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def stdFromIQR(im):
    Q1, Q3 = np.percentile(im, [25, 75])
    return 0.74130*(Q3 - Q1)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def showAgcErrorsForVisits(agcData,
                           pfs_visit_ids=None,
                           agc_exposure_ids=None,
                           byTime=False,
                           yminmax=None,
                           figure=None,
                           livePlot=None,
                           showLegend=True):
    """
    agcData: pandas DataFrame as returned by readAgcDataFromOpdb
    pfs_visit_ids: only show these visits (default: None => all)
    agc_exposure_ids: only show these agc exposures (default: None => all)
    byTime:  use time, not agc_exposure_id, as x-axis (default: False)
    yminmax: float Scale figures into +- yminmax (or 0, sqrt(2)*yminmax for rerror); or None
    figure:  matplotlib figure to use, or None
    """
    if livePlot is None:
        fig, axs = plt.subplots(3, 1, num=figure, sharex=True, sharey=False, squeeze=False)
        axs = axs.flatten()
    else:
        fig, axs = livePlot.fig, livePlot.axes[:3]

    if pfs_visit_ids is not None:
        agcData = agcData[agcData.isin(dict(pfs_visit_id=pfs_visit_ids)).pfs_visit_id]

    if agc_exposure_ids is not None:
        agcData = agcData[agcData.isin(dict(agc_exposure_id=agc_exposure_ids)).agc_exposure_id]

    if len(agcData) == 0:
        raise RuntimeError("I have no data to plot")

    agc_exposure_ids = np.array(sorted(set(agcData.agc_exposure_id)))

    grouped = agcData.groupby("agc_exposure_id")

    pfs_visit_ids = grouped.pfs_visit_id.min()
    taken_ats = grouped.taken_at.mean()
    shutter_open = grouped.shutter_open.max()

    tmp = pd.DataFrame(dict(agc_exposure_id=agcData.agc_exposure_id,
                            dx=agcData.agc_nominal_x_mm - agcData.agc_center_x_mm,
                            dy=agcData.agc_nominal_y_mm - agcData.agc_center_y_mm))
    grouped = tmp.groupby("agc_exposure_id")
    xbar, ybar = grouped.agg(xbar=('dx', 'mean')), grouped.agg(ybar=('dy', 'mean')).to_numpy()
    rbar = np.hypot(xbar, ybar)

    def plot_zbar(ax, zbar, xvec=agc_exposure_ids):
        ax.set_prop_cycle(None)
        for pfs_visit_id in sorted(set(pfs_visit_ids)):
            sel = pfs_visit_ids == pfs_visit_id
            color = ax.plot(xvec[sel], 1e3 * zbar[sel], '.-', label=f"{int(pfs_visit_id)}")[0].get_color()
            sel &= shutter_open > 0
            ax.plot(xvec[sel], 1e3 * zbar[sel], 'o', color=color)

        ax.axhline(0, color='black')
        if showLegend:
            ax.legend(ncol=6)

    for ax, vals, label in zip(axs, [rbar, xbar, ybar], ['rerror', 'xerror', 'yerror']):
        plot_zbar(ax, vals, taken_ats if byTime else agc_exposure_ids)
        ax.set_ylabel(f"{label} (microns)")

        if yminmax is not None:
            scale = np.array([-0.1, np.sqrt(2)]) if label == 'rerror' else np.array([-1, 1])
            ax.set_ylim(yminmax * scale)

    ax.set_xlabel("HST" if byTime else "agc_exposure_id")

    visits = sorted(set(pfs_visit_ids))
    fig.suptitle(f"pfs_visit_ids {visits[0]:.0f}..{visits[-1]:.0f}")


def showAgcErrorsForVisitsByCamera(opdb=None, agcData=None, visits=[],
                                   AGC=range(1, 6+1), plotBy="agc_exposure_id", colorBy="camera",
                                   showCamerasAsLegend=True,
                                   showAltInsrot=False, plotXY=False, connectDxDy=False, plotXYStride=1,
                                   nVisitMin=0,
                                   yminmax=0, what="agc_exposure_id", showFocusSets=False,
                                   alpha=0.5, figure=None, axes=None):
    """
    Break down AGC errors per camera

    Note the this routine can read from the database using readAGCStarsForVisitSetByPfsVisitId(), or it
    can reuse a passed-in dataframe.  Either way it returns the dataframe, which may be used next time
    (and the requested visits are silently ignored). It may also be passed to plotFocus

       agcData = None
       agcData = showAgcErrorsForVisitsByCamera(opdb, agcData=agcData, ...)
       agcData = showAgcErrorsForVisitsByCamera(opdb, agcData=agcData, ...)
       ...

    showAltInsrot:  show offsets as a function of altitude and insrot
    showCamerasAsLegend: If true, identify cameras in a legend not a colorbar (True)
    plotXY: `bool`  plot the per-camera offsets
    connectDxDy: `bool` Connect the offsets when plotXY is True
    plotXYStride: `int` Only plot every plotXYStride'th agc_exposure when plotXY is True
       If None, plot averages per visits
    nVisitMin: `int` only plot visits when plotXY with more than this many points in a camera
    figure: the matplotlib.Figure to use; or None
    axes: An array of the appropriate number of matplotlib.Axes to use; or None
    """
    if plotXY:
        plots = ["xy"]
    else:
        plots = ["y", "x"]

    nPanel = len(plots)
    if axes is None:
        figure, axes = plt.subplots(nPanel, 1, num=figure, sharex=True, squeeze=False)
        axes = axes.flatten()
    else:
        assert len(axes) == nPanel, f"I need {nPanel} axes; you gave me {len(axes)}"

    figure.subplots_adjust(hspace=0.025)

    possibleArrays = ["agc_exposure_id", "altitude", "insrot"]
    if plotBy not in possibleArrays:
        raise RuntimeError(f"Unknown \"plotBy\" value {plotBy} (possibilities: {', '.join(possibleArrays)})")

    possibleArrays = ["camera", "visit", "altitude", "insrot"]
    if colorBy not in possibleArrays:
        raise RuntimeError(f"Unknown \"colorBy\" value {colorBy} "
                           f"(possibilities: {', '.join(possibleArrays)})")

    if colorBy == "visit":
        colorBy = "pfs_visit_id"

    if agcData is None:
        if opdb is None:
            raise RuntimeError("You need to supply either an OPDB connection or an agcData array")

        if len(visits) == 0:
            raise RuntimeError("Please specify a set of pfs_visit_ids")

        agcData = readAGCStarsForVisitSetByPfsVisitId(opdb, visits, False)

    agcData0 = agcData

    agcData = agcData.copy()
    agcData["dx"] = agcData.agc_nominal_x_mm - agcData.agc_center_x_mm
    agcData["dy"] = agcData.agc_nominal_y_mm - agcData.agc_center_y_mm

    if True:
        # unpack 6 guide_delta_zN columns into guide_delta_z_camera
        guide_delta_z_cid = np.empty(len(agcData))
        for ic in agcData.agc_camera_id.unique():
            ic = int(ic)
            ll = agcData.agc_camera_id == ic
            guide_delta_z_cid = np.where(agcData.agc_camera_id == ic,
                                         agcData[f"guide_delta_z{ic + 1}"], guide_delta_z_cid)
        agcData["guide_delta_z_cid"] = guide_delta_z_cid

    grouped = agcData.groupby(["agc_exposure_id", "agc_camera_id"], as_index=True)
    tmp_ac = grouped.agg(dx=("dx", "mean"),
                         dy=("dy", "mean"),
                         altitude=("altitude", "first"),
                         insrot=("insrot", "first"),
                         pfs_visit_id=("pfs_visit_id", "first"),
                         guide_delta_z_cid=("guide_delta_z_cid", "first"),
                         )
    tmp_a = tmp_ac.groupby("agc_exposure_id", as_index=True).agg(dx=("dx", "mean"), dy=("dy", "mean"))
    tmp_c = tmp_ac.groupby("agc_camera_id", as_index=True).agg(dx=("dx", "median"), dy=("dy", "median"))

    tmp_ac.dx -= tmp_c.dx   # correct for unknown positions of the cameras
    tmp_ac.dy -= tmp_c.dy

    tmp_ac.dx -= tmp_a.dx   # correct for pointing errors in the telescope
    tmp_ac.dy -= tmp_a.dy

    tmp_ac.reset_index(inplace=True)

    AGC = list(AGC)
    if len(AGC) < 6:
        tmp_ac = tmp_ac[np.isin(tmp_ac.agc_camera_id + 1, AGC)]

    # Time to plot

    for ax, z in zip(axes, plots):
        if z == "xy":
            cmap = plt.matplotlib.colors.ListedColormap([f"C{i}" for i in range(6)])
            norm = plt.matplotlib.colors.Normalize(0.5, 6.5)

            if plotXYStride is None:    # plot per pfs_visit instead
                grouped = tmp_ac[np.isfinite(tmp_ac.dx + tmp_ac.dy)].groupby(
                    ["pfs_visit_id", "agc_camera_id"], as_index=False)
                tmp_acv = grouped.agg(dx=("dx", "mean"),
                                      dy=("dy", "mean"),
                                      altitude=("altitude", "first"),
                                      insrot=("insrot", "first"),
                                      nagc_exposures_per_visit=("agc_exposure_id", "count"),
                                      )
                if nVisitMin > 0:
                    tmp_acv = tmp_acv[tmp_acv.nagc_exposures_per_visit > nVisitMin]
                plotXYStride = 1
            else:
                tmp_acv = tmp_ac

            for cid in tmp_acv.agc_camera_id.unique():
                dx, dy = 1e3*tmp_acv.dx, 1e3*tmp_acv.dy
                c = tmp_acv.agc_camera_id + 1

                ll = tmp_acv.agc_camera_id == cid
                dx = dx[ll]
                dy = dy[ll]
                c = c[ll]

                if plotXYStride > 0:
                    dx = dx[::plotXYStride]
                    dy = dy[::plotXYStride]
                    c = c[::plotXYStride]

                S = ax.scatter(dx, dy, c=c, norm=norm, cmap=cmap, alpha=alpha)
                if connectDxDy:
                    plt.plot(dx, dy, color=f"C{cid}", alpha=alpha)

            if yminmax > 0:
                ax.set_xlim(ax.set_ylim(yminmax*np.array([-1, 1])))
            ax.set_aspect(1)

            colorbarLabel = "camera"
            ax.set_xlabel("Mean AGC x-position (microns)")
            ax.set_ylabel("Mean AGC y-position (microns)")
        else:
            if showAltInsrot:
                if yminmax > 0:
                    vmin = -yminmax
                    vmax = yminmax
                else:
                    vmin = None
                    vmax = None

                S = ax.hexbin(tmp_ac.altitude, tmp_ac.insrot, 1e3*tmp_ac[f"d{z}"], vmin=vmin, vmax=vmax)

                ax.set_xlabel("altitude")
                ax.set_ylabel("insrot")
                colorbarLabel = f"Mean {z} offset (microns)"
            else:
                if colorBy == "camera":
                    cmap = plt.matplotlib.colors.ListedColormap([f"C{i}" for i in range(6)])
                    norm = plt.matplotlib.colors.Normalize(0.5, 6.5)
                    c = tmp_ac.agc_camera_id + 1
                else:
                    cmap = None
                    norm = None
                    c = tmp_ac[colorBy]

                S = ax.scatter(tmp_ac[plotBy], 1e3*tmp_ac[f"d{z}"], c=c, marker='.', s=10,
                               norm=norm, cmap=cmap, alpha=alpha)
                colorbarLabel = colorBy

                ax.set_xlabel(plotBy)
                ax.set_ylabel(f"{z.upper()} offset (microns)")

                if yminmax > 0:
                    ax.set_ylim(yminmax*np.array([-1, 1]))

        if colorBy == "camera" and showCamerasAsLegend:
            if z == "y":
                for cid in np.sort(tmp_ac.agc_camera_id.unique()):
                    ax.plot([np.nan], 'o', color=f"C{cid}", label=f"AG{cid + 1}")
                ax.legend(ncol=6)
        else:
            with opaqueColorbar(S):
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                figure.colorbar(S, label=colorbarLabel, cax=cax, orientation='vertical')

        if plotBy == "agc_exposure_id":
            visits = np.sort(agcData.pfs_visit_id.unique())
            showVisitBoundaries(ax, agcData, visits)
            if opdb is not None:
                ax.format_coord = FormatCoord(plotBy, agcData, opdb)

        if z == "xy":
            ax.axhline(0, color='black', alpha=0.5)
            ax.axvline(0, color='black', alpha=0.5)
        else:
            ax.axhline(0, color='black', zorder=10)

    visits = np.sort(agcData.pfs_visit_id.unique())

    figure.suptitle("pfs_visit_id "
                    + (f"{','.join(str(v) for v in visits)}" if len(visits) < 5 else
                       f"{visits[0]}..{visits[-1]}")
                    + (f"\n{', '.join(f'AG{int(c) + 1}' for c in np.sort(tmp_ac.agc_camera_id.unique()))}"),
                    y=1
                    )

    return agcData0, tmp_ac


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class GuiderConfig:
    transforms = {}

    def __init__(self,
                 transforms={},
                 showAverageGuideStarPos=False,
                 showAverageGuideStarPath=False,
                 showGuideStars=True,
                 showGuideStarsAsPoints=True,
                 showGuideStarsAsArrows=False,
                 showGuideStarPositions=False,
                 showByVisit=True,
                 rotateToAG1Down=False,
                 modelBoresightOffset=True,
                 modelCCDOffset=True,
                 solveForAGTransforms=False,
                 onlyShutterOpen=True,
                 maxGuideError=25,
                 maxPosError=40,
                 guideErrorEstimate=50,
                 pfiScaleReduction=1,
                 gstarExpansion=10,
                 agc_exposure_idsStride=1,
                 guide_star_frac=0.1,
                 pfs_visitIdMin=0,
                 pfs_visitIdMax=0,
                 agc_exposure_idMin=0,
                 agc_exposure_idMax=0,
                 agc_exposure_cm=plt.get_cmap("viridis"),
                 showByVisitSize=5,
                 showByVisitAlpha=1):
        """
        showAverageGuideStarPos=False     plot the per-agc_exposure_id average of selected guide stars,
                                          per AG chip (ignores guide_star_frac)
        showAverageGuideStarPath=False    connect the showAverageGuideStarPos points in order
        showGuideStars=True               plot the selected guide stars
        showGuideStarsAsPoints=True       show positions of quide stars as points, not arrows
        showGuideStarsAsArrows=False
        showGuideStarPositions=False      show guide stars with correct relative positions on the CCD
        showByVisit=True                  colour code by exposure_id, not AG camera (ignored if
                                          self.showGuideStarsAsPoints is False)
        rotateToAG1Down=False              rotate the PFI so AG1 is at the bottom and AG4 at the top
        modelBoresightOffset=True         remove the mean offset and rotation/scale for each exposure
        modelCCDOffset=True               remove the mean offset and rotation/scale for each CCD
        solveForAGTransforms=False        re-solve for the CCD offsets, even if already estimated
        onlyShutterOpen=True              only show exposures when the spectrograph shutters were open
        maxGuideError=25                  only plot exposures with |guideError| < maxGuideError microns
                                          ignored if <= 0
                                          N.b. guideError is the mean of stars with
                                          guide errors < maxPosError
        maxPosError=40                    don't show guide stars with mean(error) > maxPosError (microns)
                                          ignore if <= 0
        guideErrorEstimate=50             average guide error in microns (used to show quiver scalebar)
        pfiScaleReduction=1               How much to shrink AG distances from the centre of the PFI
        gstarExpansion=10                 How much to expand the distances of guide stars from
                                          the centre of their AG camera
        agc_exposure_idsStride=1          only plot every agc_exposure_idsStride'th AG exposure
        guide_star_frac=0.1               only plot this fraction (0..1) of the guide_stars seen on an AG
        pfs_visitIdMin=0                  minimum allowed value of pfs_visit_id; ignored if <= 0
        pfs_visitIdMax=0                  minimum allowed value of pfs_visit_id; ignored if <= 0
        agc_exposure_idMin=0              minimum allowed value of agc_exposure_id; ignored if <= 0
        agc_exposure_idMax=0              maximum allowed value of agc_exposure_id; ignored if <= 0
        agc_exposure_cm=plt.matplotlib.cm.get_cmap("viridis")
        showByVisitSize=5
        showByVisitAlpha=1
        """
        self.transforms = transforms

        self.showAverageGuideStarPos = showAverageGuideStarPos
        self.showAverageGuideStarPath = showAverageGuideStarPath
        self.showGuideStars = showGuideStars
        self.showGuideStarsAsPoints = showGuideStarsAsPoints
        self.showGuideStarsAsArrows = showGuideStarsAsArrows
        self.showGuideStarPositions = showGuideStarPositions
        self.showByVisit = showByVisit
        self.rotateToAG1Down = rotateToAG1Down
        self.modelBoresightOffset = modelBoresightOffset
        self.modelCCDOffset = modelCCDOffset
        self.solveForAGTransforms = solveForAGTransforms
        self.onlyShutterOpen = onlyShutterOpen
        self.maxGuideError = maxGuideError
        self.maxPosError = maxPosError
        self.guideErrorEstimate = guideErrorEstimate
        self.pfiScaleReduction = pfiScaleReduction
        self.gstarExpansion = gstarExpansion
        self.agc_exposure_idsStride = agc_exposure_idsStride
        self.guide_star_frac = guide_star_frac
        self.pfs_visitIdMin = pfs_visitIdMin
        self.pfs_visitIdMax = pfs_visitIdMax
        self.agc_exposure_idMin = agc_exposure_idMin
        self.agc_exposure_idMax = agc_exposure_idMax
        self.agc_exposure_cm = agc_exposure_cm
        self.showByVisitSize = showByVisitSize
        self.showByVisitAlpha = showByVisitAlpha

        self.validate()

    def validate(self):
        #
        # Check options
        #
        if not (self.showGuideStarsAsPoints or self.showGuideStarsAsArrows or
                self.showAverageGuideStarPos or self.showAverageGuideStarPath):
            print("You haven't asked for any plots; proceeding")

        if self.showGuideStarsAsPoints:
            if self.showGuideStarsAsArrows:
                print("Disabling GuiderConfig.showGuideStarsAsArrows as showGuideStarsAsPoints is True")
                self.showGuideStarsAsArrows = False
            if self.showAverageGuideStarPos:
                print("Ignoring GuiderConfig.showAverageGuideStarPos as showGuideStarsAsPoints is True")

        if not (self.showGuideStarsAsPoints or self.showAverageGuideStarPos):
            if self.showByVisit:
                print("Disabling GuiderConfig.showByVisit as showGuideStarsAsPoints is False")
                self.showByVisit = False

        self.guide_star_frac = min([self.guide_star_frac, 1])  # sanity check

        if self.agc_exposure_idsStride <= 0:
            print(f"GuiderConfig.agc_exposure_idsStride (== {self.agc_exposure_idsStride})"
                  " <= 0 is not supported; setting to 1")
            self.agc_exposure_idsStride = 1

    def selectStars(self, agcData, agc_camera_id=-1, guidingErrors=None):
        agc_exposure_ids = sorted(set(agcData.agc_exposure_id))

        aids = agc_exposure_ids[self.agc_exposure_id0::self.agc_exposure_idsStride]

        sel = agcData.isin(dict(agc_exposure_id=aids)).agc_exposure_id.to_numpy()
        if agc_camera_id >= 0:
            sel &= agcData.agc_camera_id == agc_camera_id

        if guidingErrors is not None and self.maxGuideError > 0:
            dx = agcData.agc_center_x_mm - agcData.agc_nominal_x_mm
            dy = agcData.agc_center_y_mm - agcData.agc_nominal_y_mm
            dr = np.hypot(dx, dy)

            tmp = pd.DataFrame(dict(agc_exposure_id=agcData.agc_exposure_id, dr=dr))
            grouped = tmp.groupby("agc_exposure_id")
            _guidingErrors = grouped.agg(guidingErrors=('dr', 'mean'))
            tmp = _guidingErrors.merge(agcData, on="agc_exposure_id").guidingErrors

            sel &= tmp.to_numpy() < 1e-3*self.maxGuideError

        if self.onlyShutterOpen:
            sel &= agcData.shutter_open > 0
        if self.pfs_visitIdMin > 0:
            sel &= agcData.pfs_visit_id >= self.pfs_visitIdMin
        if self.pfs_visitIdMax > 0:
            sel &= agcData.pfs_visit_id <= self.pfs_visitIdMax
        if self.agc_exposure_idMin > 0:
            sel &= agcData.agc_exposure_id >= self.agc_exposure_idMin
        if self.agc_exposure_idMax > 0:
            sel &= agcData.agc_exposure_id <= self.agc_exposure_idMax
            # sel &= agcData.agc_match_flags == 0

        return sel

    @staticmethod
    def nameStride(stride):
        if stride == 1:
            return ""
        return f" {stride}" + {2: 'nd', 3: 'rd'}.get(stride, 'th')

    def make_title(self, agcData, name=None):
        title = ""
        if name is not None:
            title += f"{name}  "

        visits = sorted(set(agcData.pfs_visit_id))

        if len(visits) == 0:
            v0, v1 = 0, 0
        else:
            v0, v1 = visits[0], visits[-1]
        if self.pfs_visitIdMin > 0 and v0 < self.pfs_visitIdMin:
            v0 = self.pfs_visitIdMin
        if self.pfs_visitIdMax > 0 and v1 < self.pfs_visitIdMax:
            v1 = self.pfs_visitIdMax
            title += f"{v0}..{v1}"
        if self.modelBoresightOffset or self.modelCCDOffset:
            title += " "
            what = []
            if self.modelBoresightOffset:
                what.append("Boresight")
            if self.modelCCDOffset:
                what.append("AGs")
                title += f" {' and '.join(what)} offset and rotation/scale removed"
        if self.rotateToAG1Down:
            insrot = np.deg2rad(agcData.insrot).to_numpy()

            title += " Rotated"

            sel = self.selectStars(agcData)
            if np.std(np.rad2deg(insrot[sel])) < 5:
                title += f"  {np.mean(np.rad2deg(-insrot[sel])):.1f}" r"$^\circ$"
                title += f" alt,az = {np.mean(agcData.altitude[sel]):.1f}," \
                    f"{np.mean(agcData.azimuth[sel]):.1f}"
        if self.maxGuideError > 0 or self.maxPosError > 0:
            title += '\n' if True else ' '
            if self.maxGuideError > 0:
                title += f"Max guide error: {self.maxGuideError} microns"
            if self.maxPosError > 0:
                title += f" Max per-star positional error: {self.maxPosError} microns"
                title += f"\nEvery{self.nameStride(self.agc_exposure_idsStride)} AG exposure"
        if self.showGuideStars:
            title += f"; {100*self.guide_star_frac:.0f}% of guide stars"
        if not self.onlyShutterOpen:
            title += " (including open shutter)"
        if (self.showAverageGuideStarPos or self.showAverageGuideStarPath) and \
           not (self.showGuideStarsAsArrows or self.showGuideStarsAsPoints):
            title += "  Mean guide error"

        return title


def rotXY(angle, x, y):
    """Rotate (x, y) by angle in a +ve direction
    angle in radians"""
    c, s = np.cos(angle), np.sin(angle)
    x, y = c*x - s*y, s*x + c*y

    return x, y


def showGuiderErrors(agcData, config,
                     verbose=False,
                     agc_camera_ids=range(6),
                     name=None,
                     livePlot=None):
    """
    Show the guiders for the data in agcData

    The mean nominal position of the guide stars per AGC is shown as a cross (often red), as
    is the boresight.

    agcData: pandas DataFrame as returned by readAgcDataFromOpdb
    config: a GuiderConfig
    agc_camera_ids:  a list of 0-indexed cameras to display; default 0..5
    name:  a string to show in the header; or None
    """
    if livePlot is not None:
        fig, ax, colorbar = livePlot.fig, livePlot.axes[-1], livePlot.colorbar
    else:
        colorbar = None

    agc_exposure_ids = np.sort(agcData.agc_exposure_id.unique())
    #
    # Fix the mean guiding offset for each exposure?
    #
    for aid in agc_exposure_ids:
        if config.solveForAGTransforms or (config.modelBoresightOffset and aid not in config.transforms):
            if aid in config.transforms:
                continue

            sel = agcData.agc_exposure_id == aid

            if verbose:
                print(f"Solving for offsets/rotations for {aid}")
            transform = MeasureXYRot(agcData.agc_center_x_mm[sel], agcData.agc_center_y_mm[sel],
                                     agcData.agc_nominal_x_mm[sel], agcData.agc_nominal_y_mm[sel],
                                     nsigma=3)

            res = scipy.optimize.minimize(transform, transform.getArgs(), method='Powell')
            transform.setArgs(res.x)
            config.transforms[aid] = transform

    if config.modelBoresightOffset:
        for aid in agc_exposure_ids:
            sel = agcData.agc_exposure_id == aid

            transform = config.transforms[aid]
            agcData.loc[sel, "agc_nominal_x_mm"], agcData.loc[sel, "agc_nominal_y_mm"] = \
                transform.distort(agcData.agc_nominal_x_mm[sel], agcData.agc_nominal_y_mm[sel], inverse=True)

    agcData["dx"] = agcData.agc_center_x_mm - agcData.agc_nominal_x_mm
    agcData["dy"] = agcData.agc_center_y_mm - agcData.agc_nominal_y_mm
    agcData["dr"] = np.hypot(agcData.dx, agcData.dy)
    #
    # Find the mean guiding errors for each exposure
    #
    grouped = agcData.groupby("agc_exposure_id")
    guidingErrors = grouped.agg(guidingErrors=('dr', 'mean'))

    config.agc_exposure_id0 = 0      # start with this agc_exposure_id (if config.agc_exposure_idsStride != 1)
    if config.agc_exposure_idsStride > 0 and (config.onlyShutterOpen or config.maxGuideError > 0):
        for aid in agc_exposure_ids:
            if (config.onlyShutterOpen and (agcData.shutter_open == 0)[agcData.agc_exposure_id == aid].any()):
                continue
            if config.maxGuideError > 0 and \
               float(guidingErrors.loc[aid].squeeze()) > 1e-3*config.maxGuideError:
                continue

            config.agc_exposure_id0 = np.where(agc_exposure_ids == aid)[0][0]
            break

    if config.rotateToAG1Down:
        insrot = np.deg2rad(agcData.insrot)

        agcData.agc_nominal_x_mm, agcData.agc_nominal_y_mm = \
            rotXY(-insrot, agcData.agc_nominal_x_mm, agcData.agc_nominal_y_mm)
        agcData.agc_center_x_mm, agcData.agc_center_y_mm = \
            rotXY(-insrot, agcData.agc_center_x_mm, agcData.agc_center_y_mm)
        agcData.dx, agcData.dy = rotXY(-insrot, agcData.dx, agcData.dy)

    for agc_camera_id in agc_camera_ids:
        sel = config.selectStars(agcData, agc_camera_id, guidingErrors)

        if sum(sel) == 0:
            continue

        if config.solveForAGTransforms or (config.modelCCDOffset and agc_camera_id not in config.transforms):
            #
            # Solve for a per-CCD offset and rotation/scale
            #
            if agc_camera_id not in config.transforms:
                print(f"Solving for offsets/rotations for AG{agc_camera_id + 1}")

                transform = MeasureXYRot(agcData.agc_center_x_mm, agcData.agc_center_y_mm,
                                         agcData.agc_nominal_x_mm, agcData.agc_nominal_y_mm)
                config.transforms[agc_camera_id] = transform

                res = scipy.optimize.minimize(transform, transform.getArgs(), method='Powell')
                transform.setArgs(res.x)

        if config.modelCCDOffset:
            agcDataCam = agcData[sel].copy()   # pandas doesn't guarantee a copy or a view, so be specific

            transform = config.transforms[agc_camera_id]

            agcDataCam.agc_nominal_x_mm, agcDataCam.agc_nominal_y_mm = \
                transform.distort(agcDataCam.agc_nominal_x_mm, agcDataCam.agc_nominal_y_mm, inverse=True)

            agcDataCam.dx = agcDataCam.agc_center_x_mm - agcDataCam.agc_nominal_x_mm
            agcDataCam.dy = agcDataCam.agc_center_y_mm - agcDataCam.agc_nominal_y_mm
            agcDataCam.dr = np.hypot(agcDataCam.dx, agcDataCam.dy)

            agcData.loc[sel] = agcDataCam

    sel = config.selectStars(agcData, guidingErrors=guidingErrors)
    tmp = agcData[sel]
    if config.maxPosError > 0:
        grouped = tmp.groupby("guide_star_id")
        _tmp = grouped.agg(
            drbar=("dr", "mean"),
        )
        tmp = tmp.merge(_tmp, on="guide_star_id")
        tmp = tmp[tmp.drbar < 1e-3*config.maxPosError]

    grouped = tmp.groupby(["agc_exposure_id", "agc_camera_id"], as_index=False)
    guideErrorByCamera = grouped.agg(
        agc_nominal_x_mm=("agc_nominal_x_mm", "mean"),
        agc_nominal_y_mm=("agc_nominal_y_mm", "mean"),
        dx=("dx", "mean"),
        dy=("dy", "mean"),
    )
    guideErrorByCamera.sort_values("agc_exposure_id", inplace=True)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    S = None                  # returned by scatter
    Q = None                  # and quiver
    ax.set_prop_cycle(None)

    agc_ring_R = np.mean(np.hypot(*np.array(list(agcCameraCenters.values())).T))  # Approx. radius of AGs

    for agc_camera_id in agc_camera_ids:
        color = f"C{agc_camera_id}"
        AGlabel = f"AG{agc_camera_id + 1}"

        sel = config.selectStars(agcData, agc_camera_id, guidingErrors)

        if sum(sel) == 0:
            continue

        plotData = agcData[sel].copy()   # pandas doesn't guarantee a copy or a view, so be specific
        plotData = plotData[plotData.isin(dict(
            agc_exposure_id=agc_exposure_ids[
                config.agc_exposure_id0::config.agc_exposure_idsStride])).agc_exposure_id]
        plotData.sort_values("agc_exposure_id", inplace=True)

        plotData["x"] = plotData.agc_nominal_x_mm/config.pfiScaleReduction
        plotData["y"] = plotData.agc_nominal_y_mm/config.pfiScaleReduction

        guide_star_ids = np.sort(plotData.guide_star_id.unique())
        np.random.seed(666)
        nguide_star = len(guide_star_ids)
        usedGuideStars = np.random.choice(guide_star_ids,
                                          max([1, int(config.guide_star_frac*nguide_star)]), replace=False)
        if len(usedGuideStars) != len(plotData):
            plotData = plotData[plotData.isin(dict(
                guide_star_id=usedGuideStars)).guide_star_id]

        if config.rotateToAG1Down:
            grouped = plotData.groupby("agc_exposure_id")
            tmp = grouped.agg(
                xbar=("x", "mean"),
                ybar=("y", "mean"),
            )
            plotData = plotData.merge(tmp, on="agc_exposure_id")

            rbar = np.hypot(plotData.xbar, plotData.ybar)

            plotData.xbar *= agc_ring_R/rbar
            plotData.ybar *= agc_ring_R/rbar
        else:
            plotData["xbar"] = np.mean(plotData.agc_nominal_x_mm)
            plotData["ybar"] = np.mean(plotData.agc_nominal_y_mm)

        plotData.sort_values("guide_star_id", inplace=True)
        #
        # OK, ready to plot
        #
        if config.rotateToAG1Down:
            ax.add_patch(Circle((0, 0), agc_ring_R, fill=False, color="red"))
        else:
            ax.plot(plotData.xbar, plotData.ybar, '+',
                    color='red' if
                    config.showByVisit or (config.showAverageGuideStarPos and
                                           not (config.showGuideStarsAsArrows or
                                                config.showGuideStarsAsPoints)) else 'black',
                    zorder=10)

        labelled = False
        if config.showGuideStarPositions:
            xg = plotData.xbar + (plotData.x - plotData.xbar)*config.gstarExpansion
            yg = plotData.ybar + (plotData.y - plotData.ybar)*config.gstarExpansion
        else:
            xg, yg = plotData.xbar, plotData.ybar

        if config.showGuideStars:
            label = None if labelled else AGlabel
            xend, yend = xg + 1e3*plotData.dx, yg + 1e3*plotData.dy
            if config.showGuideStarsAsPoints:
                if config.showByVisit:
                    S = ax.scatter(xend, yend, s=config.showByVisitSize,
                                   alpha=config.showByVisitAlpha,
                                   vmin=np.min(agcData.agc_exposure_id),
                                   vmax=np.max(agcData.agc_exposure_id),
                                   c=plotData.agc_exposure_id, cmap=config.agc_exposure_cm)
                else:
                    ax.plot(xend, yend, '.', color=color, label=label, alpha=config.showByVisitAlpha,
                            markersize=config.showByVisitSize)
                    labelled = True
            elif config.showGuideStarsAsArrows:
                Q = ax.quiver(xg, yg, xend - xg, yend - yg, alpha=0.5, color=color, label=label)
                labelled = True
            else:
                pass   # useful code path if config.showAverageGuideStarPos is true

        if config.showAverageGuideStarPos or config.showAverageGuideStarPath:
            tmp = guideErrorByCamera[guideErrorByCamera.agc_camera_id == agc_camera_id]
            xa = np.mean(tmp.agc_nominal_x_mm)/config.pfiScaleReduction + 1e3*tmp.dx
            ya = np.mean(tmp.agc_nominal_y_mm)/config.pfiScaleReduction + 1e3*tmp.dy

            if config.showAverageGuideStarPath:
                ax.plot(xa.iloc[0], ya.iloc[0], '.', color='black', zorder=-10)
                ax.plot(xa, ya, '-', color='black', alpha=0.25, zorder=-10)

            if config.showAverageGuideStarPos:
                S = ax.scatter(xa, ya, s=config.showByVisitSize, alpha=config.showByVisitAlpha,
                               vmin=tmp.agc_exposure_id.min(), vmax=tmp.agc_exposure_id.max(),
                               c=tmp.agc_exposure_id, cmap=config.agc_exposure_cm)

    if S is not None:
        if colorbar is None:
            a = S.get_alpha()
            S.set_alpha(1)
            colorbar = fig.colorbar(S, ax=ax)
            colorbar.set_label("agc_exposure_id")
            S.set_alpha(a)
        else:
            colorbar.update_normal(S)

        if livePlot:
            livePlot.colorbar = colorbar

    if Q is not None:
        qlen = config.guideErrorEstimate   # microns
        ax.quiverkey(Q, 0.1, 0.9, qlen, f"{qlen} micron", color='black')

    if not config.showGuideStarsAsArrows:
        showAGCameraCartoon(showInstrot=True, showUp=config.rotateToAG1Down, ax1=ax)
    elif labelled:
        L = ax.legend(loc="lower right", markerscale=1)
        for lh in L.legendHandles:
            lh.set_alpha(1)

    ax.set_aspect(1)
    #
    # Fiddle limits
    #
    if False:      # make plot square, and expand a little to fit in the quivers
        axScale = 1.1
        lims = axScale*np.array([plt.xlim(), plt.ylim()])
    else:
        lims = 350/config.pfiScaleReduction*np.array([-1, 1])
    ax.set_xlim(ax.set_ylim(np.min(lims), np.max(lims)))

    ax.plot([0], [0], '+', color='red')

    if Q is not None:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    else:
        ax.set_xlabel(r"$\delta$x (microns)")
        ax.set_ylabel(r"$\delta$y (microns)")

    fig.suptitle(config.make_title(agcData, name))

    return guideErrorByCamera


def showGuiderErrorsByParams(agcData, guideErrorByCamera, params, config, figure=None, name=None):
    """Repeat the showGuiderError plots, but coloured by the list of quantities in params

    N.b.: changing non-plotting values in config will have no effect
    """
    agc_ring_R = np.mean(np.hypot(*np.array(list(agcCameraCenters.values())).T))
    #
    # Look up our AGC exposure IDs and positions/errors
    #
    aids = np.sort(guideErrorByCamera.agc_exposure_id.unique())
    xas = guideErrorByCamera.agc_nominal_x_mm/config.pfiScaleReduction + 1e3*guideErrorByCamera.dx
    yas = guideErrorByCamera.agc_nominal_y_mm/config.pfiScaleReduction + 1e3*guideErrorByCamera.dy
    #
    # Lookup the desired parameters
    subset = agcData[agcData.isin(dict(agc_exposure_id=aids)).agc_exposure_id]
    grouped = subset.groupby("agc_exposure_id")

    aggList = {}
    for p in params:
        aggList[p] = (p, "first")   # e.g. grouped.altitude.first()
        paramValues = grouped.agg(**aggList)
        #
        # and join them back to the guider errors
        #
    paramValues = guideErrorByCamera.merge(paramValues, on="agc_exposure_id")

    n = len(params)
    ny = int(np.sqrt(n))
    nx = n//ny
    if nx*ny < n:
        ny += 1

    fig, axs = plt.subplots(ny, nx, num=figure, sharex=True, sharey=True, squeeze=False)
    axs = axs.flatten()

    for i, (ax, what) in enumerate(zip(axs, params)):
        plt.sca(ax)

        p = paramValues[what]

        vmin, vmax = np.percentile(p, [1, 99])

        S = plt.scatter(xas, yas, s=config.showByVisitSize, alpha=config.showByVisitAlpha,
                        vmin=vmin, vmax=vmax, c=p, cmap=config.agc_exposure_cm)

        a = S.get_alpha()
        S.set_alpha(1)
        C = plt.colorbar(S, shrink=1/nx if ny == 1 else 1)
        C.set_label(what)
        S.set_alpha(a)

        if config.rotateToAG1Down:
            plt.gca().add_patch(Circle((0, 0), agc_ring_R, fill=False, color="red"))

        plt.plot([0], [0], '+', color='red')

        if True:
            lims = 350/config.pfiScaleReduction*np.array([-1, 1])
            plt.xlim(plt.ylim(np.min(lims), np.max(lims)))

        plt.gca().set_aspect(1)

        if not config.rotateToAG1Down:
            showAGCameraCartoon(showInstrot=True, showUp=False)

    fig.supxlabel(r"$\delta$x (microns)")
    fig.supylabel(r"$\delta$y (microns)")

    plt.suptitle(config.make_title(agcData, name), y=1.0)


def showTelescopeErrors(agcData, config, showTheta=False, figure=None, livePlot=None):
    """
    """
    agc_ring_R = np.mean(np.hypot(*np.array(list(agcCameraCenters.values())).T))  # Approx. radius of AGs

    agc_exposure_ids = np.sort(agcData.agc_exposure_id.unique())
    subset = agcData[agcData.isin(dict(agc_exposure_id=agc_exposure_ids)).agc_exposure_id]
    grouped = subset.groupby("agc_exposure_id")
    altitude = grouped.altitude.mean()
    azimuth = grouped.azimuth.mean()
    shutter_open = grouped.shutter_open.max()
    guide_delta_altitude = grouped.guide_delta_altitude.mean()
    guide_delta_azimuth = grouped.guide_delta_azimuth.mean()
    guide_delta_insrot = grouped.guide_delta_insrot.mean()

    sel = shutter_open.to_numpy() > 0

    # Lookup the guide errors
    dx = guide_delta_altitude
    dy = guide_delta_azimuth
    theta = guide_delta_insrot

    if livePlot is None:
        nx, ny = 2, 2
        fig, axs = plt.subplots(nx, ny, num=figure, sharex=False, sharey=False, squeeze=False)
        axs = axs.flatten()
    else:
        fig, axs = livePlot

    for i, ax in enumerate(axs):
        plt.sca(ax)

        shrink = 0.45 if ny == 1 else 1
        thetaScale = 30   # scale for plotting theta (and microns@AGCs)
        if i in [0, 1]:
            if i == 0:
                S = plt.hexbin(dx[sel], dy[sel], gridsize=min(10, int(np.sqrt(sum(sel)))))
                plt.colorbar(S, shrink=shrink).set_label("N")
            elif i == 1:
                S = plt.scatter(dx[sel], dy[sel], s=10, c=agc_exposure_ids[sel])
                plt.colorbar(S, shrink=shrink).set_label("agc_exposure_id")
                ax.set_facecolor('black')

                axs[0].sharex(ax)
                axs[0].sharey(ax)

            plt.gca().set_aspect(1)
            plt.xlabel(r"$\delta$alt (asec)")
            plt.ylabel(r"$\delta$az (asec)")
        elif i == 2:
            if showTheta:
                yvec, ylabel = theta, r"$\theta$ (arcsec)"
                ylim = thetaScale*np.array([-1, 1])
            else:
                yvec, ylabel = (1e3*agc_ring_R)*np.deg2rad(theta/3600), \
                    f"guide error @{1e-1*agc_ring_R:.2f}cm (microns)"
                ylim = thetaScale*np.array([-1, 1])
                S = plt.scatter(agc_exposure_ids[sel], yvec[sel], c=altitude[sel])
                plt.colorbar(S).set_label("altitude")

            plt.ylim(ylim)
            plt.xlabel("agc_exposure_id")
            plt.ylabel(ylabel)
        elif i == 3:
            vmin, vmax = 0.5*thetaScale*np.array([-1, 1])
            S = plt.scatter(azimuth[sel], altitude[sel], c=theta[sel], s=5, vmin=vmin, vmax=vmax)
            plt.colorbar(S).set_label(r"$\theta$ (arcsec)")

            plt.xlabel("azimuth")
            plt.ylabel("altitude")

    title = ""
    title += f"$\\langle\\delta (az, alt)\\rangle$ = ({np.mean(dx):.1f}, {np.mean(dy):.2f}) arcsec "
    title += f"   $\\langle\\theta\\rangle$ = {np.mean(theta):.1f} arcsec"
    plt.suptitle(title)


def plotDriftRate(agcData, agc_exposure_ids=None, radialTangential=True,
                  guideStrategy="nominal", showNominal=False,
                  fitTrend=True, robust=True, rates={}, plot=True,
                  agcVisitSmoothing=1, subtractMeanOffset=True, byTime=True, byCamera=True,
                  showCamera=True, visitName=None, figure=None):
    """Fit and plot the guide cameras drifts, based on the values in agcData
    Only use the agc_exposures in agc_exposure_ids (if provided)

    agcData: `pd.DataFrame` as returned by readAgcDataFromOpdb()
    agc_exposure_ids : `list of int` only show these agc_exposure_ids; or None
    radialTangential :`bool`  decompose x/y offsets into radial/tangential components
    showNominal: show the difference between the nominal (rather than the center) and the "guide" positions
    fitTrend: `bool` if true, fit the drift against time
    robust: `bool` if fitTrend, use scipy.stats.siegelslopes's robust line-fitter
    rates: `dict` set with the results of the fitting (if fitTrend is True)
    agcVisitSmoothing: Smooth the sequence of errors by a boxcar of length agcVisitSmoothing
    subtractMeanOffset: `bool` Subtract mean offset for each camera
    byTime:  `bool` plot against time, not agc_exposure_id
    byCamera: `bool` show separate symbols for each AG camera
    showCamera: `bool` colour points by agc_camera_id
    plot: Don't actually make any plots if False (useful with rates)
    figure: `matplotlib.Figure` reuse this figure

    """
    if not byCamera:
        showCamera = False

    visits = sorted(agcData.pfs_visit_id.unique())

    if agc_exposure_ids is not None:
        agcData = agcData[agcData.isin(dict(agc_exposure_id=agc_exposure_ids)).agc_exposure_id]

        if len(agcData) == 0:
            raise RuntimeError("No relevant agc_exposures are available in the agcData object")

    if showNominal:
        agcData = set_nominal0(agcData)

        agcData["dx"] = agcData.agc_nominal_x_mm - agcData.agc_nominal_x_mm0
        agcData["dy"] = agcData.agc_nominal_y_mm - agcData.agc_nominal_y_mm0
    else:
        if guideStrategy == "nominal":
            agcData["dx"] = agcData.agc_center_x_mm - agcData.agc_nominal_x_mm
            agcData["dy"] = agcData.agc_center_y_mm - agcData.agc_nominal_y_mm
        elif guideStrategy == "nominal0":
            agcData = set_nominal0(agcData)
            agcData["dx"] = agcData.agc_center_x_mm - agcData.agc_nominal_x_mm0
            agcData["dy"] = agcData.agc_center_y_mm - agcData.agc_nominal_y_mm0
        else:
            raise RuntimeError(f"Valid guideStrategy values are nominal and nominal0; saw {guideStrategy}")

    agcData = agcData[agcData.shutter_open == 1]

    if agcVisitSmoothing > 1:
        agcData = smoothAgcData(agcData, agcVisitSmoothing, copy=True)

    if byCamera:
        grouped = agcData.groupby(["agc_exposure_id", "agc_camera_id"], as_index=False)
    else:
        grouped = agcData.groupby(["agc_exposure_id"], as_index=False)

    tmp = grouped.agg(
        taken_at=("taken_at", "first"),
        agc_nominal_x_mm=("agc_nominal_x_mm", "mean"),
        agc_nominal_y_mm=("agc_nominal_y_mm", "mean"),
        dx=("dx", "mean"),
        dy=("dy", "mean"),
    )

    if byCamera:
        if subtractMeanOffset:
            grouped = tmp.groupby("agc_camera_id")
            ntmp = grouped.agg(
                dxbar=("dx", "mean"),
                dybar=("dy", "mean"),
            )
            tmp = ntmp.merge(tmp, on="agc_camera_id")
            tmp.dx -= tmp.dxbar
            tmp.dy -= tmp.dybar
    else:
        tmp.dx = tmp.dx - np.mean(tmp.dx)
        tmp.dy = tmp.dy - np.mean(tmp.dy)

    tmp.sort_values("taken_at", inplace=True)

    dx, dy = tmp.dx, tmp.dy
    if len(tmp) == 0:
        print(f"No valid points were found for {agcData.pfs_visit_id.unique()}")
        return
    time = (tmp.taken_at - tmp.taken_at.iloc[0]).dt.total_seconds().to_numpy()/60

    if radialTangential:
        theta = np.arctan2(tmp.agc_nominal_y_mm, tmp.agc_nominal_x_mm)

        c, s = np.cos(theta), np.sin(theta)
        dradial, dtangential = c*dx + s*dy, -s*dx + c*dy
        #
        # OK!  time to do the fit
        #
    if fitTrend:
        def func(x, a, b):
            return a + b*x

        z1, z2 = (dradial, dtangential) if radialTangential else (dx, dy)
        if robust:
            # offset and rate;  microns and microns/sec
            br, ar = scipy.stats.siegelslopes(z1, time)
            bt, at = scipy.stats.siegelslopes(z2, time)
        else:
            ar, br = scipy.optimize.curve_fit(func, time, z1)[0]
            at, bt = scipy.optimize.curve_fit(func, time, z2)[0]

        visit = visits[0]
        if visit not in rates:
            rates[visit] = {}
            rates[visit]["radial" if radialTangential else "x"] = br
            rates[visit]["tangential" if radialTangential else "x"] = bt
            rates[visit]["azimuth"] = np.mean(agcData.azimuth)
            rates[visit]["altitude"] = np.mean(agcData.altitude)
            rates[visit]["insrot"] = np.mean(agcData.insrot)
            rates[visit]["exptime"] = np.mean(agcData.exptime)

    #
    # And make the plots
    #
    if not plot:
        return

    fig, axs = plt.subplots(2, 1, num=figure, sharex=True, sharey=True, squeeze=False)
    axs = axs.flatten()

    if radialTangential:
        vectors, labels = [dradial, dtangential], ["radial", "tangential"]
    else:
        vectors, labels = [dy, dx], ["y", "x"]

    for ax, vec, xy in zip(axs, vectors, labels):
        plt.sca(ax)
        vecm = 1e3*vec  # convert mm to microns
        if showCamera:
            for agc_camera_id in range(6):
                ll = tmp.agc_camera_id == agc_camera_id

                kwargs = dict(color=f"C{agc_camera_id}", label=f"AG{agc_camera_id + 1}")
                if byTime:
                    plt.plot(time[ll], vecm[ll], 'o', **kwargs)
                else:
                    plt.plot(tmp.agc_exposure_id[ll], vecm[ll], 'o', **kwargs)
                    plt.legend(ncol=3, loc='upper center')
        else:
            if byTime:
                plt.plot(time, vecm, 'o')
            else:
                plt.plot(tmp.agc_exposure_id, vecm, 'o')

        plt.ylabel(xy)

        if fitTrend:
            a, b = (ar, br) if xy in ("x", "radial") else (at, bt)
            tt = np.array([time[0], time[-1]])
            xx = tt if byTime else np.array([tmp.agc_exposure_id.iloc[0], tmp.agc_exposure_id.iloc[-1]])
            plt.plot(xx, 1e3*(a + b*tt), color='black')

            plt.text(0.05, 1.05, f"Rate: {1e3*b:.3f} microns/min", transform=ax.transAxes)

    fig.supxlabel("time (min)" if byTime else "agc_exposure_id")

    if showNominal:
        what = "nominal - nominal0"
    else:
        what = f"center - {guideStrategy}"
        fig.supylabel(f"{what} (microns)")

    title = []
    title.append(f"visit{'s' if len(visits) > 1 else ''} {visits[0]:.0f}")
    if len(visits) > 1:
        title[-1] += f"..{visits[-1]:.0f}"
    if visitName:
        title[-1] += f" {visitName}"
        title.append("")

    title[-1] += f"alt, az ({np.mean(agcData.altitude):.1f}, {np.mean(agcData.azimuth):.1f})"
    title[-1] += f" insrot {np.mean(agcData.insrot):.1f}"
    title.append("")

    if agcVisitSmoothing > 1:
        title[-1] += f" smoothed {agcVisitSmoothing}"

    title = '\n'.join([t for t in title if t])

    plt.suptitle(title)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  New pandas-native routines.  Should eventually replace the above (maybe? I've
# also worked on making the above routines more-or-less pandas-compliant)


def set_center0(agcData, xy0Stat="mean", setVisit0=False):
    if "agc_center_x_mm0" not in agcData:
        grouped = agcData.groupby(["guide_star_id"], as_index=False)
        agcData = grouped.agg(
            agc_center_x_mm0=pd.NamedAgg("agc_center_x_mm", xy0Stat),
            agc_center_y_mm0=pd.NamedAgg("agc_center_y_mm", xy0Stat),
        ).merge(agcData, on="guide_star_id")

    if setVisit0 and "agc_center_x_mm_visit0" not in agcData:
        raise RuntimeError("Test me")

        grouped = agcData.groupby(["pfs_visit_id", "guide_star_id"], as_index=False)
        _agcData = grouped.agg(
            agc_center_x_mm_visit0=pd.NamedAgg("agc_center_x_mm", xy0Stat),
            agc_center_y_mm_visit0=pd.NamedAgg("agc_center_y_mm", xy0Stat),
        ).merge(agcData, on=["pfs_visit_id", "guide_star_id"])
        agcData["agc_center_x_mm_visit0"] = _agcData.agc_center_x_mm_visit0
        agcData["agc_center_y_mm_visit0"] = _agcData.agc_center_y_mm_visit0

    return agcData


def set_nominal0(agcData, xy0Stat="mean", setVisit0=False):
    """Return agcData with columns agc_nominal_[xy]_mm0 added"""

    if "agc_nominal_x_mm0" not in agcData:
        grouped = agcData.groupby(["guide_star_id"], as_index=False)
        agcData = grouped.agg(
            agc_nominal_x_mm0=pd.NamedAgg("agc_nominal_x_mm", xy0Stat),
            agc_nominal_y_mm0=pd.NamedAgg("agc_nominal_y_mm", xy0Stat),
        ).merge(agcData, on="guide_star_id")

    if setVisit0 and "agc_nominal_x_mm_visit0" not in agcData:
        raise RuntimeError("Test me")

        grouped = agcData.groupby(["pfs_visit_id", "guide_star_id"], as_index=False)
        agcData = grouped.agg(
            agc_nominal_x_mm_visit0=pd.NamedAgg("agc_nominal_x_mm", xy0Stat),
            agc_nominal_y_mm_visit0=pd.NamedAgg("agc_nominal_y_mm", xy0Stat),
        ).merge(agcData, on=["pfs_visit_id", "guide_star_id"])

    return agcData


def smoothAgcData(agcData, agcVisitSmoothing, copy=True):
    """
    Smooth agcData by a boxcar of length agcVisitSmoothing

    copy: `bool`: return a copy (ignored if agcVisitSmoothing <= 1)
    """

    if agcVisitSmoothing <= 1:
        return agcData

    if copy:
        agcData = agcData.copy()

    taken_at = agcData["taken_at"]
    del agcData["taken_at"]         # required by agcData.rolling.  I should sort this out

    agcData = agcData.groupby("agc_camera_id", as_index=False)
    agcData = agcData.rolling(agcVisitSmoothing,
                              on="agc_exposure_id",
                              min_periods=1, center=True).mean()

    agcData["taken_at"] = taken_at

    return agcData


def estimateGuideErrors(agcData, plot=False, guideStrategy="center0", showNominal=False,
                        byVisit=False, labelByTime=False, showAGMean=True, drawTrack=False,
                        recenterPerVisit=False, agcVisitSmoothing=1,
                        subtractMedian=False, rotateToAG1Down=False, expand=1,
                        showClosedShutter=False, xy0Stat="median", visitName=None, showCartoon=True):
    """Estimate (and maybe plot) the mean guide errors for each AG camera and exposure
    Only exposures with the spectrograph shutters open are considered unless showClosedShutter is True

    agcData: `pd.DataFrame` as returned by readAgcDataFromOpdb()
    plot: `bool` plot the results?
    guideStrategy: how to define the reference position of the guide stars
        center0:          Use the average of the first visit's agc_center_[xy]_mm (see xy0Stat)
        center0PerVisit:  Use the average of agc_center_[xy]_mm for each visit (see xy0Stat)
        boresight:        Use nominal, but corrected by xy0Stat for each visit
        nominal:          Use agc_nominal_[xy]_mm
        nominal0:         Use agc_nominal_[xy]_mm at start of sequence
        nominal0PerVisit: Use agc_nominal_[xy]_mm at start of each visit
    showNominal: show the difference between the nominal (rather than the center) and the "guide" positions
    byVisit: group on pfs_visit_id rather than agc_exposure_id
    labelByTime: label plots by elapsed time rather than agc_exposure_id
    showAGMean: Show the mean of the AG guide signals at the centre of the plot
    drawTrack: connect the dots, helping see the order of points
    agcVisitSmoothing: Smooth the sequence of errors by a boxcar of length agcVisitSmoothing
    recenterPerVisit: `bool` Subtract the per-visit mean position
    subtractMedian:       Subtract the median from each camera's guide errors
    rotateToAG1Down:     `bool` Rotate PFS so AG1 is down and AG4 is up?
    expand:             Enlarge the residual plot by a factor of expand
    showClosedShutter:    Include data with the spectrograph shutter closed
    xy0Stat: The name of the statistic to use for center0/center0PerVisit (must be supported by
             pd.DataFrame.agg, and also in validXy0StatStrategies list)
    showCartoon: show the cartoon of the AG cameras

    Returns:
       agcData:  As updated, possibly with added columns
                 d[xy]                        offsets, meaning defined by arguments to this function (!)
                 agc_center_[xy]_mm0          Values of agc_center_[xy]_mm at start of sequence
                 agc_nominal_[xy]_mm0         Values of agc_nominal_[xy]_mm at start of sequence
                 agc_nominal_[xy]y_mm_visit0  Values of agc_nominal_[xy]_mm at start of each visit
       agcGuideErrors: pandas DataFrame with desired errors
    """
    validXy0StatStrategies = ["mean", "median", "first", "last"]
    if xy0Stat not in validXy0StatStrategies:
        raise RuntimeError(f"Unknown xy0Stat {xy0Stat}"
                           f" (valid: {', '.join(validXy0StatStrategies)})")

    validGuideStrategies = ["boresight",
                            "center0", "center0PerVisit", "nominal", "nominal0PerVisit", "nominal0"]
    if guideStrategy not in validGuideStrategies:
        raise RuntimeError(f"Unknown guideStrategy {guideStrategy}"
                           f" (valid: {', '.join(validGuideStrategies)})")

    shutterMin = -1 if showClosedShutter else 0

    agcData = set_center0(agcData, xy0Stat)
    agcData = set_nominal0(agcData, xy0Stat)

    agcData = agcData[agcData.shutter_open > 0]

    if guideStrategy == "boresight":
        tmp = pd.DataFrame(dict(pfs_visit_id=agcData.pfs_visit_id,
                                guide_star_id=agcData.guide_star_id,
                                agc_camera_id=agcData.agc_camera_id,
                                agc_nominal_x_mm_offset=agcData.agc_nominal_x_mm - agcData.agc_nominal_x_mm0,
                                agc_nominal_y_mm_offset=agcData.agc_nominal_y_mm - agcData.agc_nominal_y_mm0,
                                ))
        #
        # Find the stars that are detected in all the exposures
        # (actually the ones that appear most often, but that should usually be the same thing)
        #
        grouped = agcData.groupby(["guide_star_id"], as_index=False)
        tmp2 = grouped.agg(
            nobs=("agc_exposure_id", "count"),
        )
        tmp2 = tmp2[tmp2.nobs == tmp2.nobs.max()]
        tmp = tmp[tmp.isin(dict(guide_star_id=tmp2.guide_star_id.to_numpy())).guide_star_id]
        #
        # Use those stars to calculate the per-visit offsets
        #
        if "agc_nominal_x_mm_visit" not in agcData:
            grouped = tmp.groupby(["pfs_visit_id"], as_index=False)
            tmp = grouped.agg(
                agc_nominal_x_mm_visit=("agc_nominal_x_mm_offset", xy0Stat),
                agc_nominal_y_mm_visit=("agc_nominal_y_mm_offset", xy0Stat),
            )
            agcData = tmp.merge(agcData, on=["pfs_visit_id"])

        refPos_x = agcData.agc_nominal_x_mm0 + agcData.agc_nominal_x_mm_visit
        refPos_y = agcData.agc_nominal_y_mm0 + agcData.agc_nominal_y_mm_visit
    elif guideStrategy == "center0":
        refPos_x = agcData.agc_center_x_mm0
        refPos_y = agcData.agc_center_y_mm0
    elif guideStrategy == "center0PerVisit":
        refPos_x = agcData.agc_center_x_mm_visit0
        refPos_y = agcData.agc_center_y_mm_visit0
    elif guideStrategy == "nominal":
        refPos_x = agcData.agc_nominal_x_mm
        refPos_y = agcData.agc_nominal_y_mm
    elif guideStrategy == "nominal0":
        refPos_x = agcData.agc_nominal_x_mm0
        refPos_y = agcData.agc_nominal_y_mm0
    elif guideStrategy == "nominal0PerVisit":
        refPos_x = agcData.agc_nominal_x_mm_visit0
        refPos_y = agcData.agc_nominal_y_mm_visit0
    else:
        raise RuntimeError("You can't get here; complain to RHL")

    if showNominal:
        agcData["dx"] = agcData.agc_nominal_x_mm - refPos_x
        agcData["dy"] = agcData.agc_nominal_y_mm - refPos_y
    else:
        agcData["dx"] = agcData.agc_center_x_mm - refPos_x
        agcData["dy"] = agcData.agc_center_y_mm - refPos_y

    agcData_smoothed = smoothAgcData(agcData, agcVisitSmoothing)

    grouped = agcData_smoothed.groupby(
        ["pfs_visit_id" if byVisit else "agc_exposure_id" , "agc_camera_id"], as_index=False)
    agcGuideErrors = grouped.agg(
        agc_exposure_id=pd.NamedAgg("agc_exposure_id", "mean"),
        pfs_visit_id=pd.NamedAgg("pfs_visit_id", "mean"),
        agc_nominal_x_mm0=pd.NamedAgg("agc_nominal_x_mm", "mean"),
        agc_nominal_y_mm0=pd.NamedAgg("agc_nominal_y_mm", "mean"),
        altitude=pd.NamedAgg("altitude", "mean"),
        azimuth=pd.NamedAgg("azimuth", "mean"),
        insrot=pd.NamedAgg("insrot", "mean"),
        taken_at=pd.NamedAgg("taken_at", "mean"),
        guide_delta_altitude=pd.NamedAgg("guide_delta_altitude", "mean"),
        guide_delta_azimuth=pd.NamedAgg("guide_delta_azimuth", "mean"),
        guide_delta_insrot=pd.NamedAgg("guide_delta_insrot", "mean"),
        xbar=pd.NamedAgg("dx", "mean"),
        ybar=pd.NamedAgg("dy", "mean"),
        shutter_open=pd.NamedAgg("shutter_open", "max"),
    )

    if subtractMedian:
        for agc_camera_id in sorted(set(agcGuideErrors.agc_camera_id)):
            sel = (agcGuideErrors.agc_camera_id == agc_camera_id) & (agcGuideErrors.shutter_open > shutterMin)
            agcGuideErrors.xbar = np.where(sel, agcGuideErrors.xbar - np.median(agcGuideErrors.xbar[sel]),
                                           agcGuideErrors.xbar)
            agcGuideErrors.ybar = np.where(sel, agcGuideErrors.ybar - np.median(agcGuideErrors.ybar[sel]),
                                           agcGuideErrors.ybar)
    #
    if recenterPerVisit:
        grouped = agcGuideErrors.groupby("pfs_visit_id")
        tmp = grouped.agg(
            xbarMean=pd.NamedAgg("xbar", lambda x: np.nanmean(x)),
            ybarMean=pd.NamedAgg("ybar", lambda x: np.nanmean(x)),
        ).merge(agcGuideErrors, on="pfs_visit_id")
        agcGuideErrors.xbar = agcGuideErrors.xbar - tmp.xbarMean
        agcGuideErrors.ybar = agcGuideErrors.ybar - tmp.ybarMean
    #
    # Plot the results?
    #

    def selectPoints(df, i):
        if i == 1 and showClosedShutter:
            ll = df.shutter_open == 0
            s = 5
        else:
            ll = df.shutter_open > 0
            s = None

        return ll, s

    if plot:
        if "elased_time" not in agcGuideErrors:
            agcGuideErrors["elapsed_time"] = \
                (agcGuideErrors.taken_at - agcGuideErrors.taken_at.iloc[0]).dt.total_seconds().to_numpy()

        _agcGuideErrors = agcGuideErrors[["agc_exposure_id", "pfs_visit_id", "agc_camera_id", "shutter_open",
                                          "insrot", "taken_at", "elapsed_time",
                                          "agc_nominal_x_mm0", "agc_nominal_y_mm0", "xbar", "ybar"]]

        if True:   # draw each guide star at the centre of its CCD
            grouped = _agcGuideErrors.groupby("agc_camera_id")
            _agcGuideErrors = grouped.agg(
                agc_camera_x_mm0=("agc_nominal_x_mm0", "median"),
                agc_camera_y_mm0=("agc_nominal_y_mm0", "median"),
            ).merge(_agcGuideErrors, on="agc_camera_id")

            agc_camera_x_mm = _agcGuideErrors.agc_camera_x_mm0
            agc_camera_y_mm = _agcGuideErrors.agc_camera_y_mm0
        else:    # draw each guide star at its position within the CCD
            agc_camera_x_mm = _agcGuideErrors.agc_nominal_x_mm0
            agc_camera_y_mm = _agcGuideErrors.agc_nominal_y_mm0

        insrotBar = np.deg2rad(np.mean(agcData.insrot))  # mean insrot for data

        for i in range(2):
            ll, s = selectPoints(_agcGuideErrors, i)

            x = agc_camera_x_mm/expand + 1e3*_agcGuideErrors.xbar
            y = agc_camera_y_mm/expand + 1e3*_agcGuideErrors.ybar

            if rotateToAG1Down:
                x, y = rotXY(-insrotBar, x, y)

            S = plt.scatter(x[ll], y[ll], marker='o', s=s,
                            c=_agcGuideErrors.pfs_visit_id[ll] if byVisit else
                            _agcGuideErrors.elapsed_time[ll] if labelByTime else
                            _agcGuideErrors.agc_exposure_id[ll])

            if drawTrack:
                for agc_camera_id in range(6):
                    lll = ll & (_agcGuideErrors.agc_camera_id == agc_camera_id)
                    plt.plot(x[lll], y[lll], color='black', alpha=0.25, zorder=-1)

            if i == 0:
                plt.plot(agc_camera_x_mm/expand, agc_camera_y_mm/expand, '+', color='red', zorder=10)

            if showAGMean:
                grouped = _agcGuideErrors.groupby("agc_exposure_id", as_index=False)
                tmp = grouped.agg(
                    pfs_visit_id=("pfs_visit_id", "first"),
                    xbar=("xbar", "mean"),
                    ybar=("ybar", "mean"),
                    shutter_open=("shutter_open", "first"),
                    elapsed_time=("elapsed_time", "mean"),
                    insrot=("insrot", "mean"),
                )
                x = 1e3*tmp.xbar
                y = 1e3*tmp.ybar
                ll, s = selectPoints(tmp, i)

                if rotateToAG1Down:
                    a = np.deg2rad(tmp.insrot)
                    x, y = rotXY(-a, x, y)

                S = plt.scatter(x[ll], y[ll], marker='o', s=s, alpha=0.5,
                                c=tmp.pfs_visit_id[ll] if byVisit else
                                tmp.elapsed_time[ll] if labelByTime else
                                tmp.agc_exposure_id[ll])
                if drawTrack:
                    plt.plot(x[ll], y[ll], color='black', alpha=0.25, zorder=-1)

        a = S.get_alpha()
        S.set_alpha(1)
        plt.colorbar(S).set_label("pfs_visit_id" if byVisit else
                                  "time (s)" if labelByTime else "agc_exposure_id")
        S.set_alpha(a)

        plt.plot([0], [0], '+', color='red')

        plt.gca().set_aspect(1)
        plt.xlim(plt.ylim(280/expand*np.array([-1, 1])))

        delta = f"{'nominal' if showNominal else 'center'} - {guideStrategy}"
        plt.xlabel(f"x (microns)  ({delta})")
        plt.ylabel(f"y (microns)  ({delta})")

        _visits = [int(v) for v in sorted(agcData.pfs_visit_id.unique())]
        title = []
        title.append(f"{_visits[0]:.0f}")

        if len(_visits) > 1:
            title[-1] += f"..{_visits[-1]:.0f}"
        if visitName:
            title[-1] += f" {visitName}"

        title.append("")

        if guideStrategy in ("boresight", "center0", "center0PerVisit"):
            title[-1] += f" xy0Stat:{xy0Stat}"
        if recenterPerVisit:
            title[-1] += " per-visit centre removed"
        if agcVisitSmoothing > 1:
            title[-1] += f" smoothed {agcVisitSmoothing}"
        if rotateToAG1Down:
            title[-1] += f" rotated {-np.rad2deg(insrotBar):.1f}"

        title.append("")

        title[-1] += f"alt, az ({np.mean(agcData.altitude):.1f}, {np.mean(agcData.azimuth):.1f})"
        title[-1] += f" insrot {np.rad2deg(insrotBar):.1f}"

        title = '\n'.join([_ for _ in title if _])

        if showCartoon:
            showAGCameraCartoon(showInstrot=not rotateToAG1Down, showUp=True,
                                insrot=insrotBar if rotateToAG1Down else None)

        plt.suptitle(title)

    return agcData, agcGuideErrors


def compareAGCPfsUtils(opdb, pfs_visit_id, nAgcExposures=10, instPaCorrection=0,
                       alignCenterPosition=False,
                       plot=True, compress=10, plotUsingScatter=False, showCartoon=True):
    """
    Look at the difference between AG and pfs_utils transformations

    opdb: opdb postgres connection
    pfs_visit_id: desired visit
    nAgcExposures: average over AGC positions for this many exposures
    instPaCorrection: fiddle factor to add to inst_pa; asec
    plot: `bool` plot the results?
    compress: how much to compress the PFI for plotting
    plotUsingScatter: use plt.scatter
    showCartoon: show the cartoon of the AG cameras
    """
    pfs_design_id = readPfsDesign(opdb, pfs_visit_id).pfs_design_id.iloc[0]

    # read the positions of the guide stars from the design file
    agcStars = readAGCStars(opdb, pfs_design_id, pfs_visit_id)

    ra_center_config = agcStars.ra_center_config.mean()
    dec_center_config = agcStars.dec_center_config.mean()
    inst_pa = agcStars.pa_config.mean()

    # and the guider positions
    agcData = readAGCPositionsForVisitByAgcExposureId(opdb, pfs_visit_id, flipToHardwareCoords=False)
    del agcData["agc_camera_id"]       # also present in agcStars
    del agcData["pfs_visit_id"]       # also present in agcStars

    # average together then first nAgcExposures exposures.
    # just using the first can miss a guide camera, and while the averaging averages over the trailed
    # nominal position bug that was fixed on 2023-07-26, it's a small effect
    grouped = agcData[agcData.agc_exposure_id < agcData.agc_exposure_id.iloc[0] + nAgcExposures].groupby(
        ["agc_exposure_id", "guide_star_id"])
    agcData = grouped.agg(
        agc_nominal_x_mm=("agc_nominal_x_mm", "median"),
        agc_nominal_y_mm=("agc_nominal_y_mm", "median"),
    ).merge(agcData)
    agcStars = agcStars.merge(agcData, on="guide_star_id")

    #
    # Transform to PFI coordinates using pfs_utils (including optional offset to inst_pa)
    #
    inst_pa += instPaCorrection/3600

    raDecIn = np.array(list(zip(agcStars.guide_star_ra, agcStars.guide_star_dec))).T
    par = agcStars.guide_star_parallax.to_numpy()
    pm = np.array(list(zip(agcStars.guide_star_pm_ra, agcStars.guide_star_pm_dec))).T

    # Altitude, time at start of agc_exposure.  Shouldn't matter providing they are consistent
    altitude = agcStars.altitude.iloc[0]
    time = agcStars.taken_at.iloc[0]
    xy = ct.CoordinateTransform(raDecIn, "sky_pfi", za=90.0 - altitude,
                                cent=np.array([[ra_center_config], [dec_center_config]]),
                                time=time, pa=inst_pa, pm=pm, par=par)
    agcStars["pfs_utils_x_mm"], agcStars["pfs_utils_y_mm"] = xy[0:2]

    if alignCenterPosition:
        # force the AGC and pfs_utils's centres to agree
        dx = agcStars.pfs_utils_x_mm - agcStars.agc_nominal_x_mm
        dy = agcStars.pfs_utils_y_mm - agcStars.agc_nominal_y_mm

        dxbar, dybar = dx.mean(), dy.mean()
        agcStars.pfs_utils_x_mm -= dxbar
        agcStars.pfs_utils_y_mm -= dybar

        plt.arrow(0, 0.75, dxbar, dybar, length_includes_head=True, color='red')
    else:
        dxbar, dybar = None, None

    # Calculate the angle between the agc and pfs_utils positions
    agcStars["agc_nominal_theta"] = np.arctan2(agcStars.agc_nominal_y_mm, agcStars.agc_nominal_x_mm)
    agcStars["pfs_utils_theta"] = np.arctan2(agcStars.pfs_utils_y_mm, agcStars.pfs_utils_x_mm)

    agcStars["delta_theta"] = 3600*np.rad2deg(agcStars["agc_nominal_theta"] - agcStars["pfs_utils_theta"])

    # ----

    if plot:
        grouped = agcStars.groupby("agc_camera_id")
        agcStars = grouped.agg(
            agc_camera_x_mm=("pfs_utils_x_mm", "mean"),
            agc_camera_y_mm=("pfs_utils_y_mm", "mean"),
        ).merge(agcStars, on="agc_camera_id")

        for x, y, label in [(agcStars.pfs_utils_x_mm, agcStars.pfs_utils_y_mm, "pfs_utils"),
                            (agcStars.agc_nominal_x_mm, agcStars.agc_nominal_y_mm, "agc")]:
            if plotUsingScatter:
                S = plt.scatter((x - agcStars.agc_camera_x_mm) + agcStars.agc_camera_x_mm/compress,
                                (y - agcStars.agc_camera_y_mm) + agcStars.agc_camera_y_mm/compress,
                                c=agcStars.agc_camera_id,
                                marker='+' if label == 'agc' else 'o', label=label)
                if label == 'agc':
                    plt.colorbar(S)
            else:
                plt.plot((x - agcStars.agc_camera_x_mm) + agcStars.agc_camera_x_mm/compress,
                         (y - agcStars.agc_camera_y_mm) + agcStars.agc_camera_y_mm/compress, 'o', label=label)

        plt.legend()

        plt.plot([0], [0], '+', color='red')

        plt.xlim(plt.ylim(300/compress*np.array([-1, 1])))
        plt.gca().set_aspect(1)

        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")

        if showCartoon:
            showAGCameraCartoon()

        title = [f"visit: {pfs_visit_id}"]
        title += [f"alt, az ({agcStars.altitude.mean():.1f}, {agcStars.azimuth.mean():.1f}) "
                  f"insrot {agcStars.insrot.mean():.1f}"]
        title += [f"d(theta) = {agcStars.delta_theta.median():.1f} asec"]

        if dxbar is not None:
            title[-1] += f"  (dx, dy) = ({1e3*dxbar:.0f}, {1e3*dybar:.0f}) microns"

        if False and instPaCorrection != 0:
            title += [f"PA incremented by {instPaCorrection} asec"]

        plt.title('\n'.join(title))

    return agcStars

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def MomentDifferenceToMmPiston(deltaMxx1D, includePistonCorrection=False):
    """From Kawanomoto-san's ics_agActor code

    deltaMxx1D is the 1-D image second moment (e.g. <x^2>) """
    deltaMxx2D = 2*deltaMxx1D

    piston = 0.0086*deltaMxx2D
    if includePistonCorrection:
        piston -= 0.026

    return piston


def getMA(npt, forceAlpha=None, nptCrit=100):
    small = npt < nptCrit
    marker = 'o' if small else '.'
    alpha = forceAlpha if forceAlpha is not None else 0.5 if small else 0.25

    return marker, alpha


def showVisitBoundaries(ax, tmp, visits):
    for v in visits[:-1]:
        ax.axvline(np.max(tmp.agc_exposure_id[tmp.pfs_visit_id == v]), color='black', zorder=-1, alpha=0.25)


def averageArraysByFocusPosition(focusPosition, xx, yy, useMedian=True):
    """Return xx, yy grouped by focusPosition"""

    focusPositions = np.sort(list(set(focusPosition)))
    focusPositions = (1e3*focusPositions + 0.5).astype(int)/1e3  # round to the nearest micron

    op = np.median if useMedian else np.mean

    _xx = np.empty_like(focusPositions, dtype=float)
    _yy = np.empty_like(_xx)
    for i, fp in enumerate(focusPositions):
        ll = focusPosition == fp
        _xx[i] = op(xx[ll])
        _yy[i] = op(yy[ll])

    return _xx, _yy


def find_W_M2OFF3(butler, visit, nTry=100, key="W_M2OFF3"):
    """Search back from visit looking for metadata"""
    if butler is None:
        raise RuntimeError(f"I need a butler in order to find {key}")

    md = None
    for i in range(nTry):
        dateRefList = list(butler.registry.queryDatasets('raw', visit=visit))
        if len(dateRefList) > 0:
            md = butler.get("raw.metadata", dateRefList[0].dataId)
            return md[key]

        visit -= 1

    raise RuntimeError(f"Failed to find a value of {key} in visit, "
                       f"or within {nTry - 1} visits preceding, {visit}")


def setImageSizes(tmp, useTraceRadius=True):
    # calculate 1-D RMS
    if useTraceRadius:
        bad = (tmp.mxx < 0) | (tmp.myy < 0)
        tmp["rms"] = np.sqrt(np.where(bad, np.nan, 0.5*(tmp.mxx + tmp.myy)))
    else:
        bad = (tmp.mxx*tmp.myy - tmp.mxy**2) < 0
        tmp["rms"] = np.where(bad, np.nan, (tmp.mxx*tmp.myy - tmp.mxy**2)**0.25)

    tmp["FWHM"] = 2*np.sqrt(2*np.log(2))*tmp.rms   # Gaussian equivalent
    tmp.FWHM *= 0.137                              # Convert to arcsec; 13micron pixels
    pass                                           # The scale at the AG cameras is 94.7micron/arcsec (JEG)
    tmp["left"] = (tmp.agc_data_flags & 0x1) == 0x0

    return tmp


def overplotFocusSets(what, focus, axes):
    focusPosition = np.round(1e3*focus.focusPosition.to_numpy())/1e3  # round to nearest micron
    focusPosition = np.where(np.isfinite(focusPosition), focusPosition, -1)

    change = focus[what][:-1].to_numpy()[np.where(np.abs(np.diff(focusPosition)) > 1e-3)]
    what0 = np.min(focus[what])
    for i, what1 in enumerate(list(change) + [np.max(focus[what])]):
        for ax in axes:
            ax.axvspan(what0, what1, color='black' if i%2 == 0 else 'brown', alpha=0.1, zorder=-1)
        what0 = what1
#
# Define a formatter to add the pfs_visit_id to the cursor readout
#


class FormatCoord:
    def __init__(self, plotBy, dataFrame, opdb=None):
        self._plotBy = plotBy
        self._dataFrame = dataFrame
        self._opdb = opdb

    def __call__(self, x, y):
        x = int(x + 0.5)
        label = "(x, y) = ("
        label += f"{x:.0f}" if self._plotBy == "agc_exposure_id" else f"{x:.2f}"
        label += f", y: {y:.2f})"

        pfs_visit_ids = self._dataFrame[self._dataFrame.agc_exposure_id == x].pfs_visit_id.unique()
        if len(pfs_visit_ids) > 0:
            pfs_visit_id = pfs_visit_ids[0]

            label += f"  pfs_visit_id: {pfs_visit_id}"
            if self._opdb is not None:
                design_name = readPfsDesign(self._opdb, pfs_visit_id).design_name.iloc[0]
                label += f" ({design_name})"

        return label


class ShowFocusFit:
    def __init__(self, axes, indicateFocusPosition=False):

        self.indicateFocusPosition = indicateFocusPosition
        self._axes = axes
        ax = axes[0]
        self._text = ax.text(0.99, 0.02, "Click to set M2_OFF3",
                             ha="right", va="bottom", transform=ax.transAxes)
        self._focus = None
        self._focusLines = []

    def __call__(self, event):
        if event.inaxes:
            self._focus = event.xdata
            self._text.set_text(f"M2_OFF3 = {self._focus:.2f}mm")

            for fl in self._focusLines:
                fl.remove()
            self._focusLines = []

            ax = self._axes[0]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            m2_off3 = np.array(xlim)
            guiderFocus_to_M2_OFF3 = get_guiderFocus_to_M2_OFF3()
            self._focusLines = ax.plot(m2_off3, 1/guiderFocus_to_M2_OFF3*(m2_off3 - self._focus),
                                       color='black')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            if self.indicateFocusPosition:
                for ax in self._axes:
                    self._focusLines.append(ax.axvline(self._focus, color='black', alpha=0.5))


def plotFocus(opdb, visits, agcData=None,
              AGC=range(1, 6+1),
              plotBy="focus",
              colorBy="camera",
              showAGActorFocus=True,
              showOpdbFocus=True,
              showFWHM=True,
              showPfiFocusPosition=False,
              averageByFocusPosition=False,
              showMedian=False,
              showOnlyMedian=False,
              connectMedian=True,
              showCameraId=False,
              showFocusSets=False,
              onlyGuideStars=True,
              plotFrac=1,
              ditherScale=5e-3,
              yLimitsMicron=180,
              indicateFocusPosition=False,
              useTraceRadius=True,
              magMin=None,
              magMax=None,
              minFWHM=None,
              maxFWHM=None,
              mmToMicrons=1e3,
              useM2Off3=True,
              butler=None,
              forceAlpha=None,
              figure=None,
              axes=None,
              ):
    """
    Note the this routine can read from the database using readAGCStarsForVisitSetByPfsVisitId(), or it
    can reuse a passed-in dataframe.  Either way it returns the dataframe, which may be used next time
    (and the requested visits are silently ignored). It may also be passed to showAgcErrorsForVisitsByCamera

       agcData = None
       agcData = plotFocus(opdb, agcData=agcData, ...)
       agcData = plotFocus(opdb, agcData=agcData, ...)
       ...

    Parameters:
    opdb:
        Connection to opdb
    visits: `list` of `int`
        The visits of interest
    AGC:  `list` of `int`
        The AG cameras to show (1-indexed; default all)
    showAGActorFocus: `bool`
        Show the AG's estimate of the focus position [default: True]
    showOpdbFocus: `bool`
        Show this routine's estimate of the focus position [default: True]
    showFWHM: `bool`
        Show the raw FWHM of the stars [default: True]
    showPfiFocusPosition: `bool`
        Show the estimates PFI focus as well as the AG focus [default: False]
    averageByFocusPosition = False
    showMedian = False
    showOnlyMedian = False               # show the only the median of quantities; implies showMedian
    showCameraId = False                 # Applies to FWHM panel;  if False, colour by L/R
    showFocusSets = False
       Indicate which sets of data were taken with the same M2_OFF3 value
    onlyGuideStars = True

    plotFrac = 1 # 0.01    # fraction of agc star data points to show
    ditherScale = 5e-3
    mmToMicrons = 1e3      # conversion from mm to microns; if set to 1 then plots will be in mm (!)
    yLimitsMicron = 180    # if >= 0, set focus error scale to +- yLimitsMicron (or the given tuple)
    useTraceRadius: `bool`
       Use the "trace" definition of rms; sqrt(0.5*(Mxx + Myy))
       rather than "determinant" (Mxx*Myy - Mxy**2)**(1/4)
    magMin = None
    magMax = None
    minFWHM = None if False else 0.45
    maxFWHM = None if False else 1.89
    indicateFocusPosition = False   Plot a vertical line at chosen focus position when you're using the cursor
    useM2Off3 = True   # I.e. not M2POS3; almost always what you want;
              Used when plotBy == "focus"
    figure: the matplotlib.Figure to use; or None
    axes: An array of the appropriate number of matplotlib.Axes to use; or None
    """

    possibleArrays = ["focus", "agc_exposure_id", "altitude", "insrot"]
    if plotBy not in possibleArrays:
        raise RuntimeError(f"Unknown \"plotBy\" value {plotBy} (possibilities: {', '.join(possibleArrays)})")

    what = "focusPosition" if plotBy == "focus" else plotBy

    possibleArrays = ["camera", "visit", "altitude", "insrot"]
    if colorBy not in possibleArrays:
        raise RuntimeError(f"Unknown \"colorBy\" value {colorBy} "
                           f"(possibilities: {', '.join(possibleArrays)})")

    if colorBy == "visit":
        colorBy = "pfs_visit_id"

    try:
        yLimitsMicron[0]
    except TypeError:
        if yLimitsMicron is None or yLimitsMicron < 0:
            yLimitsMicron = 0

        yLimitsMicron = (-yLimitsMicron, yLimitsMicron)

    if showOnlyMedian:
        showMedian = True
    #
    # Read the data
    #
    if agcData is None:
        if opdb is None:
            raise RuntimeError("You need to supply either an OPDB connection or an agcData array")

        if len(visits) == 0:
            raise RuntimeError("Please specify a set of pfs_visit_ids")

        agcData = readAGCStarsForVisitSetByPfsVisitId(opdb, visits, False, useTraceRadius=useTraceRadius,
                                                      butler=butler)

    agcData0 = agcData

    ll = np.ones(len(agcData), dtype=bool)
    ll &= (agcData.agc_data_flags & ~0x1) == 0

    try:
        ll &= np.isfinite(agcData.agc_camera_id)
    except TypeError:   # Pandas sometimes converts the camera ID to float if it needs NaNs (from a join?)
        pass
    if magMin is not None:
        ll &= agcData.estimated_magnitude >= magMin
    if magMax is not None:
        ll &= agcData.estimated_magnitude <= magMax

    if onlyGuideStars:
        try:
            ll &= np.isfinite(agcData.guide_star_id)
        except TypeError:
            print("No guide stars are found; disabling onlyGuideStars")
            onlyGuideStars = False
        else:
            nGuide = np.sum(ll)
            ll &= (agcData.guide_star_flag & np.array(AutoGuiderStarMask.GAIA)) != 0x0
            ll &= (agcData.guide_star_flag & np.array(AutoGuiderStarMask.NON_BINARY)) != 0x0
            ll &= (agcData.guide_star_flag & np.array(AutoGuiderStarMask.GALAXY)) == 0x0

            print(f"{nGuide} guide stars, of which {np.sum(ll)} are isolated GAIA stars")

    agcData["focusPosition"] = agcData.m2_off3 if useM2Off3 else agcData.m2_pos3

    AGC = list(AGC)
    if len(AGC) != 6:
        keep = np.zeros(len(agcData), dtype=bool)
        for c in AGC:
            keep |= (agcData.agc_camera_id == c - 1)
        ll &= keep

    agcData = agcData[ll]

    agcData = agcData.sample(frac=1)

    # The Focus Offset assumed in the AGActor
    focusOffset = MomentDifferenceToMmPiston(0, includePistonCorrection=True)

    if visits[-1] < 122129:
        # Kawanomoto-san used to use 4*(a**2 + b**2) rather than (a**2 + b**2) in the focus code;
        # this was a bug (INSTRM-2501), fixed on 2025-03-23
        for c in range(0, 6):
            if c == 0:
                c = ""
            agcData[f"guide_delta_z{c}"] = focusOffset + (agcData[f"guide_delta_z{c}"] - focusOffset)/4

    grouped = agcData.groupby(["agc_exposure_id", "left", "agc_camera_id"])
    focus = grouped.agg(
        pfs_visit_id=("pfs_visit_id", "first"),
        altitude=("altitude", "mean"),
        insrot=("insrot", "mean"),
        focusPosition=("focusPosition", "mean"),
        rms=("rms", "median"),
        FWHM=("FWHM", "median"),
        guide_delta_z=("guide_delta_z", "first"),
        guide_delta_z1=("guide_delta_z1", "first"),
        guide_delta_z2=("guide_delta_z2", "first"),
        guide_delta_z3=("guide_delta_z3", "first"),
        guide_delta_z4=("guide_delta_z4", "first"),
        guide_delta_z5=("guide_delta_z5", "first"),
        guide_delta_z6=("guide_delta_z6", "first"),
    )
    focus.reset_index(inplace=True)

    height_ratios = []
    nPanel = 0
    if showAGActorFocus:
        nPanel += 1
        height_ratios.append(2)
    if showOpdbFocus:
        nPanel += 1
        height_ratios.append(2)
    if showFWHM:
        nPanel += 1
        height_ratios.append(3)

    if axes is None:
        figure, axes = plt.subplots(nPanel, 1, num=figure, sharex=True,
                                    height_ratios=height_ratios, squeeze=False)
        axes = axes.flatten()
    else:
        assert len(axes) == nPanel, f"I need {nPanel} axes; you gave me {len(axes)}"

    figure.subplots_adjust(hspace=0.025)

    #
    # Make top two plots, one from AGActor one from our private analysis from opdb guide star parameters
    #
    ai = 0

    for AGActor in [True, False]:
        if AGActor:
            if showAGActorFocus:
                ax = axes[ai]
                ai += 1
            else:
                continue
        else:
            if showOpdbFocus:
                ax = axes[ai]
                ai += 1
            else:
                continue

        marker, alpha = ('o', 1) if averageByFocusPosition else \
            getMA(len(focus)/(1 + focus.agc_camera_id.nunique()), forceAlpha)

        if AGActor:
            #
            # Use analysis written to opdb by AGActor
            #
            if colorBy == "camera":
                if False:
                    print("RHL", focus[["agc_exposure_id", "guide_delta_z",
                                        "guide_delta_z1", "guide_delta_z2", "guide_delta_z3",
                                        "guide_delta_z4", "guide_delta_z5", "guide_delta_z6"]])

                xx = focus[what]
                for ic in sorted(focus.agc_camera_id.unique()):
                    ic = int(ic)
                    ll = focus.agc_camera_id == ic
                    _x, _y = xx[ll].to_numpy(), mmToMicrons*focus[ll][f"guide_delta_z{ic + 1}"].to_numpy()
                    if averageByFocusPosition:
                        _x, _y = averageArraysByFocusPosition(focus.focusPosition[ll], _x, _y)
                    ax.plot(_x, _y, marker, alpha=alpha, color=f"C{ic}", label=f"AG{ic + 1}")
            else:
                grouped = focus.groupby(["agc_exposure_id"])
                _focus = grouped.agg(
                    pfs_visit_id=("pfs_visit_id", "first"),
                    altitude=("altitude", "first"),
                    insrot=("insrot", "first"),
                    guide_delta_z=("guide_delta_z", "first"),
                    focusPosition=("focusPosition", "first"),
                )
                _focus.reset_index(inplace=True)

                xx = _focus[what]
                yy = mmToMicrons*_focus.guide_delta_z
                cc = _focus[colorBy]

            ylab = "AG Actor Focus error"
        else:
            #
            # Estimate and plot a focus correction
            #
            # Handle the case where all the stars are on either the left or right sides
            #
            focusErrors = {}
            for ic in np.sort(focus.agc_camera_id.unique()):
                ic = int(ic)
                focusErrors[ic] = {}

                if colorBy == "camera":
                    _focus = focus[focus.agc_camera_id == ic]
                else:
                    grouped = focus.groupby(["agc_exposure_id", "left"])
                    _focus = grouped.agg(
                        pfs_visit_id=("pfs_visit_id", "first"),
                        altitude=("altitude", "mean"),
                        insrot=("insrot", "mean"),
                        rms=("rms", "median"),
                        focusPosition=("focusPosition", "first"),
                        agc_camera_id=("agc_camera_id", "first"),
                    )
                    _focus.reset_index(inplace=True)

                # noLR == not left AND right
                noLR = set(_focus.agc_exposure_id[~_focus.left]) ^ set(_focus.agc_exposure_id[_focus.left])
                _focus = _focus[~np.isin(_focus.agc_exposure_id, list(noLR))]
                x = _focus[what]

                deltaMxx1D = _focus.rms[_focus.left].to_numpy()**2 - _focus.rms[~_focus.left].to_numpy()**2
                focusError = mmToMicrons*MomentDifferenceToMmPiston(deltaMxx1D, includePistonCorrection=False)
                focusErrors[ic] = focusError

                xx = x[_focus.left]
                yy = focusError

                if colorBy != "camera":
                    cc = _focus[colorBy][_focus.left]
                    agc_camera_id = _focus.agc_camera_id[_focus.left]
                    break

                if averageByFocusPosition:
                    xx, yy = averageArraysByFocusPosition(_focus.focusPosition[_focus.left], xx, yy)

                color = f"C{ic}"
                if not (showMedian and showOnlyMedian):
                    ax.plot(xx, yy, marker, alpha=alpha, color=color)
                if len(xx) > 0:
                    ax.plot([np.nan], [np.nan], 'o', color=color, label=f"AG{ic + 1}")

                if showMedian:
                    if plotBy == "focus":
                        scaling = 1e3
                    else:
                        scaling = 1

                    # distinct rounded values of xx
                    xvals = np.sort(list(set(np.round(scaling*xx).astype(int))))/scaling
                    _xx = np.empty(len(xvals))
                    _yy = np.empty_like(_xx)

                    for i, xv in enumerate(xvals):
                        _xx[i] = xv
                        _yy[i] = np.nanmedian(yy[np.abs(xx - xv) < 1/scaling])
                    ax.plot(_xx, _yy, '-' if connectMedian else marker,
                            color=color, alpha=1 if connectMedian else alpha)

                del _focus

            ylab = r"$\Delta$ focus"

        S = None
        if colorBy == "camera":
            ax.legend(ncols=6)
        else:
            if len(focus[colorBy].unique()) == 0:
                ax.plot(xx, yy, marker, alpha=alpha, color='black', zorder=3)
            else:
                S = ax.scatter(xx, yy, c=cc, marker=marker, alpha=alpha, zorder=10)

        ax.axhline(0, color='black', alpha=0.5)
        if showPfiFocusPosition:
            if AGActor:
                # N.b. AG already corrects for an estimate of AG -> PFI correction
                ax.axhline(0, color='red', label="PFI")
            else:
                # correction for AG -> PFI focal planes
                ax.axhline(-mmToMicrons*focusOffset, color='red', label="PFI")
            ax.legend(ncols=7)

        if yLimitsMicron[0] != 0:
            ax.set_ylim(1e-3*mmToMicrons*np.array(yLimitsMicron))

        ax.set_ylabel(f"{ylab}\n({'mm' if mmToMicrons == 1 else 'micron'})")

    if S is None:
        secay = axes[0].secondary_yaxis('right',
                                        functions=(lambda x: x*guiderFocus_to_M2_OFF3,
                                                   lambda x: x/guiderFocus_to_M2_OFF3))
        secay.set_ylabel(r"$\Delta$ M2_OFF3 (mm)")
    else:
        with opaqueColorbar(S):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            figure.colorbar(S, label=colorBy, cax=cax, orientation='vertical')

    # -------------

    if showFWHM:
        ax = axes[ai]
        ai += 1

        x = agcData[what]

        if not showOnlyMedian:
            dither = ditherScale*(np.max(x) - np.min(x))/np.nanmean(agcData.centroid_y_pix)  # dither a little

            xx = x + dither*(agcData.centroid_y_pix - np.nanmean(agcData.centroid_y_pix))
            yy = agcData.FWHM
            if plotFrac < 1:
                ll = np.random.choice([True, False], len(x), p=[plotFrac, 1 - plotFrac])
                xx = xx[ll]
                yy = yy[ll]
            else:
                ll = np.ones(len(agcData), dtype=bool)

            marker, alpha = getMA(len(xx), forceAlpha)
            if forceAlpha is not None:
                alpha = forceAlpha
            if showCameraId:
                cmap = plt.matplotlib.colors.ListedColormap([f"C{i}" for i in range(6)])
                norm = plt.matplotlib.colors.Normalize(0.5, 6.5)

                for show_left in [True, False]:
                    lll = ll & (agcData.left == show_left)
                    m = marker if show_left else '*'
                    s = (6 if show_left else 4.5)**2

                    ax.scatter(xx[lll], yy[lll], c=agcData.agc_camera_id[lll] + 1,
                               s=s, marker=m, alpha=alpha, cmap=cmap, norm=norm)
                    ax.scatter([np.nan], [np.nan], marker=m, s=s,
                               label="left" if show_left else "right", c="black")

                for agc_camera_id in range(6):
                    if np.sum(ll & (agcData.agc_camera_id == agc_camera_id)) > 0:
                        color = f"C{agc_camera_id}"
                        ax.plot([np.nan], [np.nan], 'o', color=color, label=f"AG{agc_camera_id + 1}")

                ax.legend(ncol=8, columnspacing=1.3)
            else:
                ax.scatter(xx, yy, c=np.where(agcData.left, 'red', 'green'), marker=marker, alpha=alpha)
                if what == "focusPosition":
                    for lr, color in zip([agcData.left, ~agcData.left], ["red", "green"]):
                        ii = np.argsort(agcData.agc_exposure_id[lr])
                        _xx, _yy = xx[lr].iloc[ii], yy[lr].iloc[ii]
                        if maxFWHM is not None:
                            _yy[_yy > maxFWHM] = np.nan

                        ax.plot(_xx, _yy, 'o', color=color, alpha=alpha)

        if showOnlyMedian and not showCameraId:
            ax.plot([np.nan], [np.nan], 'o', color="red", label="left")
            ax.plot([np.nan], [np.nan], 'o', color="green", label="right")
            ax.legend()

        if False:
            f_ratio = 2.2
            pixelSize = 13     # microns
            plateScale = 0.14  # arcsec/pixel
            delta = 150

            for ic in np.sort(agcData.agc_camera_id[ll]):
                dfocus = focusErrors[ic][ll]    # need to broadcast to all stars for this agc_exposure_id
                ll = agcData.agc_camera_id[ll] == ic
                FWHM = yy

                for isLeft in [True, False]:
                    R = (dfocus + delta*(-1 if isLeft else 1))/(2*f_ratio)  # Radius of pupil image in microns
                    R *= plateScale/pixelSize

                    # Moment of inertia of disk is R^2/2
                    ax.plot(xx, np.sqrt(FWHM**2 - (R**2)/2), '.', color="black")

        x = focus[what]
        if showMedian or showOnlyMedian:
            if showCameraId:
                marker, alpha = getMA(len(focus)/6)[0], 1
                for c in range(1, 6+1):
                    color = f"C{c - 1}"
                    ll = focus.agc_camera_id == c - 1
                    xx, yy = x[ll].to_numpy(), focus.FWHM[ll].to_numpy()

                    for left in [True, False]:
                        lll = focus.left[ll] == left
                        ax.plot(xx[lll], yy[lll], '*' if left else '.', alpha=alpha, color=color)
                    # just for the legend
                    if c == 1:
                        ax.plot([np.nan], [np.nan], '*', color='black', label="left")
                        ax.plot([np.nan], [np.nan], '.', color='black', label="right")

                    if sum(ll) > 0:
                        ax.plot([np.nan], [np.nan], 'o', color=color, label=f"AG{c}")

                ax.legend(ncol=8, columnspacing=1.3)

            else:
                focusOffsets = (np.sort(list(set(1000*x))).astype(int) + 0.5)/1000  # round to the nearest um
                _x = np.empty(len(focusOffsets))
                _FWHM = np.empty_like(_x)
                for left in [True, False]:
                    ll = focus.left == left
                    for i, f in enumerate(focusOffsets):
                        _x[i] = f
                        _FWHM[i] = np.nanmedian(focus.FWHM[ll & (np.abs(x - f) < 2e-3)])
                    ax.plot(_x, _FWHM, '-', color='red' if left else 'green')

        ax.set_ylim(minFWHM, maxFWHM)

        ax.set_ylabel(r"FWHM (arcsec)")

    visits = np.sort(focus.pfs_visit_id.unique())
    magLimitsStr = ""
    if magMin is None:
        if magMax is not None:
            magLimitsStr = f"mag < {magMax}"
    else:
        magLimitsStr = f"{magMin} < mag"
        if magMax is not None:
            magLimitsStr += f" < {magMax}"

    figure.supxlabel((("M2_OFF3" if useM2Off3 else "M2_POS3") + " (mm)") if what == "focusPosition" else what)
    figure.suptitle("pfs_visit_id "
                    + (f"{','.join(str(v) for v in visits)}" if len(visits) < 5 else
                       f"{visits[0]}..{visits[-1]}")
                    + (" only GAIA stars" if onlyGuideStars else "")
                    + (f"\n{', '.join(f'AG{int(c) + 1}' for c in np.sort(focus.agc_camera_id.unique()))}")
                    + (f"  {magLimitsStr}" if magLimitsStr else "")
                    + "  " + ("traceRadius" if useTraceRadius else "detRadius"),
                    y=1
                    )

    if showFocusSets:
        overplotFocusSets(what, focus, axes)

    if showAGActorFocus and showOpdbFocus:
        axes[1].sharey(axes[0])

    for ax in axes:
        if what == "agc_exposure_id":
            showVisitBoundaries(ax, agcData, visits)
            ax.format_coord = FormatCoord(plotBy, agcData, opdb)

    if plotBy == "focus":
        ff = ShowFocusFit(axes, indicateFocusPosition=indicateFocusPosition)
        figure.canvas.mpl_connect('button_press_event', ff)

    return agcData0


def plotFocusByAG(opdb, visits,
                  onlyGuideStars=True,
                  byExposureId=False,
                  byCamera=True,
                  maxFocusError=0,
                  minAgcExposureId=0,
                  maxAgcExposureId=0,
                  mmToMicrons=1e3,
                  showLegend=False
                  ):
    """
    byExposureId = False # Plot against agc_exposure_id as opposed to focus position
    """
    #
    # Read the data
    #
    tmp = readAGCStarsForVisitSetByPfsVisitId(opdb, visits, False)

    ff = []
    for c in range(1, 6+1):
        grouped = tmp[c == tmp.agc_camera_id + 1].groupby(["agc_exposure_id", "left",])
        focus = grouped.agg(
            agc_exposure_id=("agc_exposure_id", "first"),
            pfs_visit_id=("pfs_visit_id", "first"),
            agc_camera_id=("agc_camera_id", "first"),
            left=("left", "first"),
            altitude=("altitude", "mean"),
            focusPosition=("m2_off3", "mean"),
            rms=("rms", "median"),
            FWHM=("FWHM", "median"),
        )

        #
        # Estimate and plot a focus correction
        #
        # Handle the case where all the stars are on either the left or right sides
        # noLR == not left AND right
        noLR = set(focus.agc_exposure_id[~focus.left]) ^ set(focus.agc_exposure_id[focus.left])
        _focus = focus[~np.isin(focus.agc_exposure_id, list(noLR))].copy()

        focusError = _focus.rms[_focus.left].to_numpy()**2 - _focus.rms[~_focus.left].to_numpy()**2
        focusError = mmToMicrons*MomentDifferenceToMmPiston(focusError, includePistonCorrection=False)

        # Extend focusError to be the same length as _focus
        # It'd probably be OK to do this with numpy stacking,
        # but I'm not quite sure that pandas defines the order

        tmpVec = np.empty(len(_focus))
        tmpVec[_focus.left] = focusError
        tmpVec[~_focus.left] = focusError
        _focus["focusError"] = tmpVec
        del tmpVec

        ff.append(_focus)

    _focus = pd.concat(ff)
    del ff

    ll = np.ones(len(_focus), dtype=bool)
    if maxFocusError > 0:
        ll &= np.abs(_focus.focusError) < maxFocusError
    if minAgcExposureId > 0:
        ll &= _focus.agc_exposure_id >= minAgcExposureId
    if maxAgcExposureId > 0:
        ll &= _focus.agc_exposure_id <= maxAgcExposureId

    if False:
        print(f"RHL {len(_focus)} {sum(ll)}")
        print(f"RHL {np.sort(_focus.agc_exposure_id.unique())}")

    if sum(ll) != len(ll):
        _focus = _focus[ll]

    if byExposureId:
        xname = "agc_exposure_id"
    else:
        xname = "pfs_visit_id"

    _focus.reset_index(drop=True, inplace=True)
    vgrouped = _focus.groupby([xname, "agc_camera_id"], as_index=True)
    focusByVisit = vgrouped.agg(
        focusError=("focusError", "median"),
        focusPosition=("focusPosition", "median"),
        nAGCExposure=("agc_camera_id", "count"),
        _pfs_visit_id=("pfs_visit_id", "first"),  # might also be an index field
        _agc_camera_id=("agc_camera_id", "first"),  # also an index field
    )
    #
    # subtract off the per-visit values
    #
    agc = 1                             # Ignore AG1 when taking the average
    grouped = focusByVisit[focusByVisit._agc_camera_id + 1 != agc].groupby(xname)
    focusByVisit.focusError -= grouped.agg(fp=("focusError", "mean")).fp
    focusByVisit.focusError -= np.mean(focusByVisit.focusError)

    focusByVisit.reset_index(inplace=True)

    ll = focusByVisit.nAGCExposure > 20

    focusByVisit.focusError *= -1  # Match Kawanomoto-san's plot

    if byCamera:
        for v in np.sort(focusByVisit[xname].unique()):
            ll = (v == focusByVisit[xname])
            plt.plot(focusByVisit.agc_camera_id[ll] + 1, focusByVisit.focusError[ll],
                     '-o', label=f"{v}", alpha=0.5)

    focusErrorPerCamera = np.full(6, np.nan)
    for c in range(1, 6+1):
        ll = (focusByVisit.agc_camera_id == c - 1)
        if sum(ll) == 0:
            continue

        if not byCamera:
            plt.plot(focusByVisit[xname][ll], focusByVisit.focusError[ll], '-o',
                     color=f"C{c - 1}", label=f"AG{c}", alpha=1)

        focusErrorPerCamera[c - 1] = np.nanmean(focusByVisit.focusError[ll])
        plt.axhline(focusErrorPerCamera[c - 1], color=f"C{c-1}", alpha=1,
                    label=None if not byCamera else f"AG{c}")

    if byCamera:
        plt.plot(range(1, 6+1), focusErrorPerCamera, marker="*", markersize=10, color="black")

    if showLegend:
        plt.legend(ncols=2)

    xlab = "AG Camera" if byCamera else xname
    plt.xlabel(xlab)
    plt.ylabel(r"$\Delta$ focus AG (micron)")

    visits = np.sort(_focus[xname].unique())
    plt.suptitle(f"{xname} "
                 + (f"{','.join(str(v) for v in visits)}" if len(visits) < 5 else
                    f"{visits[0]}..{visits[-1]}")
                 + (" only GAIA" if onlyGuideStars else "")
                 + (f"\n{', '.join(f'AG{int(c) + 1}' for c in np.sort(tmp.agc_camera_id.unique()))}")
                 )


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Plotting utils


def drawCircularArrow(radius, cen, theta12, clockwise=True, angle=0, ax=None, **kwargs):
    """Draw a circular arrow
    e.g.
       drawCircularArrow(30, (0, 0), (0, 250), clockwise=True)
    """
    # modified from https://stackoverflow.com/questions/37512502/how-to-make-arrow-that-loops-in-matplotlib
    theta1, theta2 = theta12
    if ax is None:
        ax = plt.gca()

    # Draw the line
    arc = Arc(cen, radius, radius, angle=angle, theta1=theta1, theta2=theta2, capstyle='round', **kwargs)
    ax.add_patch(arc)

    kwargs.update(color=arc.get_edgecolor())

    # Create and draw the arrow head as a triangle
    theta_end = np.deg2rad((theta1 if clockwise else theta2) + angle)
    endX = cen[0] + (radius/2)*np.cos(theta_end)  # determine end position
    endY = cen[1] + (radius/2)*np.sin(theta_end)

    ax.add_patch(RegularPolygon((endX, endY), numVertices=3, radius=radius/9,
                                orientation=np.deg2rad(angle + theta2 + (180 if clockwise else 0)),
                                **kwargs))


def showAGCameraCartoon(ax1, showInstrot=False, showUp=False, lookingAtHardware=True, insrot=None):
    ax = ax1.inset_axes([0.01, 0.01, 0.2, 0.2])
    ax.set_aspect(1)
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_zorder(-1)

    for agc_camera_id in range(6):
        color = f"C{agc_camera_id}"

        xbar, ybar = agcCameraCenters[agc_camera_id]

        if insrot is not None:
            xbar, ybar = rotXY(-insrot, np.array([xbar]), np.array([ybar]))
            xbar, ybar = xbar[0], ybar[0]

        if not lookingAtHardware:
            ybar *= -1

        ax.text(xbar, ybar, f"{agc_camera_id + 1}",
                ha="center", va="center", color=color)
        ax.set_xlim(ax.set_ylim(-300, 300))

    if showInstrot:
        drawCircularArrow(100, (0, 10), (0, 250), clockwise=lookingAtHardware, ax=ax)

    if showUp:
        ax.text(0, 5, r"0$^\circ$", ha="center", va="center", size=7)
        for i in [-1, 1]:
            ax.text(-260, i*150, r'$\uparrow$', va='center', rotation=90)
