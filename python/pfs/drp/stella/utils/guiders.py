import warnings
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, RegularPolygon
import pandas as pd
import psycopg2

from pfs.utils.coordinates.transform import MeasureDistortion


__all__ = ["agcCameraCenters", "showAgcErrorsForVisits",
           "readAgcDataFromOpdb", "getAGCPositionsForVisitByAgcExposureId", ]

# Approximate centers of AG camera (indexed by agc_camera_id)
agcCameraCenters = {
    0: ( 237.58,   -0.50),              # noqa E201, E241
    1: ( 120.19,  212.49),              # noqa E201, E241
    2: (-120.02,  212.10),              # noqa E201, E241
    3: (-242.08,    2.00),              # noqa E201, E241
    4: (-122.58, -211.67),              # noqa E201, E241
    5: ( 119.23, -209.79),              # noqa E201, E241
}

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# Communicate with the DB


def pd_read_sql(sql_query: str, db_conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """Execute SQL Query and get Dataframe with pandas"""
    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        # create pandas DataFrame from database query
        df = pd.read_sql_query(sql_query, db_conn)
    return df


def getAGCPositionsForVisitByAgcExposureId(obdb, pfs_visit_id, flipToHardwareCoords, agc_exposure_idStride=1):
    with obdb:
        tmp = pd_read_sql(f'''
           SELECT
               min(pfs_visit_id) as pfs_visit_id, agc_exposure.agc_exposure_id,
               min(agc_exposure.taken_at) as taken_at,
               avg(agc_exposure.altitude) as altitude, avg(agc_exposure.azimuth) as azimuth,
               avg(agc_exposure.insrot) as insrot,
               agc_match.guide_star_id, min(m2_pos3) as m2_pos3,
               avg(agc_match.agc_camera_id) as agc_camera_id,
               avg(agc_nominal_x_mm) as agc_nominal_x_mm, avg(agc_center_x_mm) as agc_center_x_mm,
               avg(agc_nominal_y_mm) as agc_nominal_y_mm, avg(agc_center_y_mm) as agc_center_y_mm,
               min(agc_match.flags) as agc_match_flags, min(agc_data.flags) as agc_data_flags
           FROM agc_exposure
           JOIN agc_data ON agc_data.agc_exposure_id = agc_exposure.agc_exposure_id
           JOIN agc_match ON agc_match.agc_exposure_id = agc_data.agc_exposure_id AND
                             agc_match.agc_camera_id = agc_data.agc_camera_id AND
                             agc_match.spot_id = agc_data.spot_id
           WHERE
               pfs_visit_id = {pfs_visit_id}
           GROUP BY guide_star_id, agc_exposure.agc_exposure_id
           ''', obdb)

    if flipToHardwareCoords:
        tmp.agc_nominal_y_mm *= -1
        tmp.agc_center_y_mm *= -1

    return tmp


def readAgcDataFromOpdb(opdb, visits, agc_exposure_idStride=1, butler=None, dataId=None):
    """Read a useful set of data about the AGs from the opdb
    opdb: connection to the opdb
    visits: list of the desired visits
    butler: a butler to read INST-PA; or None
    """
    dd = []
    for v in visits:
        dd.append(getAGCPositionsForVisitByAgcExposureId(opdb, v, flipToHardwareCoords=True,
                                                         agc_exposure_idStride=1))
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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def stdFromIQR(im):
    Q1, Q3 = np.percentile(im, [25, 75])
    return 0.74130*(Q3 - Q1)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def showAgcErrorsForVisits(agcData,
                           byTime=False,
                           figure=None
                           ):
    """
    agcData: pandas DataFrame as returned by readAgcDataFromOpdb
    byTime = False    # use time, not agc_exposure_id, as x-axis
    matplotlib figure to use, or None
    """
    fig, axs = plt.subplots(3, 1, num=figure, sharex=True, sharey=False, squeeze=False)
    axs = axs.flatten()

    agc_exposure_ids = np.array(sorted(set(agcData.agc_exposure_id)))

    pfs_visit_ids = np.empty(len(agc_exposure_ids))
    taken_ats = np.empty(len(agc_exposure_ids), dtype=agcData.taken_at.dtype)
    for i, eid in enumerate(agc_exposure_ids):
        sel = agcData.agc_exposure_id == eid
        pfs_visit_ids[i] = agcData.pfs_visit_id[sel].to_numpy()[0]
        taken_ats[i] = agcData.taken_at[sel].to_numpy()[0]

    xbar = np.empty(len(agc_exposure_ids))
    ybar = np.full_like(xbar, np.NaN)

    for i, eid in enumerate(agc_exposure_ids):
        present = (agcData.agc_exposure_id == eid).to_numpy()
        xerr = agcData.agc_nominal_x_mm - agcData.agc_center_x_mm
        xbar[i] = 1e3*np.nanmean(xerr[present])

        yerr = agcData.agc_nominal_y_mm - agcData.agc_center_y_mm
        ybar[i] = 1e3*np.nanmean(yerr[present])

    def plot_zbar(zbar, xvec=agc_exposure_ids):
        plt.gca().set_prop_cycle(None)
        for pfs_visit_id in sorted(set(pfs_visit_ids)):
            sel = pfs_visit_ids == pfs_visit_id
            plt.plot(xvec[sel], zbar[sel], '.-', label=f"{int(pfs_visit_id)}")

        plt.axhline(0, color='black')
        plt.legend(ncol=6)

    j = 0
    plt.sca(axs[j]); j += 1             # noqa E702

    plot_zbar(np.hypot(xbar, ybar), taken_ats if byTime else agc_exposure_ids)
    plt.ylabel("rerror (microns)")

    plt.sca(axs[j]); j += 1             # noqa E702

    plot_zbar(xbar, taken_ats if byTime else agc_exposure_ids)
    plt.ylabel("xerror (microns)")

    plt.sca(axs[j]); j += 1             # noqa E702
    plot_zbar(ybar, taken_ats if byTime else agc_exposure_ids)
    plt.ylabel("yerror (microns)")

    plt.xlabel("HST" if byTime else "agc_exposure_id")

    visits = sorted(set(pfs_visit_ids))
    plt.suptitle(f"{visits[0]:.0f}..{visits[-1]:.0f}")

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


class GuiderConfig:
    def __init__(self,
                 showAverageGuideStarPos=False,
                 showGuideStars=True,
                 showGuideStarsAsPoints=True,
                 showGuideStarsAsArrows=False,
                 showGuideStarPositions=False,
                 showByVisit=True,
                 rotateToZenith=True,
                 flipInsrot=False,
                 fixBoresightOffset=True,
                 fixCCDOffset=True,
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
                 agc_exposure_cm=plt.matplotlib.cm.get_cmap("viridis"),
                 showByVisitSize=5,
                 showByVisitAlpha=1):
        """
        showAverageGuideStarPos=False     plot the per-agc_exposure_id average of selected guide stars,
                                          per AG chip (ignores guide_star_frac)
        showGuideStars=True               plot the selected guide stars
        showGuideStarsAsPoints=True       show positions of quide stars as points, not arrows
        showGuideStarsAsArrows=False
        showGuideStarPositions=False      show guide stars with correct relative positions on the CCD
        showByVisit=True                  colour code by exposure_id, not AG camera (ignored if
                                          self.showGuideStarsAsPoints is False)
        rotateToZenith=True
        flipInsrot=False
        fixBoresightOffset=True           remove the mean offset and rotation/scale for each exposure
        fixCCDOffset=True                 remove the mean offset and rotation/scale for each CCD
        maxGuideError=25                  only plot exposures with |guideError| < maxGuideError microns;
                                          ignored if <= 0
                                          N.b. guideError is the mean of stars with
                                          guide errors < maxPosError
        maxPosError=40                    don't show guide stars with mean(error) > maxPosError (micros);
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
        self.showAverageGuideStarPos = showAverageGuideStarPos
        self.showGuideStars = showGuideStars
        self.showGuideStarsAsPoints = showGuideStarsAsPoints
        self.showGuideStarsAsArrows = showGuideStarsAsArrows
        self.showGuideStarPositions = showGuideStarPositions
        self.showByVisit = showByVisit
        self.rotateToZenith = rotateToZenith
        self.flipInsrot = flipInsrot
        self.fixBoresightOffset = fixBoresightOffset
        self.fixCCDOffset = fixCCDOffset
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


def showGuiderErrors(agcData, transforms={}, verbose=False,
                     showAverageGuideStarPos=False,
                     showGuideStars=True,
                     showGuideStarsAsPoints=True,
                     showGuideStarsAsArrows=False,
                     showGuideStarPositions=False,
                     showByVisit=True,
                     rotateToZenith=True,
                     flipInsrot=False,
                     fixBoresightOffset=True,
                     fixCCDOffset=True,
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
                     agc_exposure_cm=plt.matplotlib.cm.get_cmap("viridis"),
                     showByVisitSize=5,
                     showByVisitAlpha=1):
    """
        agcData: pandas DataFrame as returned by readAgcDataFromOpdb
        showAverageGuideStarPos=False, # plot the per-agc_exposure_id average of selected guide stars,
                                             per AG chip (ignores guide_star_frac)
        showGuideStars=True,          # plot the selected guide stars
        showGuideStarsAsPoints=True,  # show positions of quide stars as points, not arrows
        showGuideStarsAsArrows=False,
        showGuideStarPositions=False, # show guide stars with correct relative positions on the CCD
        showByVisit=True,  # colour code by exposure_id, not AG camera (ignored if showGuideStarsAsPoints
                             is False)
        rotateToZenith=True,
        flipInsrot=False,
        fixBoresightOffset=True,  # remove the mean offset and rotation/scale for each exposure
        fixCCDOffset=True,        # remove the mean offset and rotation/scale for each CCD
        maxGuideError=25,     # only plot exposures with |guideError| < maxGuideError microns; ignore if <= 0
                              # N.b. guideError is the mean of stars with guide errors < maxPosError
        maxPosError=40,       # don't show guide stars with mean(error) > maxPosError (micros); ignore if <= 0
        guideErrorEstimate=50,     # average guide error in microns (used to show quiver scalebar)
        pfiScaleReduction=1, # 2.4  # How much to shrink AG distances from the centre of the PFI
        gstarExpansion=10,  # How much to expand the distances of guide stars relative to the centre of
                            # their AG camera
        agc_exposure_idsStride=1,  # only plot every agc_exposure_idsStride'th AG exposure
        guide_star_frac=0.1,       # only plot this fraction (0..1) of the guide_stars seen on an AG
        pfs_visitIdMin=0,          # minimum allowed value of pfs_visit_id; ignored if <= 0
        pfs_visitIdMax=0,          # minimum allowed value of pfs_visit_id; ignored if <= 0
        agc_exposure_idMin=0,      # minimum allowed value of agc_exposure_id; ignored if <= 0
        agc_exposure_idMax=0,      # maximum allowed value of agc_exposure_id; ignored if <= 0
        agc_exposure_cm=plt.matplotlib.cm.get_cmap("viridis"),
        showByVisitSize=5,
                     showByVisitAlpha=1):
    """
    #
    # Check options
    #
    if not showGuideStarsAsPoints:
        if showByVisit:
            print("Disabling showByVisit")
            showByVisit = False

    guide_star_frac = min([guide_star_frac, 1])  # sanity check

    agc_exposure_ids = np.array(sorted(set(agcData.agc_exposure_id)))

    agc_nominal_x_mm = agcData.agc_nominal_x_mm.to_numpy().copy()
    agc_nominal_y_mm = agcData.agc_nominal_y_mm.to_numpy().copy()
    agc_center_x_mm = agcData.agc_center_x_mm.to_numpy().copy()
    agc_center_y_mm = agcData.agc_center_y_mm.to_numpy().copy()

    if rotateToZenith:
        insrot = np.deg2rad(agcData.insrot).to_numpy()

    def rotXY(angle, x, y):
        """Rotate (x, y) by angle in a +ve direction
        insrot in radians"""
        c, s = np.cos(angle), np.sin(angle)
        x, y = c*x - s*y, s*x + c*y

        return x, y
    #
    # Fix the mean guiding offset for each exposure?
    #
    try:
        transforms
    except (NameError, KeyError):
        print("Unable to find transforms; building")
        transforms = {}

    if fixBoresightOffset:
        dx = agc_center_x_mm - agc_nominal_x_mm
        dy = agc_center_y_mm - agc_nominal_y_mm
        dr = np.hypot(dx, dy)

        for aid in agc_exposure_ids:
            sel = agcData.agc_exposure_id == aid

            # sel &= dr < 200*1e-3  # remove egregious errors in centroids

            if aid in transforms:
                transform = transforms[aid]
            else:
                if verbose:
                    print(f"Solving for offsets/rotations for {aid}")
                transform = MeasureDistortion(agc_center_x_mm[sel], agc_center_y_mm[sel], -1,
                                              agc_nominal_x_mm[sel], agc_nominal_y_mm[sel], None)

                res = scipy.optimize.minimize(transform, transform.getArgs(), method='Powell')
                transform.setArgs(res.x)
                transforms[aid] = transform

            agc_nominal_x_mm[sel], agc_nominal_y_mm[sel] = \
                transform.distort(agc_nominal_x_mm[sel], agc_nominal_y_mm[sel], inverse=True)

    dx = agc_center_x_mm - agc_nominal_x_mm
    dy = agc_center_y_mm - agc_nominal_y_mm
    dr = np.hypot(dx, dy)
    #
    # Find the mean guiding errors for each exposure
    #
    guidingErrors = {}
    for aid in agc_exposure_ids:
        sel = agcData.agc_exposure_id == aid
        guidingErrors[aid] = np.mean(dr[sel][dr[sel] < 1e-3*maxPosError])
    #
    # Work in coordinates aligned with polar angle
    #
    if rotateToZenith and flipInsrot:
        insrot = -insrot
    if False and rotateToZenith:
        dx, dy = rotXY(-insrot, dx, dy)
    #
    # Set the agc_nominal_[xy]_mm values for each star to have the same position as in the first exposure_id,
    # to take out dithers
    # N.b. only used for plotting -- you must have set dx, dy first!
    #
    agc_nominal_x_mm0 = np.empty_like(agc_nominal_x_mm)
    agc_nominal_y_mm0 = np.empty_like(agc_nominal_y_mm)
    for gid in set(agcData.guide_star_id):
        sel = agcData.guide_star_id == gid
        agc_nominal_x_mm0[sel] = agc_nominal_x_mm[sel][0]
        agc_nominal_y_mm0[sel] = agc_nominal_y_mm[sel][0]

    agc_exposure_id0 = 0         # start with this agc_exposure_ids (useful when agc_exposure_idsStride != 1)
    if agc_exposure_idsStride > 0 and maxGuideError > 0:
        for aid in agc_exposure_ids:
            if guidingErrors[aid] < 1e-3*maxGuideError:
                agc_exposure_id0 = np.where(agc_exposure_ids == aid)[0][0]
                break

    guideErrorByCamera = {}   # per-exposure mean guide error
    S = None                  # returned by scatter
    Q = None                  # and quiver
    plt.gca().set_prop_cycle(None)
    for agc_camera_id in range(6):
        if False and agc_camera_id + 1 not in (1,):
            continue
        color = f"C{agc_camera_id}"
        AGlabel = f"AG{agc_camera_id + 1}"

        sel = np.zeros(len(agcData.agc_camera_id), dtype=int)

        for aid in agc_exposure_ids[agc_exposure_id0::agc_exposure_idsStride]:
            if maxGuideError > 0:
                if guidingErrors[aid] > 1e-3*maxGuideError:
                    continue
            sel |= agcData.agc_exposure_id == aid

        sel &= (agcData.agc_camera_id == agc_camera_id).to_numpy()

        if pfs_visitIdMin > 0:
            sel &= agcData.pfs_visit_id >= pfs_visitIdMin
        if pfs_visitIdMax > 0:
            sel &= agcData.pfs_visit_id <= pfs_visitIdMax
        if agc_exposure_idMin > 0:
            sel &= agcData.agc_exposure_id >= agc_exposure_idMin
        if agc_exposure_idMax > 0:
            sel &= agcData.agc_exposure_id <= agc_exposure_idMax
        # sel &= agcData.agc_match_flags == 0

        if sum(sel) > 0:
            guide_star_ids = sorted(set(agcData.guide_star_id[sel]))
            np.random.seed(666)
            nguide_star = len(guide_star_ids)
            useGuideStars = np.random.choice(range(nguide_star),
                                             max([1, int(guide_star_frac*nguide_star)]), replace=False)

            if fixCCDOffset:
                #
                # Solve for a per-CCD offset and rotation/scale
                #
                print(f"Solving for offsets/rotations for {AGlabel}")
                transform = MeasureDistortion(agc_center_x_mm[sel], agc_center_y_mm[sel], -1,
                                              agc_nominal_x_mm[sel], agc_nominal_y_mm[sel], None)

                res = scipy.optimize.minimize(transform, transform.getArgs(), method='Powell')
                transform.setArgs(res.x)

                agc_nominal_x_mm[sel], agc_nominal_y_mm[sel] = \
                    transform.distort(agc_nominal_x_mm[sel], agc_nominal_y_mm[sel], inverse=True)

                dx[sel] = (agc_center_x_mm - agc_nominal_x_mm)[sel]
                dy[sel] = (agc_center_y_mm - agc_nominal_y_mm)[sel]

            if False:
                print(f"{AGlabel}  ({np.mean(1e3*dx[sel]):4.1f}, {np.mean(1e3*dy[sel]):4.1f}) +- "
                      f"({stdFromIQR(1e3*dx[sel]):4.1f}, {stdFromIQR(1e3*dy[sel]):4.1f})")

            if rotateToZenith:
                agc_nominal_x_mm[sel], agc_nominal_y_mm[sel] = \
                    rotXY(-insrot[sel], agc_nominal_x_mm[sel], agc_nominal_y_mm[sel])
                agc_center_x_mm[sel], agc_center_y_mm[sel] = \
                    rotXY(-insrot[sel], agc_center_x_mm[sel], agc_center_y_mm[sel])
                dx[sel], dy[sel] = rotXY(-insrot[sel], dx[sel], dy[sel])

            agc_nominal_x_mm0[sel] = agc_nominal_x_mm[sel][0]
            agc_nominal_y_mm0[sel] = agc_nominal_y_mm[sel][0]

            x, y = agc_center_x_mm[sel], agc_center_y_mm[sel]
            x /= pfiScaleReduction
            y /= pfiScaleReduction

            if rotateToZenith:
                xbar, ybar = np.empty_like(agc_center_x_mm), np.empty_like(agc_center_y_mm)

                for aid in set(agcData.agc_exposure_id[sel]):
                    la = sel & (agcData.agc_exposure_id.to_numpy() == aid)
                    xbar[la] = np.mean(agc_center_x_mm[la])/pfiScaleReduction
                    ybar[la] = np.mean(agc_center_y_mm[la])/pfiScaleReduction
                xbar, ybar = xbar[sel], ybar[sel]

                agc_ring_R = np.median(np.hypot(agc_nominal_x_mm, agc_nominal_y_mm))
                rbar = np.hypot(xbar, ybar)

                xbar *= agc_ring_R/rbar
                ybar *= agc_ring_R/rbar
            else:
                xbar = np.full_like(x, np.mean(x))
                ybar = np.full_like(xbar, np.mean(y))

            if rotateToZenith:
                plt.gca().add_patch(Circle((0, 0), agc_ring_R, fill=False, color="red"))
            else:
                plt.plot(xbar, ybar, '+',
                         color='red' if (showAverageGuideStarPos and
                                         not (showGuideStarsAsArrows or showGuideStarsAsPoints)) else 'black',
                         zorder=10)

            labelled = False
            xgs, ygs = {}, {}
            for i, gid in enumerate(guide_star_ids):
                gl = agcData.guide_star_id[sel] == gid

                if sum(gl) == 0:
                    print(f"Skipping {gid}")
                    continue

                xg, yg = x[gl], y[gl]
                dxg, dyg = dx[sel][gl], dy[sel][gl]

                if maxPosError > 0 and np.mean(np.hypot(dxg, dyg)) > 1e-3*maxPosError:
                    continue

                if showGuideStarPositions:
                    xg = xbar[gl] + (xg - xbar[gl])*gstarExpansion
                    yg = ybar[gl] + (yg - ybar[gl])*gstarExpansion
                else:
                    xg, yg = 0*xg + xbar[gl], 0*yg + ybar[gl]

                _agc_exposure_id = list(agcData.agc_exposure_id[sel][gl])
                for aid in agc_exposure_ids[::agc_exposure_idsStride]:
                    if aid in _agc_exposure_id:
                        if aid not in xgs:
                            xgs[aid], ygs[aid] = [], []

                        j = _agc_exposure_id.index(aid)
                        xgs[aid].append(xg[j] + 1e3*dxg[j])
                        ygs[aid].append(yg[j] + 1e3*dyg[j])

                if showGuideStars and i in useGuideStars:
                    label = None if labelled else AGlabel
                    xend, yend = xg + 1e3*dxg, yg + 1e3*dyg
                    if showGuideStarsAsPoints:
                        if showByVisit:
                            S = plt.scatter(xend, yend, s=showByVisitSize, alpha=showByVisitAlpha,
                                            vmin=agc_exposure_ids[0], vmax=agc_exposure_ids[-1],
                                            c=agcData.agc_exposure_id[sel][gl], cmap=agc_exposure_cm)
                        else:
                            plt.plot(xend, yend, '.', color=color, label=label, alpha=showByVisitAlpha,
                                     markersize=showByVisitSize)
                    elif showGuideStarsAsArrows:
                        Q = plt.quiver(xg, yg, xend - xg, yend - yg, alpha=0.5, color=color, label=label)
                    else:
                        pass   # useful code path if showAverageGuideStarPos is true
                    labelled = True

            xa = np.empty(len(xgs))
            ya = np.empty_like(xa)
            for i, aid in enumerate(xgs):
                xa[i] = np.mean(xgs[aid])
                ya[i] = np.mean(ygs[aid])

            guideErrorByCamera[agc_camera_id] = (list(xgs.keys()), xa, ya)

            if showAverageGuideStarPos:
                if showGuideStarsAsPoints or showGuideStarsAsArrows:
                    plt.plot(xa[0], ya[0], '.', color='black', zorder=-1)
                    plt.plot(xa, ya, '-', color='black', alpha=0.5, zorder=-1)
                else:
                    S = plt.scatter(xa, ya, s=showByVisitSize, alpha=showByVisitAlpha,
                                    vmin=agc_exposure_ids[0], vmax=agc_exposure_ids[-1],
                                    c=list(xgs.keys()), cmap=agc_exposure_cm)

    if S is not None:
        a = S.get_alpha()
        S.set_alpha(1)
        plt.colorbar(S).ax.set_title("agc_exposure_id")
        S.set_alpha(a)

    if Q is not None:
        qlen = guideErrorEstimate   # microns
        plt.quiverkey(Q, 0.1, 0.9, qlen, f"{qlen} micron", color='black')

    if not rotateToZenith and (showByVisit or (showAverageGuideStarPos and not showGuideStarsAsArrows)):
        showAGCameraCartoon(showInstrot=True, showUp=True)
    else:
        L = plt.legend(loc="lower right", markerscale=1)
        for lh in L.legendHandles:
            lh.set_alpha(1)

    plt.gca().set_aspect(1)
    #
    # Fiddle limits
    #
    if False:      # make plot square, and expand a little to fit in the quivers
        axScale = 1.1
        lims = axScale*np.array([plt.xlim(), plt.ylim()])
    else:
        lims = 350/pfiScaleReduction*np.array([-1, 1])
    plt.xlim(plt.ylim(np.min(lims), np.max(lims)))

    if Q is not None:
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    else:
        plt.xlabel(r"$\delta$x (microns)")
        plt.ylabel(r"$\delta$y (microns)")

    def nameStride(stride):
        if stride == 1:
            return ""
        return f" {stride}" + {2: 'nd', 3: 'rd'}.get(stride, 'th')

    visits = sorted(set(agcData.pfs_visit_id))
    v0, v1 = visits[0], visits[-1]
    if pfs_visitIdMin > 0 and v0 < pfs_visitIdMin:
        v0 = pfs_visitIdMin
    if pfs_visitIdMax > 0 and v1 < pfs_visitIdMax:
        v1 = pfs_visitIdMax
    title = f"{v0}..{v1}"
    if fixBoresightOffset or fixCCDOffset:
        title += " "
        if fixBoresightOffset:
            title += " Boresight"
        if fixCCDOffset:
            title += " AGs"
        title += " offset and rotation/scale removed"
    if rotateToZenith:
        title += " Rotated"
        if flipInsrot:
            title += " (flipped) "
        if np.std(np.rad2deg(insrot[sel])) < 5:
            title += f"  {np.mean(np.rad2deg(-insrot[sel])):.1f}" r"$^\circ$"
            title += f" alt,az = {np.mean(agcData.altitude[sel]):.1f}, {np.mean(agcData.azimuth[sel]):.1f}"
    if maxGuideError > 0 or maxPosError > 0:
        title += '\n' if True else ' '
        if maxGuideError > 0:
            title += f"Max guide error: {maxGuideError} microns"
        if maxPosError > 0:
            title += f" Max per-star positional error: {maxPosError} microns"
    title += f"\nEvery{nameStride(agc_exposure_idsStride)} AG exposure"
    if showGuideStars:
        title += f"; {100*guide_star_frac:.0f}% of guide stars"
    if showAverageGuideStarPos and not (showGuideStarsAsArrows or showGuideStarsAsPoints):
        title += "  Mean guide error"
    plt.suptitle(title)

    return title, guideErrorByCamera


def showGuiderErrorsByParams(agcData, guideErrorByCamera, params, title, figure=None,
                             plotScatter=True,
                             rotateToZenith=True,
                             pfiScaleReduction=1,
                             showByVisitSize=5,
                             showByVisitAlpha=1,
                             agc_exposure_cm=plt.matplotlib.cm.get_cmap("viridis")):
    n = len(params)
    ny = int(np.sqrt(n))
    nx = n//ny
    if nx*ny < n:
        ny += 1

    fig, axs = plt.subplots(ny, nx, num=figure, sharex=True,
                            sharey=True if plotScatter else False, squeeze=False)
    axs = axs.flatten()

    agc_ring_R = np.mean(np.hypot(*np.array(list(agcCameraCenters.values())).T))

    for i, (ax, what) in enumerate(zip(axs, params)):
        plt.sca(ax)

        if plotScatter:
            xas, yas, params = [], [], []
            for agc_camera_id in guideErrorByCamera:
                aids, xa, ya = guideErrorByCamera[agc_camera_id]
                p = getattr(agcData, what)
                if what == "distance_from_ag_center" and i == 2:
                    op = np.std
                    opname = "std"
                elif True:
                    op = np.mean
                    opname = "mean"
                else:
                    op = np.median
                    opname = "median"

                param = np.array([op(p[agcData.agc_exposure_id == aid]) for aid in aids])
                xas.append(list(xa))
                yas.append(list(ya))
                params.append(list(param))

            what = f"{opname}({what})"

            xas = np.array(sum(xas, []))
            yas = np.array(sum(yas, []))
            params = np.array(sum(params, []))

            vmin, vmax = np.min(params), np.max(params)
            vmin, vmax = np.percentile(params, [10, 90])

            S = plt.scatter(xas, yas, s=showByVisitSize, alpha=showByVisitAlpha, vmin=vmin, vmax=vmax,
                            c=params, cmap=agc_exposure_cm)

            a = S.get_alpha()
            S.set_alpha(1)
            C = plt.colorbar(S, shrink=1/nx if ny == 1 else 1)
            C.ax.set_title(what)
            S.set_alpha(a)

            if rotateToZenith:
                plt.gca().add_patch(Circle((0, 0), agc_ring_R, fill=False, color="red"))

            if True:
                lims = 350/pfiScaleReduction*np.array([-1, 1])
                plt.xlim(plt.ylim(np.min(lims), np.max(lims)))

            plt.gca().set_aspect(1)

            if not rotateToZenith:
                showAGCameraCartoon(showInstrot=True, showUp=False)
        else:
            if what in ("rms_pix", "distance_from_ag_center"):
                for agc_camera_id in guideErrorByCamera:
                    sel = agcData.agc_camera_id == agc_camera_id
                    plt.plot(agcData.agc_exposure_id[sel], getattr(agcData, what)[sel], 'o',
                             color=f"C{agc_camera_id}",
                             label=f"{agc_camera_id+1} {what}")
            else:
                plt.plot(agcData.agc_exposure_id, getattr(agcData, what), 'o', label=what)
            plt.legend()

    fig.supxlabel(r"$\delta$x (microns)")
    fig.supylabel(r"$\delta$y (microns)")
    plt.suptitle(title, y=1.0)

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


def showAGCameraCartoon(showInstrot=False, showUp=False, lookingAtHardware=True):
    ax = plt.gca().inset_axes([0.01, 0.01, 0.2, 0.2])
    ax.set_aspect(1)
    ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    for agc_camera_id in range(6):
        color = f"C{agc_camera_id}"

        xbar, ybar = agcCameraCenters[agc_camera_id]

        if not lookingAtHardware:
            ybar *= -1

        ax.text(xbar, ybar, f"{agc_camera_id + 1}",
                ha="center", va="center", color=color)
        ax.set_xlim(ax.set_ylim(-300, 300))

    if showInstrot:
        drawCircularArrow(100, (0, 10), (0, 250), clockwise=lookingAtHardware, ax=ax)

    if showUp:
        for i in [-1, 1]:
            ax.text(-260, i*150, r'$\uparrow$', va='center', rotation=90)
