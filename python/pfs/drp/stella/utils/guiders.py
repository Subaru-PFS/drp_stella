import warnings
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, RegularPolygon
import pandas as pd
import psycopg2

from pfs.utils.coordinates.transform import MeasureDistortion


__all__ = ["GuiderConfig", "agcCameraCenters", "showAgcErrorsForVisits",
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
    transforms = {}

    def __init__(self,
                 showAverageGuideStarPos=False,
                 showGuideStars=True,
                 showGuideStarsAsPoints=True,
                 showGuideStarsAsArrows=False,
                 showGuideStarPositions=False,
                 showByVisit=True,
                 rotateToZenith=True,
                 modelBoresightOffset=True,
                 modelCCDOffset=True,
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
        modelBoresightOffset=True           remove the mean offset and rotation/scale for each exposure
        modelCCDOffset=True                 remove the mean offset and rotation/scale for each CCD
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
        self.modelBoresightOffset = modelBoresightOffset
        self.modelCCDOffset = modelCCDOffset
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
        #
        # Check options
        #
        if not self.showGuideStarsAsPoints:
            if self.showByVisit:
                print("Disabling showByVisit")
                self.showByVisit = False

        self.guide_star_frac = min([self.guide_star_frac, 1])  # sanity check

    def selectStars(self, agcData, guidingErrors=None):
        sel = np.zeros(len(agcData.agc_camera_id), dtype=int)

        agc_exposure_ids = np.array(sorted(set(agcData.agc_exposure_id)))

        for aid in agc_exposure_ids[self.agc_exposure_id0::self.agc_exposure_idsStride]:
            if guidingErrors is not None and self.maxGuideError > 0:
                if guidingErrors[aid] > 1e-3*self.maxGuideError:
                    continue
            sel |= agcData.agc_exposure_id == aid

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

    def make_title(self, agcData):
        visits = sorted(set(agcData.pfs_visit_id))

        v0, v1 = visits[0], visits[-1]
        if self.pfs_visitIdMin > 0 and v0 < self.pfs_visitIdMin:
            v0 = self.pfs_visitIdMin
        if self.pfs_visitIdMax > 0 and v1 < self.pfs_visitIdMax:
            v1 = self.pfs_visitIdMax
        title = f"{v0}..{v1}"
        if self.modelBoresightOffset or self.modelCCDOffset:
            title += " "
            if self.modelBoresightOffset:
                title += " Boresight"
            if self.modelCCDOffset:
                title += " AGs"
            title += " offset and rotation/scale removed"
        if self.rotateToZenith:
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
        if self.showAverageGuideStarPos and \
           not (self.showGuideStarsAsArrows or self.showGuideStarsAsPoints):
            title += "  Mean guide error"

        return title


def showGuiderErrors(agcData, config,
                     verbose=False,
                     agc_camera_ids=range(6)):
    """
        agcData: pandas DataFrame as returned by readAgcDataFromOpdb
    """
    agc_exposure_ids = np.array(sorted(set(agcData.agc_exposure_id)))

    agc_nominal_x_mm = agcData.agc_nominal_x_mm.to_numpy().copy()
    agc_nominal_y_mm = agcData.agc_nominal_y_mm.to_numpy().copy()
    agc_center_x_mm = agcData.agc_center_x_mm.to_numpy().copy()
    agc_center_y_mm = agcData.agc_center_y_mm.to_numpy().copy()

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
    if config.modelBoresightOffset:
        dx = agc_center_x_mm - agc_nominal_x_mm
        dy = agc_center_y_mm - agc_nominal_y_mm
        dr = np.hypot(dx, dy)

        for aid in agc_exposure_ids:
            sel = agcData.agc_exposure_id == aid

            # sel &= dr < 200*1e-3  # remove egregious errors in centroids

            if aid in config.transforms:
                transform = config.transforms[aid]
            else:
                if verbose:
                    print(f"Solving for offsets/rotations for {aid}")
                transform = MeasureDistortion(agc_center_x_mm[sel], agc_center_y_mm[sel], -1,
                                              agc_nominal_x_mm[sel], agc_nominal_y_mm[sel], None)

                res = scipy.optimize.minimize(transform, transform.getArgs(), method='Powell')
                transform.setArgs(res.x)
                config.transforms[aid] = transform

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
        guidingErrors[aid] = np.mean(dr[sel][dr[sel] < 1e-3*config.maxPosError])
    #
    # Set the agc_nominal_[xy]_mm values for each star to have the same position as in the first exposure_id,
    # to take out dithers
    # N.b. only used for plotting
    #
    agc_nominal_x_mm0 = np.empty_like(agc_nominal_x_mm)
    agc_nominal_y_mm0 = np.empty_like(agc_nominal_y_mm)
    for gid in set(agcData.guide_star_id):
        sel = agcData.guide_star_id == gid
        agc_nominal_x_mm0[sel] = agc_nominal_x_mm[sel][0]
        agc_nominal_y_mm0[sel] = agc_nominal_y_mm[sel][0]

    config.agc_exposure_id0 = 0      # start with this agc_exposure_id (if config.agc_exposure_idsStride != 1)
    if config.agc_exposure_idsStride > 0 and config.maxGuideError > 0:
        for aid in agc_exposure_ids:
            if guidingErrors[aid] < 1e-3*config.maxGuideError:
                config.agc_exposure_id0 = np.where(agc_exposure_ids == aid)[0][0]
                break

    guideErrorByCamera = {}   # per-exposure mean guide error
    S = None                  # returned by scatter
    Q = None                  # and quiver
    plt.gca().set_prop_cycle(None)

    for agc_camera_id in agc_camera_ids:
        color = f"C{agc_camera_id}"
        AGlabel = f"AG{agc_camera_id + 1}"

        sel = config.selectStars(agcData, guidingErrors) & (agcData.agc_camera_id == agc_camera_id).to_numpy()

        if sum(sel) > 0:
            guide_star_ids = sorted(set(agcData.guide_star_id[sel]))
            np.random.seed(666)
            nguide_star = len(guide_star_ids)
            useGuideStars = np.random.choice(range(nguide_star),
                                             max([1, int(config.guide_star_frac*nguide_star)]), replace=False)

            if config.modelCCDOffset:
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

            if config.rotateToZenith:
                agc_nominal_x_mm[sel], agc_nominal_y_mm[sel] = \
                    rotXY(-insrot[sel], agc_nominal_x_mm[sel], agc_nominal_y_mm[sel])
                agc_center_x_mm[sel], agc_center_y_mm[sel] = \
                    rotXY(-insrot[sel], agc_center_x_mm[sel], agc_center_y_mm[sel])
                dx[sel], dy[sel] = rotXY(-insrot[sel], dx[sel], dy[sel])

            agc_nominal_x_mm0[sel] = agc_nominal_x_mm[sel][0]
            agc_nominal_y_mm0[sel] = agc_nominal_y_mm[sel][0]

            x, y = agc_center_x_mm[sel], agc_center_y_mm[sel]
            x /= config.pfiScaleReduction
            y /= config.pfiScaleReduction

            if config.rotateToZenith:
                xbar, ybar = np.empty_like(agc_center_x_mm), np.empty_like(agc_center_y_mm)

                for aid in set(agcData.agc_exposure_id[sel]):
                    la = sel & (agcData.agc_exposure_id.to_numpy() == aid)
                    xbar[la] = np.mean(agc_center_x_mm[la])/config.pfiScaleReduction
                    ybar[la] = np.mean(agc_center_y_mm[la])/config.pfiScaleReduction
                xbar, ybar = xbar[sel], ybar[sel]

                agc_ring_R = np.median(np.hypot(agc_nominal_x_mm, agc_nominal_y_mm))
                rbar = np.hypot(xbar, ybar)

                xbar *= agc_ring_R/rbar
                ybar *= agc_ring_R/rbar
            else:
                xbar = np.full_like(x, np.mean(x))
                ybar = np.full_like(xbar, np.mean(y))

            if config.rotateToZenith:
                plt.gca().add_patch(Circle((0, 0), agc_ring_R, fill=False, color="red"))
            else:
                plt.plot(xbar, ybar, '+',
                         color='red' if (config.showAverageGuideStarPos and
                                         not (config.showGuideStarsAsArrows or
                                              config.showGuideStarsAsPoints)) else 'black',
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

                if config.maxPosError > 0 and np.mean(np.hypot(dxg, dyg)) > 1e-3*config.maxPosError:
                    continue

                if config.showGuideStarPositions:
                    xg = xbar[gl] + (xg - xbar[gl])*config.gstarExpansion
                    yg = ybar[gl] + (yg - ybar[gl])*config.gstarExpansion
                else:
                    xg, yg = 0*xg + xbar[gl], 0*yg + ybar[gl]

                _agc_exposure_id = list(agcData.agc_exposure_id[sel][gl])
                for aid in agc_exposure_ids[::config.agc_exposure_idsStride]:
                    if aid in _agc_exposure_id:
                        if aid not in xgs:
                            xgs[aid], ygs[aid] = [], []

                        j = _agc_exposure_id.index(aid)
                        xgs[aid].append(xg[j] + 1e3*dxg[j])
                        ygs[aid].append(yg[j] + 1e3*dyg[j])

                if config.showGuideStars and i in useGuideStars:
                    label = None if labelled else AGlabel
                    xend, yend = xg + 1e3*dxg, yg + 1e3*dyg
                    if config.showGuideStarsAsPoints:
                        if config.showByVisit:
                            S = plt.scatter(xend, yend, s=config.showByVisitSize,
                                            alpha=config.showByVisitAlpha,
                                            vmin=agc_exposure_ids[0], vmax=agc_exposure_ids[-1],
                                            c=agcData.agc_exposure_id[sel][gl], cmap=config.agc_exposure_cm)
                        else:
                            plt.plot(xend, yend, '.', color=color, label=label, alpha=config.showByVisitAlpha,
                                     markersize=config.showByVisitSize)
                            labelled = True
                    elif config.showGuideStarsAsArrows:
                        Q = plt.quiver(xg, yg, xend - xg, yend - yg, alpha=0.5, color=color, label=label)
                        labelled = True
                    else:
                        pass   # useful code path if config.showAverageGuideStarPos is true

            xa = np.empty(len(xgs))
            ya = np.empty_like(xa)
            for i, aid in enumerate(xgs):
                xa[i] = np.mean(xgs[aid])
                ya[i] = np.mean(ygs[aid])

            guideErrorByCamera[agc_camera_id] = (list(xgs.keys()), xa, ya)

            if config.showAverageGuideStarPos:
                if config.showGuideStarsAsPoints or config.showGuideStarsAsArrows:
                    plt.plot(xa[0], ya[0], '.', color='black', zorder=-1)
                    plt.plot(xa, ya, '-', color='black', alpha=0.5, zorder=-1)
                else:
                    S = plt.scatter(xa, ya, s=config.showByVisitSize, alpha=config.showByVisitAlpha,
                                    vmin=agc_exposure_ids[0], vmax=agc_exposure_ids[-1],
                                    c=list(xgs.keys()), cmap=config.agc_exposure_cm)

    if S is not None:
        a = S.get_alpha()
        S.set_alpha(1)
        plt.colorbar(S).set_label("agc_exposure_id")
        S.set_alpha(a)

    if Q is not None:
        qlen = config.guideErrorEstimate   # microns
        plt.quiverkey(Q, 0.1, 0.9, qlen, f"{qlen} micron", color='black')

    if not config.rotateToZenith and (config.showByVisit or
                                      (config.showAverageGuideStarPos and not config.showGuideStarsAsArrows)):
        showAGCameraCartoon(showInstrot=True, showUp=True)
    elif labelled:
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
        lims = 350/config.pfiScaleReduction*np.array([-1, 1])
    plt.xlim(plt.ylim(np.min(lims), np.max(lims)))

    if Q is not None:
        plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    else:
        plt.xlabel(r"$\delta$x (microns)")
        plt.ylabel(r"$\delta$y (microns)")

    plt.suptitle(config.make_title(agcData))

    return guideErrorByCamera


def showGuiderErrorsByParams(agcData, guideErrorByCamera, params, config, figure=None,
                             plotScatter=True):
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

                assert len(set(aids)) == len(aids)

                subset = agcData[agcData.isin(dict(agc_exposure_id=aids)).agc_exposure_id]
                grouped = subset.groupby("agc_exposure_id")
                # Return what for each group -- there's only one value, but we still need to aggregate
                param = getattr(grouped, what).mean() # e.g. grouped.altitude.mean()

                xas.append(list(xa))
                yas.append(list(ya))
                params.append(list(param))

            xas = np.array(sum(xas, []))
            yas = np.array(sum(yas, []))
            params = np.array(sum(params, []))

            vmin, vmax = np.min(params), np.max(params)
            vmin, vmax = np.percentile(params, [10, 90])

            S = plt.scatter(xas, yas, s=config.showByVisitSize, alpha=config.showByVisitAlpha,
                            vmin=vmin, vmax=vmax, c=params, cmap=config.agc_exposure_cm)

            a = S.get_alpha()
            S.set_alpha(1)
            C = plt.colorbar(S, shrink=1/nx if ny == 1 else 1)
            C.set_label(what)
            S.set_alpha(a)

            if config.rotateToZenith:
                plt.gca().add_patch(Circle((0, 0), agc_ring_R, fill=False, color="red"))

            if True:
                lims = 350/config.pfiScaleReduction*np.array([-1, 1])
                plt.xlim(plt.ylim(np.min(lims), np.max(lims)))

            plt.gca().set_aspect(1)

            if not config.rotateToZenith:
                showAGCameraCartoon(showInstrot=True, showUp=False)

    fig.supxlabel(r"$\delta$x (microns)")
    fig.supylabel(r"$\delta$y (microns)")

    plt.suptitle(config.make_title(agcData), y=1.0)


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
