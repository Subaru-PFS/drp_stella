import warnings
import numpy as np
import scipy.optimize
import scipy.stats
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


def getAGCPositionsForVisitByAgcExposureId(opdb, pfs_visit_id, flipToHardwareCoords, agc_exposure_idStride=1):
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
                           figure=None,
                           showLegend=True):
    """
    agcData: pandas DataFrame as returned by readAgcDataFromOpdb
    byTime = False    # use time, not agc_exposure_id, as x-axis
    matplotlib figure to use, or None
    """
    fig, axs = plt.subplots(3, 1, num=figure, sharex=True, sharey=False, squeeze=False)
    axs = axs.flatten()

    agc_exposure_ids = np.array(sorted(set(agcData.agc_exposure_id)))

    subset = agcData[agcData.isin(dict(agc_exposure_id=agc_exposure_ids)).agc_exposure_id]
    grouped = subset.groupby("agc_exposure_id")

    pfs_visit_ids = grouped.pfs_visit_id.min()
    taken_ats = grouped.taken_at.mean()
    shutter_open = grouped.shutter_open.max()

    tmp = pd.DataFrame(dict(agc_exposure_id=agcData.agc_exposure_id,
                            dx=agcData.agc_nominal_x_mm - agcData.agc_center_x_mm,
                            dy=agcData.agc_nominal_y_mm - agcData.agc_center_y_mm))
    grouped = tmp.groupby("agc_exposure_id")
    xbar, ybar = grouped.agg(xbar=('dx', 'mean')), grouped.agg(ybar=('dy', 'mean')).to_numpy()

    def plot_zbar(zbar, xvec=agc_exposure_ids):
        plt.gca().set_prop_cycle(None)
        for pfs_visit_id in sorted(set(pfs_visit_ids)):
            sel = pfs_visit_ids == pfs_visit_id
            color = plt.plot(xvec[sel], zbar[sel], '.-', label=f"{int(pfs_visit_id)}")[0].get_color()
            sel &= shutter_open > 0
            plt.plot(xvec[sel], zbar[sel], 'o', color=color)

        plt.axhline(0, color='black')
        if showLegend:
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
                 agc_exposure_cm=plt.matplotlib.cm.get_cmap("viridis"),
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
    insrot in radians"""
    c, s = np.cos(angle), np.sin(angle)
    x, y = c*x - s*y, s*x + c*y

    return x, y


def showGuiderErrors(agcData, config,
                     verbose=False,
                     agc_camera_ids=range(6),
                     name=None):
    """
    agcData: pandas DataFrame as returned by readAgcDataFromOpdb
    config: a GuiderConfig
    agc_camera_ids:  a list of 0-indexed cameras to display; default 0..5
    name:  a string to show in the header; or None
    """
    agc_exposure_ids = np.sort(agcData.agc_exposure_id.unique())

    if "agc_nominal_x_mm0" in agcData:  # restore original values from the opdb
        agcData.agc_nominal_x_mm = agcData.agc_nominal_x_mm0
        agcData.agc_nominal_y_mm = agcData.agc_nominal_y_mm0
    else:
        assert "agc_nominal_y_mm0" not in agcData

        agcData["agc_nominal_x_mm0"] = agcData.agc_nominal_x_mm
        agcData["agc_nominal_y_mm0"] = agcData.agc_nominal_y_mm
    #
    # Fix the mean guiding offset for each exposure?
    #
    if config.modelBoresightOffset:
        for aid in agc_exposure_ids:
            sel = agcData.agc_exposure_id == aid

            if aid in config.transforms:
                transform = config.transforms[aid]
            else:
                if verbose:
                    print(f"Solving for offsets/rotations for {aid}")
                transform = MeasureDistortion(agcData.agc_center_x_mm[sel], agcData.agc_center_y_mm[sel], -1,
                                              agcData.agc_nominal_x_mm[sel], agcData.agc_nominal_y_mm[sel],
                                              None, nsigma=3)

                res = scipy.optimize.minimize(transform, transform.getArgs(), method='Powell')
                transform.setArgs(res.x)
                config.transforms[aid] = transform

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
            if (config.maxGuideError > 0 and float(guidingErrors.loc[aid]) > 1e-3*config.maxGuideError):
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

        if config.modelCCDOffset:
            agcDataCam = agcData[sel].copy()   # pandas doesn't guarantee a copy or a view, so be specific

            if agc_camera_id in config.transforms and not config.solveForAGTransforms:
                transform = config.transforms[agc_camera_id]
            else:
                #
                # Solve for a per-CCD offset and rotation/scale
                #
                print(f"Solving for offsets/rotations for AG{agc_camera_id + 1}")

                transform = MeasureDistortion(agcDataCam.agc_center_x_mm, agcDataCam.agc_center_y_mm, -1,
                                              agcDataCam.agc_nominal_x_mm, agcDataCam.agc_nominal_y_mm, None)
                config.transforms[agc_camera_id] = transform

                res = scipy.optimize.minimize(transform, transform.getArgs(), method='Powell')
                transform.setArgs(res.x)

            agcDataCam.agc_nominal_x_mm, agcDataCam.agc_nominal_y_mm = \
                transform.distort(agcDataCam.agc_nominal_x_mm, agcDataCam.agc_nominal_y_mm, inverse=True)

            agcDataCam.dx = agcDataCam.agc_center_x_mm - agcDataCam.agc_nominal_x_mm
            agcDataCam.dy = agcDataCam.agc_center_y_mm - agcDataCam.agc_nominal_y_mm
            agcDataCam.dr = np.hypot(agcDataCam.dy, agcDataCam.dy)

            agcData.loc[sel] = agcDataCam

    sel = config.selectStars(agcData, guidingErrors=guidingErrors)
    tmp = agcData[sel]
    if config.maxPosError > 0:
        grouped = tmp.groupby(["guide_star_id"])
        _tmp = grouped.agg(
            drbar=("dr", "mean"),
        )
        tmp = tmp.merge(_tmp, on="guide_star_id")
        tmp = tmp[tmp.drbar > 1e-3*config.maxPosError]

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
    plt.gca().set_prop_cycle(None)

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
            plt.gca().add_patch(Circle((0, 0), agc_ring_R, fill=False, color="red"))
        else:
            plt.plot(plotData.xbar, plotData.ybar, '+',
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
                    S = plt.scatter(xend, yend, s=config.showByVisitSize,
                                    alpha=config.showByVisitAlpha,
                                    vmin=np.min(agcData.agc_exposure_id),
                                    vmax=np.max(agcData.agc_exposure_id),
                                    c=plotData.agc_exposure_id, cmap=config.agc_exposure_cm)
                else:
                    plt.plot(xend, yend, '.', color=color, label=label, alpha=config.showByVisitAlpha,
                             markersize=config.showByVisitSize)
                    labelled = True
            elif config.showGuideStarsAsArrows:
                Q = plt.quiver(xg, yg, xend - xg, yend - yg, alpha=0.5, color=color, label=label)
                labelled = True
            else:
                pass   # useful code path if config.showAverageGuideStarPos is true

        if config.showAverageGuideStarPos or config.showAverageGuideStarPath:
            tmp = guideErrorByCamera[guideErrorByCamera.agc_camera_id == agc_camera_id]
            xa = np.mean(tmp.agc_nominal_x_mm)/config.pfiScaleReduction + 1e3*tmp.dx
            ya = np.mean(tmp.agc_nominal_y_mm)/config.pfiScaleReduction + 1e3*tmp.dy

            if config.showAverageGuideStarPath:
                plt.plot(xa.iloc[0], ya.iloc[0], '.', color='black', zorder=-10)
                plt.plot(xa, ya, '-', color='black', alpha=0.25, zorder=-10)

            if config.showAverageGuideStarPos:
                S = plt.scatter(xa, ya, s=config.showByVisitSize, alpha=config.showByVisitAlpha,
                                vmin=tmp.agc_exposure_id.min(), vmax=tmp.agc_exposure_id.max(),
                                c=tmp.agc_exposure_id, cmap=config.agc_exposure_cm)

    if S is not None:
        a = S.get_alpha()
        S.set_alpha(1)
        plt.colorbar(S).set_label("agc_exposure_id")
        S.set_alpha(a)

    if Q is not None:
        qlen = config.guideErrorEstimate   # microns
        plt.quiverkey(Q, 0.1, 0.9, qlen, f"{qlen} micron", color='black')

    if not config.showGuideStarsAsArrows:
        showAGCameraCartoon(showInstrot=True, showUp=config.rotateToAG1Down)
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

    plt.suptitle(config.make_title(agcData, name))

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

        if True:
            lims = 350/config.pfiScaleReduction*np.array([-1, 1])
            plt.xlim(plt.ylim(np.min(lims), np.max(lims)))

        plt.gca().set_aspect(1)

        if not config.rotateToAG1Down:
            showAGCameraCartoon(showInstrot=True, showUp=False)

    fig.supxlabel(r"$\delta$x (microns)")
    fig.supylabel(r"$\delta$y (microns)")

    plt.suptitle(config.make_title(agcData, name), y=1.0)


def showTelescopeErrors(agcData, config, showTheta=False, figure=None):
    """
    """
    agc_exposure_ids = set(agcData.agc_exposure_id)

    agc_ring_R = np.mean(np.hypot(*np.array(list(agcCameraCenters.values())).T))  # Approx. radius of AGs

    diff = agc_exposure_ids.difference(config.transforms)
    if len(diff) > 0:
        print(f"Warning: {diff} not present in config.transforms; ignoring")
        agc_exposure_ids = agc_exposure_ids.intersection(config.transforms)

    agc_exposure_ids = np.array(sorted(agc_exposure_ids))

    # unpack the first three values from config.transforms for our agc_exposure_ids
    dx, dy, theta = np.array([config.transforms[aid].getArgs() for aid in agc_exposure_ids])[:, :3].T

    subset = agcData[agcData.isin(dict(agc_exposure_id=agc_exposure_ids)).agc_exposure_id]
    grouped = subset.groupby("agc_exposure_id")
    altitude = grouped.altitude.mean()
    azimuth = grouped.azimuth.mean()
    shutter_open = grouped.shutter_open.max()

    sel = shutter_open.to_numpy() > 0

    nx, ny = 2, 2
    fig, axs = plt.subplots(nx, ny, num=figure, sharex=False, sharey=False, squeeze=False)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        plt.sca(ax)

        shrink = 0.45 if ny == 1 else 1
        if i == 0:
            S = plt.hexbin(1e3*dx[sel], 1e3*dy[sel], gridsize=50)
            plt.colorbar(S, shrink=shrink)
            plt.gca().set_aspect(1)
            plt.xlabel(r"$\delta$x (microns)")
            plt.ylabel(r"$\delta$y (microns)")
        elif i == 1:
            S = plt.scatter(1e3*dx[sel], 1e3*dy[sel], s=10, c=agc_exposure_ids[sel])
            plt.colorbar(S, shrink=shrink).set_label("agc_exposure_id")
            ax.set_facecolor('black')
            plt.gca().set_aspect(1)
            plt.xlabel(r"$\delta$x (microns)")
            plt.ylabel(r"$\delta$y (microns)")

            ax.get_shared_x_axes().join(ax, axs[0])
            ax.get_shared_y_axes().join(ax, axs[0])
        elif i == 2:
            if showTheta:
                yvec, ylabel = 3600*theta, r"$\theta$ (arcsec)"
                ylim = 40*np.array([-1, 1])
            else:
                yvec, ylabel = agc_ring_R*1e4*np.deg2rad(theta), f"guide error @{agc_ring_R}cm (microns)"
                ylim = 40*np.array([-1, 1])
            S = plt.scatter(agc_exposure_ids[sel], yvec[sel], c=altitude[sel])
            plt.colorbar(S).set_label("altitude")

            plt.ylim(ylim)
            plt.xlabel("agc_exposure_id")
            plt.ylabel(ylabel)
        elif i == 3:
            vmin, vmax = 40*np.array([-1, 1])
            S = plt.scatter(azimuth[sel], altitude[sel], c=3600*theta[sel], s=5, vmin=vmin, vmax=vmax)
            plt.colorbar(S).set_label(r"$\theta$ (arcsec)")

            plt.xlabel("azimuth")
            plt.ylabel("altitude")

    title = ""
    title += f"({1e3*np.mean(dx):.1f}, {1e3*np.mean(dy):.1f}) microns   {3600*np.mean(theta):.1f} arcsec"
    plt.suptitle(title)


def plotDriftRate(agcData, agc_exposure_ids=None,
                  robust=True, subtractMeanOffset=True, byTime=True, showCamera=True,
                  figure=None):
    """Fit and plot the guide cameras drifts, based on the values in agcData
    Only use the agc_exposures in agc_exposure_ids (if provided)

    agcData : as updated by showGuiderErrors, with dx/dy columns
    agc_exposure_ids : only show these agc_exposure_ids; or None
    robust: `bool` use scipy.stats.siegelslopes's robust line-fitter
    subtractMeanOffset: `bool` Subtract mean offset for each camera
    byTime:  `bool` plot against time, not agc_exposure_id
    showCamera: `bool` colour points by agc_camera_id
    figure: `matplotlib.Figure` reuse this figure

    """
    if agc_exposure_ids is not None:
        agcData = agcData[agcData.isin(dict(agc_exposure_id=agc_exposure_ids)).agc_exposure_id]

        if len(agcData) == 0:
            raise RuntimeError("No relevant agc_exposures are available in the agcData object")

    agcData = agcData[agcData.shutter_open == 1]

    grouped = agcData.groupby(["agc_exposure_id", "agc_camera_id"], as_index=False)
    tmp = grouped.agg(
        taken_at=("taken_at", "first"),
        agc_nominal_x_mm=("agc_nominal_x_mm", "mean"),
        agc_nominal_y_mm=("agc_nominal_y_mm", "mean"),
        dx=("dx", "mean"),
        dy=("dy", "mean"),
    )

    if subtractMeanOffset:
        grouped = tmp.groupby("agc_camera_id")
        ntmp = grouped.agg(
            dxbar=("dx", "mean"),
            dybar=("dy", "mean"),
        )
        tmp = ntmp.merge(tmp, on="agc_camera_id")
        tmp.dx -= tmp.dxbar
        tmp.dy -= tmp.dybar

    tmp.sort_values("taken_at", inplace=True)

    dx, dy = tmp.dx, tmp.dy
    theta = np.arctan2(tmp.agc_nominal_y_mm, tmp.agc_nominal_x_mm)

    c, s = np.cos(theta), np.sin(theta)
    dradial, dtangential = c*dx + s*dy, -s*dx + c*dy
    time = (tmp.taken_at - tmp.taken_at.iloc[0]).dt.total_seconds().to_numpy()
    #
    # OK!  time to do the fit
    #

    def func(x, a, b):
        return a + b*x

    if robust:
        br, ar = scipy.stats.siegelslopes(dradial, time)  # offset and rate;  microns and microns/sec
        bt, at = scipy.stats.siegelslopes(dtangential, time)  # offset and rate;  microns and microns/sec
    else:
        ar, br = scipy.optimize.curve_fit(func, time, dradial)[0]
        at, bt = scipy.optimize.curve_fit(func, time, dtangential)[0]
    #
    # And make the plots
    #
    fig, axs = plt.subplots(2, 1, num=figure, sharex=True, sharey=True, squeeze=False)
    axs = axs.flatten()

    for ax, vec, xy in zip(axs, [1e3*dradial, 1e3*dtangential], ["radial", "tangential"]):
        plt.sca(ax)
        if showCamera:
            for agc_camera_id in range(6):
                ll = tmp.agc_camera_id == agc_camera_id

                kwargs = dict(color=f"C{agc_camera_id}", label=f"{agc_camera_id + 1}")
                if byTime:
                    plt.plot(time[ll], vec[ll], 'o', **kwargs)
                else:
                    plt.plot(tmp.agc_exposure_id[ll], vec[ll], 'o', **kwargs)
            plt.legend(ncol=3, loc='upper center')
        else:
            if byTime:
                plt.plot(time, vec, 'o')
            else:
                plt.plot(tmp.agc_exposure_id, vec, 'o')

        plt.ylabel(xy)

        a, b = (ar, br) if xy == "radial" else (at, bt)
        tt = np.array([time[0], time[-1]])
        xx = tt if byTime else np.array([tmp.agc_exposure_id.iloc[0], tmp.agc_exposure_id.iloc[-1]])
        plt.plot(xx, 1e3*(a + b*tt), color='black')

        plt.text(0.05, 0.9, f"Rate: {60*1e3*b:.3f} microns/min", transform=ax.transAxes)

    visits = sorted(set(agcData.pfs_visit_id))

    fig.supxlabel("time (s)" if byTime else "agc_exposure_id")
    fig.supylabel("drift of nominal position during visit (microns)")
    plt.suptitle(f"visits {visits[0]}..{visits[-1]}")

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#  New pandas-native routines.  Should eventually replace the above


def estimateGuideErrors(agcData, guideStrategy="center0", byVisit=False, subtractMedian=False,
                        showClosedShutter=False, xy0Stat="median"):
    """Estimate the mean guide errors for each AG camera and exposure
    Only exposures with the spectrograph shutters open are considered

    guideStrategy: how to define the correct position of the guide stars
        center0:          Use the average of the first visit's agc_center_[xy]_mm (see xy0Stat)
        nominal:          Use agc_nominal_[xy]_mm
        nominal0:         Use agc_nominal_[xy]_mm at start of sequence
        nominal0PerVisit: Use agc_nominal_[xy]_mm at start of each visit
    byVisit: group on pfs_visit_id rather than agc_exposure_id
    subtractMedian:       Subtract the median from each camera's guide errors
    showClosedShutter:    Include data with the spectrograph shutter closed
    xy0Stat: The name of the statistic to use for center0 (must be supported by pd.DataFrame.agg,
             and also in validXy0StatStrategies list)

    Returns:
       agcData:  As updated, with added columns
                 agc_center_[xy]_mm0          Values of agc_center_[xy]_mm at start of sequence
                 agc_nominal_[xy]_mm0         Values of agc_nominal_[xy]_mm at start of sequence
                 agc_nominal_[xy]y_mm_visit0  Values of agc_nominal_[xy]_mm at start of each visit
       agcGuideErrors: pandas DataFrame with desired errors
    """

    validXy0StatStrategies = ["mean", "median", "first", "last"]
    if xy0Stat not in validXy0StatStrategies:
        raise RuntimeError(f"Unknown xy0Stat {xy0Stat}"
                           f" (valid: {', '.join(validXy0StatStrategies)})")

    validGuideStrategies = ["center0", "nominal", "nominal0PerVisit", "nominal0"]
    if guideStrategy not in validGuideStrategies:
        raise RuntimeError(f"Unknown guideStrategy {guideStrategy}"
                           f" (valid: {', '.join(validGuideStrategies)})")

    shutterMin = -1 if showClosedShutter else 0

    grouped = agcData.groupby(["guide_star_id"], as_index=False)
    agcData = grouped.agg(
        agc_nominal_x_mm0=pd.NamedAgg("agc_nominal_x_mm", xy0Stat),
        agc_nominal_y_mm0=pd.NamedAgg("agc_nominal_y_mm", xy0Stat),
    ).merge(agcData, on=["guide_star_id"])

    grouped = agcData[agcData.shutter_open > shutterMin].groupby(["pfs_visit_id", "guide_star_id"],
                                                                 as_index=False)
    agcData = grouped.agg(
        agc_center_x_mm0=pd.NamedAgg("agc_center_x_mm", xy0Stat),
        agc_center_y_mm0=pd.NamedAgg("agc_center_y_mm", xy0Stat),
        agc_nominal_x_mm_visit0=pd.NamedAgg("agc_nominal_x_mm", xy0Stat),
        agc_nominal_y_mm_visit0=pd.NamedAgg("agc_nominal_y_mm", xy0Stat),
    ).merge(agcData, on=["pfs_visit_id", "guide_star_id"])

    if guideStrategy == "center0":
        agcData["dx"] = agcData.agc_center_x_mm0 - agcData.agc_center_x_mm
        agcData["dy"] = agcData.agc_center_y_mm0 - agcData.agc_center_y_mm
    elif guideStrategy == "nominal":
        agcData["dx"] = agcData.agc_nominal_x_mm - agcData.agc_center_x_mm
        agcData["dy"] = agcData.agc_nominal_y_mm - agcData.agc_center_y_mm
    elif guideStrategy == "nominal0PerVisit":
        agcData["dx"] = agcData.agc_nominal_x_mm_visit0 - agcData.agc_center_x_mm
        agcData["dy"] = agcData.agc_nominal_y_mm_visit0 - agcData.agc_center_y_mm
    elif guideStrategy == "nominal0":
        agcData["dx"] = agcData.agc_nominal_x_mm0 - agcData.agc_center_x_mm
        agcData["dy"] = agcData.agc_nominal_y_mm0 - agcData.agc_center_y_mm
    else:
        raise RuntimeError("You can't get here; complain to RHL")

    grouped = agcData[agcData.shutter_open > shutterMin].groupby(
        ["pfs_visit_id" if byVisit else "agc_exposure_id" , "agc_camera_id"], as_index=False)
    agcGuideErrors = grouped.agg(
        agc_exposure_id=pd.NamedAgg("agc_exposure_id", "mean"),
        pfs_visit_id=pd.NamedAgg("pfs_visit_id", "mean"),
        agc_nominal_x_mm0=pd.NamedAgg("agc_nominal_x_mm", "mean"),
        agc_nominal_y_mm0=pd.NamedAgg("agc_nominal_y_mm", "mean"),
        altitude=pd.NamedAgg("altitude", "mean"),
        azimuth=pd.NamedAgg("azimuth", "mean"),
        insrot=pd.NamedAgg("insrot", "mean"),
        guide_delta_altitude=pd.NamedAgg("guide_delta_altitude", "mean"),
        guide_delta_azimuth=pd.NamedAgg("guide_delta_azimuth", "mean"),
        guide_delta_insrot=pd.NamedAgg("guide_delta_insrot", "mean"),
        xbar=pd.NamedAgg("dx", "mean"),
        ybar=pd.NamedAgg("dy", "mean"),
        shutter_open=pd.NamedAgg("shutter_open", "max")
    )

    if subtractMedian:
        for agc_camera_id in sorted(set(agcGuideErrors.agc_camera_id)):
            sel = (agcGuideErrors.agc_camera_id == agc_camera_id) & (agcGuideErrors.shutter_open > shutterMin)
            agcGuideErrors.xbar = np.where(sel, agcGuideErrors.xbar - np.median(agcGuideErrors.xbar[sel]),
                                           agcGuideErrors.xbar)
            agcGuideErrors.ybar = np.where(sel, agcGuideErrors.ybar - np.median(agcGuideErrors.ybar[sel]),
                                           agcGuideErrors.ybar)

    return agcData, agcGuideErrors


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
        ax.text(0, 5, r"0$^\circ$", ha="center", va="center", size=7)
        for i in [-1, 1]:
            ax.text(-260, i*150, r'$\uparrow$', va='center', rotation=90)


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# OLD! OLD!

def _showGuiderErrors(agcData, config,
                      verbose=False,
                      agc_camera_ids=range(6),
                      name=None):
    """
    agcData: pandas DataFrame as returned by readAgcDataFromOpdb
    config: a GuiderConfig
    agc_camera_ids:  a list of 0-indexed cameras to display; default 0..5
    name:  a string to show in the header; or None
    """
    agc_exposure_ids = np.array(sorted(set(agcData.agc_exposure_id)))

    agc_nominal_x_mm = agcData.agc_nominal_x_mm.to_numpy().copy()
    agc_nominal_y_mm = agcData.agc_nominal_y_mm.to_numpy().copy()
    agc_center_x_mm = agcData.agc_center_x_mm.to_numpy().copy()
    agc_center_y_mm = agcData.agc_center_y_mm.to_numpy().copy()

    insrot = np.deg2rad(agcData.insrot).to_numpy()
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
                                              agc_nominal_x_mm[sel], agc_nominal_y_mm[sel], None, nsigma=3)

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
    tmp = pd.DataFrame(dict(agc_exposure_id=agcData.agc_exposure_id, dr=dr))
    grouped = tmp.groupby("agc_exposure_id")
    guidingErrors = grouped.agg(guidingErrors=('dr', 'mean'))
    guidingErrors = guidingErrors.to_dict()['guidingErrors']
    #
    config.agc_exposure_id0 = 0      # start with this agc_exposure_id (if config.agc_exposure_idsStride != 1)
    if config.agc_exposure_idsStride > 0 and (config.onlyShutterOpen or config.maxGuideError > 0):
        for aid in agc_exposure_ids:
            if (config.onlyShutterOpen and (agcData.shutter_open == 0)[agcData.agc_exposure_id == aid].any()):
                continue
            if (config.maxGuideError > 0 and guidingErrors[aid] > 1e-3*config.maxGuideError):
                continue

            config.agc_exposure_id0 = np.where(agc_exposure_ids == aid)[0][0]
            break

    guideErrorByCamera = {}   # per-exposure mean guide error
    S = None                  # returned by scatter
    Q = None                  # and quiver
    plt.gca().set_prop_cycle(None)

    for agc_camera_id in agc_camera_ids:
        color = f"C{agc_camera_id}"
        AGlabel = f"AG{agc_camera_id + 1}"

        sel = config.selectStars(agcData, agc_camera_id, guidingErrors)

        labelled = False
        if sum(sel) > 0:
            guide_star_ids = sorted(set(agcData.guide_star_id[sel]))
            np.random.seed(666)
            nguide_star = len(guide_star_ids)
            useGuideStars = np.random.choice(range(nguide_star),
                                             max([1, int(config.guide_star_frac*nguide_star)]), replace=False)

            if config.modelCCDOffset and \
               (config.solveForAGTransforms or agc_camera_id not in config.transforms):
                #
                # Solve for a per-CCD offset and rotation/scale
                #
                print(f"Solving for offsets/rotations for {AGlabel}")
                transform = MeasureDistortion(agc_center_x_mm[sel], agc_center_y_mm[sel], -1,
                                              agc_nominal_x_mm[sel], agc_nominal_y_mm[sel], None)
                config.transforms[agc_camera_id] = transform

                res = scipy.optimize.minimize(transform, transform.getArgs(), method='Powell')
                transform.setArgs(res.x)

                agc_nominal_x_mm[sel], agc_nominal_y_mm[sel] = \
                    transform.distort(agc_nominal_x_mm[sel], agc_nominal_y_mm[sel], inverse=True)

                dx[sel] = (agc_center_x_mm - agc_nominal_x_mm)[sel]
                dy[sel] = (agc_center_y_mm - agc_nominal_y_mm)[sel]

            if False:
                print(f"{AGlabel}  ({np.mean(1e3*dx[sel]):4.1f}, {np.mean(1e3*dy[sel]):4.1f}) +- "
                      f"({stdFromIQR(1e3*dx[sel]):4.1f}, {stdFromIQR(1e3*dy[sel]):4.1f})")

            if config.rotateToAG1Down:
                agc_nominal_x_mm[sel], agc_nominal_y_mm[sel] = \
                    rotXY(-insrot[sel], agc_nominal_x_mm[sel], agc_nominal_y_mm[sel])
                agc_center_x_mm[sel], agc_center_y_mm[sel] = \
                    rotXY(-insrot[sel], agc_center_x_mm[sel], agc_center_y_mm[sel])
                dx[sel], dy[sel] = rotXY(-insrot[sel], dx[sel], dy[sel])

            x, y = agc_center_x_mm[sel], agc_center_y_mm[sel]
            x /= config.pfiScaleReduction
            y /= config.pfiScaleReduction

            if config.rotateToAG1Down:
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
            #
            # OK, ready to plot
            #
            if config.rotateToAG1Down:
                plt.gca().add_patch(Circle((0, 0), agc_ring_R, fill=False, color="red"))
            else:
                plt.plot(xbar, ybar, '+',
                         color='red' if
                         config.showByVisit or (config.showAverageGuideStarPos and
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

            guideErrorByCamera[agc_camera_id] = (np.array(list(xgs.keys())), xa, ya)

            if config.showAverageGuideStarPos:
                if config.showGuideStarsAsPoints or config.showGuideStarsAsArrows:
                    plt.plot(xa[0], ya[0], '.', color='black', zorder=-1)
                    plt.plot(xa, ya, '-', color='black', alpha=0.5, zorder=-1)
                else:
                    agc_exp_ids = np.array(list(xgs.keys()))
                    sid = np.argsort(agc_exp_ids)
                    S = plt.scatter(xa[sid], ya[sid], s=config.showByVisitSize, alpha=config.showByVisitAlpha,
                                    vmin=agc_exp_ids.min(), vmax=agc_exp_ids.max(),
                                    c=agc_exp_ids[sid], cmap=config.agc_exposure_cm)

    if S is not None:
        a = S.get_alpha()
        S.set_alpha(1)
        plt.colorbar(S).set_label("agc_exposure_id")
        S.set_alpha(a)

    if Q is not None:
        qlen = config.guideErrorEstimate   # microns
        plt.quiverkey(Q, 0.1, 0.9, qlen, f"{qlen} micron", color='black')

    if not config.rotateToAG1Down and \
       (config.showByVisit or (config.showAverageGuideStarPos and not config.showGuideStarsAsArrows)):
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

    plt.suptitle(config.make_title(agcData, name))

    return guideErrorByCamera

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
