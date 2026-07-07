import io
import re
import sys
import importlib
import numpy as np
import pandas as pd
import psycopg2
import warnings

import astropy.io.fits
import lsst.afw.fits

from pfs.datamodel.utils import astropyHeaderFromDict
from pfs.utils.coordinates.transform import makePfiTransform
from pfs.datamodel import FiberStatus, TargetType

__all__ = ["headerToMetadata", "metadataToHeader", "getPfsVersions", "processConfigListFromCmdLine",
           "pd_read_sql", "makePfiTransformFromOpdb", "getHomeVisits", "getTimeForVisits", "calculateNxNy"]


def pd_read_sql(sql_query: str, db_conn: psycopg2.extensions.connection,
                showQuery: bool = False) -> pd.DataFrame:
    """Execute SQL Query and get Dataframe with pandas

    Works around (harmless but annoying) pandas warning telling me to use sqlalchemy to access postgres
    """

    if showQuery:
        print(sql_query)

    with warnings.catch_warnings():
        # ignore warning for non-SQLAlchemy Connecton
        # see github.com/pandas-dev/pandas/issues/45660
        warnings.simplefilter('ignore', UserWarning)
        # create pandas DataFrame from database query
        df = pd.read_sql_query(sql_query, db_conn)
    return df


def getPfsVersions(prefix="VERSION_"):
    """Retrieve the software versions of PFS DRP-2D software

    The version values come from the ``<module>.version.__version__``
    attribute, which gets set at build time.

    The versions are in a format suitable for inclusion in a FITS header.

    Parameters
    ----------
    prefix : `str`, optional
        Prefix to add to software product name, to distinguish in the FITS
        header.

    Returns
    -------
    versions : `dict` (`str`: `str`)
        Versions of software products.
    """
    versions = {}
    for name, module in (("datamodel", "pfs.datamodel"),
                         ("obs_pfs", "lsst.obs.pfs"),
                         ("drp_stella", "pfs.drp.stella"),
                         ):
        importlib.import_module(module + ".version")
        key = (prefix + name).upper()
        if len(key) > 8:
            key = "HIERARCH " + key
        versions[key] = sys.modules[module + ".version"].__version__
    return versions


def headerToMetadata(header):
    """Convert FITS header to LSST metadata

    Parameters
    ----------
    header : `dict` or `astropy.io.fits.Header`
        FITS header.

    Returns
    -------
    metadata : `lsst.daf.base.PropertyList`
        LSST metadata object.
    """
    if isinstance(header, dict):
        header = astropyHeaderFromDict(header)
    # Read the primary header with lsst.afw.fits
    # This requires writing the FITS file into memory and reading it from there
    fits = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(header=header)])
    buffer = io.BytesIO()
    fits.writeto(buffer)
    ss = buffer.getvalue()
    size = len(ss)
    ff = lsst.afw.fits.MemFileManager(size)
    ff.setData(ss, size)
    return ff.readMetadata(0)


def metadataToHeader(metadata):
    """Convert LSST metadata to FITS header dict

    Parameters
    ----------
    metadata : `lsst.daf.base.PropertyList`
        LSST metadata object.

    Returns
    -------
    header : `dict`
        FITS header.
    """
    header = {}
    for key in metadata.names():
        if len(key) > 8:
            key = "HIERARCH " + key
        header[key] = metadata.get(key)
    return header


def processConfigListFromCmdLine(cmdLineString):
    """Handle setting lists of strings on the command line, converting a string to a list

    E.g. "[AA, 'BB', CC]" -> ["AA", "BB", "CC"]  ("BB" is also supported)

    Parameters
    ----------
    cmdLineString : `str`
       The string to process

    Returns
    -------
        The list if it fits the pattern, or the initial string otherwise
    """
    if cmdLineString and \
       cmdLineString[0] == '[' and cmdLineString[-1] == ']':  # command line string line [A, B]
        cmdLineString = "".join(cmdLineString[1:-1])
        what = []
        for el in cmdLineString.split(','):
            mat = re.match(r"\s*[\"'](.*)[\"']\s*$", el)
            if mat:
                el = el.group(1)
            what.append(el)

        return what
    else:
        return cmdLineString


def frameId(visitId, subVisitId=0):
    """Return a frameId given an visitId and optionally a subVisitId"""
    return "%06d%02d" % (visitId, subVisitId)


def makePfiTransformFromOpdb(opdb, visitId, subVisitId=0):
    """Make a PfiTransform object for the given frameId by reading the opdb

    opdb: `psycopg2.extensions.connection`
       Connection to opdb
    visitId: `int`
       Desired visit
    subVisitId: `int`
       Desired subvisit (default: 0)
    """
    with opdb:
        tmp = pd_read_sql(f'''
            SELECT altitude, insrot
            FROM mcs_exposure
            WHERE mcs_frame_id = {frameId(visitId, subVisitId)};
        ''', opdb)
    altitude, insrot = tmp.iloc[0]

    with opdb:
        tmp = pd_read_sql(f"""
        SELECT
           *
        FROM
           mcs_pfi_transformation
        WHERE
           mcs_frame_id = {frameId(visitId, subVisitId)}
        """, opdb)

    mcs_frame_id, x0, y0, dscale, scale2, theta, alpha_rot, camera_name = tmp.iloc[0]

    mpt = makePfiTransform(camera_name, altitude=altitude, insrot=insrot)

    mpt.mcsDistort.setArgs([x0, y0, theta, dscale, scale2])

    return mpt


def getHomeVisits(opdb, dateStart=None, dateEnd=None, arm=None, pfsVisits=None,
                  returnDataFrame=False, limit=None):
    """Return an array of fiber trace home visits (using the r arm) with insrot == 0

    opdb:
       connection to opdb
    dateStart `str`:
       Only return visits taken on or after dateStart (default: None; no constraint)
    dateEnd `str`:
       Only return visits taken before dateStart (default: None; no constraint)
    pfsVisits: `list` of `int`
       Only return visits included in this list (default: None; no constraint)
    returnDataFrame: `bool`
       Return the pd.DataFrame if True else pfs_visit_id as a numpy array (default: False)
    limit `int`
       Return at most limit visits
    """

    LIMIT = "" if limit is None else f"LIMIT {limit}"

    where = []
    if dateStart is not None:
        where.append(f"time_exp_start  AT TIME ZONE 'UTC' AT TIME ZONE 'HST' >= '{dateStart}'")
    if dateEnd is not None:
        where.append(f"time_exp_start  AT TIME ZONE 'UTC' AT TIME ZONE 'HST' < '{dateEnd}'")
    if arm is not None:
        where.append(f"sps_camera.arm = '{arm}'")
    if pfsVisits is not None:
        where.append(f"sps_exposure.pfs_visit_id IN ({', '.join(str(_) for _ in pfsVisits)})")

    if False:
        with opdb:
            tmp = pd_read_sql(f'''
            SELECT DISTINCT
               sps_exposure.pfs_visit_id
               ,(SELECT DISTINCT
                   count (*) -- FILTER (WHERE pcf.is_on_source IS TRUE)
                FROM pfs_config AS pc
                JOIN pfs_config_sps AS pcs ON pcs.visit0 = pc.visit0
                JOIN pfs_config_fiber AS pcf ON pcf.pfs_design_id = pc.pfs_design_id AND
                                                pcf.visit0 = pc.visit0
                JOIN pfs_design_fiber AS pdf ON pdf.pfs_design_id = pcf.pfs_design_id AND
                                                pdf.fiber_id = pcf.fiber_id
                WHERE
                   pcs.pfs_visit_id = sps_exposure.pfs_visit_id AND
                   pcf.fiber_status = {int(FiberStatus.GOOD)} AND
                   pdf.target_type != {int(TargetType.ENGINEERING)}
                ) as n_fiber
            FROM pfs_config
            JOIN pfs_design ON pfs_design.pfs_design_id = pfs_config.pfs_design_id
            JOIN pfs_config_sps ON pfs_config_sps.visit0 = pfs_config.visit0
            JOIN sps_exposure ON sps_exposure.pfs_visit_id = pfs_config_sps.pfs_visit_id
            JOIN visit_set ON visit_set.pfs_visit_id = pfs_config_sps.pfs_visit_id
            JOIN iic_sequence ON iic_sequence.iic_sequence_id = visit_set.iic_sequence_id
            JOIN sps_camera ON sps_camera.sps_camera_id = sps_exposure.sps_camera_id
            JOIN tel_status ON tel_status.pfs_visit_id = pfs_config_sps.pfs_visit_id
            WHERE
               design_name = 'cobraHome' AND
               (cmd_str LIKE '%trace%' OR cmd_str LIKE '%scienceTrace%') AND
               abs(insrot) < 1 AND altitude > 89.9 AND
               pfs_config.visit0 = sps_exposure.pfs_visit_id
               {" AND " + " AND ".join(where) if where else ''}
            -- GROUP BY sps_exposure.pfs_visit_id
            -- , pfs_config.visit0
            ORDER BY sps_exposure.pfs_visit_id
            {LIMIT}
               ''', opdb)

        return tmp

    with opdb:
        tmp = pd_read_sql(f'''
        SELECT DISTINCT
           sps_exposure.pfs_visit_id
           ,(SELECT DISTINCT
             count (*) -- FILTER (WHERE pcf.is_on_source IS TRUE)
             FROM pfs_config AS pc
             JOIN pfs_config_fiber AS pcf ON pcf.pfs_design_id = pc.pfs_design_id AND pcf.visit0 = pc.visit0
             JOIN pfs_design_fiber AS pdf ON pdf.pfs_design_id = pcf.pfs_design_id AND
                                             pdf.fiber_id = pcf.fiber_id
             WHERE
                pc.visit0 = pfs_config.visit0 AND
                pcf.fiber_status = {int(FiberStatus.GOOD)} AND
                pdf.target_type != {int(TargetType.ENGINEERING)}
             ) as n_fiber
           -- , cmd_str
           , sps_exposure.time_exp_start AS time_exp_start
        FROM pfs_config
        JOIN pfs_design ON pfs_design.pfs_design_id = pfs_config.pfs_design_id
        JOIN pfs_config_sps ON pfs_config_sps.visit0 = pfs_config.visit0
        JOIN sps_exposure ON sps_exposure.pfs_visit_id = pfs_config_sps.pfs_visit_id
        JOIN visit_set ON visit_set.pfs_visit_id = pfs_config_sps.pfs_visit_id
        JOIN iic_sequence ON iic_sequence.iic_sequence_id = visit_set.iic_sequence_id
        JOIN sps_camera ON sps_camera.sps_camera_id = sps_exposure.sps_camera_id
        JOIN tel_status ON tel_status.pfs_visit_id = pfs_config_sps.pfs_visit_id
        WHERE
           design_name = 'cobraHome' AND
           (cmd_str LIKE '%trace%' OR cmd_str LIKE '%scienceTrace%') AND
           abs(insrot) < 1 AND altitude > 89.9
           {" AND " + " AND ".join(where) if where else ''}
        ORDER BY sps_exposure.pfs_visit_id
        {LIMIT}
           ''', opdb)

    return tmp if returnDataFrame else tmp.pfs_visit_id.to_numpy()


def getTimeForVisits(opdb, visits):
    """Return a table with the times for the given list of visits

    opdb: `psycopg2.extensions.connection`
       Connection to opdb
    visits: `array[int]`
       The desired visit numbers

    N.b. opdb time zone is/was wrong between visits 136339 and 136631; remove correction when it's fixed
    """
    with opdb:
        tmp = pd_read_sql(f'''
           SELECT DISTINCT
               pfs_visit_id,
           MIN(time_exp_start
               AT TIME ZONE CASE WHEN (pfs_visit_id BETWEEN 136339 AND 136631) THEN 'UTC' ELSE 'HST' END
               AT TIME ZONE 'HST') as time_exp_start
               -- , MIN(exptime) as exptime
            FROM
               sps_exposure
            WHERE
               pfs_visit_id IN ({",".join([str(v) for v in visits])})
            GROUP BY pfs_visit_id
            ORDER BY pfs_visit_id ASC
           ''', opdb)

    return tmp


def calculateNxNy(n):
    """Calculate nx, ny for subplots

    n: `int`
       Desired number of panels
    """
    ny = int(np.sqrt(n))
    nx = n//ny

    while ny*nx < n:
        nx += 1

    return nx, ny
