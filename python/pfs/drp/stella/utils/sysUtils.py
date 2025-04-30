import io
import re
import sys
import importlib
import pandas as pd
import psycopg2
import warnings

import astropy.io.fits
import lsst.afw.fits

from pfs.datamodel.utils import astropyHeaderFromDict
from pfs.utils.coordinates.transform import makePfiTransform

__all__ = ["headerToMetadata", "metadataToHeader", "getPfsVersions", "processConfigListFromCmdLine",
           "pd_read_sql", "makePfiTransformFromOpdb"]


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
