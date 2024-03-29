import io
import re
import sys
import importlib

import astropy.io.fits
import lsst.afw.fits

from pfs.datamodel.utils import astropyHeaderFromDict

__all__ = ["headerToMetadata", "metadataToHeader", "getPfsVersions", "processConfigListFromCmdLine"]


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
