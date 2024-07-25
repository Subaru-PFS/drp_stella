from typing import Union, Dict, Any, Iterable

import getpass
import platform
import time

from lsst.daf.base import PropertyList

__all__ = ("setCalibHeader", "recordCalibInputs")


def setCalibHeader(header: Union[PropertyList, Dict], calibName: str, visitList: Iterable[int],
                   outputId: Dict[str, Any]) -> None:
    """Set header keys for calibs

    We record the type, the time, the inputs, and the output.

    Parameters
    ----------
    header : `lsst.daf.base.PropertyList` or `dict`
        Header/metadata for calibration; modified.
    visitList : iterable of `int`
        List of visits for data that went into the calib.
    outputId : `dict` [`str`: POD]
        Data identifier for output. Should include at least ``spectrograph`` and
        ``arm``.
    """
    header["OBSTYPE"] = calibName  # Used by ingestCalibs.py

    now = time.localtime()
    header["CALIB_CREATION_DATE"] = time.strftime("%Y-%m-%d", now)
    header["CALIB_CREATION_TIME"] = time.strftime("%X %Z", now)
    try:
        hostname = platform.node()
    except Exception:
        hostname = None
    header["CALIB_CREATION_HOST"] = hostname if hostname else "unknown host"
    try:
        username = getpass.getuser()
    except Exception:
        username = None
    header["CALIB_CREATION_USER"] = username if username else "unknown user"

    # Clobber any existing CALIB_INPUT_*
    names = list(header.keys())
    for key in names:
        if key.startswith("CALIB_INPUT_"):
            header.remove(key)
    # Set new CALIB_INPUT_*
    for ii, vv in enumerate(sorted(set(visitList))):
        header[f"CALIB_INPUT_{ii}"] = vv

    header["CALIB_ID"] = " ".join(f"{key}={value}" for key, value in outputId.items())
    header["SPECTROGRAPH"] = outputId["spectrograph"]
    header["ARM"] = outputId["arm"]


def recordCalibInputs(self, calib, dataIdList, outputId):
    """Record metadata including the inputs and creation details

    This metadata will go into the FITS header.

    Parameters
    ----------
    calib : `lsst.afw.image.Exposure`
        Combined calib exposure.
    dataIdList : iterable of `dict`
        List of data identifiers for calibration inputs.
    outputId : `dict`
        Data identifier for output.
    """
    setCalibHeader(calib.getMetadata(), self.calibName, [dataId["visit"] for dataId in dataIdList], outputId)
