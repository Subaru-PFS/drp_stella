from typing import Optional

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.constants
import astropy.units as u
import numpy as np

from pfs.utils.location import SUBARU
from pfs.datamodel import PfsArm, PfsConfig

__all__ = ("calculateBarycentricCorrection", "applyBarycentricCorrection")


def calculateBarycentricCorrection(
    pfsArm: PfsArm, pfsConfig: PfsConfig, location: Optional[EarthLocation] = None
):
    """Calculate barycentric correction for a pfsArm

    The barycentric correction is calculated for each fiber and recorded in the
    pfsArm notes.

    Parameters
    ----------
    pfsArm : `PfsArm`
        pfsArm for which to calculate the barycentric correction.
    pfsConfig : `PfsConfig`
        Fiber configuration.
    location : `EarthLocation`, optional
        Observatory location. If ``None``, the Subaru location is used.
    """
    if location is None:
        location = SUBARU.location
    time = Time(pfsArm.identity.obsTime)
    pfsConfig = pfsConfig.select(fiberId=pfsArm.fiberId)
    if not np.array_equal(pfsArm.fiberId, pfsConfig.fiberId):
        raise RuntimeError("fiberId mismatch")
    for ii in range(len(pfsArm)):
        coord = SkyCoord(ra=pfsConfig.ra[ii]*u.deg, dec=pfsConfig.dec[ii]*u.deg)
        corr = coord.radial_velocity_correction("barycentric", obstime=time, location=location)
        pfsArm.notes.barycentricCorrection[ii] = corr.to(u.km/u.s).value


def applyBarycentricCorrection(pfsArm: PfsArm, *, inverse=False):
    """Apply previously calculated barycentric correction

    Parameters
    ----------
    pfsArm : `PfsArm`
        pfsArm to which to apply the barycentric correction.
    inverse : `bool`, optional
        If `True`, apply the inverse correction. Default is `False`.

    Returns
    -------
    noValue : `set`
        fiberIds for which no barycentric correction was available.
    """
    speedOfLight = astropy.constants.c.to(u.km/u.s).value
    noValue = set()
    for ii in range(len(pfsArm)):
        corr = pfsArm.notes.barycentricCorrection[ii]  # km/s
        if not np.isfinite(corr):
            noValue.add(pfsArm.fiberId[ii])
            continue
        if inverse:
            corr *= -1
        pfsArm.wavelength[ii] *= 1 + (corr/speedOfLight)
    return noValue
