from typing import Optional

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.constants
import astropy.units as u
import numpy as np

from pfs.datamodel import PfsArm, PfsConfig


def barycentricCorrection(pfsArm: PfsArm, pfsConfig: PfsConfig, location: Optional[EarthLocation] = None):
    """Apply barycentric correction to a pfsArm

    Parameters
    ----------
    pfsArm : `PfsArm`
        pfsArm to correct.
    pfsConfig : `PfsConfig`
        Fiber configuration.
    location : `EarthLocation`, optional
        Observatory location. If ``None``, the Subaru location is used.
    """
    if location is None:
        # From https://subarutelescope.org/en/about/
        location = EarthLocation.from_geodetic("-155d28m34s", "19d49m32s", 4163.0*u.m)
    time = Time(pfsArm.identity.obsTime)
    pfsConfig = pfsConfig.select(fiberId=pfsArm.fiberId)
    if not np.array_equal(pfsArm.fiberId, pfsConfig.fiberId):
        raise RuntimeError("fiberId mismatch")
    for ii in range(len(pfsArm)):
        coord = SkyCoord(ra=pfsConfig.ra[ii]*u.deg, dec=pfsConfig.dec[ii]*u.deg)
        corr = coord.radial_velocity_correction("barycentric", obstime=time, location=location)
        pfsArm.notes.barycentricCorrection[ii] = corr.to(u.km/u.s).value
        if not np.isfinite(pfsArm.notes.barycentricCorrection[ii]):
            continue
        pfsArm.wavelength[ii] *= 1 + (corr/astropy.constants.c).value
