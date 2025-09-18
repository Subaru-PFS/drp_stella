import numpy as np

from pfs.datamodel import PfsConfig
from pfs.datamodel import PfsFiberArraySet
from .focalPlaneFunction import FocalPlaneFunction

__all__ = ("subtractSky1d",)


def subtractSky1d(spectra: PfsFiberArraySet, pfsConfig: PfsConfig, sky1d: FocalPlaneFunction) -> None:
    """Subtract sky model from spectra

    Parameters
    ----------
    spectra : `PfsFiberArraySet`
        Spectra from which to subtract sky model. The spectra are modified.
    pfsConfig : `PfsConfig`
        Top-end configuration.
    sky1d : `FocalPlaneFunction`
        Sky model.
    """
    sky = sky1d(spectra.wavelength, pfsConfig.select(fiberId=spectra.fiberId))
    skyValues = sky.values*spectra.norm
    skyVariances = sky.variances*spectra.norm**2
    spectra.flux -= skyValues
    spectra.sky += skyValues
    bitmask = spectra.flags.add("BAD_SKY")
    spectra.mask[np.array(sky.masks)] |= bitmask
####    spectra.covar[:, 0, :] += skyVariances
