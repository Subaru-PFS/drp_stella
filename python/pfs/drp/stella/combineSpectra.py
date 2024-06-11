from typing import List

import numpy as np

from lsst.pipe.base import Struct
from pfs.datamodel import PfsFiberArraySet, MaskHelper


def combineSpectraSets(spectra: List[PfsFiberArraySet], flags: MaskHelper, badMasks: List[str]) -> Struct:
    """Combine spectra

    Parameters
    ----------
    spectra : iterable of `pfs.datamodel.PfsFiberArraySet`
        List of spectra to combine. These should already have been
        resampled to a common wavelength representation.
    flags : `pfs.datamodel.MaskHelper`
        Mask interpreter, for identifying bad pixels.
    badMasks : `list` of `str`
        Mask planes to exclude from the combination.

    Returns
    -------
    wavelength : `numpy.ndarray` of `float`
        Wavelengths for combined spectrum.
    flux : `numpy.ndarray` of `float`
        Normalised flux measurements for combined spectrum.
    sky : `numpy.ndarray` of `float`
        Sky measurements for combined spectrum.
    norm : `numpy.ndarray` of `float`
        Normalisation of combined spectrum.
    covar : `numpy.ndarray` of `float`
        Covariance matrix for combined spectrum.
    mask : `numpy.ndarray` of `int`
        Mask for combined spectrum.
    """
    archetype = spectra[0]
    mask = np.zeros_like(archetype.mask)
    flux = np.zeros_like(archetype.flux)
    sky = np.zeros_like(archetype.sky)
    norm = np.zeros_like(archetype.norm)
    covar = np.zeros_like(archetype.covar)
    sumWeights = np.zeros_like(archetype.flux)

    for ss in spectra:
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = ss.variance/ss.norm**2
            good = ((ss.mask & ss.flags.get(*badMasks)) == 0) & (variance > 0)

        weight = np.zeros_like(ss.flux)
        weight[good] = 1.0/variance[good]
        with np.errstate(invalid="ignore"):
            flux[good] += ss.flux[good]*weight[good]/ss.norm[good]
            sky[good] += ss.sky[good]*weight[good]/ss.norm[good]
            norm[good] += ss.norm[good]*weight[good]
        mask[good] |= ss.mask[good]
        sumWeights += weight

    good = sumWeights > 0
    flux[good] /= sumWeights[good]
    sky[good] /= sumWeights[good]
    norm[good] /= sumWeights[good]
    covar[:, 0][good] = 1.0/sumWeights[good]
    covar[:, 0][~good] = np.inf
    covar[:, 1:] = np.where(good, 0.0, np.inf)[:, np.newaxis]

    for ss in spectra:
        mask[~good] |= ss.mask[~good]
    mask[~good] |= flags["NO_DATA"]
    covar2 = np.zeros((1, 1), dtype=archetype.covar.dtype)
    with np.errstate(invalid="ignore"):
        return Struct(
            wavelength=archetype.wavelength,
            flux=flux*norm,
            sky=sky*norm,
            norm=norm,
            covar=covar*norm[:, np.newaxis, :]**2,
            mask=mask,
            covar2=covar2
        )
