from typing import List, Optional

import numpy as np

from lsst.pipe.base import Struct
from pfs.datamodel import PfsFiberArraySet, MaskHelper


def combineSpectraSets(
    spectra: List[PfsFiberArraySet],
    flags: MaskHelper,
    badMasks: List[str],
    suspectMasks: Optional[List[str]] = None,
    weights: Optional[List[np.ndarray]] = None,
) -> Struct:
    """Combine spectra

    Parameters
    ----------
    spectra : iterable of `pfs.datamodel.PfsFiberArraySet`
        List of spectra to combine. These should already have been
        resampled to a common wavelength representation.
    flags : `pfs.datamodel.MaskHelper`
        Mask interpreter, for identifying bad pixels.
    badMasks : `list` of `str`
        Mask planes to absolutely exclude from the combination.
    suspectMasks : `list` of `str`, optional
        Mask planes to exclude from the combination unless they provide the only
        non-bad pixels.
    weights : `list` of `numpy.ndarray` of `float`, optional
        Inverse-variance weights for each of the ``spectra`` (in units of the
        inverse of the ``spectra`` variance), e.g., from ``ModelWeightsTask``.
        If not provided, the inverse of the measured variance is used.

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
    if weights is not None and len(weights) != len(spectra):
        raise RuntimeError(f"Length mismatch between spectra ({len(spectra)}) and weights ({len(weights)})")

    def makeData():
        """Make a Struct to hold the combination data"""
        return Struct(
            mask=np.zeros_like(archetype.mask),
            flux=np.zeros_like(archetype.flux),
            sky=np.zeros_like(archetype.sky),
            norm=np.zeros_like(archetype.norm),
            covar=np.zeros_like(archetype.covar),
            count=np.zeros_like(archetype.flux, dtype=int),
            sumWeights=np.zeros_like(archetype.flux),
        )

    goodData = makeData()
    badData = makeData()
    suspectData = None
    if suspectMasks:
        badMasks = list(badMasks) + list(suspectMasks)  # Don't use suspect masks unless necessary
        suspectData = makeData()

    def accumulate(data, spectrum, weight, variance, select):
        """Accumulate a spectrum into the combination

        Parameters
        ----------
        data : `Struct`
            Data structure to accumulate into.
        spectrum : `pfs.datamodel.PfsFiberArraySet`
            Spectrum to accumulate.
        weight : `numpy.ndarray` of `float`
            Weights for each pixel.
        variance : `numpy.ndarray` of `float`
            Normalised variance for each pixel.
        select : `numpy.ndarray` of `bool`
            Pixels to accumulate.
        """
        with np.errstate(invalid="ignore"):
            factor = weight[select]/spectrum.norm[select]
            data.flux[select] += spectrum.flux[select]*factor
            data.sky[select] += spectrum.sky[select]*factor
            data.norm[select] += spectrum.norm[select]*weight[select]
            data.covar[:, 0][select] += variance[select]*weight[select]**2
        data.mask[select] |= spectrum.mask[select]
        data.count[select] += 1
        data.sumWeights[select] += weight[select]

    for ii, ss in enumerate(spectra):
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = ss.variance/ss.norm**2
            goodVariance = np.isfinite(variance) & (variance > 0)
            if weights is not None:
                weight = weights[ii]*ss.norm**2
                goodVariance &= np.isfinite(weight) & (weight > 0)
            else:
                weight = 1.0/variance
        good = ((ss.mask & ss.flags.get(*badMasks)) == 0) & goodVariance  # Use freely
        suspect = None
        if suspectMasks:
            suspect = ((ss.mask & ss.flags.get(*suspectMasks)) != 0) & goodVariance  # Use if we're desperate
        bad = ((ss.mask & ss.flags.get(*badMasks)) != 0) & goodVariance

        weight = np.where(goodVariance, weight, 0.0)
        accumulate(goodData, ss, weight, variance, good)
        if suspectMasks and np.any(suspect):
            accumulate(suspectData, ss, weight, variance, suspect)
        accumulate(badData, ss, weight, variance, bad)

    def calculate(target, source, select):
        """Calculate the final combined spectrum

        Parameters
        ----------
        target : `Struct`
            Data structure in which to put calculation result.
        source : `Struct`
            Data structure from which to calculate.
        select : `numpy.ndarray` of `bool`
            Pixels to calculate.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            factor = 1.0/source.sumWeights[select]
            target.flux[select] = source.flux[select]*factor
            target.sky[select] = source.sky[select]*factor
            target.norm[select] = source.norm[select]*factor
            target.covar[:, 0][select] = source.covar[:, 0][select]*factor**2

    good = (goodData.count > 0) & (goodData.sumWeights > 0)
    calculate(goodData, goodData, good)
    goodData.mask[good] &= ~goodData.mask.dtype.type(flags.get(*badMasks))
    bad = ~good

    if suspectMasks:
        useSuspect = (goodData.count == 0) & (suspectData.count > 0)
        if np.any(useSuspect):
            calculate(goodData, suspectData, useSuspect)
            goodData.mask[useSuspect] |= suspectData.mask[useSuspect]
        bad &= ~useSuspect

    calculate(goodData, badData, bad & (badData.sumWeights > 0))
    goodData.mask[bad] |= flags["NO_DATA"]
    for ss in spectra:
        goodData.mask[bad] |= ss.mask[bad]

    covar2 = np.zeros((1, 1), dtype=archetype.covar.dtype)
    with np.errstate(invalid="ignore"):
        return Struct(
            wavelength=archetype.wavelength,
            flux=goodData.flux*goodData.norm,
            sky=goodData.sky*goodData.norm,
            norm=goodData.norm,
            covar=goodData.covar*goodData.norm[:, np.newaxis, :]**2,
            mask=goodData.mask,
            covar2=covar2,
        )
