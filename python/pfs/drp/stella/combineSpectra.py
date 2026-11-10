from typing import List, Optional, Sequence

import numpy as np

from lsst.pipe.base import Struct
from pfs.datamodel import PfsFiberArraySet, MaskHelper

__all__ = ("combineSpectraSets", "mergeWeights", "coaddWeights", "accumulateWeighted")


def accumulateWeighted(
    rows: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    selects: Sequence[np.ndarray],
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted sum of rows: the common step of the spectral combinations

    Each input's selected pixels contribute ``row*weight``. The other pixels
    contribute nothing, not even a non-finite value. The inputs are added in
    order. Normalisation (e.g., by the sum of the weights) is left to the
    caller, as it is part of the combination's policy.

    Parameters
    ----------
    rows : sequence of `numpy.ndarray`
        The values of each input, e.g. its flux. They may have leading
        dimensions (e.g. covariance diagonals); the pixels are the last axes.
    weights : sequence of `numpy.ndarray`
        The weight of each pixel of each input (pixels only).
    selects : sequence of `numpy.ndarray` of `bool`
        The pixels of each input to use.
    out : `numpy.ndarray`, optional
        Array to accumulate into; if not provided, a zero array like the
        first row.

    Returns
    -------
    sum : `numpy.ndarray`
        The weighted sum.
    """
    if out is None:
        out = np.zeros_like(rows[0])
    for row, weight, select in zip(rows, weights, selects):
        with np.errstate(invalid="ignore"):
            out[..., select] += row[..., select]*weight[select]
    return out


def mergeWeights(
    spectra: List[PfsFiberArraySet],
    flags: MaskHelper,
    badMasks: List[str],
    suspectMasks: Optional[List[str]] = None,
) -> Struct:
    """The weights with which `combineSpectraSets` combines spectra

    Each pixel of each input has weight ``norm**2/variance`` (zero if the
    variance isn't positive and finite). An output pixel is the combination of:
    - the inputs that are good there (not masked as bad or suspect);
    - if there are none, the inputs that are suspect there;
    - failing that, the inputs that are masked as bad there (it is then
      flagged ``NO_DATA``).

    Parameters
    ----------
    spectra : iterable of `pfs.datamodel.PfsFiberArraySet`
        Spectra to combine, resampled to a common wavelength sampling.
    flags : `pfs.datamodel.MaskHelper`
        Mask interpreter, for identifying bad pixels.
    badMasks : `list` of `str`
        Mask planes to exclude from the combination.
    suspectMasks : `list` of `str`, optional
        Mask planes to exclude from the combination unless they provide the only
        non-bad pixels.

    Returns
    -------
    weights : `list` of `numpy.ndarray` of `float`
        Each input's weight for each pixel.
    selects : `list` of `numpy.ndarray` of `bool`
        Each input's pixels that contribute to the combination.
    sumWeights : `numpy.ndarray` of `float`
        Sum of the weights of the contributing inputs, for each pixel.
    calculate : `numpy.ndarray` of `bool`
        Output pixels that are calculated (the others are zero).
    mask : `numpy.ndarray` of `int`
        Mask of the combination.
    """
    archetype = spectra[0]
    if suspectMasks:
        badMasks = list(badMasks) + list(suspectMasks)  # Don't use suspect masks unless necessary

    def makeData():
        """Make a Struct for one category of pixels (good, suspect or bad)"""
        return Struct(
            mask=np.zeros_like(archetype.mask),
            count=np.zeros_like(archetype.flux, dtype=int),
            sumWeights=np.zeros_like(archetype.flux),
            selects=[],
        )

    goodData = makeData()
    suspectData = makeData() if suspectMasks else None
    badData = makeData()

    def accumulate(data, spectrum, weight, select):
        """Count a spectrum's pixels into a category"""
        data.mask[select] |= spectrum.mask[select]
        data.count[select] += 1
        data.sumWeights[select] += weight[select]
        data.selects.append(select)

    weights = []
    for ss in spectra:
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = ss.variance/ss.norm**2
            goodVariance = np.isfinite(variance) & (variance > 0)
        good = ((ss.mask & ss.flags.get(*badMasks)) == 0) & goodVariance  # Use freely
        bad = ((ss.mask & ss.flags.get(*badMasks)) != 0) & goodVariance

        weight = np.zeros_like(ss.flux)
        weight[goodVariance] = 1.0/variance[goodVariance]
        weights.append(weight)
        accumulate(goodData, ss, weight, good)
        if suspectMasks:
            suspect = ((ss.mask & ss.flags.get(*suspectMasks)) != 0) & goodVariance  # Use if we're desperate
            accumulate(suspectData, ss, weight, suspect)
        accumulate(badData, ss, weight, bad)

    # Choose the category of each output pixel
    useGood = (goodData.count > 0) & (goodData.sumWeights > 0)
    mask = goodData.mask.copy()
    mask[useGood] &= ~mask.dtype.type(flags.get(*badMasks))
    useBad = ~useGood
    useSuspect = np.zeros_like(useGood)
    if suspectMasks:
        useSuspect = (goodData.count == 0) & (suspectData.count > 0)
        mask[useSuspect] |= suspectData.mask[useSuspect]
        useBad &= ~useSuspect
    mask[useBad] |= flags["NO_DATA"]
    for ss in spectra:
        mask[useBad] |= ss.mask[useBad]
    useBad &= badData.sumWeights > 0  # Pixels with no data at all aren't calculated

    sumWeights = np.zeros_like(archetype.flux)
    sumWeights[useGood] = goodData.sumWeights[useGood]
    sumWeights[useBad] = badData.sumWeights[useBad]
    selects = [gg & useGood for gg in goodData.selects]
    if suspectMasks:
        sumWeights[useSuspect] = suspectData.sumWeights[useSuspect]
        selects = [sel | (sus & useSuspect) for sel, sus in zip(selects, suspectData.selects)]
    selects = [sel | (bb & useBad) for sel, bb in zip(selects, badData.selects)]

    return Struct(weights=weights, selects=selects, sumWeights=sumWeights,
                  calculate=useGood | useSuspect | useBad, mask=mask)


def combineSpectraSets(
    spectra: List[PfsFiberArraySet],
    flags: MaskHelper,
    badMasks: List[str],
    suspectMasks: Optional[List[str]] = None,
) -> Struct:
    """Combine spectra

    The weights are those of `mergeWeights`. The flux and sky are combined
    normalised (divided by the norm), and the result is multiplied by the
    combined norm.

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
    weights : `lsst.pipe.base.Struct`
        The weights used (from `mergeWeights`).
    """
    archetype = spectra[0]
    weights = mergeWeights(spectra, flags, badMasks, suspectMasks)
    with np.errstate(invalid="ignore", divide="ignore"):
        fluxWeights = [ww/ss.norm for ww, ss in zip(weights.weights, spectra)]  # flux and sky: normalised
    flux = accumulateWeighted([ss.flux for ss in spectra], fluxWeights, weights.selects)
    sky = accumulateWeighted([ss.sky for ss in spectra], fluxWeights, weights.selects)
    norm = accumulateWeighted([ss.norm for ss in spectra], weights.weights, weights.selects)

    select = weights.calculate
    covar = np.zeros_like(archetype.covar)
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = 1.0/weights.sumWeights[select]
        flux[select] = flux[select]*factor
        sky[select] = sky[select]*factor
        norm[select] = norm[select]*factor
        covar[:, 0][select] = factor

    covar2 = np.zeros((1, 1), dtype=archetype.covar.dtype)
    with np.errstate(invalid="ignore"):
        return Struct(
            wavelength=archetype.wavelength,
            flux=flux*norm,
            sky=sky*norm,
            norm=norm,
            covar=covar*norm[:, np.newaxis, :]**2,
            mask=weights.mask,
            covar2=covar2,
            weights=weights,
        )


def coaddWeights(spectra: List, badMasks: List[str]) -> Struct:
    """The weights with which `CoaddSpectraTask.combine` combines spectra

    Each input pixel that is not masked as bad and has positive variance has
    weight ``1/variance``; the others are not used.

    Parameters
    ----------
    spectra : iterable of `pfs.datamodel.PfsFiberArray`
        Spectra to combine, resampled to a common wavelength sampling.
    badMasks : `list` of `str`
        Mask planes to exclude from the combination (interpreted with each
        spectrum's own ``flags``).

    Returns
    -------
    weights : `list` of `numpy.ndarray` of `float`
        Each input's weight for each pixel (zero where not used).
    selects : `list` of `numpy.ndarray` of `bool`
        Each input's pixels that contribute to the combination.
    sumWeights : `numpy.ndarray` of `float`
        Sum of the weights, for each pixel.
    """
    weights = []
    selects = []
    sumWeights = np.zeros_like(spectra[0].flux)
    for ss in spectra:
        weight = np.zeros_like(sumWeights)
        with np.errstate(invalid="ignore", divide="ignore"):
            good = ((ss.mask & ss.flags.get(*badMasks)) == 0) & (ss.covar[0] > 0)
            weight[good] = 1.0/ss.covar[0][good]
        weights.append(weight)
        selects.append(good)
        sumWeights += weight
    return Struct(weights=weights, selects=selects, sumWeights=sumWeights)
