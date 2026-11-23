from typing import List, Optional, Sequence

import numpy as np

from lsst.pipe.base import Struct
from pfs.datamodel import PfsFiberArraySet, MaskHelper

__all__ = ("combineSpectraSets", "calculateCombineWeights", "calculateCoaddWeights", "accumulateWeighted")


def accumulateWeighted(
    arrays: Sequence[np.ndarray],
    weights: Sequence[np.ndarray],
    selects: Sequence[np.ndarray],
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Weighted sum of arrays: the common step of the spectral combinations

    Each input's selected pixels contribute ``array*weight``. The other pixels
    contribute nothing, not even a non-finite value. The inputs are added in
    order. Normalisation (e.g., by the sum of the weights) is left to the
    caller, as it is part of the combination's policy.

    Parameters
    ----------
    arrays : sequence of `numpy.ndarray`
        The values of each input, e.g. its flux. They may have leading
        dimensions (e.g. covariance diagonals); the pixels are the last axes.
    weights : sequence of `numpy.ndarray`
        The weight of each pixel of each input (pixels only).
    selects : sequence of `numpy.ndarray` of `bool`
        The pixels of each input to use.
    out : `numpy.ndarray`, optional
        Array to accumulate into; if not provided, a zero array like the
        first array.

    Returns
    -------
    sum : `numpy.ndarray`
        The weighted sum.
    """
    if out is None:
        out = np.zeros_like(arrays[0])
    for array, weight, select in zip(arrays, weights, selects):
        with np.errstate(invalid="ignore"):
            out[..., select] += array[..., select]*weight[select]
    return out


def calculateCombineWeights(
    spectra: List[PfsFiberArraySet],
    flags: MaskHelper,
    badMasks: List[str],
    suspectMasks: Optional[List[str]] = None,
    modelWeights: Optional[List[np.ndarray]] = None,
) -> Struct:
    """The weights with which `combineSpectraSets` combines spectra

    Each pixel of each input has weight ``norm**2/variance``, or, with
    ``modelWeights``, ``modelWeight*norm**2`` (zero if the variance, or the
    model weight, isn't positive and finite). An output pixel is the
    combination of:
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
    modelWeights : `list` of `numpy.ndarray` of `float`, optional
        Inverse-variance weights for each of the ``spectra`` (in units of the
        inverse of the ``spectra`` variance), e.g., from ``ModelWeightsTask``.
        If not provided, the inverse of the measured variance is used.

    Returns
    -------
    weights : `list` of `numpy.ndarray` of `float`
        Each input's weight for each pixel.
    variances : `list` of `numpy.ndarray` of `float`
        Each input's normalised variance (``variance/norm**2``), for the
        propagation of errors.
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
    if modelWeights is not None and len(modelWeights) != len(spectra):
        raise RuntimeError(
            f"Length mismatch between spectra ({len(spectra)}) and weights ({len(modelWeights)})"
        )
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
    variances = []
    for ii, ss in enumerate(spectra):
        with np.errstate(invalid="ignore", divide="ignore"):
            variance = ss.variance/ss.norm**2
            goodVariance = np.isfinite(variance) & (variance > 0)
            if modelWeights is not None:
                weight = modelWeights[ii]*ss.norm**2
                goodVariance &= np.isfinite(weight) & (weight > 0)
            else:
                weight = 1.0/variance
        good = ((ss.mask & ss.flags.get(*badMasks)) == 0) & goodVariance  # Use freely
        bad = ((ss.mask & ss.flags.get(*badMasks)) != 0) & goodVariance

        weight = np.where(goodVariance, weight, 0.0)
        weights.append(weight)
        variances.append(variance)
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

    return Struct(weights=weights, variances=variances, selects=selects, sumWeights=sumWeights,
                  calculate=useGood | useSuspect | useBad, mask=mask)


def combineSpectraSets(
    spectra: List[PfsFiberArraySet],
    flags: MaskHelper,
    badMasks: List[str],
    suspectMasks: Optional[List[str]] = None,
    weights: Optional[List[np.ndarray]] = None,
) -> Struct:
    """Combine spectra

    The weights are those of `calculateCombineWeights`. The flux and sky are combined
    normalised (divided by the norm), and the result is multiplied by the
    combined norm. The variance is propagated for the weights used:
    ``sum(variance*weight**2)/sum(weight)**2``.

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
    weights : `lsst.pipe.base.Struct`
        The weights used (from `calculateCombineWeights`).
    """
    archetype = spectra[0]
    weights = calculateCombineWeights(spectra, flags, badMasks, suspectMasks, weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        fluxWeights = [ww/ss.norm for ww, ss in zip(weights.weights, spectra)]  # flux and sky: normalised
        varianceWeights = [ww**2 for ww in weights.weights]
    flux = accumulateWeighted([ss.flux for ss in spectra], fluxWeights, weights.selects)
    sky = accumulateWeighted([ss.sky for ss in spectra], fluxWeights, weights.selects)
    norm = accumulateWeighted([ss.norm for ss in spectra], weights.weights, weights.selects)
    variance = accumulateWeighted(weights.variances, varianceWeights, weights.selects)

    select = weights.calculate
    covar = np.zeros_like(archetype.covar)
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = 1.0/weights.sumWeights[select]
        flux[select] = flux[select]*factor
        sky[select] = sky[select]*factor
        norm[select] = norm[select]*factor
        covar[:, 0][select] = variance[select]*factor**2

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


def calculateCoaddWeights(
    spectra: List, badMasks: List[str], modelWeights: Optional[Sequence[Optional[np.ndarray]]] = None
) -> Struct:
    """The weights with which `CoaddSpectraTask.combine` combines spectra

    Each input pixel that is not masked as bad and has positive variance has
    weight ``1/variance``, or its model weight if provided (and positive and
    finite); the others are not used.

    Parameters
    ----------
    spectra : iterable of `pfs.datamodel.PfsFiberArray`
        Spectra to combine, resampled to a common wavelength sampling.
    badMasks : `list` of `str`
        Mask planes to exclude from the combination (interpreted with each
        spectrum's own ``flags``).
    modelWeights : iterable of (`numpy.ndarray` of `float` or `None`), optional
        Inverse-variance weights for each spectrum, resampled like the spectra
        (see ``ModelWeightsTask``). If `None` (for all spectra, or for an
        individual spectrum), the inverse of the measured variance is used.

    Returns
    -------
    weights : `list` of `numpy.ndarray` of `float`
        Each input's weight for each pixel (zero where not used).
    selects : `list` of `numpy.ndarray` of `bool`
        Each input's pixels that contribute to the combination.
    sumWeights : `numpy.ndarray` of `float`
        Sum of the weights, for each pixel.
    """
    if modelWeights is None:
        modelWeights = [None]*len(spectra)
    weights = []
    selects = []
    sumWeights = np.zeros_like(spectra[0].flux)
    for ss, model in zip(spectra, modelWeights, strict=True):
        with np.errstate(invalid="ignore", divide="ignore"):
            pixelWeights = 1.0/ss.covar[0] if model is None else model
            good = ((ss.mask & ss.flags.get(*badMasks)) == 0) & (ss.covar[0] > 0)
            good &= np.isfinite(pixelWeights) & (pixelWeights > 0)
        # The weights multiply the inputs at their own precision, and are summed at the flux's
        weight = np.zeros(sumWeights.shape, dtype=np.result_type(sumWeights, pixelWeights))
        weight[good] = pixelWeights[good]
        weights.append(weight)
        selects.append(good)
        sumWeights += weight.astype(sumWeights.dtype)
    return Struct(weights=weights, selects=selects, sumWeights=sumWeights)
