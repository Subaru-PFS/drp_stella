import itertools
from types import SimpleNamespace
from collections import defaultdict

import numpy as np

from pfs.datamodel import MaskHelper
from .datamodel import FluxTable, interpolateFlux

__all__ = ["makeFluxTable"]


def coadd(flux, variance, mask, good, noData):
    """Coadd spectra arrays

    We perform a simple weighted average.

    Parameters
    ----------
    flux : `numpy.ndarray` of `float`, dimensions ``(M,N)``
        Flux arrays; will do a weighted average.
    variance : `numpy.ndarray` of `float`, dimensions ``(M,N)``
        Variance arrays; used for the weights.
    mask : `numpy.ndarray` of `int`, dimensions ``(M,N)``
        Mask arrays; will be ``OR``-ed together.
    good : `numpy.ndarray` of `bool`, dimensions ``(M,N)``
        Selection array indicating values to use.
    noData : `int`
        Mask value to give pixels with no good data.

    Returns
    -------
    flux : `numpy.ndarray` of `float`, dimension ``(N)``
        Coadded flux array.
    variance : `numpy.ndarray` of `float`, dimension ``(N)``
        Coadded variance array.
    mask : `numpy.ndarray` of `int`, dimension ``(N)``
        Coadded mask array.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = np.zeros_like(flux)
        weight[good] = 1.0/variance[good]
        sumWeight = weight.sum(axis=0)
        coaddFlux = (flux*weight).sum(axis=0)/sumWeight
        anyGood = np.bitwise_or.reduce(good, axis=0)
        coaddMask = np.bitwise_or.reduce(np.where(good, mask, 0), axis=0)
        coaddMask[~anyGood] |= noData
        coaddVariance = 1.0/sumWeight
    return SimpleNamespace(flux=coaddFlux, variance=coaddVariance, mask=coaddMask)


def maskOverlaps(wavelength1, sn1, mask1, wavelength2, sn2, mask2, maskVal):
    """Mask overlapping spectra

    We identify the overlapping region(s) to mask through having a lower
    signal-to-noise ratio.

    Wavelength vectors are assumed to be sorted from low to high.

    Parameters
    ----------
    wavelength1 : `numpy.ndarray` of `float`, dimension ``(M)``
        Wavelength vector for first arm.
    sn1 : `numpy.ndarray` of `float`, dimension ``(M)``
        Signal-to-noise vector for first arm.
    mask1 : `numpy.ndarray` of `int`, dimension ``(M)``
        Mask vector for first arm; will be updated in-place.
    wavelength2 : `numpy.ndarray` of `float`, dimension ``(N)``
        Wavelength vector for second arm.
    sn2 : `numpy.ndarray` of `float`, dimension ``(N)``
        Signal-to-noise vector for second arm.
    mask2 : `numpy.ndarray` of `int`, dimension ``(N)``
        Mask vector for second arm; will be updated in-place.
    maskVal : `int`
        Mask value to ``OR`` into the mask in the overlapping region(s).
    """
    min1, max1 = wavelength1[0], wavelength1[-1]
    min2, max2 = wavelength2[0], wavelength2[-1]
    assert min1 < max1 and min2 < max2, "Wavelength vectors are sorted from low to high"
    if min1 > max2 or min2 > max1:
        return  # No overlap

    # Make sure number 1 has the lowest wavelength, for convenience
    if min2 < min1:
        wavelength1, wavelength2 = wavelength2, wavelength1
        sn1, sn2 = sn2, sn1
        mask1, mask2 = mask2, mask1
        min1, max1, min2, max2 = min2, max2, min1, max1

    # Identify the crossover points from the signal-to-noise
    sn1 = interpolateFlux(wavelength1, sn1, wavelength2, fill=0.0)  # S/N of arm1 in the frame of arm2

    with np.errstate(invalid="ignore"):
        better = sn2 > sn1  # arm2 signal-to-noise is better than arm1
    if np.all(better):
        # arm2 is always better than arm1
        overlap1 = slice(0, len(wavelength1))
        overlap2 = None
    elif np.all(~better):
        # arm1 is always better than arm2
        overlap1 = None
        overlap2 = slice(0, len(wavelength2))
    elif max1 < max2:
        # Overlap is at the end of arm1 and beginning of arm2
        crossover2 = np.argwhere(better)[-1][-1]  # Last index with lesser variance
        crossover1 = np.searchsorted(wavelength1, wavelength2[crossover2])
        overlap1 = slice(crossover1, len(wavelength1))
        overlap2 = slice(0, crossover2)
    else:
        # arm2 is completely contained somewhere in the middle of arm1
        indices = np.argwhere(better)[-1]
        crossover1 = indices[0]
        crossover2 = indices[-1]
        overlap2 = slice(crossover1, crossover2)
        overlap1 = slice(np.searchsorted(wavelength1, wavelength2[crossover1]),
                         np.searchsorted(wavelength1, wavelength2[crossover2]))

    # Mask the nominated pixels
    if overlap1 is not None:
        mask1[overlap1] |= maskVal
    if overlap2 is not None:
        mask2[overlap2] |= maskVal


def makeFluxTable(identities, spectra, flags, ignoreFlags=None, iterations=3, sigma=3.0):
    """Create a FluxTable from multiple spectra

    Parameters
    ----------
    identities : iterable of `dict`
        Key-value pairs describing the identity of each spectrum. Requires at
        least the ``visit`` and ``arm`` keywords.
    spectra : iterable of `pfs.datamodel.PfsSpectrum`
        Spectra to coadd.
    flags : `pfs.datamodel.MaskHelper`
        Helper for dealing with symbolic names for mask values.
    ignoreFlags : iterable of `str`, optional
        Masks for which to ignore pixels; ``NO_DATA`` is always included.
    iterations : `int`
        Number of rejection iterations.
    sigma : `float`
        k-sigma rejection threshold (standard deviations).

    Returns
    -------
    fluxTable : `pfs.datamodel.FluxTable`
        Fluxes at near the native resolution.
    """
    if len(identities) != len(spectra):
        raise RuntimeError("Length mismatch: %d vs %d" % (len(identities), len(spectra)))
    armSpectra = defaultdict(list)
    for ident, spec in zip(identities, spectra):
        armSpectra[ident["arm"]].append(spec)

    if ignoreFlags is None:
        ignoreFlags = []
    maskVal = flags.get("NO_DATA", *ignoreFlags)
    noData = flags.get("NO_DATA")

    armWavelengths = {}
    armCoadds = {}
    for arm in armSpectra:
        lengths = set(len(ss.wavelength) for ss in armSpectra[arm])
        assert len(lengths) == 1

        if len(armSpectra[arm]) == 1:
            ss = armSpectra[arm][0]
            armCoadds[arm] = SimpleNamespace(flux=ss.flux, variance=ss.variance, mask=ss.mask)
            armWavelengths[arm] = ss.wavelength
            continue

        armWavelengths[arm] = np.array([ss.wavelength for ss in armSpectra[arm]]).mean(axis=0)

        # Resample all spectra to the common wavelength scale
        resampled = [ss.resample(armWavelengths[arm]) for ss in armSpectra[arm]]

        # Convert lists to arrays, for convenience and speed
        resampledFlux = np.array([ss.flux for ss in resampled])
        resampledVariance = np.array([ss.variance for ss in resampled])
        resampledMask = np.array([ss.mask for ss in resampled])

        with np.errstate(invalid="ignore"):
            good = ((resampledMask & maskVal) == 0) & (resampledVariance > 0) & np.isfinite(resampledFlux)
        resampledFlux[~good] = 0.0  # avoid NANs leaking into the summations

        keep = np.ones_like(resampledFlux, dtype=bool)
        for _ in range(iterations):
            use = good & keep
            data = coadd(resampledFlux, resampledVariance, resampledMask, use, noData)
            with np.errstate(divide='ignore', invalid='ignore'):
                diff = (resampledFlux - data.flux[np.newaxis, :])/np.sqrt(resampledVariance)
            lq, uq = np.percentile(diff[use], (25.0, 75.0))
            stdev = 0.741*(uq - lq)
            with np.errstate(invalid='ignore'):
                keep = np.abs(diff) < sigma*stdev
            if np.all(~keep):
                keep = np.ones_like(resampledFlux, dtype=bool)
                break

        use = good & keep
        armCoadds[arm] = coadd(resampledFlux, resampledVariance, resampledMask, use, noData)

    # Mask overlaps in wavelength
    overlap = flags.add("OVERLAP")
    for arm1, arm2 in itertools.combinations(list(armSpectra.keys()), 2):
        with np.errstate(invalid="ignore", divide="ignore"):
            sn1 = armCoadds[arm1].flux/np.sqrt(armCoadds[arm1].variance)
            sn2 = armCoadds[arm2].flux/np.sqrt(armCoadds[arm2].variance)
        maskOverlaps(armWavelengths[arm1], sn1, armCoadds[arm1].mask,
                     armWavelengths[arm2], sn2, armCoadds[arm2].mask,
                     overlap)

    # Smash everything together, then sort by wavelength
    armList = list(armCoadds.keys())
    select = [((armCoadds[arm].mask & overlap) == 0) for arm in armList]
    lengths = [ss.sum() for ss in select]
    wavelength = np.concatenate([armWavelengths[arm] for arm, nn in zip(armList, lengths) if nn != 0])
    flux = np.concatenate([armCoadds[arm].flux for arm, nn in zip(armList, lengths) if nn != 0])
    variance = np.concatenate([armCoadds[arm].variance for arm, nn in zip(armList, lengths) if nn != 0])
    mask = np.concatenate([armCoadds[arm].mask for arm, nn in zip(armList, lengths) if nn != 0])

    indices = np.argsort(wavelength)
    flags = MaskHelper.fromMerge([ss.flags for ss in spectra])

    return FluxTable(wavelength[indices], flux[indices], np.sqrt(variance[indices]), mask[indices], flags)
