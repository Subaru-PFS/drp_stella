from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.polynomial.chebyshev import chebvander
from scipy.ndimage import median_filter

from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import Task

from .utils.math import robustRms

if TYPE_CHECKING:
    from pfs.datamodel import PfsFiberArray, PfsFiberArraySet

__all__ = ("ModelWeightsConfig", "ModelWeightsTask", "calculateModelVariance", "resampleWeights")


class ModelWeightsConfig(Config):
    """Configuration for ModelWeightsTask"""

    mask = ListField(
        dtype=str,
        default=["NO_DATA", "SAT", "BAD_FLAT", "CR", "BAD_FIBERNORMS", "BAD_SKY", "BAD_FLUXCAL"],
        doc="Mask planes to exclude from the object model and the variance fit",
    )
    smoothWidth = Field(
        dtype=int,
        default=51,
        doc=(
            "Width (pixels) of the median filter used to model the object flux. The residual "
            "correlation between a pixel's weight and its noise scales roughly inversely with this."
        ),
    )
    order = Field(
        dtype=int,
        default=3,
        doc="Chebyshev order (in wavelength) of the variance model coefficients",
    )
    rejIterations = Field(dtype=int, default=3, doc="Number of rejection iterations for the variance fit")
    rejThreshold = Field(dtype=float, default=4.0, doc="Rejection threshold (robust sigma) for variance fit")
    minNumPixels = Field(
        dtype=int,
        default=100,
        doc="Minimum number of good pixels in a spectrum to fit the variance model",
    )

    def validate(self):
        super().validate()
        if self.smoothWidth < 1:
            raise ValueError(f"smoothWidth ({self.smoothWidth}) must be positive")
        if self.order < 0:
            raise ValueError(f"order ({self.order}) must be non-negative")


def calculateModelVariance(
    wavelength: np.ndarray,
    flux: np.ndarray,
    sky: np.ndarray,
    variance: np.ndarray,
    good: np.ndarray,
    smoothWidth: int,
    order: int,
    rejIterations: int,
    rejThreshold: float,
    minNumPixels: int,
) -> np.ndarray:
    """Calculate a model variance for a single spectrum

    The variance is modeled as a function of the expected signal in each pixel,
    ``A(wavelength) + K(wavelength)*model``, where ``A`` and ``K`` are smooth
    functions of wavelength (Chebyshev polynomials) fit to the measured
    variance as a function of the measured signal (``flux + sky``), and the
    expected signal ``model`` is the ``sky`` plus a median-smoothed ``flux``.

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`, shape ``(M,)``
        Wavelength array.
    flux : `numpy.ndarray` of `float`, shape ``(M,)``
        Flux array (sky-subtracted, if ``sky`` is non-zero).
    sky : `numpy.ndarray` of `float`, shape ``(M,)``
        Sky array (subtracted from the flux).
    variance : `numpy.ndarray` of `float`, shape ``(M,)``
        Measured variance array.
    good : `numpy.ndarray` of `bool`, shape ``(M,)``
        Pixels with good data (finite values and positive variance, unmasked).
    smoothWidth : `int`
        Width (pixels) of the median filter used to model the object flux.
    order : `int`
        Chebyshev order of the variance model coefficients.
    rejIterations : `int`
        Number of rejection iterations for the variance fit.
    rejThreshold : `float`
        Rejection threshold (robust sigma) for the variance fit.
    minNumPixels : `int`
        Minimum number of good pixels required to fit the variance model.

    Returns
    -------
    modelVariance : `numpy.ndarray` of `float`, shape ``(M,)``
        Model variance. Pixels for which no model can be calculated are NaN.
    """
    length = flux.size
    result = np.full(length, np.nan, dtype=float)
    numGood = good.sum()
    if numGood == 0:
        return result

    # Model the object flux: median-filter the good pixels, and interpolate over the bad pixels.
    # We filter only the good pixels (rather than filling bad pixels first), and use mirror boundary
    # conditions, so that no single pixel is repeated within the filter window: repeated pixels would
    # dominate the median, making the model (and hence the weights) correlated with their noise.
    indices = np.arange(length)
    goodFlux = flux[good]
    objectModel = np.interp(
        indices,
        indices[good],
        median_filter(goodFlux, size=min(smoothWidth, 2 * goodFlux.size - 1), mode="mirror"),
    )
    skyModel = np.where(np.isfinite(sky), sky, 0.0)
    expected = np.clip(skyModel + objectModel, 0.0, None)

    if numGood < minNumPixels:
        # Not enough data for a model: use a constant, so the weights aren't correlated with the noise
        result[:] = np.median(variance[good])
        return result

    # Fit variance = A(wavelength) + K(wavelength)*signal
    wlMin = wavelength[good].min()
    wlMax = wavelength[good].max()
    xx = 2.0 * (wavelength - wlMin) / (wlMax - wlMin) - 1.0 if wlMax > wlMin else np.zeros_like(wavelength)
    cheby = chebvander(xx, order)  # shape (M, order + 1)
    measured = np.where(good, flux + skyModel, 0.0)

    def makeDesign(signal: np.ndarray) -> np.ndarray:
        """Construct the design matrix for the variance model"""
        return np.concatenate((cheby, cheby * signal[:, np.newaxis]), axis=1)

    design = makeDesign(measured)
    use = good.copy()
    for ii in range(rejIterations + 1):
        # Normalise by the variance, so the fit is in relative terms
        scale = 1.0 / np.abs(variance[use])
        coeffs = np.linalg.lstsq(design[use] * scale[:, np.newaxis], variance[use] * scale, rcond=None)[0]
        if ii == rejIterations:
            break
        fit = design @ coeffs
        with np.errstate(invalid="ignore", divide="ignore"):
            resid = (variance - fit) / fit
        rms = robustRms(resid[use])
        newUse = good & (np.abs(resid - np.median(resid[use])) < rejThreshold * rms)
        if newUse.sum() < minNumPixels or np.array_equal(newUse, use):
            break
        use = newUse

    result[:] = makeDesign(expected) @ coeffs

    # Guard against a poorly-constrained model: replace non-positive values with a constant
    bad = ~(result > 0)
    if np.any(bad):
        result[bad] = np.median(variance[good])
    return result


class ModelWeightsTask(Task):
    """Calculate inverse-variance weights from a model of the variance

    The variance in extracted spectra is estimated from the data, so pixels
    that fluctuate low have a lower variance (and pixels that fluctuate high
    have a higher variance) than they should. Weighting by the inverse of that
    variance biases weighted averages low, by of order one count per pixel.

    This task provides inverse-variance weights that are not correlated with
    the noise in each pixel, by predicting the variance from the expected
    signal (sky plus a smoothed object model) rather than the measured signal.
    The relationship between signal and variance is fit to the data for each
    spectrum, so no knowledge of the read noise, gain or extraction is
    required.

    The fit assumes that the spectra are related to detected counts by a factor
    that varies smoothly with wavelength. That's not the case for
    flux-calibrated spectra close to a dichroic, where the throughput falls
    steeply; in that case, the caller should provide that factor (``scale``)
    so the fit can be performed in units of counts.

    The input spectra are not modified.
    """

    ConfigClass = ModelWeightsConfig
    _DefaultName = "modelWeights"

    def run(
        self, spectra: "PfsFiberArraySet | PfsFiberArray", scale: np.ndarray | None = None
    ) -> np.ndarray:
        """Calculate weights for spectra

        Parameters
        ----------
        spectra : `PfsFiberArraySet` or `PfsFiberArray`
            Spectra for which to calculate weights. The ``sky`` should contain
            the sky that has been subtracted from the ``flux`` (or be zero, if
            the sky has not been subtracted).
        scale : `numpy.ndarray` of `float`, optional
            Factor converting detected counts to the units of ``spectra``
            (i.e., ``flux = counts*scale``), with the same shape as
            ``spectra.flux``. If provided, the variance is modeled in units of
            counts, which is necessary if the factor varies steeply with
            wavelength. Pixels with a non-finite or non-positive ``scale`` get
            zero weight.

        Returns
        -------
        weights : `numpy.ndarray` of `float`
            Inverse-variance weights, with the same shape as ``spectra.flux``.
            The weights are provided for masked pixels (so that they may be
            used for those pixels when nothing better is available), but are
            zero where a model cannot be calculated (e.g., non-finite values,
            or a spectrum without any good pixels).
        """
        wavelength = np.atleast_2d(spectra.wavelength)
        flux = np.atleast_2d(spectra.flux)
        sky = np.atleast_2d(spectra.sky)
        variance = np.atleast_2d(spectra.variance)
        mask = np.atleast_2d(spectra.mask)
        badBitmask = spectra.flags.get(*[mm for mm in self.config.mask if mm in spectra.flags])
        if scale is None:
            scale = np.ones_like(flux, dtype=float)
        else:
            scale = np.atleast_2d(scale)
            if scale.shape != flux.shape:
                raise RuntimeError(f"Shape mismatch between scale {scale.shape} and flux {flux.shape}")
        with np.errstate(invalid="ignore"):
            goodScale = np.isfinite(scale) & (scale > 0)
        scale = np.where(goodScale, scale, 1.0)

        weights = np.zeros_like(flux, dtype=float)
        numFailed = 0
        for ii in range(flux.shape[0]):
            with np.errstate(invalid="ignore"):
                good = (
                    ((mask[ii] & badBitmask) == 0)
                    & np.isfinite(flux[ii])
                    & np.isfinite(sky[ii])
                    & np.isfinite(variance[ii])
                    & (variance[ii] > 0)
                    & goodScale[ii]
                )
            # Model in units of counts, then convert back
            modelVariance = calculateModelVariance(
                wavelength[ii],
                flux[ii] / scale[ii],
                sky[ii] / scale[ii],
                variance[ii] / scale[ii] ** 2,
                good,
                smoothWidth=self.config.smoothWidth,
                order=self.config.order,
                rejIterations=self.config.rejIterations,
                rejThreshold=self.config.rejThreshold,
                minNumPixels=self.config.minNumPixels,
            ) * scale[ii] ** 2
            valid = np.isfinite(modelVariance) & (modelVariance > 0) & goodScale[ii]
            if not np.any(valid):
                numFailed += 1
                continue
            weights[ii][valid] = 1.0 / modelVariance[valid]

        if numFailed > 0:
            self.log.debug("Unable to calculate model weights for %d/%d spectra", numFailed, flux.shape[0])

        return weights.reshape(np.shape(spectra.flux))


def resampleWeights(
    spectra: "PfsFiberArraySet | PfsFiberArray",
    weights: np.ndarray,
    wavelength: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """Resample inverse-variance weights to a new wavelength sampling

    Resampling mixes neighboring pixels, so the variance of the resampled
    spectrum differs from the variance of the original pixels (in a manner
    that depends on the position of each new pixel relative to the original
    pixels). To get the corresponding weights for the resampled spectrum, we
    resample the variance corresponding to the weights in the same way as the
    variance of the spectrum, with the same treatment of bad pixels.

    Parameters
    ----------
    spectra : `PfsFiberArraySet` or `PfsFiberArray`
        Spectra for which the weights were calculated (at their original
        sampling). These are not modified.
    weights : `numpy.ndarray` of `float`
        Inverse-variance weights, with the same shape as ``spectra.flux``.
        Pixels with zero weight are treated as bad.
    wavelength : `numpy.ndarray` of `float`
        Wavelength array to which to resample.
    **kwargs
        Additional arguments for ``spectra.resampleVariance`` (``order``,
        ``bad``, ``minWeight``); these should be the same as those used for
        resampling the spectra.

    Returns
    -------
    resampledWeights : `numpy.ndarray` of `float`
        Inverse-variance weights for the resampled spectra. Pixels for which
        the weights can't be calculated (including those affected by pixels
        with zero weight, or by bad pixels in the spectra) have zero weight.
    """
    if weights.shape != spectra.flux.shape:
        raise RuntimeError(f"Shape mismatch between weights {weights.shape} and flux {spectra.flux.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = np.where(weights > 0, 1.0 / weights, np.nan)
    resampled = spectra.resampleVariance(wavelength, variance, numCovar=1, **kwargs)
    resampledVariance = resampled.covariance[..., 0, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        # Resampled pixels affected by bad pixels can have wild variance values: give them zero weight
        good = (resampled.mask & spectra.flags.get("NO_DATA")) == 0
        good &= np.isfinite(resampledVariance) & (resampledVariance > 0)
        return np.where(good, 1.0 / resampledVariance, 0.0)
