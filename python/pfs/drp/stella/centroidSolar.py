import os

import numpy as np
from scipy.optimize import minimize_scalar

from lsst.afw.image import VisitInfo
from lsst.pipe.base import Task, Struct
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.utils import getPackageDir

from pfs.datamodel.drp import PfsArm
from pfs.datamodel.pfsFiberArray import PfsFiberArray
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.datamodel import PfsConfig

from .arcLine import ArcLineSet
from .atmosphere import AtmosphericTransmission
from .datamodel import PfsSingle
from .DetectorMapContinued import DetectorMap
from .interpolate import Interpolator
from .lsf import LsfDict
from .referenceLine import ReferenceLineSource, ReferenceLineStatus
from .selectFibers import SelectFibersTask


__all__ = ("CentroidSolarConfig", "CentroidSolarTask")


def safeInterpolationMask(
    interpolator: Interpolator,
    bad: np.ndarray,
    wavelength: np.ndarray,
    minOffset: float,
    maxOffset: float,
    halfSize: int,
) -> np.ndarray:
    """Identify points that can be safely interpolated for any trial offset

    A point is unsafe if, for some offset within ``[minOffset, maxOffset]``,
    the interpolation kernel's window would include a masked template sample
    or reach outside the template's wavelength domain. Excluding such points
    once, up front, for the entire offset search range (rather than
    recomputing which points are usable at each trial offset) keeps the set
    of points used in the fit fixed: if it were instead recomputed at each
    trial offset, a masked sample entering or leaving the kernel window as
    the offset varies would flip that point in or out of the fit, producing
    a genuine discontinuity in the log-likelihood that corrupts the
    finite-difference curvature estimate used for ``offsetErr``.

    Parameters
    ----------
    interpolator : `Interpolator`
        Interpolator for the template.
    bad : `numpy.ndarray` of `bool`
        Bad-pixel mask for the template, aligned with its wavelength array.
    wavelength : `numpy.ndarray` of `float`
        Wavelengths (nm) of the points to be interpolated, unshifted.
    minOffset, maxOffset : `float`
        Bounds of the wavelength offset to be searched over (nm).
    halfSize : `int`
        Half-size of the interpolation kernel window (in template samples).

    Returns
    -------
    safe : `numpy.ndarray` of `bool`
        `True` for points that are safe to use at every offset in
        ``[minOffset, maxOffset]``.
    """
    loIndex = interpolator.indices(wavelength - maxOffset)
    hiIndex = interpolator.indices(wavelength - minOffset)
    inDomain = np.isfinite(loIndex) & np.isfinite(hiIndex)
    numTemplate = bad.size
    start = np.clip(np.floor(np.where(inDomain, loIndex, 0)).astype(int) - halfSize, 0, numTemplate)
    stop = np.clip(np.ceil(np.where(inDomain, hiIndex, 0)).astype(int) + halfSize + 1, 0, numTemplate)
    cumBad = np.concatenate([[0], np.cumsum(bad.astype(np.int64))])
    noBadInRange = (cumBad[stop] - cumBad[start]) == 0
    return inDomain & noBadInRange


def replaceFlux(template: PfsSimpleSpectrum, flux: np.ndarray) -> PfsSimpleSpectrum:
    """Return a shallow copy of a template spectrum with a different flux array

    Parameters
    ----------
    template : `PfsSimpleSpectrum`
        Template spectrum to copy; only its ``flux`` is replaced.
    flux : `numpy.ndarray` of `float`
        Replacement flux array, aligned with ``template.wavelength``.

    Returns
    -------
    result : `PfsSimpleSpectrum`
        Copy of ``template`` with ``flux`` replacing the original flux array.
    """
    return PfsSimpleSpectrum(
        template.target, template.wavelength, flux, template.mask, template.flags, template.metadata
    )


def fitWavelengthOffset(
    spectrum: PfsFiberArray,
    targetTemplate: PfsSimpleSpectrum,
    skyTemplate: PfsSimpleSpectrum,
    minWavelength: float,
    maxWavelength: float,
    spectrumMask: list[str],
    targetMask: list[str],
    skyMask: list[str],
    resampleOrder: int,
    targetFluxOrder: int,
    skyFluxOrder: int,
    minOffset: float,
    maxOffset: float,
    priorOffset: float,
    curvatureStepFraction: float,
    curvatureNumPoints: int,
    *,
    computeOffsetErr: bool = True,
) -> Struct:
    """Fit a wavelength offset between the observed spectrum and two templates

    The observed spectrum is modeled as a linear combination of two
    templates (e.g., a solar continuum/absorption template and a night-sky
    emission template), each with its own flux-matching polynomial, sharing a
    single wavelength offset (since the offset reflects a wavelength
    calibration error common to both).

    Parameters
    ----------
    spectrum : `PfsFiberArray`
        Observed spectrum.
    targetTemplate : `PfsSimpleSpectrum`
        Target template spectrum. This should already be LSF-convolved to
        match the observed spectrum.
    skyTemplate : `PfsSimpleSpectrum`
        Sky template spectrum. This should already be LSF-convolved to match
        the observed spectrum.
    minWavelength, maxWavelength : `float`
        Bounds of the wavelength range to use for fitting (nm).
    spectrumMask : `list` of `str`
        Mask planes to use for masking bad pixels in the observed spectrum.
    targetMask : `list` of `str`
        Mask planes to use for masking bad pixels in ``targetTemplate``.
    skyMask : `list` of `str`
        Mask planes to use for masking bad pixels in ``skyTemplate``.
    resampleOrder : `int`
        Order of the resampling kernel to use (1 is linear, >=2 are Lanczos).
    targetFluxOrder : `int`
        Order of the polynomial to fit the flux ratio between the observed
        spectrum and ``targetTemplate``.
    skyFluxOrder : `int`
        Order of the polynomial to fit the flux ratio between the observed
        spectrum and ``skyTemplate``.
    minOffset, maxOffset : `float`
        Bounds of the wavelength offset to search for (nm).
    priorOffset : `float`
        Width (sigma) of the Gaussian prior on the wavelength offset (nm),
        centered on zero offset.
    curvatureStepFraction : `float`
        Fraction of the offset search range (``maxOffset - minOffset``) to use
        as the spacing between points sampled around the maximum when
        estimating the curvature of the log-likelihood (see Notes).
    curvatureNumPoints : `int`
        Number of points (spaced by ``curvatureStepFraction*(maxOffset -
        minOffset)``) to sample around the maximum for the curvature estimate.
    computeOffsetErr : `bool`, optional
        Estimate ``offsetErr`` from the log-likelihood curvature, raising if
        that curvature is non-negative at the maximum? Set `False` to skip
        this (returning ``offsetErr=nan`` unconditionally) when only
        ``logLikelihood`` is needed and a marginal/positive curvature
        shouldn't invalidate an otherwise-valid fit -- e.g., when summing
        ``logLikelihood`` across many (fiber, window) pairs for an outer,
        non-linear fit of a parameter (such as atmospheric PWV) shared across
        the whole exposure, where excluding whichever pairs happen to have
        ill-determined offset uncertainties at a given trial would bias the
        sum toward trial values that happen to leave more pairs well-behaved.

    Notes
    -----
    ``interpolateFlux`` uses a truncated (finite-window) Lanczos kernel, which
    is not perfectly shift-invariant: as the offset varies continuously, the
    pixel window used for interpolation shifts by integer steps of the
    template's wavelength sampling, producing tiny ripples in the
    log-likelihood on that length scale. This limits the precision with which
    its curvature can be estimated from finite differences, which is why the
    curvature is instead estimated from a least-squares parabola fit to
    several points around the maximum (see ``curvatureStepFraction`` and
    ``curvatureNumPoints``): this averages over the ripples, rather than
    differencing across them.

    Returns
    -------
    offset : `float`
        The fitted wavelength offset (nm).
    offsetErr : `float`
        The uncertainty in the fitted wavelength offset (nm).
    logLikelihood : `float`
        The maximum log-likelihood of the fit.
    flux : `float`
        The mean flux of the observed spectrum over the fitting window.
    fluxErr : `float`
        The uncertainty in the mean flux of the observed spectrum over the
        fitting window.
    """
    # Compute in double precision throughout: spectrum/template flux and variance are typically
    # stored as float32, and the resulting quantization noise is small in absolute terms but is
    # large enough, relative to the tiny steps used to numerically differentiate the log-likelihood
    # below, to corrupt the curvature (and hence the offset uncertainty) estimate.
    select = (spectrum.wavelength >= minWavelength) & (spectrum.wavelength <= maxWavelength)
    select &= (spectrum.mask & spectrum.flags.get(*spectrumMask)) == 0
    spectrumWavelength = spectrum.wavelength[select]
    spectrumFlux = spectrum.flux[select].astype(float)
    spectrumVariance = spectrum.variance[select].astype(float)
    spectrumMask = spectrum.mask[select]

    flux = np.mean(spectrumFlux) if spectrumFlux.size > 0 else np.nan
    fluxErr = np.sqrt(np.sum(spectrumVariance))/spectrumFlux.size if spectrumFlux.size > 0 else np.nan

    targetWavelength = targetTemplate.wavelength.astype(float)
    skyWavelength = skyTemplate.wavelength.astype(float)

    # Identify bad template pixels; these are passed through to `Interpolator.interpolateFlux` below
    # (as `fromMask`) so it can exclude them from its interpolation kernel and renormalize over the
    # remaining weight, rather than poisoning the whole output sample. (An earlier version of this
    # code instead set bad samples to NaN and relied on that poisoning a plain floating-point kernel
    # sum, but that makes a single bad template pixel near the edge of the offset search range flip
    # the interpolated flux discretely between a finite value and NaN as the offset crosses the
    # sample's position -- a real jump in the log-likelihood, not just numerical noise, that
    # corrupts the finite-difference curvature estimate used for `offsetErr` below.)
    targetBad = (targetTemplate.mask & targetTemplate.flags.get(*targetMask)) != 0
    targetFlux = targetTemplate.flux.astype(float)
    skyBad = (skyTemplate.mask & skyTemplate.flags.get(*skyMask)) != 0
    skyFlux = skyTemplate.flux.astype(float)

    # The two templates are typically stored in very different absolute flux units (e.g., a solar
    # spectrum in physical flux units vs. a sky-emission template many orders of magnitude fainter),
    # so their design matrix columns can differ in scale enormously. Left unscaled, this drives the
    # normal-equations matrix `chi` (below) to a condition number well past double precision's
    # useful range, corrupting both the fit and the finite-difference curvature estimate used for
    # `offsetErr`. Rescaling each template to order-unity flux fixes the conditioning without
    # changing the fit result: the linear flux-scaling coefficients solved for below simply absorb
    # the rescaling. We use the RMS flux within the fitting window rather than the median: the sky
    # template is emission lines on a zero background, so most samples are zero/near-zero and the
    # median would be dominated by that background instead of reflecting the lines' amplitude.
    targetWindow = (targetWavelength >= minWavelength) & (targetWavelength <= maxWavelength) & ~targetBad
    skyWindow = (skyWavelength >= minWavelength) & (skyWavelength <= maxWavelength) & ~skyBad
    targetScale = np.sqrt(np.mean(targetFlux[targetWindow]**2)) if np.any(targetWindow) else np.nan
    skyScale = np.sqrt(np.mean(skyFlux[skyWindow]**2)) if np.any(skyWindow) else np.nan
    if not np.isfinite(targetScale) or targetScale == 0:
        targetScale = 1.0
    if not np.isfinite(skyScale) or skyScale == 0:
        skyScale = 1.0
    targetFlux = targetFlux/targetScale
    skyFlux = skyFlux/skyScale

    # Constructing the Interpolator is much more expensive than evaluating it (the templates are
    # high-resolution spectra with ~1e6 points), so build them once here and reuse them for every
    # evaluation of the log-likelihood below (of which there are many: both the offset optimizer and
    # the curvature estimate call it repeatedly), rather than rebuilding them on every evaluation.
    targetInterpolator = Interpolator(targetWavelength)
    skyInterpolator = Interpolator(skyWavelength)

    # Permanently exclude points whose interpolation kernel window could ever include a masked
    # template sample (or run outside the template's domain) for some offset in the search range;
    # see `safeInterpolationMask` for why this must be fixed once here rather than recomputed at
    # each trial offset.
    kernelHalfSize = 1 if resampleOrder <= 1 else resampleOrder
    targetSafe = safeInterpolationMask(
        targetInterpolator, targetBad, spectrumWavelength, minOffset, maxOffset, kernelHalfSize
    )
    skySafe = safeInterpolationMask(
        skyInterpolator, skyBad, spectrumWavelength, minOffset, maxOffset, kernelHalfSize
    )
    safe = targetSafe & skySafe
    spectrumWavelength = spectrumWavelength[safe]
    spectrumFlux = spectrumFlux[safe]
    spectrumVariance = spectrumVariance[safe]

    def calculateLogLikelihood(offset):
        """Calculate the log-likelihood of the fit for a given wavelength offset

        Parameters
        ----------
        offset : `float`
            Wavelength offset (nm).

        Returns
        -------
        logLikelihood : `float`
            The log-likelihood of the fit.
        """
        # A positive offset means the observed spectrum's features are redshifted relative to the
        # templates' rest wavelengths, so we look up the templates at the corresponding bluer
        # wavelength. Both templates share the same offset: it reflects a wavelength calibration
        # error common to both, not a property of either source.
        shiftedTargetFlux = targetInterpolator.interpolateFlux(
            targetFlux, spectrumWavelength - offset, fill=np.nan, order=resampleOrder, fromMask=targetBad
        )
        shiftedSkyFlux = skyInterpolator.interpolateFlux(
            skyFlux, spectrumWavelength - offset, fill=np.nan, order=resampleOrder, fromMask=skyBad
        )
        good = np.isfinite(shiftedTargetFlux) & np.isfinite(shiftedSkyFlux)
        if not np.any(good):
            return -np.inf

        # Fit the flux ratio between the observed spectrum and each template as a polynomial (with
        # its own order), linearly combine the two templates, and calculate the profile
        # log-likelihood of the fit (linear parameters marginalized out).
        normWavelength = (
            2*(spectrumWavelength[good] - minWavelength)/(maxWavelength - minWavelength) - 1
        )
        targetBasis = np.polynomial.chebyshev.chebvander(normWavelength, targetFluxOrder)
        skyBasis = np.polynomial.chebyshev.chebvander(normWavelength, skyFluxOrder)
        design = np.hstack([
            shiftedTargetFlux[good][:, np.newaxis]*targetBasis,
            shiftedSkyFlux[good][:, np.newaxis]*skyBasis,
        ])
        weight = 1.0/spectrumVariance[good]
        phi = design.T @ (weight*spectrumFlux[good])
        chi = design.T @ (weight[:, np.newaxis]*design)
        # Ridge-regularize: if a component contributes ~0 signal in this window, chi is
        # near-singular, and np.linalg.lstsq's automatic rank threshold flips in/out discretely as
        # offset varies infinitesimally, injecting jumps into the log-likelihood that corrupt the
        # finite-difference curvature (offsetErr) estimate below. A small relative Tikhonov term
        # keeps chi invertible and the log-likelihood smooth without measurably biasing fits where
        # all components are well constrained.
        chi += 1.0e-8*np.diag(np.diag(chi))
        coeff = np.linalg.solve(chi, phi)
        logLikelihood = 0.5*(phi @ coeff)
        logLikelihood -= 0.5*(offset/priorOffset)**2  # Gaussian prior centered on zero offset

        return logLikelihood

    # Maximize the log-likelihood
    result = minimize_scalar(
        lambda offset: -calculateLogLikelihood(offset), bounds=(minOffset, maxOffset), method="bounded"
    )
    offset = result.x
    logLikelihood = -result.fun

    if False:
        import matplotlib.pyplot as plt

        shiftedTargetFlux = targetInterpolator.interpolateFlux(
            targetFlux, spectrumWavelength - offset, fill=np.nan, order=resampleOrder, fromMask=targetBad
        )
        shiftedSkyFlux = skyInterpolator.interpolateFlux(
            skyFlux, spectrumWavelength - offset, fill=np.nan, order=resampleOrder, fromMask=skyBad
        )
        good = np.isfinite(shiftedTargetFlux) & np.isfinite(shiftedSkyFlux)
        normWavelength = (
            2*(spectrumWavelength[good] - minWavelength)/(maxWavelength - minWavelength) - 1
        )
        targetBasis = np.polynomial.chebyshev.chebvander(normWavelength, targetFluxOrder)
        skyBasis = np.polynomial.chebyshev.chebvander(normWavelength, skyFluxOrder)
        design = np.hstack([
            shiftedTargetFlux[good][:, np.newaxis]*targetBasis,
            shiftedSkyFlux[good][:, np.newaxis]*skyBasis,
        ])
        weight = 1.0/spectrumVariance[good]
        phi = design.T @ (weight*spectrumFlux[good])
        chi = design.T @ (weight[:, np.newaxis]*design)
        chi += 1.0e-8*np.diag(np.diag(chi))
        coeff = np.linalg.solve(chi, phi)
        targetCoeff = coeff[:targetFluxOrder + 1]
        skyCoeff = coeff[targetFluxOrder + 1:]
        targetModel = (shiftedTargetFlux[good][:, np.newaxis]*targetBasis) @ targetCoeff
        skyModel = (shiftedSkyFlux[good][:, np.newaxis]*skyBasis) @ skyCoeff
        model = targetModel + skyModel

        plt.errorbar(
            spectrumWavelength[good], spectrumFlux[good], yerr=np.sqrt(spectrumVariance[good]),
            fmt="k.", label="Observed", alpha=0.5,
        )
        plt.plot(spectrumWavelength[good], targetModel, "b--", label="Target component")
        plt.plot(spectrumWavelength[good], skyModel, "g--", label="Sky component")
        plt.plot(spectrumWavelength[good], model, "r-", label="Best-fit model")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Flux")
        plt.title(f"Centroiding fit: offset={offset:.3f} nm ({'SUCCESS' if result.success else 'FAILED'})")
        plt.legend()
        plt.show()

    if not result.success:
        raise RuntimeError(f"Failed to maximize log-likelihood: {result}")

    if computeOffsetErr:
        # Estimate the offset uncertainty from the curvature of the log-likelihood at its maximum, by
        # fitting a parabola (least squares) to several points spanning a modest range around the
        # maximum, rather than differencing the function at just two or three points (e.g., via
        # ``scipy.differentiate.hessian``). ``interpolateFlux``'s kernel window shifts by integer steps
        # of the template's wavelength sampling as the offset varies continuously (see Notes), producing
        # small-scale ripples in the log-likelihood; a finite-difference estimate is sensitive to exactly
        # these ripples; and an adaptive step-growth strategy intended to average over them can grow the
        # step so far that it leaves the locally-quadratic region around the peak altogether, sometimes
        # returning a wrong-signed curvature. A least-squares parabola fit over several points instead
        # averages over the ripples directly, and -- since it only ever evaluates the fixed set of points
        # below -- can't run away like an adaptive algorithm can.
        curvatureStep = curvatureStepFraction*(maxOffset - minOffset)
        curvatureRelOffset = curvatureStep*np.linspace(
            -(curvatureNumPoints - 1)/2, (curvatureNumPoints - 1)/2, curvatureNumPoints
        )
        curvatureOffset = np.clip(offset + curvatureRelOffset, minOffset, maxOffset)
        curvatureLogLikelihood = np.array([calculateLogLikelihood(oo) for oo in curvatureOffset])
        quadratic = np.polyfit(curvatureOffset - offset, curvatureLogLikelihood, 2)
        curvature = 2*quadratic[0]

        if False:
            import matplotlib.pyplot as plt
            offsetArray = np.linspace(minOffset, maxOffset, 100)
            plt.plot(offsetArray, [calculateLogLikelihood(oo) for oo in offsetArray])
            plt.plot(curvatureOffset, curvatureLogLikelihood, "o")
            plt.axvline(offset, color="k", ls="--")
            plt.xlabel("Wavelength Offset (nm)")
            plt.ylabel("Log-Likelihood")
            plt.title(f"Centroiding fit: offset={offset:.3f} nm, curvature={curvature:.3e}")
            plt.show()

        if not curvature < 0:
            raise RuntimeError(
                f"Log-likelihood curvature is non-negative at its maximum (offset={offset}): {curvature}"
            )
        offsetErr = np.sqrt(-1.0/curvature)
    else:
        offsetErr = np.nan

    return Struct(offset=offset, offsetErr=offsetErr, logLikelihood=logLikelihood, flux=flux, fluxErr=fluxErr)


class CentroidSolarConfig(Config):
    """Configuration for `CentroidSolarTask`"""
    selectFibers = ConfigurableField(target=SelectFibersTask, doc="Task to select fibers")
    wavelengths = ListField(
        dtype=float,
        default=[
            # Blue arm, hitting all those lovely solar absorption lines
            400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 560.0,
            # Red/MR arm
            655.0, 859.0, # Halpha and Calcium triplet
            735.0, 795.0, 836.0, 890.0, 945.0, # Sky lines, avoiding telluric bands
        ],
        doc="Central wavelengths to use for centroiding (nm)",
    )
    halfWidth = ListField(
        dtype=float,
        default=(
            [10.0] * 9  # Blue arm
            + [10, 11, 15, 15, 10, 15, 15]  # Red/MR arm
        ),
        doc="Half-width of the region to use for centroiding (nm)",
    )
    targetTemplate = Field(
        dtype=str,
        default=os.path.join(getPackageDir("drp_pfs_data"), "templates", "solar_spectrum.fits"),
        doc="Path for the raw (pre-LSF-convolution) target (solar) spectrum template; "
        "it is LSF-convolved at run time (see the ``lsf`` parameter of ``run``), "
        "after multiplication by the atmospheric transmission if ``doFitAtmosphere`` is set",
    )
    skyTemplate = Field(
        dtype=str,
        default=os.path.join(getPackageDir("drp_pfs_data"), "templates", "sky_spectrum.fits"),
        doc="Path for the LSF-convolved sky (night-sky emission) spectrum template",
    )
    mask = ListField(
        dtype=str,
        default=["BAD", "SAT", "CR", "SUSPECT", "NO_DATA"],
        doc="Mask planes to use for masking bad pixels in the observed spectrum",
    )
    targetMask = ListField(
        dtype=str,
        default=["NO_DATA"],
        doc="Mask planes to use for masking bad pixels in the target template",
    )
    skyMask = ListField(
        dtype=str,
        default=["BAD", "SAT", "CR", "SUSPECT", "NO_DATA"],
        doc="Mask planes to use for masking bad pixels in the sky template",
    )
    resampleOrder = Field(
        dtype=int,
        default=1,
        doc="Order of the resampling kernel to use for interpolating the templates "
        "(1 is linear, >=2 are Lanczos)",
    )
    targetFluxOrder = Field(
        dtype=int,
        default=2,
        doc="Order of the polynomial used to fit the flux ratio between the observed spectrum "
        "and the target template",
    )
    skyFluxOrder = Field(
        dtype=int,
        default=2,
        doc="Order of the polynomial used to fit the flux ratio between the observed spectrum "
        "and the sky template",
    )
    maxOffset = Field(
        dtype=float,
        default=0.1,
        doc="Maximum wavelength offset to search for, +/- (nm)",
    )
    priorOffset = Field(
        dtype=float,
        default=0.05,
        doc="Width (sigma) of the Gaussian prior on the wavelength offset, centered on zero (nm)",
    )
    curvatureStepFraction = Field(
        dtype=float,
        default=0.02,
        doc="Fraction of the offset search range (2*maxOffset) to use as the spacing between "
        "points sampled around the maximum when estimating the curvature of the log-likelihood "
        "(used to derive offsetErr)",
    )
    curvatureNumPoints = Field(
        dtype=int,
        default=7,
        doc="Number of points to sample around the maximum for the curvature estimate (see "
        "curvatureStepFraction)",
    )
    xErr = Field(
        dtype=float,
        default=1.0,
        doc="Uncertainty in the x (spatial) position to assign to each fitted line (pixels), "
        "since we don't measure it",
    )
    doFitAtmosphere = Field(
        dtype=bool,
        default=True,
        doc="Fit a shared precipitable water vapor (PWV) parameter, and multiply the atmospheric "
        "transmission it implies into the target (solar) template, before LSF convolution?",
    )
    atmosphereTransmission = Field(
        dtype=str,
        default=os.path.join(getPackageDir("drp_pfs_data"), "atmosphere", "pfs_atmosphere.fits"),
        doc="Path for the atmospheric transmission model FITS grid (see AtmosphericTransmission.fromFits)",
    )
    pwvMin = Field(
        dtype=float,
        default=0.0,
        doc="Minimum precipitable water vapor to consider when fitting (mm); clamped to the range "
        "spanned by the atmosphereTransmission grid",
    )
    pwvMax = Field(
        dtype=float,
        default=30.0,
        doc="Maximum precipitable water vapor to consider when fitting (mm); clamped to the range "
        "spanned by the atmosphereTransmission grid",
    )
    priorPwv = Field(
        dtype=float,
        default=None,
        optional=True,
        doc="Width (sigma) of an optional Gaussian prior on PWV, centered on zero (mm); "
        "a numerical-stability aid only, since PWV's own accuracy isn't of interest "
        "(just its consistency across the exposure)",
    )

    def setDefaults(self):
        super().setDefaults()
####        self.selectFibers.targetType = ["SKY"]  # Don't want contamination from bright science targets

    def validate(self):
        super().validate()
        if len(self.wavelengths) != len(self.halfWidth):
            raise RuntimeError(
                f"Length mismatch between wavelengths ({len(self.wavelengths)} "
                f"and halfWidth ({len(self.halfWidth)})"
            )


class CentroidSolarTask(Task):
    """Centroid absorption lines in an exposure

    We fit a linear combination of a target (solar) template and a sky
    (night-sky emission) template to sky fiber spectra.
    """

    ConfigClass = CentroidSolarConfig
    _DefaultName = "centroidSolar"

    selectFibers: SelectFibersTask

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("selectFibers")

    def run(
        self,
        pfsArm: PfsArm,
        pfsConfig: PfsConfig,
        detectorMap: DetectorMap,
        lsf: LsfDict,
        visitInfo: VisitInfo = None,
    ) -> Struct:
        """Centroid using cross-correlation of solar spectral template

        Parameters
        ----------
        pfsArm : `PfsArm`
            The PFS arm spectra.
        pfsConfig : `PfsConfig`
            Top-end fiber configuration.
        detectorMap : `DetectorMap`
            Mapping between fiberId,wavelength and detector coordinates.
        lsf : `LsfDict`
            Line-spread functions, indexed by fiberId. Used to LSF-convolve the
            (raw, pre-convolution) target template at run time; see
            ``CentroidSolarConfig.targetTemplate``.
        visitInfo : `VisitInfo`, optional
            Visit information for the exposure being centroided, used to
            determine the zenith distance. Required if
            ``config.doFitAtmosphere`` is set; the caller should supply the
            exposure's own ``visitInfo``, not ``detectorMap.visitInfo`` (which
            may be a stale, calibration-time value).

        Returns
        -------
        results : `Struct`
            The results of the centroiding, including the following fields:
            - `lines` (`ArcLineSet`): The template fitting results, presented as
              a set of arc lines.
            - `offset` (`numpy.ndarray` of `float`): The fitted wavelength offsets
              for each fiber (nm).
            - `offsetErr` (`numpy.ndarray` of `float`): The uncertainties in the
              fitted wavelength offsets for each fiber (nm).
            - `logLikelihood` (`numpy.ndarray` of `float`): The maximum log-likelihoods of
              the fits for each fiber.
            - `pwv` (`float`): The fitted precipitable water vapor (mm), or `nan`
              if ``config.doFitAtmosphere`` is not set. This is a nuisance
              parameter (its own uncertainty isn't estimated); what matters is
              that a single value is shared across the whole exposure.
        """
        subConfig = self.selectFibers.run(pfsConfig.select(fiberId=pfsArm.fiberId))
        self.log.debug("Selected %d fibers for solar centroiding", len(subConfig.fiberId))
        targetTemplate = self.loadTargetTemplate()
        skyTemplate = self.loadSkyTemplate()
        self.log.debug(
            "Loaded target template (%d points) and sky template (%d points)",
            len(targetTemplate.wavelength), len(skyTemplate.wavelength),
        )

        detMapWavelength = detectorMap.getWavelength()
        minWl = detMapWavelength.min()
        maxWl = detMapWavelength.max()

        spectra = {ff: pfsArm.extractFiber(PfsSingle, pfsConfig, ff) for ff in subConfig.fiberId}
        windows = [
            (centerWavelength, halfWidth)
            for centerWavelength, halfWidth in zip(self.config.wavelengths, self.config.halfWidth)
            if centerWavelength - halfWidth >= minWl and centerWavelength + halfWidth <= maxWl
        ]
        self.log.debug(
            "Fitting %d fibers x %d wavelength windows = %d (fiber, window) pairs",
            len(subConfig.fiberId), len(windows), len(subConfig.fiberId)*len(windows),
        )

        # Re-express each fiber's LSF on the (high-resolution) target template's own native
        # wavelength grid, once -- this doesn't depend on PWV, so it's never repeated per trial.
        warpedLsf = {
            ff: lsf[ff].warp(detectorMap.getWavelength(ff), targetTemplate.wavelength)
            for ff in subConfig.fiberId
        }
        # Fibers sharing the same LSF (the common case today) share a single convolution per PWV
        # trial below, instead of paying for one convolution per fiber per trial.
        widthKey = {ff: getattr(warpedLsf[ff], "width", None) for ff in subConfig.fiberId}
        representative = {}
        for ff, key in widthKey.items():
            if key is not None and key not in representative:
                representative[key] = warpedLsf[ff]
        self.log.debug(
            "Grouped %d fibers into %d distinct LSF widths (%d fibers ungrouped)",
            len(subConfig.fiberId), len(representative),
            sum(1 for key in widthKey.values() if key is None),
        )

        transmissionInterpolator = None
        pwvBounds = None
        if self.config.doFitAtmosphere:
            if visitInfo is None:
                raise RuntimeError("visitInfo is required when config.doFitAtmosphere is set")
            atmosphere = AtmosphericTransmission.fromFits(self.config.atmosphereTransmission)
            zd = 90.0 - visitInfo.getBoresightAzAlt().getLatitude().asDegrees()
            transmissionInterpolator = atmosphere.makeInterpolator(zd, targetTemplate.wavelength)
            pwvBounds = (
                max(self.config.pwvMin, atmosphere.pwv[0]),
                min(self.config.pwvMax, atmosphere.pwv[-1]),
            )
            self.log.debug("Zenith distance=%f deg; searching PWV in %s mm", zd, pwvBounds)

        def computeTargetFlux(pwv):
            """Compute this trial's transmission-attenuated, LSF-convolved target flux

            Returns a pair of dicts (by width-group key, and a per-fiberId fallback
            for fibers whose LSF doesn't expose a ``width``), either of which
            ``targetTemplateFor`` can look up from.
            """
            transmission = transmissionInterpolator(pwv) if pwv is not None else 1.0
            attenuated = targetTemplate.flux*transmission
            fluxByWidth = {key: rep.convolve(attenuated) for key, rep in representative.items()}
            fluxByFiber = {
                ff: warpedLsf[ff].convolve(attenuated) for ff, key in widthKey.items() if key is None
            }
            return fluxByWidth, fluxByFiber

        def targetTemplateFor(fluxByWidth, fluxByFiber, ff):
            key = widthKey[ff]
            flux = fluxByWidth[key] if key is not None else fluxByFiber[ff]
            return replaceFlux(targetTemplate, flux)

        def fitAll(pwv, *, computeOffsetErr=True):
            """Fit every (fiber, window) pair for a trial PWV, returning the results

            A (fiber, window) pair that fails to fit (raises `RuntimeError`) is
            reported as `None`; it is simply omitted from that trial's total
            log-likelihood in the outer PWV objective, rather than being
            permanently excluded from every trial.

            ``computeOffsetErr=False`` (used by the outer PWV objective) skips
            the offset-uncertainty/curvature estimate, which can spuriously
            fail (raising `RuntimeError`) even when the fitted log-likelihood
            itself is perfectly valid; since that failure rate is itself
            PWV-dependent, letting it exclude items would bias the summed
            log-likelihood toward whichever trial PWV happens to keep more
            items "valid", rather than whichever fits best.
            """
            self.log.debug("Fitting all (fiber, window) pairs for trial pwv=%s", pwv)
            fluxByWidth, fluxByFiber = computeTargetFlux(pwv)
            results = []
            for ii, ff in enumerate(subConfig.fiberId):
                spectrum = spectra[ff]
                fiberTargetTemplate = targetTemplateFor(fluxByWidth, fluxByFiber, ff)
                numGood = 0
                for centerWavelength, halfWidth in windows:
                    try:
                        result = self.fitTemplate(
                            spectrum, fiberTargetTemplate, skyTemplate, centerWavelength, halfWidth,
                            computeOffsetErr=computeOffsetErr,
                        )
                    except RuntimeError as exc:
                        self.log.debug(
                            "Fitting for fiberId=%d, wavelength=%f failed: %s", ff, centerWavelength, exc
                        )
                        result = None
                    else:
                        numGood += 1
                    results.append((ff, centerWavelength, result))
                self.log.debug(
                    "Fiber %d/%d (fiberId=%d): %d/%d windows fit successfully",
                    ii + 1, len(subConfig.fiberId), ff, numGood, len(windows),
                )
            return results

        pwv = np.nan
        if self.config.doFitAtmosphere:
            def objective(trialPwv):
                total = 0.0
                numGood = 0
                numItems = 0
                for _, _, result in fitAll(trialPwv, computeOffsetErr=False):
                    numItems += 1
                    if result is None:
                        continue
                    total += result.logLikelihood
                    numGood += 1
                if numGood == 0:
                    self.log.debug("PWV trial: pwv=%f mm, no successful fits", trialPwv)
                    return np.inf
                if self.config.priorPwv is not None:
                    total -= 0.5*(trialPwv/self.config.priorPwv)**2
                self.log.debug(
                    "PWV trial: pwv=%f mm, logLikelihood=%f (%d/%d fits succeeded)",
                    trialPwv, total, numGood, numItems,
                )
                return -total

            self.log.debug("Starting outer PWV search over bounds=%s", pwvBounds)
            fitResult = minimize_scalar(objective, bounds=pwvBounds, method="bounded")
            if not fitResult.success:
                raise RuntimeError(f"Failed to fit PWV: {fitResult}")
            pwv = fitResult.x
            self.log.info("Fitted PWV=%f mm", pwv)

        self.log.debug("Running final fit pass at pwv=%s to measure offsets", pwv)
        fiberId = []
        wavelength = []
        xList = []
        yList = []
        yErrList = []
        fluxList = []
        fluxErrList = []
        flag = []
        offset = []
        offsetErr = []
        logLikelihood = []
        for ff, centerWavelength, result in fitAll(pwv if self.config.doFitAtmosphere else None):
            if result is None:
                continue
            good = np.isfinite(result.offset) and np.isfinite(result.offsetErr)
            self.log.debug("Fitting for fiberId=%d, wavelength=%f: %s", ff, centerWavelength, result)

            point = detectorMap.findPoint(ff, centerWavelength)
            dispersion = detectorMap.getDispersion(ff, centerWavelength)  # nm/pixel

            fiberId.append(ff)
            wavelength.append(centerWavelength)
            xList.append(point.getX())
            yList.append(point.getY() + result.offset/dispersion)
            yErrList.append(result.offsetErr/dispersion)
            fluxList.append(result.flux)
            fluxErrList.append(result.fluxErr)
            flag.append(not good)
            offset.append(result.offset)
            offsetErr.append(result.offsetErr)
            logLikelihood.append(result.logLikelihood)

        num = len(fiberId)
        empty = np.full(num, np.nan)
        lines = ArcLineSet.fromColumns(
            fiberId=np.array(fiberId, dtype=np.int32),
            wavelength=np.array(wavelength, dtype=float),
            x=np.array(xList, dtype=float),
            y=np.array(yList, dtype=float),
            xErr=np.full(num, self.config.xErr),
            yErr=np.array(yErrList, dtype=float),
            xx=empty, yy=empty, xy=empty,
            flux=np.array(fluxList, dtype=float),
            fluxErr=np.array(fluxErrList, dtype=float),
            fluxNorm=empty,
            flag=np.array(flag, dtype=bool),
            status=np.full(num, ReferenceLineStatus.GOOD.value, dtype=np.int32),
            description=["solar"]*num,
            transition=["UNKNOWN"]*num,
            source=np.full(num, ReferenceLineSource.MANUAL.value, dtype=np.int32),
        )

        self.log.info("Measured %d/%d solar centroids", (~lines.flag).sum(), len(lines))

        return Struct(
            lines=lines,
            offset=np.array(offset, dtype=float),
            offsetErr=np.array(offsetErr, dtype=float),
            logLikelihood=np.array(logLikelihood, dtype=float),
            pwv=pwv,
        )

    def loadTargetTemplate(self):
        """Load the target (solar) template spectrum

        Returns
        -------
        template : `pfs.datamodel.PfsSimpleSpectrum`
            The target template spectrum.
        """
        targetTemplate = PfsSimpleSpectrum.readFits(self.config.targetTemplate)
        if False:
            import matplotlib.pyplot as plt
            plt.plot(targetTemplate.wavelength, targetTemplate.flux, "k-")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Flux (arbitrary)")
            plt.title("Target template")
            plt.show()
        return targetTemplate

    def loadSkyTemplate(self):
        """Load the sky (night-sky emission) template spectrum

        Returns
        -------
        template : `pfs.datamodel.PfsSimpleSpectrum`
            The sky template spectrum.
        """
        skyTemplate = PfsSimpleSpectrum.readFits(self.config.skyTemplate)
        if False:
            import matplotlib.pyplot as plt
            plt.plot(skyTemplate.wavelength, skyTemplate.flux, "k-")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Flux (arbitrary)")
            plt.title("Sky template")
            plt.show()
        return skyTemplate

    def fitTemplate(
        self,
        spectrum: PfsFiberArray,
        targetTemplate: PfsSimpleSpectrum,
        skyTemplate: PfsSimpleSpectrum,
        centerWavelength: float,
        halfWidth: float,
        *,
        computeOffsetErr: bool = True,
    ) -> Struct:
        """Fit the templates to the observed spectrum in a wavelength window

        Parameters
        ----------
        spectrum : `PfsFiberArray`
            Observed spectrum.
        targetTemplate : `PfsSimpleSpectrum`
            Target template spectrum. This should already be LSF-convolved to
            match the observed spectrum.
        skyTemplate : `PfsSimpleSpectrum`
            Sky template spectrum. This should already be LSF-convolved to
            match the observed spectrum.
        centerWavelength : `float`
            Central wavelength of the fitting window (nm).
        halfWidth : `float`
            Half-width of the fitting window (nm).
        computeOffsetErr : `bool`, optional
            Estimate ``offsetErr`` from the log-likelihood curvature? See
            `fitWavelengthOffset`.

        Returns
        -------
        offset : `float`
            The fitted wavelength offset (nm).
        offsetErr : `float`
            The uncertainty in the fitted wavelength offset (nm).
        logLikelihood : `float`
            The maximum log-likelihood of the fit.
        flux : `float`
            The mean flux of the observed spectrum over the fitting window.
        fluxErr : `float`
            The uncertainty in the mean flux of the observed spectrum over the
            fitting window.
        """
        return fitWavelengthOffset(
            spectrum,
            targetTemplate,
            skyTemplate,
            centerWavelength - halfWidth,
            centerWavelength + halfWidth,
            self.config.mask,
            self.config.targetMask,
            self.config.skyMask,
            self.config.resampleOrder,
            self.config.targetFluxOrder,
            self.config.skyFluxOrder,
            -self.config.maxOffset,
            self.config.maxOffset,
            self.config.priorOffset,
            self.config.curvatureStepFraction,
            self.config.curvatureNumPoints,
            computeOffsetErr=computeOffsetErr,
        )
