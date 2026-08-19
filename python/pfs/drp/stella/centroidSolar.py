import os

import numpy as np
from scipy import differentiate
from scipy.optimize import minimize_scalar

from lsst.pipe.base import Task, Struct
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.utils import getPackageDir

from pfs.datamodel.drp import PfsArm
from pfs.datamodel.pfsFiberArray import PfsFiberArray
from pfs.datamodel.pfsSimpleSpectrum import PfsSimpleSpectrum
from pfs.datamodel import PfsConfig

from .arcLine import ArcLineSet
from .datamodel import PfsSingle
from .DetectorMapContinued import DetectorMap
from .interpolate import Interpolator
from .referenceLine import ReferenceLineSource, ReferenceLineStatus
from .selectFibers import SelectFibersTask


__all__ = ("CentroidSolarConfig", "CentroidSolarTask")


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
    hessianStepFraction: float,
    hessianRtol: float,
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
    hessianStepFraction : `float`
        Fraction of the offset search range (``maxOffset - minOffset``) to use
        as the initial finite-difference step size when estimating the
        curvature of the log-likelihood (via ``scipy.differentiate.hessian``).
    hessianRtol : `float`
        Relative tolerance on the curvature estimate from
        ``scipy.differentiate.hessian``. The default tolerance used by that
        function (close to machine precision) is unattainable here because the
        log-likelihood is not perfectly smooth (see Notes), so this needs to be
        relaxed.

    Notes
    -----
    ``interpolateFlux`` uses a truncated (finite-window) Lanczos kernel, which
    is not perfectly shift-invariant: as the offset varies continuously, the
    pixel window used for interpolation shifts by integer steps of the
    template's wavelength sampling, producing tiny ripples in the
    log-likelihood on that length scale. This limits the precision with which
    its curvature can be estimated by finite differences.

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
    # remaining weight, rather than poisoning the whole output sample.
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
        coeff, *_ = np.linalg.lstsq(chi, phi, rcond=None)
        logLikelihood = 0.5*(phi @ coeff)
        logLikelihood -= 0.5*(offset/priorOffset)**2  # Gaussian prior centered on zero offset

        return logLikelihood

    # Maximize the log-likelihood
    result = minimize_scalar(
        lambda offset: -calculateLogLikelihood(offset), bounds=(minOffset, maxOffset), method="bounded"
    )
    if not result.success:
        raise RuntimeError(f"Failed to maximize log-likelihood: {result}")
    offset = result.x
    logLikelihood = -result.fun

    # Calculate the error from the curvature of the log-likelihood at its maximum.
    # ``scipy.differentiate.hessian`` requires a function that accepts an array of shape
    # (m, ...) and returns shape (...); since we're differentiating a scalar function (m=1),
    # we take the sole row and evaluate it elementwise, regardless of the shape of "..."
    # (which varies, since hessian is implemented as nested calls to jacobian).
    def vectorizedLogLikelihood(offsetArray):
        row = offsetArray[0]
        result = np.array([calculateLogLikelihood(oo) for oo in row.ravel()])
        return result.reshape(row.shape)

    # ``differentiate.hessian``'s default initial step (0.5) is on the scale of the *default*
    # search bounds, not the wavelength offsets (nm) we're actually differentiating with respect
    # to; left at its default, the finite-difference step can be so much larger than the width of
    # the likelihood peak that the estimated curvature comes out with the wrong sign. Scale it to
    # the actual offset search range instead. And because the log-likelihood has the ripples
    # described above, we use step_factor<1 so the algorithm *grows* the step on successive
    # iterations (away from the ripples) rather than its default of shrinking (into them), and we
    # relax the convergence tolerance accordingly (see ``hessianRtol`` docs).
    initialStep = hessianStepFraction*(maxOffset - minOffset)
    hessian = differentiate.hessian(
        vectorizedLogLikelihood,
        np.array([offset]),
        initial_step=initialStep,
        tolerances=dict(rtol=hessianRtol),
    )
    offsetErr = np.sqrt(-1.0/hessian.ddf[0, 0])

    if False:
        import matplotlib.pyplot as plt
        offsetArray = np.linspace(minOffset, maxOffset, 100)
        plt.plot(offsetArray, [calculateLogLikelihood(oo) for oo in offsetArray])
        plt.axvline(offset, color="k", ls="--")
        plt.axvline(offset - offsetErr, color="k", ls=":")
        plt.axvline(offset + offsetErr, color="k", ls=":")
        plt.xlabel("Wavelength Offset (nm)")
        plt.ylabel("Log-Likelihood")
        plt.title(f"Centroiding fit: offset={offset:.3f} +/- {offsetErr:.3f} nm")
        plt.show()

    if not hessian.success:
        raise RuntimeError(
            f"Failed to estimate curvature of log-likelihood at maximum: {hessian}"
        )

    return Struct(offset=offset, offsetErr=offsetErr, logLikelihood=logLikelihood, flux=flux, fluxErr=fluxErr)


class CentroidSolarConfig(Config):
    """Configuration for `CentroidSolarTask`"""
    selectFibers = ConfigurableField(target=SelectFibersTask, doc="Task to select fibers")
    wavelengths = ListField(
        dtype=float,
        default=[400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 560.0, 655.0, 859.0],
        doc="Central wavelengths to use for centroiding (nm)",
    )
    halfWidth = ListField(
        dtype=float,
        default=[10.0] * 10 + [11],
        doc="Half-width of the region to use for centroiding (nm)",
    )
    targetTemplate = Field(
        dtype=str,
        default=os.path.join(getPackageDir("drp_pfs_data"), "templates", "solar_spectrum.fits"),
        doc="Path for the LSF-convolved target (solar) spectrum template",
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
    hessianStepFraction = Field(
        dtype=float,
        default=0.02,
        doc="Fraction of the offset search range (2*maxOffset) to use as the initial "
        "finite-difference step size when estimating the curvature of the log-likelihood "
        "(used to derive offsetErr)",
    )
    hessianRtol = Field(
        dtype=float,
        default=0.01,
        doc="Relative tolerance on the curvature estimate used to derive offsetErr (see "
        "fitWavelengthOffset for why this needs to be relaxed from scipy's default)",
    )
    xErr = Field(
        dtype=float,
        default=1.0,
        doc="Uncertainty in the x (spatial) position to assign to each fitted line (pixels), "
        "since we don't measure it",
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

    def run(self, pfsArm: PfsArm, pfsConfig: PfsConfig, detectorMap: DetectorMap) -> Struct:
        """Centroid using cross-correlation of solar spectral template

        Parameters
        ----------
        pfsArm : `PfsArm`
            The PFS arm spectra.
        pfsConfig : `PfsConfig`
            Top-end fiber configuration.

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
        """
        subConfig = self.selectFibers.run(pfsConfig.select(fiberId=pfsArm.fiberId))
        targetTemplate = self.loadTargetTemplate()
        skyTemplate = self.loadSkyTemplate()

        detMapWavelength = detectorMap.getWavelength()
        minWl = detMapWavelength.min()
        maxWl = detMapWavelength.max()

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
        for ff in subConfig.fiberId:
            spectrum = pfsArm.extractFiber(PfsSingle, pfsConfig, ff)

            if False:
                import matplotlib.pyplot as plt
                plt.plot(spectrum.wavelength, spectrum.flux, "k-")
                cmap = plt.matplotlib.cm.get_cmap("rainbow")
                wavelengths = np.array(self.config.wavelengths)
                halfWidth = np.array(self.config.halfWidth)
                select = (wavelengths + halfWidth > minWl) & (wavelengths - halfWidth < maxWl)
                colors = cmap(np.linspace(0, 1, select.sum()))
                for wl, hw, col in zip(
                    wavelengths[select], halfWidth[select], colors
                ):
                    plt.axvspan(
                        wl - hw,
                        wl + hw,
                        label=str(wl),
                        color=col,
                        alpha=0.2,
                    )
                plt.legend()
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Flux (normalized)")
                plt.show()

            for centerWavelength, halfWidth in zip(self.config.wavelengths, self.config.halfWidth):
                if centerWavelength - halfWidth < minWl or centerWavelength + halfWidth > maxWl:
                    continue
                try:
                    result = self.fitTemplate(
                        spectrum, targetTemplate, skyTemplate, centerWavelength, halfWidth
                    )
                except RuntimeError as exc:
                    self.log.debug(
                        "Fitting for fiberId=%d, wavelength=%f failed: %s", ff, centerWavelength, exc
                    )
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
        )

    def loadTargetTemplate(self):
        """Load the target (solar) template spectrum

        Returns
        -------
        template : `pfs.datamodel.PfsSimpleSpectrum`
            The target template spectrum.
        """
        return PfsSimpleSpectrum.readFits(self.config.targetTemplate)

    def loadSkyTemplate(self):
        """Load the sky (night-sky emission) template spectrum

        Returns
        -------
        template : `pfs.datamodel.PfsSimpleSpectrum`
            The sky template spectrum.
        """
        return PfsSimpleSpectrum.readFits(self.config.skyTemplate)

    def fitTemplate(
        self,
        spectrum: PfsFiberArray,
        targetTemplate: PfsSimpleSpectrum,
        skyTemplate: PfsSimpleSpectrum,
        centerWavelength: float,
        halfWidth: float,
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
            self.config.hessianStepFraction,
            self.config.hessianRtol,
        )
