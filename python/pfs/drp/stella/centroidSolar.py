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
from .interpolate import interpolateFlux
from .referenceLine import ReferenceLineSource, ReferenceLineStatus
from .selectFibers import SelectFibersTask


__all__ = ("CentroidSolarConfig", "CentroidSolarTask")


def fitWavelengthOffset(
    spectrum: PfsFiberArray,
    template: PfsSimpleSpectrum,
    minWavelength: float,
    maxWavelength: float,
    spectrumMask: list[str],
    templateMask: list[str],
    resampleOrder: int,
    fluxOrder: int,
    minOffset: float,
    maxOffset: float,
    priorOffset: float,
    hessianStepFraction: float,
    hessianRtol: float,
) -> Struct:
    """Fit a wavelength offset between the observed spectrum and a template

    Parameters
    ----------
    spectrum : `PfsFiberArray`
        Observed spectrum.
    template : `PfsSimpleSpectrum`
        Template spectrum. This should already be LSF-convolved to match the
        observed spectrum.
    minWavelength, maxWavelength : `float`
        Bounds of the wavelength range to use for fitting (nm).
    spectrumMask : `list` of `str`
        Mask planes to use for masking bad pixels.
    resampleOrder : `int`
        Order of the resampling kernel to use (1 is linear, >=2 are Lanczos).
    fluxOrder : `int`
        Order of the polynomial to fit the flux ratio between the observed
        spectrum and the template.
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

    templateWavelength = template.wavelength.astype(float)
    templateFlux = template.flux.astype(float)

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
        # template's rest wavelengths, so we look up the template at the corresponding bluer wavelength.
        shiftedTemplateFlux = interpolateFlux(
            templateWavelength, templateFlux, spectrumWavelength - offset, fill=np.nan, order=resampleOrder
        )
        good = np.isfinite(shiftedTemplateFlux)
        if not np.any(good):
            return -np.inf

        # Fit the flux ratio between the observed spectrum and the template as a polynomial,
        # and calculate the profile log-likelihood of the fit (linear parameters marginalized out).
        normWavelength = (
            2*(spectrumWavelength[good] - minWavelength)/(maxWavelength - minWavelength) - 1
        )
        basis = np.polynomial.chebyshev.chebvander(normWavelength, fluxOrder)
        design = shiftedTemplateFlux[good][:, np.newaxis]*basis
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
        default=[400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 560.0, 660.0, 720, 740, 780, 800, 820, 840, 860, 880, 900],
        doc="Central wavelengths to use for centroiding (nm)",
    )
    halfWidth = ListField(
        dtype=float,
        default=[10.0] * 11,
        doc="Half-width of the region to use for centroiding (nm)",
    )
    template = Field(
        dtype=str,
        default=os.path.join(getPackageDir("drp_pfs_data"), "templates", "solar_spectrum.fits"),
        doc="Path for the LSF-convolved solar spectrum template",
    )
    mask = ListField(
        dtype=str,
        default=["BAD", "SAT", "CR", "SUSPECT", "NO_DATA"],
        doc="Mask planes to use for masking bad pixels",
    )
    resampleOrder = Field(
        dtype=int,
        default=1,
        doc="Order of the resampling kernel to use for interpolating the template "
        "(1 is linear, >=2 are Lanczos)",
    )
    fluxOrder = Field(
        dtype=int,
        default=2,
        doc="Order of the polynomial used to fit the flux ratio between the observed spectrum "
        "and the template",
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


class CentroidSolarTask(Task):
    """Centroid absorption lines in an exposure

    We fit a solar template to sky fiber spectra
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
        template = self.loadTemplate()

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
            for centerWavelength, halfWidth in zip(self.config.wavelengths, self.config.halfWidth):
                if centerWavelength - halfWidth < minWl or centerWavelength + halfWidth > maxWl:
                    continue
                try:
                    result = self.fitTemplate(spectrum, template, centerWavelength, halfWidth)
                except RuntimeError as exc:
                    self.log.debug(
                        "Fitting for fiberId=%d, wavelength=%f failed: %s", ff, centerWavelength, exc
                    )
                    continue

                good = np.isfinite(result.offset) and np.isfinite(result.offsetErr)

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

    def loadTemplate(self):
        """Load the solar template spectrum

        Returns
        -------
        template : `pfs.datamodel.PfsSimpleSpectrum`
            The solar template spectrum.
        """
        return PfsSimpleSpectrum.readFits(self.config.template)

    def fitTemplate(
        self, spectrum: PfsFiberArray, template: PfsSimpleSpectrum, centerWavelength: float, halfWidth: float
    ) -> Struct:
        """Fit the template to the observed spectrum in a wavelength window

        Parameters
        ----------
        spectrum : `PfsFiberArray`
            Observed spectrum.
        template : `PfsSimpleSpectrum`
            Template spectrum. This should already be LSF-convolved to match
            the observed spectrum.
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
            template,
            centerWavelength - halfWidth,
            centerWavelength + halfWidth,
            self.config.mask,
            self.config.mask,
            self.config.resampleOrder,
            self.config.fluxOrder,
            -self.config.maxOffset,
            self.config.maxOffset,
            self.config.priorOffset,
            self.config.hessianStepFraction,
            self.config.hessianRtol,
        )
