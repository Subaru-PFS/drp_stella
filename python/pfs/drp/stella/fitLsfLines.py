"""Fit the line-spread function to individual arc lines in extracted spectra

Each selected arc line is fit, in pixel space, with the disk (x) Edgeworth
profile of `pfs.drp.stella.lsfProfile.diskProfile`: the fibre face, a disk of
radius ``radius``, blurred by a kernel of rms ``blur`` with standardised
cumulants ``S, K, ...`` (up to ``blurOrder``). The fit is a variable
projection: for fixed ``(mu, radius, blur, cumulants)`` the model is linear in
the line's amplitude, the amplitudes of catalogue companions within the window
(same profile, at their catalogue offsets) and a B-spline continuum, which are
solved by weighted least squares, while the nonlinear parameters are optimised
on the outside.

These are the per-line measurements from which `pfs.drp.stella.pipelines.FitLsfTask`
builds an `pfs.drp.stella.lsfModel.ArmLsfModel`.
"""
import concurrent.futures
import multiprocessing
import os
import types
from dataclasses import dataclass

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.sparse

from .lsfProfile import MAX_BLUR_ORDER, diskProfile, diskSigma

__all__ = ("DiskLineFit", "fitDiskLine", "selectLsfLines", "findCompanions", "measureLsf", "measureLsfMany",
           "lineCentroids", "clippedMean", "centroidZeroPoint", "pinnedCentres", "findBroadLines")


def continuumBasis(lam, lamLo, lamHi, knotSpacing, order=3):
    """Sparse B-spline design matrix evaluated at ``lam``

    Only values of ``lam`` in [lamLo, lamHi] are supported; other rows are zero.

    Parameters
    ----------
    lam : `numpy.ndarray`
        Positions at which to evaluate the basis.
    lamLo, lamHi : `float`
        Range covered by the basis.
    knotSpacing : `float`
        (Approximate) spacing of the knots.
    order : `int`
        Order of the B-splines (3: cubic).

    Returns
    -------
    basis : `scipy.sparse.csc_matrix`, shape ``(len(lam), nBasis)``
        The design matrix.
    """
    nInterval = max(1, int(np.ceil((lamHi - lamLo)/knotSpacing)))
    interior = np.linspace(lamLo, lamHi, nInterval + 1)
    knots = np.concatenate([[lamLo]*order, interior, [lamHi]*order])
    nBasis = len(knots) - order - 1

    inside = (lam >= lamLo) & (lam <= lamHi)
    basis = scipy.sparse.lil_matrix((lam.size, nBasis))
    if np.any(inside):
        dm = scipy.interpolate.BSpline.design_matrix(lam[inside], knots, order)
        basis[np.nonzero(inside)[0], :] = dm
    return basis.tocsc()


@dataclass
class DiskLineFit:
    """Disk (x) Edgeworth fit to a single arc line

    Parameters
    ----------
    wavelength : `float`
        Wavelength (nm) of the fitted centre, from the spectrum's wavelength array.
    pixel : `float`
        Fitted centre (= first moment) of the line (pixels).
    radius, radiusErr : `float`
        Disk radius and its 1-sigma error (pixels).
    blur, blurErr : `float`
        Blur rms and its 1-sigma error (pixels).
    cumulants, cumulantsErr : `numpy.ndarray`
        The blur's standardised cumulants ``[S, K, ...]`` (empty for a Gaussian
        blur) and their errors.
    amplitude, amplitudeErr : `float`
        Line flux (the profile has unit area) and its 1-sigma error.
    pix : `numpy.ndarray`
        Pixels of the fit window.
    model, continuum, data : `numpy.ndarray`
        Model (line, companions and continuum), continuum alone, and data in
        the window.
    chi2 : `float`
        chi^2 of the fit.
    ndof : `int`
        Degrees of freedom.
    success : `bool`
        Did the fit succeed (and the line's core is not masked)?
    extraction : `str`
        ``"optimal"`` or ``"boxcar"``: the extraction that the profile models.
    nBadCore : `int`
        Number of masked pixels near the line's centre (saturation).
    dispersion : `float`
        nm per pixel at the line.
    pixelErr : `float`
        1-sigma error in ``pixel`` (for a fit with its centre held fixed, the
        error of the fit that determined it).
    catWavelength, catPixel : `float`
        Catalogue wavelength (nm) and its position in the spectrum (pixels).
    peak : `float`
        Peak flux of the line's model.
    companionWavelength, companionAmp : `numpy.ndarray`
        Catalogue wavelengths and fitted amplitudes of the companion lines.
    element : `str`
        Lamp (catalogue description).
    status : `int`
        Catalogue status.
    fiberId : `int`
        Fibre.
    x, y : `float`
        Position on the detector (pixels).
    pinned : `bool`
        Was the centre held fixed?
    dataCentroid, modelCentroid : `float`
        First moments (pixels) of the data and the model about ``pixel`` over
        the same window (see `lineCentroids`); set by `measureLsf`.
    lorentz, lorentzErr : `float`
        FWHM (pixels) of the Lorentzian convolved with the profile, and its
        error; 0 and NaN if not fit.
    """
    wavelength: float
    pixel: float
    radius: float
    radiusErr: float
    blur: float
    blurErr: float
    cumulants: np.ndarray
    cumulantsErr: np.ndarray
    amplitude: float
    amplitudeErr: float
    pix: np.ndarray
    model: np.ndarray
    continuum: np.ndarray
    data: np.ndarray
    chi2: float
    ndof: int
    success: bool
    extraction: str = "optimal"
    nBadCore: int = 0
    dispersion: float = np.nan
    pixelErr: float = np.nan
    catWavelength: float = np.nan
    catPixel: float = np.nan
    peak: float = np.nan
    companionWavelength: np.ndarray = None
    companionAmp: np.ndarray = None
    element: str = ""
    status: int = 0
    fiberId: int = -1
    x: float = np.nan
    y: float = np.nan
    pinned: bool = False
    dataCentroid: float = np.nan
    modelCentroid: float = np.nan
    lorentz: float = 0.0
    lorentzErr: float = np.nan

    @property
    def sigma(self):
        """RMS width of the profile before pixel integration, without any Lorentzian (pixels)"""
        return float(diskSigma(self.radius, self.blur))

    def profile(self, dx):
        """The unit-area, pixel-integrated LSF at contiguous pixel offsets ``dx`` from the centre"""
        return diskProfile(np.asarray(dx, dtype=float), 0.0, self.radius, self.blur, self.cumulants,
                           self.extraction, self.lorentz)


def fitDiskLine(wavelength, flux, variance, lam0, *, radius0=1.5, blur0=0.6, blurOrder=4, halfWindow=12,
                continuumKnotSpacing=None, coreHalf=1.5, maxBadCore=0, companions=None, extraction="optimal",
                fixCentre=None, fitLorentz=False):
    """Fit one arc line near ``lam0`` (nm) with the disk (x) Edgeworth profile

    Parameters
    ----------
    wavelength, flux : `numpy.ndarray`
        The fibre's spectrum (nm, flux); NaN flux is ignored.
    variance : `numpy.ndarray`, optional
        Variance of ``flux``; `None` for equal weights.
    lam0 : `float`
        Catalogue wavelength (nm) of the line.
    radius0, blur0 : `float`
        Starting disk radius and blur rms (pixels).
    blurOrder : `int`
        Edgeworth order of the blur: ``2`` is a Gaussian; ``4`` adds skewness
        and excess kurtosis (``3..6`` supported).
    halfWindow : `float`
        The fit uses pixels within ``halfWindow`` of the catalogue position.
    continuumKnotSpacing : `float`, optional
        Knot spacing (nm) of the continuum; `None` for a single cubic over the
        window.
    coreHalf : `float`
        Masked pixels within ``coreHalf`` of the catalogue position count as a
        saturated core...
    maxBadCore : `int`
        ... and more than ``maxBadCore`` of them fail the fit.
    companions : array-like of `float`, optional
        Catalogue wavelengths (nm) of nearby lines to fit at the same time:
        each has the same profile, at a fixed offset from the main line (from
        the wavelength solution), with its own free amplitude. Companions
        centred more than a few pixels outside the window are ignored.
    extraction : `str`
        ``"optimal"`` or ``"boxcar"``; see `pfs.drp.stella.lsfProfile.diskProfile`.
    fixCentre : `float`, optional
        Hold the line's centre fixed at this position (pixels); its error is
        then reported as 0.
    fitLorentz : `bool`
        Also convolve the profile with a Lorentzian of free FWHM (``lorentz``,
        pixels; 0..5)?

    Returns
    -------
    fit : `DiskLineFit`
        The fit (check ``fit.success``).
    """
    if blurOrder != 2 and not 3 <= blurOrder <= MAX_BLUR_ORDER:
        raise ValueError(f"blurOrder must be 2 or in 3..{MAX_BLUR_ORDER}, not {blurOrder}")
    nCum = max(0, blurOrder - 2)
    nTheta = 3 + nCum + int(fitLorentz)          # [mu, log radius, log blur, cumulants..., (lorentz)]
    iLor = 3 + nCum

    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    pixels = np.arange(flux.size, dtype=float)
    good = np.isfinite(wavelength)
    p0 = float(np.interp(lam0, wavelength[good], pixels[good]))

    lo = max(0, int(np.floor(p0 - halfWindow)))
    hi = min(flux.size, int(np.ceil(p0 + halfWindow)) + 1)
    pix = pixels[lo:hi]
    lam = wavelength[lo:hi]
    data = flux[lo:hi]
    var = None if variance is None else np.asarray(variance, dtype=float)[lo:hi]

    use = np.isfinite(data) & np.isfinite(lam)
    if var is not None:
        use &= np.isfinite(var) & (var > 0)
    nBadCore = int(np.sum((np.abs(pix - p0) <= coreHalf) & ~use))   # masked core => saturated

    def failed():
        nan = np.full(nCum, np.nan)
        fit = DiskLineFit(lam0, p0, radius0, np.nan, blur0, np.nan, nan, nan,
                          np.nan, np.nan, pix, None, None, np.asarray(data, dtype=float), np.inf, 0, False,
                          extraction=extraction, nBadCore=nBadCore, catWavelength=lam0, catPixel=p0)
        return fit

    compLam = np.asarray([] if companions is None else companions, dtype=float)
    compOffset = np.interp(compLam, wavelength[good], pixels[good]) - p0
    keep = np.abs(compOffset) <= halfWindow + 4
    compLam, compOffset = compLam[keep], compOffset[keep]
    nComp = compLam.size
    if use.sum() < nTheta + 3:
        return failed()
    weight = np.zeros(pix.size)
    weight[use] = 1.0 if var is None else 1.0/np.sqrt(var[use])
    dataUse = np.where(use, data, 0.0)

    lamLo, lamHi = np.min(lam[use]), np.max(lam[use])
    knot = continuumKnotSpacing if continuumKnotSpacing is not None else max(1e-3, lamHi - lamLo)
    contBasis = continuumBasis(np.where(np.isfinite(lam), lam, -np.inf), lamLo, lamHi, knot).toarray()

    def column(theta, dp=0.0):
        return diskProfile(pix, theta[0] + dp, np.exp(theta[1]), np.exp(theta[2]), theta[3:3 + nCum],
                           extraction, theta[iLor] if fitLorentz else 0.0)

    def solveLinear(theta):
        cols = [column(theta)] + [column(theta, dp) for dp in compOffset]
        design = np.hstack([np.column_stack(cols), contBasis])
        Aw = design*weight[:, None]
        if not np.all(np.isfinite(Aw)):              # degenerate trial profile: reject the step, skip LAPACK
            return np.full(design.shape[1], np.nan), np.full(dataUse.size, np.nan), design, Aw
        coef, *_ = np.linalg.lstsq(Aw, dataUse*weight, rcond=None)
        return coef, Aw @ coef - dataUse*weight, design, Aw

    fixed = fixCentre is not None
    free = slice(1, None) if fixed else slice(None)       # the optimised part of theta

    def full(x):
        return np.concatenate([[float(fixCentre)], x]) if fixed else x

    def residual(x):
        return solveLinear(full(x))[1]

    # radius, blur in 0.05..10 pix; |cumulant| <= 10
    lower = np.array([p0 - halfWindow, np.log(0.05), np.log(0.05)] + [-10.0]*nCum
                     + [0.0]*int(fitLorentz))[free]
    upper = np.array([p0 + halfWindow, np.log(10.0), np.log(10.0)] + [10.0]*nCum
                     + [5.0]*int(fitLorentz))[free]
    x0 = np.array([p0, np.log(radius0), np.log(blur0)] + [0.0]*nCum + [0.05]*int(fitLorentz))[free]
    x0 = np.clip(x0, lower + 1e-9, upper - 1e-9)        # strictly inside the bounds (for least_squares)
    xScale = np.array([1.0, 0.3, 0.3] + [0.3]*nCum + [0.05]*int(fitLorentz))[free]
    try:
        res = scipy.optimize.least_squares(residual, x0, method="trf", x_scale=xScale, bounds=(lower, upper),
                                           max_nfev=2000)
    except Exception:
        return failed()

    theta = full(res.x)
    coef, resid, design, Aw = solveLinear(theta)
    if not np.all(np.isfinite(coef)):
        return failed()

    # 1-sigma errors on the nonlinear parameters from the Gauss-Newton covariance (J^T J)^-1 of the
    # variable-projection residual (amplitude, companions and continuum profiled out); residuals are
    # variance-weighted, so these are absolute errors
    try:
        thetaErr = np.sqrt(np.clip(np.diag(np.linalg.inv(res.jac.T @ res.jac)), 0.0, np.inf))
    except Exception:
        thetaErr = np.full(res.x.size, np.nan)
    if fixed:
        thetaErr = np.concatenate([[0.0], thetaErr])
    try:                                              # amplitude error with the shape held fixed
        amplitudeErr = float(np.sqrt(np.linalg.inv(Aw[use].T @ Aw[use])[0, 0]))
    except Exception:
        amplitudeErr = np.nan

    mu = float(theta[0])
    radius, blur = float(np.exp(theta[1])), float(np.exp(theta[2]))
    amplitude = float(coef[0])
    success = bool(res.success and np.isfinite(amplitude) and amplitude > 0 and nBadCore <= maxBadCore)
    return DiskLineFit(
        wavelength=float(np.interp(mu, pixels[good], wavelength[good])),
        pixel=mu,
        radius=radius,
        radiusErr=radius*float(thetaErr[1]),
        blur=blur,
        blurErr=blur*float(thetaErr[2]),
        cumulants=np.asarray(theta[3:3 + nCum], dtype=float),
        cumulantsErr=np.asarray(thetaErr[3:3 + nCum], dtype=float),
        amplitude=amplitude,
        amplitudeErr=amplitudeErr,
        pix=pix.copy(),
        model=design @ coef,
        continuum=design[:, 1 + nComp:] @ coef[1 + nComp:],
        data=np.asarray(data, dtype=float),
        chi2=float(np.sum(resid**2)),
        ndof=int(use.sum() - (coef.size + res.x.size)),
        success=success,
        extraction=extraction,
        nBadCore=nBadCore,
        dispersion=float(abs(np.interp(mu, pixels[good], np.gradient(wavelength[good], pixels[good])))),
        pixelErr=float(thetaErr[0]),
        catWavelength=float(lam0),
        catPixel=p0,
        peak=float(coef[0]*np.max(design[:, 0])),
        companionWavelength=compLam,
        companionAmp=np.asarray(coef[1:1 + nComp], dtype=float),
        pinned=fixed,
        lorentz=float(theta[iLor]) if fitLorentz else 0.0,
        lorentzErr=float(thetaErr[iLor]) if fitLorentz else np.nan,
    )


def selectLsfLines(lines, wavelength, *, lamMin=None, lamMax=None, isolation=12.0, blendFrac=0.05,
                   coreSep=4.0, coreFrac=0.02, minFrac=0.02, minIntensity=None, norm=None, minThroughput=0.3,
                   edge=25, excludeMask=0):
    """Pick bright, isolated catalogue lines

    Selection uses only the catalogue line positions and (approximate)
    intensities: a line is a candidate if it lands on the detector, is bright,
    and has no catalogue neighbour of comparable strength nearby.

    Parameters
    ----------
    lines : `pfs.drp.stella.ReferenceLineSet`
        Catalogue lines for the lamp.
    wavelength : `numpy.ndarray`
        The fibre's wavelength array (nm), to map lines to pixels.
    lamMin, lamMax : `float`, optional
        Wavelength range (nm); `None` uses the spectrum's coverage.
    isolation : `float`
        A line is rejected if another catalogue line within this many pixels
        has intensity above ``blendFrac`` times its own.
    blendFrac : `float`
        Neighbour strength, relative to the candidate, above which it is a blend.
    coreSep : `float`
        A neighbour within this many pixels is a blend if it exceeds
        ``coreFrac`` of the candidate, even when below ``blendFrac``: faint
        companions ~1 LSF width away that a fit can't separate.
    coreFrac : `float`
        Relative-strength threshold within ``coreSep``.
    minFrac : `float`
        Lines fainter than ``minFrac`` times the brightest in-range line are
        not considered.
    minIntensity : `float`, optional
        An absolute brightness floor (catalogue units).
    norm : `numpy.ndarray`, optional
        Per-pixel quartz throughput (``pfsArm.norm`` for this fibre); lines
        where it is below ``minThroughput`` times its median are dropped (the
        dichroic roll-off at the ends of the arm).
    minThroughput : `float`
        Throughput floor, as a fraction of the median ``norm``.
    edge : `int`
        Keep lines at least this many pixels from either end.
    excludeMask : `int`
        Lines with any of these catalogue status bits are not selected (they
        still count as neighbours).

    Returns
    -------
    chosen : `list` of `float`
        Catalogue wavelengths (nm) of the selected lines, sorted.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    good = np.isfinite(wavelength)
    pixels = np.arange(wavelength.size, dtype=float)
    loLam, hiLam = float(np.nanmin(wavelength[good])), float(np.nanmax(wavelength[good]))
    lamMin = loLam if lamMin is None else max(lamMin, loLam)
    lamMax = hiLam if lamMax is None else min(lamMax, hiLam)

    lam = lines.data.wavelength.to_numpy().astype(float)
    intensity = lines.data.intensity.to_numpy().astype(float)
    intensity = np.where(np.isfinite(intensity), intensity, 0.0)
    status = lines.data.status.to_numpy().astype(int)
    pc = np.interp(lam, wavelength[good], pixels[good], left=np.nan, right=np.nan)
    onChip = np.isfinite(pc)
    index = np.arange(lam.size)

    throughputOK = np.ones(lam.size, dtype=bool)
    if norm is not None:
        norm = np.asarray(norm, dtype=float)
        finite = np.isfinite(norm) & (norm > 0)
        ref = float(np.nanmedian(norm[finite])) if np.any(finite) else np.nan
        idx = np.clip(np.where(onChip, pc, 0).round().astype(int), 0, norm.size - 1)
        throughputOK = onChip & np.isfinite(norm[idx]) & (norm[idx] > minThroughput*ref)

    inRange = (onChip & (lam >= lamMin) & (lam <= lamMax) & throughputOK &
               (pc > edge) & (pc < wavelength.size - edge) & (intensity > 0))
    imax = float(np.max(intensity[inRange])) if np.any(inRange) else 0.0
    floor = max(0.0 if minIntensity is None else minIntensity, minFrac*imax)
    cand = np.nonzero(inRange & (intensity >= floor) & ((status & excludeMask) == 0))[0]

    def isolated(i):
        d = np.abs(pc - pc[i])
        near = onChip & (index != i) & (d < isolation)
        core = onChip & (index != i) & (d < coreSep)
        return not (np.any(intensity[near] > blendFrac*intensity[i]) or
                    np.any(intensity[core] > coreFrac*intensity[i]))

    return sorted(float(lam[i]) for i in cand if isolated(i))


def findCompanions(lines, wavelength, lam0, *, halfWidth, minFrac=0.002):
    """Catalogue lines near ``lam0`` that should be fit along with it

    Parameters
    ----------
    lines : `pfs.drp.stella.ReferenceLineSet`
        Catalogue for the lamp.
    wavelength : `numpy.ndarray`
        The spectrum's wavelength array (nm).
    lam0 : `float`
        Catalogue wavelength (nm) of the line being fit.
    halfWidth : `float`
        Companions within this many pixels of ``lam0`` are returned.
    minFrac : `float`
        Only lines with catalogue intensity at least ``minFrac`` times that of
        ``lam0`` (and > 0) are returned.

    Returns
    -------
    companions : `list` of `float`
        Catalogue wavelengths (nm), excluding ``lam0`` itself.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    good = np.isfinite(wavelength)
    pixels = np.arange(wavelength.size, dtype=float)
    lam = lines.data.wavelength.to_numpy().astype(float)
    intensity = lines.data.intensity.to_numpy().astype(float)
    intensity = np.where(np.isfinite(intensity), intensity, 0.0)
    i0 = int(np.argmin(np.abs(lam - lam0)))
    pc = np.interp(lam, wavelength[good], pixels[good], left=np.nan, right=np.nan)
    p0 = np.interp(lam0, wavelength[good], pixels[good])
    near = ((np.abs(pc - p0) <= halfWidth) & (np.arange(lam.size) != i0) & (intensity > 0)
            & (intensity >= minFrac*intensity[i0]))
    return sorted(float(x) for x in lam[near])


def measureLsf(wavelength, flux, variance, lines, *, element="", fiberId=-1, blurOrder=4,
               extraction="optimal", halfWindow=12, continuumKnotSpacing=None, coreHalf=1.5, maxBadCore=0,
               fitCompanions=True, companionFrac=0.002, fixCentres=None, centroidHalf=4.0, keepPixels=True,
               fitLorentz=False, **selectKwargs):
    """Select and fit LSF lines in one fibre's spectrum

    Parameters
    ----------
    wavelength, flux : `numpy.ndarray`
        The fibre's spectrum (nm, flux); NaN flux is ignored (masked).
    variance : `numpy.ndarray`, optional
        Variance of ``flux``.
    lines : `pfs.drp.stella.ReferenceLineSet`
        Catalogue lines for the lamp.
    element : `str`
        Stamped onto each fit (the lamp).
    fiberId : `int`
        Stamped onto each fit.
    blurOrder, extraction, halfWindow, continuumKnotSpacing, coreHalf, maxBadCore, fitLorentz
        See `fitDiskLine`.
    fitCompanions : `bool`
        Fit catalogue lines falling in each line's window simultaneously (see
        `findCompanions`)?
    companionFrac : `float`
        Only companions at least this fraction of the line's catalogue
        intensity are fit.
    fixCentres : `dict` [`float`, (`float`, `float`)], optional
        ``round(catalogue wavelength, 4) -> (centre, centre error)`` (pixels):
        fit only these lines, each with its centre held fixed (see
        `pinnedCentres`); the error is stamped on the fit as ``pixelErr``.
    centroidHalf : `float`
        Half-width (pixels) of the window for each fit's ``dataCentroid`` and
        ``modelCentroid`` (`lineCentroids`).
    keepPixels : `bool`
        Keep each fit's pixel arrays (``pix``, ``model``, ``continuum``,
        ``data``)?
    **selectKwargs
        Passed to `selectLsfLines`.

    Returns
    -------
    fits : `list` of `DiskLineFit`
        The successful fits.
    saturated : `list` of `float`
        Catalogue wavelengths of lines that failed because their core was
        masked.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if variance is not None:
        variance = np.asarray(variance, dtype=float)
    status = dict(zip(np.round(lines.data.wavelength.to_numpy().astype(float), 4),
                      lines.data.status.to_numpy().astype(int)))

    chosen = selectLsfLines(lines, wavelength, **selectKwargs)
    fits, saturated = [], []
    for lam0 in chosen:
        fixCentre = fixErr = None
        if fixCentres is not None:
            if round(float(lam0), 4) not in fixCentres:
                continue
            fixCentre, fixErr = fixCentres[round(float(lam0), 4)]
        comps = (findCompanions(lines, wavelength, lam0, halfWidth=halfWindow + 4, minFrac=companionFrac)
                 if fitCompanions else None)
        fit = fitDiskLine(wavelength, flux, variance, lam0, blurOrder=blurOrder, halfWindow=halfWindow,
                          continuumKnotSpacing=continuumKnotSpacing, coreHalf=coreHalf, maxBadCore=maxBadCore,
                          companions=comps, extraction=extraction, fixCentre=fixCentre, fitLorentz=fitLorentz)
        if fixErr is not None:
            fit.pixelErr = float(fixErr)
        if fit.success:
            fit.element = element
            fit.fiberId = int(fiberId)
            fit.status = int(status.get(round(float(lam0), 4), 0))
            fit.dataCentroid, fit.modelCentroid = lineCentroids(fit, centroidHalf)
            if not keepPixels:
                fit.pix = fit.model = fit.continuum = fit.data = None
            fits.append(fit)
        elif fit.nBadCore > maxBadCore:
            saturated.append(float(lam0))
    return fits, saturated


def _measureTask(task):
    """Worker for `measureLsfMany`: ``(wavelength, flux, variance, lines, kwargs) -> (fits, saturated)``"""
    wavelength, flux, variance, lines, kwargs = task
    return measureLsf(wavelength, flux, variance, lines, **kwargs)


def measureLsfMany(tasks, *, nWorkers=1):
    """Run `measureLsf` on many spectra, optionally in parallel processes

    Parameters
    ----------
    tasks : `list` of `tuple`
        ``(wavelength, flux, variance, lines, kwargs)`` per spectrum; ``kwargs``
        are passed to `measureLsf`.
    nWorkers : `int`
        Number of worker processes; ``1`` runs serially. Workers are spawned
        with one BLAS thread each.

    Returns
    -------
    results : `list` of (`list` of `DiskLineFit`, `list` of `float`)
        Per task, the successful fits and the saturated lines, in the order of
        ``tasks``.
    """
    # measureLsf only needs these columns of the catalogue: keep the inter-process traffic small
    columns = ["wavelength", "intensity", "status"]
    tasks = [(wl, fl, va, types.SimpleNamespace(data=lines.data[columns].copy()), kw)
             for wl, fl, va, lines, kw in tasks]
    if not nWorkers or nWorkers <= 1 or len(tasks) < 2:
        return [_measureTask(t) for t in tasks]
    threadVars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
    saved = {v: os.environ.get(v) for v in threadVars}
    os.environ.update({v: "1" for v in threadVars})           # inherited by the spawned workers
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nWorkers,
                                                    mp_context=multiprocessing.get_context("spawn")) as pool:
            return list(pool.map(_measureTask, tasks, chunksize=max(1, len(tasks)//(4*nWorkers))))
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val


def lineCentroids(fit, half=4.0):
    """First moments of a fitted line's data and model about its centre (pixels)

    Over the unmasked pixels with ``|pix - pixel| <= half`` (the same for data
    and model), of the line's own light (data minus the fitted continuum and
    companions) and of its fitted profile alone.

    Parameters
    ----------
    fit : `DiskLineFit`
        The fit (with its pixel arrays).
    half : `float`
        Half-width of the window (pixels).

    Returns
    -------
    data, window : `float`
        Data centroid and the model's centroid over the same pixels, each minus
        ``pixel``; NaN if unusable. ``data - window`` is the residual's first
        moment, free of the window's truncation of a skewed profile.
    """
    if fit.data is None or fit.model is None or not np.isfinite(fit.amplitude) or fit.amplitude <= 0:
        return np.nan, np.nan
    dx = fit.pix - fit.pixel
    own = fit.amplitude*fit.profile(dx)                   # this line's model alone
    data = fit.data - (fit.model - own)                   # data minus continuum and companions
    ok = np.isfinite(data) & (np.abs(dx) <= half)
    if ok.sum() < 3 or np.sum(data[ok]) <= 0:
        return np.nan, np.nan
    return float(np.sum(dx[ok]*data[ok])/np.sum(data[ok])), float(np.sum(dx[ok]*own[ok])/np.sum(own[ok]))


def clippedMean(values, nSigma=3.0, nIter=5):
    """Iteratively clipped mean (scale 1.4826*MAD about the median); NaNs ignored

    Parameters
    ----------
    values : array-like
        Values.
    nSigma : `float`
        Clipping threshold.
    nIter : `int`
        Maximum number of iterations.

    Returns
    -------
    mean, error, scatter : `float`
        The mean of the points within ``nSigma`` of the last mean, its error
        (``scatter/sqrt(N)``) and their Gaussian-equivalent scatter.
    used : `numpy.ndarray` of `bool`
        The points used.
    """
    values = np.asarray(values, dtype=float)
    used = np.isfinite(values)
    if not np.any(used):
        return np.nan, np.nan, np.nan, used
    centre = np.median(values[used])
    scale = 1.4826*np.median(np.abs(values[used] - centre))
    for _ in range(nIter):
        newUsed = np.isfinite(values) & (np.abs(values - centre) <= nSigma*scale)
        centre = np.mean(values[newUsed])
        if np.array_equal(newUsed, used):
            break
        used = newUsed
    scatter = 1.4826*np.median(np.abs(values[used] - np.median(values[used])))
    return float(centre), float(scatter/np.sqrt(used.sum())), float(scatter), used


def centroidZeroPoint(fits, nSigma=3.0, nIter=5):
    """Constant offset that makes the fitted centroids match the data's first moments

    The clipped mean over ``fits`` of the residual first moment
    (``dataCentroid - modelCentroid``: data minus model centroid over the same
    window, `lineCentroids`), i.e. the shift to add to each model's centre so
    that, on average, the data's centroid is the model's.

    Parameters
    ----------
    fits : iterable of `DiskLineFit`
        Line fits from `measureLsf`.
    nSigma, nIter
        See `clippedMean`.

    Returns
    -------
    offset, error, scatter : `float`
        As for `clippedMean` (pixels).
    used : `numpy.ndarray` of `bool`
        The fits used.
    """
    return clippedMean([ff.dataCentroid - ff.modelCentroid for ff in fits], nSigma=nSigma, nIter=nIter)


def pinnedCentres(fits, shift=0.0):
    """Fixed centres for a pinned refit (`measureLsf`'s ``fixCentres``) of one fibre's lines

    Parameters
    ----------
    fits : iterable of `DiskLineFit`
        The fibre's free fits.
    shift : `float`
        Added to each centre (pixels), e.g. `centroidZeroPoint`'s offset.

    Returns
    -------
    fixCentres : `dict` [`float`, (`float`, `float`)]
        ``round(catWavelength, 4) -> (centre + shift, the free fit's centre error)``.
    """
    return {round(float(ff.catWavelength), 4): (float(ff.pixel + shift), float(ff.pixelErr)) for ff in fits}


def findBroadLines(fits, *, window=50.0, minFibres=5, minNeighbours=4, broadFrac=0.025, nSigma=3.0):
    """Find catalogue lines that are intrinsically broader than the LSF

    Each line's rms width ``sigma`` is compared with that predicted at its
    wavelength by a straight line fit to ``sigma`` vs wavelength of the other
    lines (of every lamp) fit in the same fibre within ``window`` nm (refit once
    without neighbours > 3 sigma (1.4826 MAD) high: other broad lines). A line
    is broad if the median over fibres of its fractional excess
    ``sigma/sigma_predicted - 1`` exceeds both ``broadFrac`` and ``nSigma``
    times its error.

    Parameters
    ----------
    fits : iterable of `DiskLineFit`
        Per-line fits with ``element``, ``catWavelength`` and ``fiberId`` set,
        from several lamps fit in the same fibres.
    window : `float`
        Half-width (nm) of the neighbourhood.
    minFibres : `int`
        Lines measured in fewer fibres are not assessed.
    minNeighbours : `int`
        Fibres where a line has fewer neighbours are skipped.
    broadFrac : `float`
        Threshold on the median fractional excess.
    nSigma : `float`
        ... which must also be this significant.

    Returns
    -------
    excess : `dict` [(`str`, `float`), (`float`, `float`, `int`)]
        ``(element, round(catWavelength, 3)) -> (median fractional excess, its
        error (1.2533*1.4826*MAD/sqrt(N)), N fibres)``.
    broad : `set` [(`str`, `float`)]
        The keys of the broad lines.
    """
    byFibre = {}
    for ff in fits:
        if np.isfinite(ff.sigma) and ff.sigma > 0:
            byFibre.setdefault(ff.fiberId, []).append(ff)
    perLine = {}
    for ll in byFibre.values():
        lam = np.array([ff.catWavelength for ff in ll])
        sig = np.array([ff.sigma for ff in ll])
        for i, ff in enumerate(ll):
            near = np.abs(lam - lam[i]) <= window
            near[i] = False
            if near.sum() < minNeighbours:
                continue
            dl, sn = lam[near] - lam[i], sig[near]
            coef = np.polyfit(dl, sn, 1)
            res = sn - np.polyval(coef, dl)
            keep = res <= 3*1.4826*np.median(np.abs(res - np.median(res)))
            if keep.sum() >= minNeighbours:
                coef = np.polyfit(dl[keep], sn[keep], 1)
            perLine.setdefault((ff.element, round(ff.catWavelength, 3)), []).append(sig[i]/coef[1] - 1)
    excess = {}
    for key, values in perLine.items():
        if len(values) >= minFibres:
            values = np.array(values)
            med = float(np.median(values))
            err = float(1.2533*1.4826*np.median(np.abs(values - med))/np.sqrt(values.size))
            excess[key] = (med, err, int(values.size))
    return excess, {key for key, (med, err, _) in excess.items() if med > broadFrac and med > nSigma*err}
