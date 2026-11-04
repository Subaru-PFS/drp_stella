"""Measure arc-line intensities from PFS arc spectra, and write them into line lists

The line wavelengths are taken as known (from a `ReferenceLineSet`) and the line profiles from the LSF
(the butler's ``pfsMergedLsf``); a single linear least-squares problem gives all the line fluxes plus a
smooth continuum (`fitLineIntensities`).  `measureLampIntensities` does this for one lamp in one visit:
it combines the fibres (all of them, or only the engineering fibres for IIS lamps) of the quartz-normalised
pfsMerged spectrum, fits, optionally replaces bright lines by integrated fluxes, divides by the exposure
time (`mergedExpTime`), and optionally scales lines it could not measure from a previous list.

Intensities are in counts/s, quartz-normalised, on one global scale for all the PFI lamps; the IIS
(engineering-fibre) lamps are on their own scale.  Used for PIPE2D-1890, PIPE2D-1958 and PIPE2D-1959;
the ``measureLineIntensities.py`` script is the command-line front end.
"""
from collections.abc import Sequence
import warnings

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.linalg
import scipy.optimize
import scipy.sparse

from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus, ReferenceLineSource

__all__ = ["fitLineIntensities", "combineArms", "updateLineIntensities", "report",
           "mergedExpTime", "combineFibres", "fitLinesFromData", "integratedFluxes", "scaleUnmeasured",
           "largestResiduals", "measureLampIntensities"]


def _computeProfile(lsfs, center, length):
    """Return the LSF at ``center`` (pixels), averaged over ``lsfs`` and
    normalised to unit sum, or `None` if it can't be computed
    """
    profile = np.zeros(length)
    for lsf in lsfs:
        try:
            arr = lsf.computeArray(center)
        except Exception:
            return None
        if arr.size != length or not np.all(np.isfinite(arr)):
            return None
        profile += arr

    norm = np.sum(profile)
    if norm <= 0:
        return None

    return profile/norm


def _continuumBasis(lam, lamLo, lamHi, knotSpacing, order=3):
    """Return a sparse cubic B-spline design matrix evaluated at ``lam``

    Only values of ``lam`` in [lamLo, lamHi] are supported; other rows are zero.
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


def _solve(A, b, lower):
    """Solve min |A x - b|^2 subject to x >= lower

    Parameters
    ----------
    A : `scipy.sparse.csc_matrix`, shape (nPix, nPar)
        Weighted design matrix.
    b : `numpy.ndarray`, shape (nPix,)
        Weighted data.
    lower : `numpy.ndarray`, shape (nPar,)
        Lower bounds (``-inf`` for unconstrained parameters).

    Returns
    -------
    x : `numpy.ndarray`, shape (nPar,)
        Solution.
    covar : `numpy.ndarray`, shape (nPar, nPar)
        Covariance matrix of the unconstrained problem, ``(A^T A)^-1``.
    """
    H = (A.T @ A).toarray()
    g = A.T @ b

    # Regularise very slightly, so that e.g. a line that lies entirely in a masked region
    # doesn't make the problem singular
    diag = np.diag(H).copy()
    H[np.diag_indices_from(H)] += 1e-10*np.where(diag > 0, diag, 1)

    covar = scipy.linalg.inv(H)  # ~ 1000x1000, so this is cheap

    if np.all(np.isinf(lower)):
        x = covar @ g
    else:
        # min |A x - b|^2 == min |R x - R^-T g|^2 with H = R^T R; this is a small dense problem
        R = scipy.linalg.cholesky(H, lower=False)
        y = scipy.linalg.solve_triangular(R, g, trans='T')
        res = scipy.optimize.lsq_linear(R, y, bounds=(lower, np.inf), method='bvls')
        x = res.x

    return x, covar


def fitLineIntensities(wavelength, flux, lsf, lines, variance=None, *,
                       lamMax=None, lamMin=None,
                       continuumKnotSpacing=10.0,
                       minSeparation=0.3,
                       nonNegative=True,
                       nSigmaClip=None, nIter=1,
                       rescale=True,
                       maxVarianceRatio=25.0):
    """Fit all lines in ``lines`` simultaneously to a spectrum

    The line wavelengths are held fixed; the free parameters are the line
    fluxes and the coefficients of a smooth (B-spline) continuum.

    Parameters
    ----------
    wavelength : `numpy.ndarray` of `float`, shape (N,)
        Wavelength of each pixel (nm), e.g. ``spec.wavelength[0]``.
    flux : `numpy.ndarray` of `float`, shape (N,)
        Flux; NaNs are ignored. E.g. ``np.nanmean(spec.flux, axis=0)``.
    lsf : `pfs.drp.stella.Lsf` or sequence of `pfs.drp.stella.Lsf`
        Line-spread function; if a sequence (e.g. one per fibre that went into
        an averaged spectrum), the line profiles are averaged over the LSFs.
    lines : `pfs.drp.stella.ReferenceLineSet`
        Lines to fit, e.g. from ``ReadLineListTask.run``.
    variance : `numpy.ndarray` of `float`, shape (N,), optional
        Variance of ``flux``; if `None`, all pixels are weighted equally.
    lamMax : `float`, optional
        Maximum wavelength of lines and pixels to fit (nm); if `None`
        (the default), there is no upper limit. Fitting includes a
        simultaneous continuum, so there is no need to exclude the NIR arm;
        pass e.g. ``lamMax=950.0`` if some other data problem requires it.
    lamMin : `float`, optional
        Minimum wavelength of lines and pixels to fit (nm).
    continuumKnotSpacing : `float` or `None`, optional
        Spacing of the continuum's B-spline knots (nm); if `None`, no
        continuum is fit.
    minSeparation : `float`, optional
        Lines closer than this (pixels) are fit as a single component, with
        the relative fluxes fixed at the catalogue's intensity ratios.
    nonNegative : `bool`, optional
        Constrain the line fluxes to be non-negative?
    nSigmaClip : `float` or `None`, optional
        If not `None`, reject pixels with residuals larger than this many
        sigma and refit.
    nIter : `int`, optional
        Number of clip-and-refit iterations (only used if ``nSigmaClip``).
    rescale : `bool`, optional
        Scale the fitted fluxes to match the catalogue's (arbitrary)
        intensity units, using the median ratio for good, well-measured lines?
    maxVarianceRatio : `float`, optional
        A line (or group) is marked ``fitted=False`` if its fitted flux
        variance exceeds this many times the "naive" variance it would have
        in isolation (ignoring correlations with the continuum and other
        lines). This catches lines that are nearly collinear with the
        continuum -- typically ones close to the edge of the fit range,
        where there isn't enough data beyond them to separate the line from
        the continuum's freedom to bend -- which the small ridge
        regularisation used to solve the fit would otherwise report with a
        deceptively small error.

    Returns
    -------
    newLines : `pfs.drp.stella.ReferenceLineSet`
        Copy of ``lines`` with the ``intensity`` and ``intensityErr`` of fitted
        lines updated, and their ``source`` bitmask OR'd with
        `ReferenceLineSource.EMPIRICAL_INTENSITY` (the rest of each line's
        source information, e.g. `ReferenceLineSource.NIST` for the
        wavelength/identification, is left untouched). That flag is
        currently a placeholder: it records that the intensity was measured
        from data rather than where it came from a doi/reference/date; there
        isn't enough information here to record proper provenance.
        Lines outside [``lamMin``, ``lamMax``) or the spectrum, or whose LSF
        could not be computed, keep their original catalogue intensity and
        source.
    result : `pandas.DataFrame`
        One row per line in the fit range, with columns ``wavelength``,
        ``description``, ``status``, ``catIntensity``, ``flux``, ``fluxErr``,
        ``intensity``, ``group``, ``fitted``, ``pixel``.  ``flux`` is the
        integrated line flux; ``group`` identifies lines fit as a single
        component (-1 for lines fit on their own).
    model : `numpy.ndarray` of `float`, shape (N,)
        Model (lines + continuum) for the spectrum (NaN outside the fit range).
    continuum : `numpy.ndarray` of `float`, shape (N,)
        The continuum component of ``model``.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    length = flux.size
    lsfs = list(lsf) if isinstance(lsf, Sequence) else [lsf]
    for ll in lsfs:
        assert ll.length == length, "LSF must be the same length as the spectrum"

    if lamMin is None:
        lamMin = np.nanmin(wavelength)
    if lamMax is None:
        lamMax = np.inf

    # Pixels to fit
    sigma = np.ones(length) if variance is None else np.sqrt(np.asarray(variance, dtype=float))
    usePix = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(sigma) & (sigma > 0)
    usePix &= (wavelength >= lamMin) & (wavelength < lamMax)
    if not np.any(usePix):
        raise RuntimeError("No valid pixels to fit")
    lamLo, lamHi = np.min(wavelength[usePix]), np.max(wavelength[usePix])

    # Lines to fit
    goodLam = np.isfinite(wavelength)
    pixels = np.arange(length, dtype=float)
    lineLam = lines.data.wavelength.to_numpy()
    select = np.nonzero((lineLam >= max(lamMin, np.nanmin(wavelength))) &
                        (lineLam < min(lamMax, np.nanmax(wavelength))))[0]
    select = select[np.argsort(lineLam[select])]
    lam = lineLam[select]
    center = np.interp(lam, wavelength[goodLam], pixels[goodLam])
    catIntensity = lines.data.intensity.to_numpy()[select]

    profiles = [_computeProfile(lsfs, cc, length) for cc in center]
    fitted = np.array([pp is not None for pp in profiles], dtype=bool)

    # Group lines that are too close to be separated
    group = np.full(len(select), -1, dtype=int)
    components = []   # list of indices into ``select``
    current = []
    for ii in np.nonzero(fitted)[0]:
        if current and center[ii] - center[current[-1]] >= minSeparation:
            components.append(current)
            current = []
        current.append(ii)
    if current:
        components.append(current)

    weights = np.zeros(len(select))   # fraction of the component's flux in each line
    columns = []
    for cc, members in enumerate(components):
        ww = catIntensity[members].astype(float)
        if not np.all(np.isfinite(ww) & (ww > 0)):
            ww = np.ones(len(members))
        ww /= ww.sum()
        weights[members] = ww
        if len(members) > 1:
            group[members] = cc
        columns.append(sum(w*profiles[ii] for w, ii in zip(ww, members)))
    nLine = len(columns)

    lineMatrix = scipy.sparse.vstack([scipy.sparse.csr_matrix(cc) for cc in columns]).T.tocsc() \
        if nLine > 0 else scipy.sparse.csc_matrix((length, 0))
    if continuumKnotSpacing is not None:
        contMatrix = _continuumBasis(np.where(goodLam, wavelength, -np.inf), lamLo, lamHi,
                                     continuumKnotSpacing)
    else:
        contMatrix = scipy.sparse.csc_matrix((length, 0))
    design = scipy.sparse.hstack([lineMatrix, contMatrix]).tocsr()

    lower = np.full(design.shape[1], -np.inf)
    if nonNegative:
        lower[:nLine] = 0.0

    # Fit, with optional clipping
    weight = np.where(usePix, 1/np.where(usePix, sigma, 1), 0.0)
    data = np.where(usePix, flux, 0.0)
    for iteration in range(1 + (nIter if nSigmaClip is not None else 0)):
        use = weight > 0
        A = scipy.sparse.diags(weight[use]) @ design[use]
        x, covar = _solve(A.tocsc(), data[use]*weight[use], lower)
        if nSigmaClip is None or iteration == nIter:
            break
        chi = (flux - design @ x)*weight
        scale = 1.0 if variance is not None else 1.4826*np.median(np.abs(chi[use] - np.median(chi[use])))
        reject = use & (np.abs(chi) > nSigmaClip*scale)
        if not np.any(reject):
            break
        weight[reject] = 0.0

    # Unpack results
    compFlux = x[:nLine]
    compVarRaw = np.diag(covar)[:nLine]   # in the units assumed while solving (sigma=1 if variance is None)
    if variance is None:
        # Estimate the noise from the residuals, and rescale to real units
        use = weight > 0
        chi = (flux - design @ x)[use]
        noiseScale = np.sum(chi**2)/max(1, use.sum() - design.shape[1])
    else:
        noiseScale = 1.0
    compVar = compVarRaw*noiseScale

    lineFlux = np.full(len(select), np.nan)
    lineErr = np.full(len(select), np.nan)
    for cc, members in enumerate(components):
        lineFlux[members] = weights[members]*compFlux[cc]
        # A numerically unconstrained component can have compVar <= 0 here; the sqrt then gives NaN,
        # which is fine -- that component is flagged unconstrained below and reset to fitted=False. Silence
        # the expected warning rather than let it alarm the caller on every off-edge line.
        with np.errstate(invalid='ignore'):
            lineErr[members] = weights[members]*np.sqrt(compVar[cc])

    # A component whose profile lies entirely in rejected/masked pixels isn't really fitted
    lineCols = A[:, :nLine]
    constrained = np.asarray(abs(lineCols).sum(axis=0)).flatten() > 0

    # A component that is nearly collinear with the continuum (or another component) is not
    # usefully constrained either, even though its column touches nonzero-weight pixels: the
    # small ridge regularisation in `_solve` keeps that direction from being exactly singular,
    # but it also artificially shrinks the reported variance for it, so a naive "error looks
    # small, trust the answer" check would miss it. Instead compare the fitted (marginal)
    # variance to the variance the component would have on its own, ignoring correlations with
    # every other parameter; a much larger marginal variance means the fit can't actually tell
    # this component apart from (some combination of) the others -- typically the continuum
    # near the edge of the fit range, where there isn't enough data beyond a line to separate it
    # from the continuum's freedom to bend there.
    # Both compVarRaw and ownVar are in the same (pre-rescale) units, so noiseScale cancels
    # out of the ratio -- it doesn't matter whether ``variance`` was supplied or estimated.
    # A component with no real data under its profile (e.g. a line off the blue/red edge whose
    # profile only grazes a few weighted pixels) is numerically unconstrained: the tiny ridge in
    # `_solve` keeps H nominally invertible, but `inv` can hand back a non-positive (or NaN)
    # marginal variance for that direction, which then yields a NaN error via sqrt below. Treat a
    # non-positive or non-finite marginal variance as unconstrained (ratio -> inf): otherwise a
    # negative product slips under ``maxVarianceRatio`` and the line is wrongly kept as fitted
    # with a NaN error.
    ownVar = np.asarray(lineCols.multiply(lineCols).sum(axis=0)).flatten()
    with np.errstate(divide='ignore', invalid='ignore'):
        varianceRatio = np.where((ownVar > 0) & (compVarRaw > 0), compVarRaw*ownVar, np.inf)
    constrained &= varianceRatio < maxVarianceRatio

    for cc, members in enumerate(components):
        if not constrained[cc]:
            fitted[members] = False
            lineFlux[members] = np.nan
            lineErr[members] = np.nan

    status = lines.data.status.to_numpy()[select]
    if rescale:
        good = (fitted & (group < 0) & ((status & ReferenceLineStatus.BAD) == 0) &
                np.isfinite(catIntensity) & (catIntensity > 0) & (lineFlux > 10*lineErr))
        scale = np.median(catIntensity[good]/lineFlux[good]) if np.any(good) else 1.0
    else:
        scale = 1.0
    intensity = scale*lineFlux
    intensityErr = scale*lineErr

    result = pd.DataFrame(dict(
        wavelength=lam,
        description=lines.data.description.to_numpy()[select],
        status=status,
        catIntensity=catIntensity,
        flux=lineFlux,
        fluxErr=lineErr,
        intensity=intensity,
        group=group,
        fitted=fitted,
        pixel=center,
    ))

    newData = lines.data.copy()
    newIntensity = newData.intensity.to_numpy().copy()
    newIntensity[select[fitted]] = intensity[fitted]
    newData["intensity"] = newIntensity

    # Keep the measured uncertainty alongside the intensity, so that a faint/non-detected line can be
    # filtered on S/N downstream rather than dropped here.
    newErr = (newData["intensityErr"].to_numpy().copy() if "intensityErr" in newData
              else np.full(len(newData), np.nan))
    newErr[select[fitted]] = intensityErr[fitted]
    newData["intensityErr"] = newErr

    # Flag which lines now carry an empirically-measured intensity, without disturbing the rest of
    # each line's existing source information (e.g. NIST for the wavelength/identification).
    newSource = newData.source.to_numpy().copy()
    newSource[select[fitted]] |= int(ReferenceLineSource.EMPIRICAL_INTENSITY)
    newData["source"] = newSource

    newLines = type(lines)(newData)

    inRange = goodLam & (wavelength >= lamLo) & (wavelength <= lamHi)
    model = np.where(inRange, design @ x, np.nan)
    continuum = np.where(inRange, contMatrix @ x[nLine:], np.nan) if contMatrix.shape[1] > 0 else \
        np.where(inRange, 0.0, np.nan)

    return newLines, result, model, continuum


WLROUND = 5  # decimals to which catalogue wavelengths (nm) match between result frames and the line list


def combineArms(results):
    """Inverse-variance combine per-arm fits, keyed by (description, wavelength)

    Parameters
    ----------
    results : iterable of `pandas.DataFrame`
        ``result`` frames from ``fitLineIntensities``, one per arm.

    Returns
    -------
    combined : `pandas.DataFrame`
        One row per line fitted in at least one arm, with columns
        ``description``, ``wavelength``, ``flux``, ``fluxErr`` and ``nArm``.
        ``flux`` is the inverse-variance-weighted mean of the arms' fitted
        fluxes. Faint and non-detected lines (``flux <= 0``) are kept, not cut:
        their large errors down-weight them in the combination and flag them for
        an S/N cut later.

    Every ``fitted`` row reaching the combine must carry a finite, positive
    error: a faint/non-detected line gets a large error, not NaN, and a NaN
    error only ever appears together with ``fitted=False`` (which is skipped
    here). That invariant is asserted, so a future per-arm producer that
    violated it fails loudly here rather than being silently averaged.
    """
    perLine = {}
    for df in results:
        for _, r in df.iterrows():
            if not bool(r["fitted"]):
                continue
            flux = float(r["flux"])
            if not np.isfinite(flux):
                continue
            # Keep faint and non-detected lines (flux <= 0): a non-detection carries a large error, so the
            # inverse-variance weighting below gives it little weight where another arm detects the line,
            # and otherwise records it as ~0 +/- large for an S/N cut downstream -- rather than dropping it.
            key = (str(r["description"]), round(float(r["wavelength"]), WLROUND))
            perLine.setdefault(key, []).append((flux, float(r["fluxErr"])))

    rows = []
    for (desc, lam), meas in perLine.items():
        flux = np.array([f for f, _ in meas])
        err = np.array([e for _, e in meas])
        # A fitted line always has a finite, positive fitted error here (see the docstring); fail loudly if
        # a future per-arm producer ever breaks that, rather than silently averaging around a bad error.
        assert np.all(np.isfinite(err) & (err > 0)), \
            f"{desc} {lam}: fitted line with non-finite or non-positive error {err}"
        w = 1.0/err**2
        f = float(np.sum(w*flux)/np.sum(w))
        e = float(np.sqrt(1.0/np.sum(w)))
        rows.append(dict(description=desc, wavelength=lam, flux=f, fluxErr=e, nArm=len(meas)))

    return pd.DataFrame(rows, columns=["description", "wavelength", "flux", "fluxErr", "nArm"])


def updateLineIntensities(lines, results, *, scale=1.0):
    """Return a copy of ``lines`` with intensities from per-arm fits

    Parameters
    ----------
    lines : `pfs.drp.stella.ReferenceLineSet` or `str`
        The catalogue to update, or the path of a line list to read.
    results : iterable of `pandas.DataFrame`
        Per-arm ``result`` frames from ``fitLineIntensities`` for this lamp.
    scale : `float`, optional
        Single global multiplicative scale, the same for every lamp:
        ``intensity = scale * flux``.

    Returns
    -------
    newLines : `pfs.drp.stella.ReferenceLineSet`
        Copy of ``lines`` with fitted lines' ``intensity`` set to ``scale*flux``,
        their ``intensityErr`` set to ``scale*fluxErr``, and
        ``ReferenceLineSource.EMPIRICAL_INTENSITY`` OR'd into their source.
    combined : `pandas.DataFrame`
        The per-line combination from `combineArms`, with added
        ``intensity`` (= ``scale*flux``) and ``intensityErr`` (= ``scale*fluxErr``)
        columns and a boolean ``matched`` saying whether the line was found in
        ``lines``.
    """
    if isinstance(lines, str):
        lines = ReferenceLineSet.fromLineList(lines)

    combined = combineArms(results)
    combined["intensity"] = scale*combined["flux"]
    combined["intensityErr"] = scale*combined["fluxErr"]

    newData = lines.data.copy()
    intensity = newData.intensity.to_numpy().copy()
    source = newData.source.to_numpy().copy()
    intensityErr = (newData["intensityErr"].to_numpy().copy() if "intensityErr" in newData
                    else np.full(len(newData), np.nan))
    catKey = {(str(d), round(float(w), WLROUND)): i
              for i, (d, w) in enumerate(zip(newData.description.to_numpy(), newData.wavelength.to_numpy()))}

    matched = np.zeros(len(combined), dtype=bool)
    for j, row in combined.iterrows():
        i = catKey.get((row.description, row.wavelength))
        if i is None:
            continue
        intensity[i] = row.intensity
        intensityErr[i] = row.intensityErr
        source[i] |= int(ReferenceLineSource.EMPIRICAL_INTENSITY)
        matched[j] = True
    combined["matched"] = matched

    newData["intensity"] = intensity
    newData["intensityErr"] = intensityErr
    newData["source"] = source
    newLines = type(lines)(newData)
    newLines.sort()
    return newLines, combined


def report(combined, *, scale=1.0):
    """Print a short summary of a `combineArms`/`updateLineIntensities` result

    Intensities are written with ``%12.7g`` (7 significant figures), so magnitude
    no longer affects the stored precision -- there is no 2-dp floor to clear, and
    ``scale`` is a free global multiplier (arbitrary units; keep it the same for
    every lamp). This just reports the line counts and the intensity range.
    """
    intensity = scale*combined["flux"].to_numpy()
    nOverlap = int(np.sum(combined["nArm"].to_numpy() > 1))
    unmatched = int(np.sum(~combined["matched"].to_numpy())) if "matched" in combined else 0
    print(f"{len(combined)} lines fitted ({nOverlap} combined across >1 arm)"
          + (f", {unmatched} not in the line list" if unmatched else ""))
    print(f"scale {scale:g}: intensity min/median/max = "
          f"{intensity.min():.4g} / {np.median(intensity):.4g} / {intensity.max():.4g}")


def mergedExpTime(butler, visit):
    """Return the exposure time of a merged visit: the mean of its pfsArms' exposure times

    ``pfsMerged.identity.expTime`` is the *sum* over the merged pfsArms (PIPE2D-1957), so use the
    pfsArms' own times, read from their ``CONFIG`` HDU without reading the spectra.  The mean is
    right whether or not that is fixed.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler with the collections holding the visit's pfsArms.
    visit : `int`
        Visit number.

    Returns
    -------
    expTime : `float`
        Exposure time (s).
    """
    import astropy.io.fits
    refs = list(butler.registry.queryDatasets("pfsArm", where=f"instrument='PFS' AND visit={visit}",
                                              findFirst=True))
    if not refs:
        raise RuntimeError(f"No pfsArms found for visit {visit}")
    return float(np.mean([astropy.io.fits.getdata(butler.getURI(ref).ospath, "CONFIG")["expTime"][0]
                          for ref in refs]))


def combineFibres(spec, *, fiberId=None, spectrograph=None, nClip=5, ignoreSaturatedFibers=True,
                  normFrac=1e-2):
    """Combine the fibres of a pfsMerged spectrum into one quartz-normalised spectrum

    Each fibre is divided by its ``norm`` (the quartz flat, which removes the fibre's throughput) and
    put back on the original count scale with one scalar per visit (the median ``norm``), so the
    result is in as-measured counts.  The fibres are combined with a clipped mean.

    Parameters
    ----------
    spec : `pfs.datamodel.PfsMerged`
        Merged spectra (e.g. already restricted to the engineering fibres with ``select``).
    fiberId : `int`, optional
        Use only this fibre.
    spectrograph : `int`, optional
        Use only fibres of this spectrograph.
    nClip : `float`
        Clip pixels more than this many (robust) sigma from the median over fibres.
    ignoreSaturatedFibers : `bool`
        Exclude fibres with any saturated pixel?
    normFrac : `float`
        Ignore pixels where ``norm`` is below this fraction of its median (band edges).

    Returns
    -------
    wavelength, flux, variance : `numpy.ndarray`
        The combined spectrum.
    """
    normBar = np.nanmedian(spec.norm)
    good = spec.norm > normFrac*normBar
    with np.errstate(divide="ignore", invalid="ignore"):
        specFlux = np.where(good, spec.flux*normBar/spec.norm, np.nan)
        specVar = np.where(good, spec.variance*(normBar/spec.norm)**2, np.nan)

    use = spec.mask == 0
    if spectrograph is not None:
        use &= (np.asarray(spec.spectrograph) == spectrograph)[:, np.newaxis]
    if fiberId is not None:
        use &= (spec.fiberId == fiberId)[:, np.newaxis]
    if ignoreSaturatedFibers:
        use &= (np.sum(spec.mask & spec.flags["SAT"], axis=1) == 0)[:, np.newaxis]

    fluxMasked = np.where(use, specFlux, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        q25, q50, q75 = np.nanpercentile(fluxMasked, [25, 50, 75], axis=0)
        nFiber = np.sum(np.isfinite(fluxMasked), axis=0)   # per pixel, as masking isn't uniform
        if np.mean(nFiber) > 2:
            keep = np.abs(fluxMasked - q50) < nClip*0.741*(q75 - q25)
            flux = np.nanmean(np.where(keep, fluxMasked, np.nan), axis=0)
        else:
            flux = np.nanmedian(fluxMasked, axis=0)
        variance = np.nanmean(np.where(use, specVar, np.nan), axis=0)/nFiber

    return spec.wavelength[0], flux, variance


def fitLinesFromData(spec, lsfDict, lines, expTime, fid=-1, spectrograph=-1, nClip=5, defaultFid=1000,
                     normFrac=1e-2, ignoreSaturatedFibers=True):
    """Fit line intensities to the combined fibres of a pfsMerged spectrum

    Kept with the signature the notebooks use.  The fitted fluxes are in as-measured counts;
    divide by ``expTime`` (returned unchanged) to get counts/s.

    Returns
    -------
    wavelength, flux, variance, lines, newLines, result, model, continuum, expTime
    """
    wavelength, flux, variance = combineFibres(spec, fiberId=fid if fid > 0 else None,
                                               spectrograph=spectrograph if spectrograph > 0 else None,
                                               nClip=nClip, ignoreSaturatedFibers=ignoreSaturatedFibers,
                                               normFrac=normFrac)
    lsf = lsfDict[fid if fid > 0 else defaultFid]   # the merged LSFs are all the same for now
    newLines, result, model, continuum = fitLineIntensities(wavelength, flux, lsf, lines, variance,
                                                            rescale=False)
    return wavelength, flux, variance, lines, newLines, result, model, continuum, expTime


def integratedFluxes(result, wavelength, flux, variance, continuum, *, minSN=30, halfWidth=0.6):
    """Replace the fitted fluxes of bright line groups by integrated fluxes

    Used for the IIS lamps, whose line profile the PFI-calibrated LSF doesn't describe.  Lines whose
    windows (+-``halfWidth`` nm) overlap form a group; a group with fitted S/N > ``minSN`` gets the flux
    integrated over its window (data minus fitted continuum), split between its lines in proportion to
    their fitted fluxes.

    Returns
    -------
    result : `pandas.DataFrame`
        Copy of the fitted lines of ``result``, with ``flux``/``fluxErr`` replaced where integrated, and a
        ``method`` column (``fit``, ``integrated`` or ``integrated, split N``).
    """
    res = result[result.fitted].sort_values("wavelength").reset_index(drop=True)
    group = (np.concatenate([[0], np.cumsum(np.diff(res.wavelength) > 2*halfWidth)]) if len(res)
             else np.array([]))
    res["method"] = "fit"
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        f, fe = res.flux.to_numpy()[idx], res.fluxErr.to_numpy()[idx]
        if np.sum(f) <= 0 or np.sum(f)/np.sqrt(np.sum(fe**2)) < minSN:
            continue
        window = ((wavelength > res.wavelength[idx[0]] - halfWidth)
                  & (wavelength < res.wavelength[idx[-1]] + halfWidth))
        total, totalErr = np.nansum((flux - continuum)[window]), np.sqrt(np.nansum(variance[window]))
        share = np.clip(f, 0, None)/np.sum(np.clip(f, 0, None))
        res.loc[idx, "flux"] = total*share
        res.loc[idx, "fluxErr"] = np.hypot(totalErr*share, fe)
        res.loc[idx, "method"] = "integrated" if len(idx) == 1 else f"integrated, split {len(idx)}"
    return res


def scaleUnmeasured(lines, previous, measured, *, minSN=30):
    """Give lines that weren't measured their previous intensity, scaled to the measured ones

    For each species, the scale is the median measured/previous ratio of its lines with S/N > ``minSN``.
    Unmeasured lines get no error and lose ``EMPIRICAL_INTENSITY``.

    Parameters
    ----------
    lines : `ReferenceLineSet`
        Lines with measured intensities.
    previous : `ReferenceLineSet`
        The list to take unmeasured lines' intensities from (e.g. the PFI lamp's list for an IIS lamp).
    measured : iterable of `float`
        Catalogue wavelengths of the lines that were measured.

    Returns
    -------
    newLines : `ReferenceLineSet`
    scales : `dict` [`str`, `float`]
        The scale used for each species.
    """
    d = lines.data.copy()
    prev = dict(zip(zip(previous.data.description, previous.data.wavelength.round(WLROUND)),
                    previous.data.intensity))
    key = list(zip(d.description, d.wavelength.round(WLROUND)))
    prevI = np.array([prev.get(k, np.nan) for k in key])
    isMeasured = np.isin(d.wavelength.round(WLROUND).to_numpy(), np.round(list(measured), WLROUND))
    scales = {}
    for sp in sorted(set(d.description)):
        species = (d.description == sp).to_numpy()
        good = species & isMeasured & (d.intensity > minSN*d.intensityErr).to_numpy() & (prevI > 0)
        if good.sum() < 3:
            continue
        scales[sp] = np.median(d.intensity.to_numpy()[good]/prevI[good])
        un = species & ~isMeasured
        d.loc[un, "intensity"] = prevI[un]*scales[sp]
        d.loc[un, "intensityErr"] = np.nan
        d.loc[un, "source"] = d.source[un] & ~int(ReferenceLineSource.EMPIRICAL_INTENSITY)
    newLines = type(lines)(d)
    newLines.sort()
    return newLines, scales


def largestResiduals(wavelength, flux, variance, model, result, lines, *, nPeak=15, nBright=8, avoid=0.35):
    """Return the largest positive residuals of a fit, away from the brightest lines (to find missing lines)

    Returns
    -------
    peaks : `pandas.DataFrame`
        ``wavelength``, ``residual``, ``sigma`` and the nearest listed line and its offset.
    """
    resid = np.where(np.isfinite(flux - model), flux - model, 0)
    lam = lines.data.wavelength.to_numpy()
    bright = result.sort_values("flux", ascending=False).wavelength.to_numpy()[:nBright]
    rows, seen = [], []
    for i in np.argsort(-resid):
        if len(seen) >= nPeak:
            break
        if np.any(np.abs(bright - wavelength[i]) < avoid) or any(abs(wavelength[i] - w) < 0.3 for w in seen):
            continue
        seen.append(wavelength[i])
        k = np.argmin(np.abs(lam - wavelength[i]))
        rows.append(dict(wavelength=wavelength[i], residual=resid[i], sigma=resid[i]/np.sqrt(variance[i]),
                         nearest=lam[k], offset=wavelength[i] - lam[k]))
    return pd.DataFrame(rows)


def measureLampIntensities(butler, visit, lines, *, engineering=False, integratedSN=None, previous=None):
    """Measure one lamp's line intensities (counts/s) from one visit

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler with the visit's pfsMerged, pfsMergedLsf, pfsArms and pfsConfig.
    visit : `int`
        Visit number.
    lines : `ReferenceLineSet`
        The lamp's line list.
    engineering : `bool`
        Use only the engineering (IIS) fibres?  Otherwise all fibres.
    integratedSN : `float`, optional
        Replace line groups with S/N above this by integrated fluxes (see `integratedFluxes`).
    previous : `ReferenceLineSet`, optional
        Scale unmeasured lines from this list (see `scaleUnmeasured`).

    Returns
    -------
    newLines : `ReferenceLineSet`
        ``lines`` with measured intensities (``EMPIRICAL_INTENSITY`` set) and errors.
    info : `dict`
        ``expTime``, ``result`` (per-line fit, counts/s), ``scales``, ``residuals`` and the combined
        spectrum (``wavelength``, ``flux``, ``variance``, ``model``, ``continuum``).
    """
    spec = butler.get("pfsMerged", visit=visit)
    lsfDict = butler.get("pfsMergedLsf", visit=visit)
    defaultFid = 1000
    if engineering:
        from pfs.datamodel import TargetType
        config = butler.get("pfsConfig", visit=visit)
        eng = config.fiberId[config.targetType == TargetType.ENGINEERING]
        eng = np.intersect1d(np.intersect1d(eng, spec.fiberId), list(lsfDict))
        spec = spec.select(fiberId=eng)
        defaultFid = int(eng[len(eng)//2])
    expTime = mergedExpTime(butler, visit)
    wl, flux, var, _, _, result, model, cont, _ = fitLinesFromData(spec, lsfDict, lines, expTime,
                                                                   defaultFid=defaultFid)
    residuals = largestResiduals(wl, flux, var, model, result, lines)
    if integratedSN is not None:
        result = integratedFluxes(result, wl, flux, var, cont, minSN=integratedSN)
    result = result.copy()
    result[["flux", "fluxErr"]] = result[["flux", "fluxErr"]]/expTime
    newLines, _ = updateLineIntensities(lines, [result], scale=1.0)
    scales = {}
    if previous is not None:
        newLines, scales = scaleUnmeasured(newLines, previous, result.wavelength[result.fitted])
    return newLines, dict(expTime=expTime, result=result, scales=scales, residuals=residuals,
                          wavelength=wl, flux=flux, variance=var, model=model, continuum=cont)
