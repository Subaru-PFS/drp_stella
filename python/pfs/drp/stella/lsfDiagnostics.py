"""Claude: checks of a fitted `ArmLsf` against data

These are the measurements behind the LSF chapter of the documentation and the
PIPE2D-1940 validation; the ``pfsLsf*.py`` scripts in ``bin`` run them on a
butler repository.

- `lineResiduals`, `summarizeResiduals`: arc lines minus the model's kernel,
  in units of the line's peak.
- `gaussianEquivalents`: the widths of the Gaussians equivalent to the LSF
  (FWHM, least-squares fit, second moment).
- `checkSpectraLsf`: the LSF bound to a pfsArm (``pfsArm.lsf``) against the
  lines measured in it (positions, widths and skewness).
- `chi2Breakdown`, `parameterBreakdown`: where on a detector the per-line
  fits' chi^2 is large, and their shape parameters there.
"""
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import astropy.units as u
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.special

from .armLsf import ArmLsf
from .lsfLine import LsfLineFlag

__all__ = ("selectUsedLines", "withGroupOffsets", "LineResidual", "lineResiduals", "summarizeResiduals",
           "gaussianEquivalents", "robustStats", "SpectraLsfCheck", "checkSpectraLsf", "chi2Breakdown",
           "parameterBreakdown")


def selectUsedLines(lines, minSn: float = 0.0) -> pd.DataFrame:
    """The lines that entered the smooth fits, from an `LsfLineSet`

    Parameters
    ----------
    lines : `pfs.drp.stella.lsfLine.LsfLineSet` or `pandas.DataFrame`
        Per-line fits (the ``lsfLines`` dataset).
    minSn : `float`
        Minimum flux S/N (``flux/fluxErr``).

    Returns
    -------
    used : `pandas.DataFrame`
        Lines that were used and not rejected, with ``ndof > 0`` and
        ``flux/fluxErr > minSn``.
    """
    df = lines if isinstance(lines, pd.DataFrame) else lines.toDataFrame()
    notUsed = int(LsfLineFlag.notUsed())
    rejected = sum(int(ff) for ff in LsfLineFlag if ff.name.startswith("REJECTED"))
    select = ((df.flag & notUsed) == 0) & ((df.flag & rejected) == 0) & (df.ndof > 0)
    if minSn > 0:
        select &= df.flux/df.fluxErr > minSn
    return df[select]


class _OffsetSurface:
    """A surface plus a constant: `ArmLsf` doesn't add group offsets to the halo surfaces"""
    def __init__(self, surface, offset):
        self.surface = surface
        self.offset = offset

    def __call__(self, u, v):
        return self.surface(u, v) + self.offset

    def __getattr__(self, name):
        return getattr(self.surface, name)


def withGroupOffsets(model: ArmLsf, factor: str, label: str) -> ArmLsf:
    """The model with one group's offsets included, for every fibre

    The surfaces describe the average group (e.g. the average lamp, for
    ``factor="visit"``); this is the model as that group saw it.

    Parameters
    ----------
    model : `ArmLsf`
        The fitted model.
    factor : `str`
        Grouping factor of the fit, e.g. ``visit``.
    label : `str`
        The group, e.g. a visit number.

    Returns
    -------
    result : `ArmLsf`
        A shallow copy of ``model`` that includes the group's offsets in the
        disk radius, the rms, the cumulants and (unlike the fibre groups in
        `ArmLsf`) the halo.
    """
    result = copy.copy(model)
    result.fiberGroups = dict(model.fiberGroups)
    result.fiberGroups[factor] = {int(ff): str(label) for ff in model.fiberId}
    result.surfaces = dict(model.surfaces)
    for name in ("haloFrac", "haloWidth"):
        if name in result.surfaces:
            offset = float(result.surfaces[name].getOffset(factor, str(label)))
            result.surfaces[name] = _OffsetSurface(result.surfaces[name], offset)
    return result


@dataclass
class LineResidual:
    """An arc line minus the model's kernel, in units of the line's peak

    Attributes
    ----------
    visit : `int`
        Visit.
    lamp : `str`
        Lamp (the first word of the line's description).
    fiberId : `int`
        Fibre.
    pixel : `float`
        The line's fitted centre (pixels).
    offsets : `numpy.ndarray`
        Pixels relative to ``pixel``.
    residuals : `numpy.ndarray`
        (data - model)/peak, where the model is the kernel times a fitted
        amplitude, plus a fitted constant.
    noise : `float`
        The largest standard deviation of the data in the window, divided by
        the peak.
    """
    visit: int
    lamp: str
    fiberId: int
    pixel: float
    offsets: np.ndarray
    residuals: np.ndarray
    noise: float


def lineResiduals(model: ArmLsf, lines: pd.DataFrame, getSpectra: Callable[[int], object],
                  halfWidth: float = 12, perVisit: bool = False) -> List[LineResidual]:
    """Residuals of arc lines from the model's kernel

    For each line, the kernel at the line's fitted centre is fit to the
    spectrum with only an amplitude and a constant free.

    Parameters
    ----------
    model : `ArmLsf`
        The fitted model.
    lines : `pandas.DataFrame`
        The lines to use (e.g. from `selectUsedLines`), with columns ``visit``,
        ``fiberId``, ``pixel`` and ``description``.
    getSpectra : callable
        Returns the extracted spectra (`pfs.datamodel.PfsFiberArraySet`) of a
        visit.
    halfWidth : `float`
        Half-width of the kernel (pixels).
    perVisit : `bool`
        Use each visit's offsets (`withGroupOffsets`) rather than the average
        lamp's surfaces.

    Returns
    -------
    residuals : `list` of `LineResidual`
        Lines entirely within the spectrum, with no masked pixels.
    """
    results = []
    for visit, group in lines.groupby("visit"):
        spectra = getSpectra(int(visit))
        visitModel = withGroupOffsets(model, "visit", str(int(visit))) if perVisit else model
        index = {int(ff): ii for ii, ff in enumerate(spectra.fiberId)}
        for row in group.itertuples():
            if row.fiberId not in index or row.fiberId not in model:
                continue
            ii = index[row.fiberId]
            kernel = visitModel.computeProfile(row.fiberId, row.pixel, halfWidth=halfWidth)
            pixels = kernel.start + np.arange(len(kernel.values))
            if pixels[0] < 0 or pixels[-1] >= spectra.length:
                continue
            data = spectra.flux[ii][pixels]
            sigma = np.sqrt(spectra.variance[ii][pixels])
            if np.any(spectra.mask[ii][pixels]) or not np.all(np.isfinite(data)) or not np.all(sigma > 0):
                continue
            design = np.vstack([kernel.values, np.ones_like(kernel.values)]).T
            (amplitude, background), *_ = np.linalg.lstsq(design, data, rcond=None)
            peak = amplitude*np.max(kernel.values)
            residuals = (data - amplitude*kernel.values - background)/peak
            lamp = str(row.description).split()[0] if row.description else "?"
            results.append(LineResidual(int(visit), lamp, int(row.fiberId), float(row.pixel),
                                        pixels - row.pixel, residuals, float(np.max(sigma)/peak)))
    return results


def summarizeResiduals(residuals: List[LineResidual], binSize: float = 0.25, halfWidth: float = 12,
                       minPerBin: int = 5) -> Dict[str, dict]:
    """Summarize line residuals by lamp

    Parameters
    ----------
    residuals : `list` of `LineResidual`
        Output of `lineResiduals`.
    binSize : `float`
        Bin size (pixels) for the stacked profile.
    halfWidth : `float`
        Half-width (pixels) of the stacked profile.
    minPerBin : `int`
        Minimum number of residuals in a bin of the stacked profile.

    Returns
    -------
    summary : `dict` [`str`, `dict`]
        For each lamp: ``num`` (lines); ``maxAbs``, the median over lines of
        the largest absolute residual (which includes the noise); ``noise``,
        the median of `LineResidual.noise`; ``stack``, the stacked residual
        profile (the median in each bin of offset; the systematic misfit),
        with ``edges`` its bin edges; and ``worst`` and ``worstOffset``, the
        stacked profile's largest absolute value and its bin centre.
    """
    edges = np.arange(-halfWidth, halfWidth + binSize, binSize)
    byLamp = defaultdict(list)
    for rr in residuals:
        byLamp[rr.lamp].append(rr)
    summary = {}
    for lamp, rr in sorted(byLamp.items()):
        offsets = np.concatenate([ll.offsets for ll in rr])
        values = np.concatenate([ll.residuals for ll in rr])
        which = np.digitize(offsets, edges) - 1
        stack = np.array([np.median(values[which == bb]) if np.sum(which == bb) > minPerBin else 0.0
                          for bb in range(len(edges) - 1)])
        worst = int(np.argmax(np.abs(stack)))
        summary[lamp] = dict(
            num=len(rr),
            maxAbs=float(np.median([np.max(np.abs(ll.residuals)) for ll in rr])),
            noise=float(np.median([ll.noise for ll in rr])),
            stack=stack,
            edges=edges,
            worst=float(stack[worst]),
            worstOffset=float(edges[worst] + 0.5*binSize),
        )
    return summary


def _fwhm(values: np.ndarray, step: float) -> float:
    """FWHM of a finely sampled profile, interpolating linearly at the half-maximum"""
    peak = np.argmax(values)
    half = 0.5*values[peak]
    above = np.flatnonzero(values >= half)
    lo, hi = above[0], above[-1]
    left = lo - 1 + (half - values[lo - 1])/(values[lo] - values[lo - 1])
    right = hi + (values[hi] - half)/(values[hi] - values[hi + 1])
    return float((right - left)*step)


def gaussianEquivalents(model: ArmLsf, fiberId: int, row: float, intrinsicWidths=(1.0, 2.0, 4.0),
                        coreHalfWidth: float = 4.0, numSub: int = 10) -> Dict[str, float]:
    """The rms widths (pixels) of Gaussians equivalent to the pixel-integrated LSF

    Parameters
    ----------
    model : `ArmLsf`
        The fitted model.
    fiberId : `int`
        Fibre.
    row : `float`
        Row (pixels).
    intrinsicWidths : iterable of `float`
        Intrinsic (Gaussian) rms widths (pixels) of resolved lines.
    coreHalfWidth : `float`
        Half-width (pixels) of the window for ``core``.
    numSub : `int`
        Sub-pixel sampling for ``fwhm`` and ``core``.

    Returns
    -------
    widths : `dict` [`str`, `float`]
        ``fwhm``: FWHM/2.3548 of the pixel-integrated kernel; ``fit``: the
        least-squares Gaussian (``computeGaussianSigma(method="fit")``);
        ``fit(s)`` for each intrinsic width ``s``: the same for a line of rms
        ``s``, with ``s`` removed in quadrature; ``moment``: the second
        moment (``method="moment"``); ``core``: the second moment within
        ``coreHalfWidth``.
    """
    # The pixel-integrated kernel, finely sampled: unit-spaced kernels at sub-pixel phases, interleaved
    offsets, values = [], []
    for jj in range(numSub):
        center = row + jj/numSub
        kernel = model.computeProfile(fiberId, center)
        offsets.append(kernel.start + np.arange(len(kernel.values)) - center)
        values.append(np.asarray(kernel.values, dtype=float))
    offsets = np.concatenate(offsets)
    values = np.concatenate(values)
    order = np.argsort(offsets)
    offsets, values = offsets[order], values[order]

    widths = dict(fwhm=_fwhm(values, 1/numSub)/np.sqrt(8*np.log(2)))
    widths["fit"] = float(model.computeGaussianSigma(fiberId, row, "fit"))
    for ss in intrinsicWidths:
        widths[f"fit({ss:g})"] = float(model.computeGaussianSigma(fiberId, row, "fit", ss))
    widths["moment"] = float(model.computeGaussianSigma(fiberId, row, "moment"))
    use = np.abs(offsets) <= coreHalfWidth
    mean = np.sum(values[use]*offsets[use])/np.sum(values[use])
    widths["core"] = float(np.sqrt(np.sum(values[use]*(offsets[use] - mean)**2)/np.sum(values[use])))
    return widths


def robustStats(values) -> tuple:
    """Median, Gaussian-equivalent scatter (1.4826 MAD) and number of the finite values"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, 0
    median = np.median(values)
    return float(median), float(1.4826*np.median(np.abs(values - median))), len(values)


def _gaussianFit(offsets: np.ndarray, values: np.ndarray) -> tuple:
    """Least-squares pixel-integrated Gaussian: (centre, rms)"""
    edges = np.append(offsets - 0.5, offsets[-1] + 0.5)

    def residuals(params):
        return params[0]*np.diff(scipy.special.ndtr((edges - params[1])/abs(params[2]))) - values

    guess = np.sqrt(max(np.sum(values*offsets**2)/np.sum(values), 0.1))
    result = scipy.optimize.least_squares(residuals, [np.sum(values), 0.0, guess])
    return float(result.x[1]), abs(float(result.x[2]))


def _shape(values: np.ndarray, offsets: np.ndarray) -> tuple:
    """(first moment - Gaussian centre, Gaussian rms) of a line"""
    mean = np.sum(values*offsets)/np.sum(values)
    centre, sigma = _gaussianFit(offsets, values)
    return mean - centre, sigma


@dataclass
class SpectraLsfCheck:
    """Comparison of a pfsArm's LSF with the lines measured in it (see `checkSpectraLsf`)

    Attributes
    ----------
    position : `numpy.ndarray`
        For every good line: the kernel's first moment at the catalogue
        wavelength minus the measured centroid (pixels).
    positionErr : `numpy.ndarray`
        The measured centroids' errors (pixels).
    width : `numpy.ndarray`
        For bright, isolated lines: the data's Gaussian rms divided by the
        kernel's (same window, same fitter).
    skew : `numpy.ndarray`
        For the same lines: (first moment - Gaussian centre) of the data minus
        that of the kernel (pixels); a measure of the asymmetry.
    lines : `pandas.DataFrame`
        The bright, isolated lines: ``fiberId``, ``wavelength``,
        ``description``, ``y``, ``sn``, ``width`` and ``skew``.
    """
    position: np.ndarray
    positionErr: np.ndarray
    width: np.ndarray
    skew: np.ndarray
    lines: pd.DataFrame


def checkSpectraLsf(spectra, lsf, lines, halfWidth: float = 8, minSn: float = 100, isolation: float = 10,
                    neighbourFrac: float = 0.03, close: float = 3, fiberStep: int = 5,
                    maxPositionErr: float = 0.1) -> SpectraLsfCheck:
    """Compare the LSF bound to a pfsArm with the lines measured in it

    Parameters
    ----------
    spectra : `pfs.datamodel.PfsFiberArraySet`
        Extracted spectra (a pfsArm).
    lsf : `pfs.drp.stella.spectraLsf.SpectraLsf`
        Their LSF (``pfsArm.lsf``).
    lines : `pfs.drp.stella.arcLine.ArcLineSet`
        The lines measured in the spectra (the ``lines`` dataset); entries
        described as ``Trace`` are ignored.
    halfWidth : `float`
        Half-width (pixels) of the window for widths and skewness.
    minSn : `float`
        Minimum peak S/N (from the spectrum) for widths and skewness.
    isolation : `float`
        No other peak brighter than ``neighbourFrac`` of the line's within
        this distance (pixels)...
    neighbourFrac : `float`
        ...for widths and skewness.
    close : `float`
        No other catalogue line, however faint, within this distance (pixels):
        avoids unresolved blends such as OH Lambda-doublets.
    fiberStep : `int`
        Measure widths and skewness in every ``fiberStep``-th fibre.
    maxPositionErr : `float`
        Maximum centroid error (pixels) of the lines used for positions.

    Returns
    -------
    result : `SpectraLsfCheck`
        The comparison.
    """
    fiberIndex = {int(ff): ii for ii, ff in enumerate(spectra.fiberId)}
    description = np.asarray(lines.description)
    real = description != "Trace"    # reduceExposure's doForceTraces adds a 'Trace' entry at every row
    fiberId = np.asarray(lines.fiberId)
    yy = np.asarray(lines.y)
    yErr = np.asarray(lines.yErr)
    wavelength = np.asarray(lines.wavelength)
    good = real & ~np.asarray(lines.flag) & np.isfinite(yy) & np.isfinite(yErr) & (yErr < maxPositionErr)

    position, positionErr, records = [], [], []
    for nn, ff in enumerate(sorted(set(fiberId[good]) & set(fiberIndex))):
        ii = fiberIndex[ff]
        select = good & (fiberId == ff)
        wl = wavelength[select]
        inRange = (wl > spectra.wavelength[ii].min()) & (wl < spectra.wavelength[ii].max())
        if not np.any(inRange):
            continue
        kernels = lsf.computeProfile(ff, wl[inRange]*u.nm)
        pixels = kernels.start[:, None] + np.arange(kernels.values.shape[1])
        moment = np.sum(kernels.values*pixels, axis=1)/np.sum(kernels.values, axis=1)
        position.extend(moment - yy[select][inRange])
        positionErr.extend(yErr[select][inRange])
        if nn % fiberStep != 0:
            continue

        # Every catalogue line with a centroid (not just the unflagged ones), with its peak and S/N from the
        # spectrum: the lines' own flux/fluxErr are not reliable S/N estimates
        flux = spectra.flux[ii]
        mask = spectra.mask[ii]
        noise = np.sqrt(np.abs(spectra.variance[ii]))
        this = real & (fiberId == ff) & np.isfinite(yy)
        allY, allWl, allDesc = yy[this], wavelength[this], description[this]
        inside = (allY > 1) & (allY < len(flux) - 2)
        allY, allWl, allDesc = allY[inside], allWl[inside], allDesc[inside]
        centre = np.round(allY).astype(int)
        window = centre[:, None] + np.arange(-1, 2)
        peak = np.max(flux[window], axis=1)
        # A neighbour counts only if it is a real peak: catalogue lines in a bright line's wings aren't
        isPeak = (flux[centre] > flux[centre - 1]) & (flux[centre] > flux[centre + 1])
        sn = peak/np.maximum(np.max(noise[window], axis=1), 1e-30)
        for y0, w0, d0, s0, f0 in zip(allY, allWl, allDesc, sn, peak):
            if np.any((np.abs(allY - y0) < close) & (np.abs(allWl - w0) > 1e-4)):
                continue
            neighbours = ((np.abs(allY - y0) < isolation) & (np.abs(allY - y0) > 0.5) & isPeak
                          & (peak > neighbourFrac*f0) & (sn > 5))
            if not s0 > minSn or np.any(neighbours):
                continue
            kernel = lsf.computeProfile(ff, y0*u.pix, halfWidth=halfWidth*u.pix)
            if kernel.start < 2 or kernel.stop + 2 > len(flux) or np.any(mask[kernel.slice] != 0):
                continue
            offsets = kernel.pixels - y0
            data = flux[kernel.slice].astype(float)
            lo = np.mean(flux[kernel.start - 2:kernel.start])
            hi = np.mean(flux[kernel.stop:kernel.stop + 2])
            data -= lo + (hi - lo)*(np.arange(len(data)) + 2)/(len(data) + 3)
            skewData, sigmaData = _shape(data, offsets)
            skewModel, sigmaModel = _shape(np.asarray(kernel.values, dtype=float), offsets)
            records.append(dict(fiberId=ff, wavelength=w0, description=d0, y=y0, sn=s0,
                                width=sigmaData/sigmaModel, skew=skewData - skewModel))

    columns = ["fiberId", "wavelength", "description", "y", "sn", "width", "skew"]
    table = pd.DataFrame(records, columns=columns)
    return SpectraLsfCheck(np.array(position), np.array(positionErr), table["width"].to_numpy(),
                           table["skew"].to_numpy(), table)


def chi2Breakdown(lines: pd.DataFrame, length: Optional[int] = None, width: int = 4096,
                  numBins: int = 8, minPerBin: int = 20,
                  snEdges=(0, 50, 100, 200, 300, 1000, 3000, np.inf)) -> Dict[str, object]:
    """Where are the per-line fits' chi^2/ndof large?

    Parameters
    ----------
    lines : `pandas.DataFrame`
        The used lines (`selectUsedLines`), with columns ``x``, ``y``,
        ``chi2``, ``ndof``, ``flux``, ``fluxErr``, ``visit``,
        ``description`` and ``fiberId``.
    length : `int`, optional
        Number of rows of the detector (default: the largest ``y``, rounded
        up).
    width : `int`
        Number of columns of the detector.
    numBins : `int`
        Number of bins in ``x`` and ``y``.
    minPerBin : `int`
        Minimum number of lines in a bin.
    snEdges : iterable of `float`
        Bin edges in S/N (``flux/fluxErr``).

    Returns
    -------
    breakdown : `dict`
        ``num`` and ``median`` (chi^2/ndof of all the lines); ``x``, ``y``
        and ``sn``: lists of (bin start, median, number); ``visit``: list of
        (visit, description, median, number); ``fiber``: the per-fibre
        medians, sorted (`pandas.Series`).
    """
    df = lines.copy()
    df["ratio"] = df.chi2/df.ndof
    df["sn"] = df.flux/df.fluxErr
    length = length if length is not None else int(np.ceil(df.y.max())) + 1
    result = dict(num=len(df), median=float(df.ratio.median()))
    for name, edges in (("x", np.linspace(0, width, numBins + 1)), ("y", np.linspace(0, length, numBins + 1)),
                        ("sn", np.asarray(snEdges, dtype=float))):
        bins = np.digitize(df[name], edges) - 1
        result[name] = [(float(edges[bb]), float(df.ratio[bins == bb].median()), int(np.sum(bins == bb)))
                        for bb in range(len(edges) - 1) if np.sum(bins == bb) > minPerBin]
    byVisit = df.groupby(["visit", "description"]).ratio.agg(["median", "size"])
    result["visit"] = [(int(vv), str(dd), float(rr["median"]), int(rr["size"]))
                       for (vv, dd), rr in byVisit.iterrows() if rr["size"] > minPerBin]
    result["fiber"] = df.groupby("fiberId").ratio.median().sort_values()
    return result


def parameterBreakdown(lines: pd.DataFrame, axis: str, extent: float, numBins: int = 8, minPerBin: int = 20,
                       names=("radius", "rms", "S", "K", "haloFrac", "haloWidth")) -> pd.DataFrame:
    """The per-line fits' chi^2/ndof and shape parameters, binned in ``x`` or ``y``

    Parameters
    ----------
    lines : `pandas.DataFrame`
        The used lines (`selectUsedLines`).
    axis : `str`
        ``x`` or ``y``.
    extent : `float`
        Size of the detector (pixels) along ``axis``.
    numBins : `int`
        Number of bins.
    minPerBin : `int`
        Minimum number of lines in a bin.
    names : iterable of `str`
        Parameters (columns of ``lines``) to report, where present.

    Returns
    -------
    table : `pandas.DataFrame`
        One row per bin: ``start``, ``num``, ``chi2/ndof`` and the median of
        each parameter.
    """
    edges = np.linspace(0, extent, numBins + 1)
    bins = np.digitize(lines[axis], edges) - 1
    names = [nn for nn in names if nn in lines.columns]
    rows = []
    for bb in range(numBins):
        select = lines[bins == bb]
        if len(select) < minPerBin:
            continue
        row = {"start": edges[bb], "num": len(select), "chi2/ndof": (select.chi2/select.ndof).median()}
        row.update({nn: select[nn].median() for nn in names})
        rows.append(row)
    return pd.DataFrame(rows)
