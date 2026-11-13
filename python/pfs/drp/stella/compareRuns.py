"""Claude: compare the spectra written by two pipeline runs (a regression check)

Used to check that a change to the code leaves the products unchanged, or
changes them only as expected. Each spectrum's arrays (flux, mask, sky,
variance, ...) are compared between a reference run and a new run:

- ``summary``: which arrays are identical, and the largest change relative to
  each array's rms;
- ``noise``: the changes in units of the noise (``|new - ref|/sigma`` over the
  pixels unmasked in both runs), the number of mask pixels that changed, and
  the pixels that changed most;
- ``detail``: for each kind of array, how many arrays and pixels changed, and
  the worst pixels.

Only the datasets written by the runs themselves are compared: for a chained
collection, its first child (the run).
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
from lsst.daf.butler import Butler, DatasetRef

__all__ = ("compareArrays", "noiseChange", "ownRun", "iterSpectra", "Comparison", "compareRuns")

# Arrays compared, by kind of dataset
FIBER_ARRAYS = ("wavelength", "flux", "mask", "sky", "norm", "covar")
TARGET_ARRAYS = ("wavelength", "flux", "mask", "sky", "covar")


def compareArrays(x: np.ndarray, y: np.ndarray) -> tuple[bool, float]:
    """Compare two arrays

    NaNs compare equal.

    Parameters
    ----------
    x, y : `numpy.ndarray`
        Reference and new arrays.

    Returns
    -------
    identical : `bool`
        Are the arrays identical?
    relative : `float`
        Largest ``|y - x|`` over the finite pixels, relative to the rms of
        ``x``; infinite if the shapes differ.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        return False, np.inf
    same = np.array_equal(x, y, equal_nan=True)
    finite = np.isfinite(x) & np.isfinite(y)
    if same or not np.any(finite):
        return same, 0.0
    rms = np.sqrt(np.mean(x[finite]**2)) or 1.0
    return False, float(np.max(np.abs(x[finite] - y[finite]))/rms)


def noiseChange(
    ref: np.ndarray, new: np.ndarray, variance: np.ndarray, refMask: np.ndarray, newMask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The change between two arrays in units of the noise

    Parameters
    ----------
    ref, new : `numpy.ndarray`
        Reference and new values.
    variance : `numpy.ndarray`
        Variance of the reference values.
    refMask, newMask : `numpy.ndarray` of `int`
        Masks of the reference and new values.

    Returns
    -------
    change : `numpy.ndarray`
        ``|new - ref|/sqrt(variance)`` (zero where not used).
    use : `numpy.ndarray` of `bool`
        Pixels used: unmasked in both, with finite positive variance.
    """
    variance = np.asarray(variance, dtype=float)
    use = (refMask == 0) & (newMask == 0) & np.isfinite(variance) & (variance > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        change = np.where(use, np.abs(np.asarray(new, float) - np.asarray(ref, float))/np.sqrt(variance), 0.0)
    return change, use


def ownRun(butler: Butler, collection: str) -> str:
    """The RUN collection written by a pipeline run

    For a chained collection (as made by ``pipetask run -o``), that's its
    first child; otherwise the collection itself.
    """
    try:
        return butler.registry.getCollectionChain(collection)[0]
    except Exception:
        return collection


@dataclass
class SpectrumPair:
    """The same spectrum (or set of spectra) from two runs"""

    label: str
    """Identifies the spectrum (dataId, and target for a coadd)"""
    ref: object
    """Spectrum from the reference run"""
    new: object
    """Spectrum from the new run"""
    arrays: tuple[str, ...]
    """Names of the arrays to compare"""


def iterSpectra(
    refButler: Butler, newButler: Butler, refCollection: str, dataset: str, where: str = ""
) -> Iterator[SpectrumPair]:
    """Iterate over the spectra written by the reference run, with the new run's

    Parameters
    ----------
    refButler, newButler : `lsst.daf.butler.Butler`
        Butlers for the reference and new runs.
    refCollection : `str`
        Collection of the reference run.
    dataset : `str`
        Dataset type: a set of fibre spectra (e.g. ``pfsArm``, ``pfsMerged``)
        or of target spectra (e.g. ``pfsCoadd``).
    where : `str`, optional
        Data query restricting the datasets.

    Yields
    ------
    pair : `SpectrumPair`
        The spectra from the two runs.
    """
    run = ownRun(refButler, refCollection)
    refs: list[DatasetRef] = sorted(
        refButler.registry.queryDatasets(dataset, collections=run, where=where, findFirst=True),
        key=lambda rr: str(rr.dataId),
    )
    for ref in refs:
        old = refButler.get(ref)
        new = newButler.get(dataset, ref.dataId)
        key = {kk: vv for kk, vv in ref.dataId.mapping.items() if kk != "instrument"}
        if hasattr(old, "fiberId"):
            yield SpectrumPair(f"{dataset} {key}", old, new, FIBER_ARRAYS)
        else:
            for target in old:
                yield SpectrumPair(f"{dataset} {key} {target}", old[target], new[target], TARGET_ARRAYS)


@dataclass
class Comparison:
    """Accumulated comparison of two runs, for one dataset type"""

    dataset: str
    numArrays: int = 0
    numDiffer: int = 0
    worstRelative: float = 0.0
    numPixels: int = 0
    numMaskChanged: int = 0
    noise: dict[str, list[np.ndarray]] = field(default_factory=lambda: defaultdict(list))
    worstPixels: list[tuple] = field(default_factory=list)
    details: dict[str, list[tuple]] = field(default_factory=lambda: defaultdict(list))
    lines: list[str] = field(default_factory=list)

    def add(self, pair: SpectrumPair, detail: bool = False) -> None:
        """Add the comparison of a pair of spectra"""
        differ = []
        for name in pair.arrays:
            x, y = getattr(pair.ref, name), getattr(pair.new, name)
            same, rel = compareArrays(x, y)
            self.numArrays += 1
            if not same:
                differ.append((name, rel))
                if detail:
                    self.addDetail(name, np.asarray(x, float), np.asarray(y, float),
                                   np.asarray(pair.ref.wavelength))
        self.numDiffer += len(differ)
        maxRel = max([rel for _, rel in differ], default=0.0)
        self.worstRelative = max(self.worstRelative, maxRel)
        masks = [name for name, _ in differ if name == "mask"]
        self.lines.append(f"{pair.label}: {len(pair.arrays)} arrays, {len(differ)} differ, "
                          f"max |diff|/rms {maxRel:.2e}" + ("; MASK differs" if masks else ""))

        # Changes in units of the noise
        variance = np.asarray(pair.ref.covar)[..., 0, :]
        self.numMaskChanged += int(np.sum(np.asarray(pair.ref.mask) != np.asarray(pair.new.mask)))
        for name in ("flux", "sky"):
            change, use = noiseChange(getattr(pair.ref, name), getattr(pair.new, name), variance,
                                      pair.ref.mask, pair.new.mask)
            if name == "flux":
                self.numPixels += int(use.sum())
                if np.any(use):
                    index = np.unravel_index(np.argmax(change), change.shape)
                    sigma = np.sqrt(variance[index])
                    self.worstPixels.append((float(change[index]), pair.label,
                                             float(np.asarray(pair.ref.wavelength)[index]),
                                             float(np.asarray(pair.ref.flux)[index]/sigma)))
            self.noise[name].append(change[use])

    def addDetail(self, name: str, x: np.ndarray, y: np.ndarray, wavelength: np.ndarray) -> None:
        """Record the worst change in an array that differs"""
        differ = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        finite = np.isfinite(x) & np.isfinite(y)
        rms = (np.sqrt(np.mean(x[finite]**2)) if np.any(finite) else 1.0) or 1.0
        rel = np.where(finite, np.abs(x - y), np.inf)/rms
        worst = np.unravel_index(np.argmax(rel), rel.shape)
        wl = wavelength[worst] if wavelength.shape == rel.shape else np.nan
        nonFinite = int((~finite & differ).sum())
        self.details[name].append((float(rel[worst]), int(differ.sum()), differ.size, nonFinite, float(wl),
                                   float(x[worst]), float(y[worst])))

    def summaryLines(self) -> list[str]:
        """Lines summarising which arrays changed"""
        verdict = ("IDENTICAL" if self.numDiffer == 0 else
                   f"{self.numDiffer} of {self.numArrays} arrays differ; largest |diff|/rms "
                   f"{self.worstRelative:.2e}")
        return self.lines + [f"{self.dataset}: {verdict}"]

    def noiseLines(self, numWorst: int = 4) -> list[str]:
        """Lines summarising the changes in units of the noise"""
        lines = [f"{self.dataset}: {self.numPixels} unmasked pixels; "
                 f"mask pixels changed {self.numMaskChanged}"]
        for value, label, wl, sn in sorted(self.worstPixels, key=lambda ww: -ww[0])[:numWorst]:
            lines.append(f"  worst flux change {value:.2f} sigma at {wl:.3f} nm "
                         f"(flux/sigma {sn:.0f}): {label}")
        for name, values in self.noise.items():
            vv = np.concatenate(values) if values else np.zeros(0)
            if vv.size == 0:
                continue
            pct = np.percentile(vv, [50, 99, 99.99])
            lines.append(f"  {name}: |diff|/sigma median {pct[0]:.1e}, 99% {pct[1]:.1e}, "
                         f"99.99% {pct[2]:.1e}, max {vv.max():.1e}; "
                         f"fraction > 0.01: {np.mean(vv > 0.01):.1e}")
        return lines

    def detailLines(self, numWorst: int = 3) -> list[str]:
        """Lines describing the changes in each kind of array"""
        lines = [f"{self.dataset}:"]
        for name, rows in self.details.items():
            rows = sorted(rows, key=lambda rr: -rr[0])
            frac = sum(rr[1] for rr in rows)/sum(rr[2] for rr in rows)
            nonFinite = sum(rr[3] for rr in rows)
            lines.append(f"  {name}: {len(rows)} arrays differ; pixels differing {frac:.2%}; "
                         f"non-finite changes {nonFinite}; median max|diff|/rms "
                         f"{np.median([rr[0] for rr in rows]):.1e}; worst:")
            lines += [f"     {rr[0]:.2e} at {rr[4]:.3f} nm: {rr[5]:.6g} -> {rr[6]:.6g}"
                      for rr in rows[:numWorst]]
        return lines


def compareRuns(
    repo: str, refCollection: str, newCollection: str, datasets: list[str], where: str = "",
    detail: bool = False,
) -> list[Comparison]:
    """Compare the spectra written by two runs

    Parameters
    ----------
    repo : `str`
        Butler repository.
    refCollection, newCollection : `str`
        Collections of the reference and new runs.
    datasets : `list` of `str`
        Dataset types to compare.
    where : `str`, optional
        Data query restricting the datasets.
    detail : `bool`, optional
        Accumulate the per-array details?

    Returns
    -------
    comparisons : `list` of `Comparison`
        One per dataset type.
    """
    refButler = Butler(repo, collections=refCollection)
    newButler = Butler(repo, collections=newCollection)
    results = []
    for dataset in datasets:
        comparison = Comparison(dataset)
        for pair in iterSpectra(refButler, newButler, refCollection, dataset, where):
            comparison.add(pair, detail=detail)
        results.append(comparison)
    return results
