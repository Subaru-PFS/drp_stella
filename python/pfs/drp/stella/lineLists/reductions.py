"""Compare arc reductions, and the line lists measured from them (PIPE2D-2003, PIPE2D-2004)

The PFI line lists' intensities were measured from exposures reduced with the default cosmic-ray grouping,
which compares the exposures of a same-lamp series and masks the cores of bright lines that are brighter
in one exposure as cosmic rays (PIPE2D-2003).  These functions measure the damage:

* `compareReductions` compares the fluxes of a lamp's brightest lines between two reductions of the same
  visit (e.g. the default grouping and ``cosmicray:grouping=separate``), and reports the fraction of the
  line cores that the first reduction's ``crMask`` flags.  The mean over fibres is the sensitive
  statistic: where line cores are masked, the extraction also scrambles flux between neighbouring
  fibres, which the median hides.
* `compareLineLists` compares the measured intensities of two versions of a line list, e.g. measured
  (``measureLineIntensities.py``) from two reductions.
"""
import numpy as np
import pandas as pd

from pfs.drp.stella.referenceLine import ReferenceLineSet
from .lamps import integratedLineFluxes

__all__ = ["ARMS", "crMaskedFraction", "compareReductions", "compareLineLists"]

# Wavelength range of each arm (nm) for choosing lines
ARMS = {"b": (380, 650), "r": (630, 970), "n": (940, 1260)}


def crMaskedFraction(crMask, detectorMap, line, fiberStep=20, halfHeight=2, halfWidth=1):
    """Fraction of fibres whose line core includes a pixel flagged in the cosmic-ray mask

    Parameters
    ----------
    crMask : `lsst.afw.image.Mask`
        Cosmic-ray mask (non-zero where a cosmic ray was found).
    detectorMap : `pfs.drp.stella.DetectorMap`
        Detector map, to find the line's position in each fibre.
    line : `float`
        Line wavelength (nm).
    fiberStep : `int`
        Use every ``fiberStep``-th fibre.
    halfHeight, halfWidth : `int`
        Half-size of the box (rows, columns) around the line's position that is searched.

    Returns
    -------
    fraction : `float`
        Fraction of the fibres used whose box contains a flagged pixel (NaN if none could be used).
    """
    height, width = crMask.array.shape
    flagged = used = 0
    for fiberId in detectorMap.fiberId[::fiberStep]:
        x, y = (int(round(cc)) for cc in detectorMap.findPoint(int(fiberId), float(line)))
        if halfWidth < x < width - halfWidth - 1 and halfHeight < y < height - halfHeight - 1:
            box = crMask.array[y - halfHeight:y + halfHeight + 1, x - halfWidth:x + halfWidth + 1]
            flagged += int((box != 0).any())
            used += 1
    return flagged/used if used > 0 else np.nan


def compareReductions(oldButler, newButler, visit, lines, arms=ARMS, spectrographs=(1, 2, 3, 4), numLines=30,
                      minFibers=20):
    """Compare the fluxes of a lamp's brightest lines between two reductions of a visit

    Parameters
    ----------
    oldButler, newButler : `lsst.daf.butler.Butler`
        Butlers for the two reductions (with pfsArm; ``oldButler`` also with crMask and detectorMap).
    visit : `int`
        Visit.
    lines : `pfs.drp.stella.ReferenceLineSet`
        The lamp's line list; its brightest lines are compared.
    arms : `dict` [`str`, (`float`, `float`)]
        Arms, and the wavelength range in which to choose lines.
    spectrographs : iterable of `int`
        Spectrographs.
    numLines : `int`
        Number of the brightest catalogue lines per arm.
    minFibers : `int`
        Minimum number of fibres with a measurement.

    Returns
    -------
    comparison : `pandas.DataFrame`
        One row per (arm, spectrograph, line): ``visit``, ``arm``, ``spectrograph``, ``wavelength``,
        ``crFrac`` (`crMaskedFraction` in the old reduction), ``old`` and ``new`` (the mean flux over
        fibres), ``ratio`` (old/new of the means) and ``ratioMedian`` (old/new of the medians).
    """
    wavelength = np.array([ll.wavelength for ll in lines])
    intensity = np.array([ll.intensity for ll in lines])
    rows = []
    for arm, (low, high) in arms.items():
        select = (wavelength > low) & (wavelength < high) & np.isfinite(intensity)
        brightest = wavelength[select][np.argsort(intensity[select])[::-1][:numLines]]
        for spectrograph in spectrographs:
            dataId = dict(visit=visit, arm=arm, spectrograph=spectrograph)
            try:
                old = oldButler.get("pfsArm", dataId)
                new = newButler.get("pfsArm", dataId)
                crMask = oldButler.get("crMask", dataId)
                detectorMap = oldButler.get("detectorMap", dataId)
            except LookupError:
                continue
            # No mask bits are rejected: the damage being measured is unflagged (PIPE2D-2003)
            perFiber = [[integratedLineFluxes(spectra.wavelength[ii], spectra.flux[ii], spectra.mask[ii], 0,
                                              brightest) for ii in range(len(spectra))]
                        for spectra in (old, new)]
            for wl in brightest:
                stats = []
                for fibers in perFiber:
                    fluxes = np.array([ff[wl] for ff in fibers])
                    fluxes = fluxes[np.isfinite(fluxes)]
                    enough = len(fluxes) > minFibers
                    stats.append((np.mean(fluxes), np.median(fluxes)) if enough else (np.nan, np.nan))
                (oldMean, oldMedian), (newMean, newMedian) = stats
                rows.append(dict(visit=visit, arm=arm, spectrograph=spectrograph, wavelength=wl,
                                 crFrac=crMaskedFraction(crMask, detectorMap, wl), old=oldMean, new=newMean,
                                 ratio=oldMean/newMean, ratioMedian=oldMedian/newMedian))
    columns = ["visit", "arm", "spectrograph", "wavelength", "crFrac", "old", "new", "ratio", "ratioMedian"]
    return pd.DataFrame(rows, columns=columns)


def compareLineLists(old, new, minSN=30, tolerance=0.02):
    """Compare the measured intensities of two versions of a line list

    Parameters
    ----------
    old, new : `pfs.drp.stella.ReferenceLineSet` or `str`
        The two versions (or their paths), with the same lines in the same order.
    minSN : `float`
        Lines with ``new`` intensity / ``old`` intensityErr above this are ``bright``.
    tolerance : `float`
        Lines whose intensities differ by more than this fraction are ``changed``.

    Returns
    -------
    comparison : `pandas.DataFrame`
        One row per line measured in both (positive intensities and a positive ``intensityErr`` in
        ``old``): ``description``, ``wavelength``, ``old``, ``new``, ``ratio`` (old/new), ``sn``,
        ``bright`` and ``changed``.
    """
    if isinstance(old, str):
        old = ReferenceLineSet.fromLineList(old)
    if isinstance(new, str):
        new = ReferenceLineSet.fromLineList(new)
    if len(old) != len(new):
        raise ValueError(f"The line lists have different lengths ({len(old)}, {len(new)})")
    table = pd.DataFrame(dict(description=[ll.description for ll in old],
                              wavelength=[ll.wavelength for ll in old],
                              old=[ll.intensity for ll in old], new=[ll.intensity for ll in new],
                              oldErr=[getattr(ll, "intensityErr", np.nan) for ll in old]))
    if not np.allclose(table.wavelength, [ll.wavelength for ll in new]):
        raise ValueError("The line lists' wavelengths differ")
    table = table[np.isfinite(table.oldErr) & (table.oldErr > 0) & (table.old > 0) & (table.new > 0)].copy()
    table["ratio"] = table.old/table.new
    table["sn"] = table.new/table.oldErr
    table["bright"] = table.sn > minSN
    table["changed"] = np.abs(table.ratio - 1) > tolerance
    return table.drop(columns="oldErr").reset_index(drop=True)
