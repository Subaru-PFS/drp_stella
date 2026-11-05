"""Pressure-broadened resonance families: identification, BROAD flags, widths and shifts (PIPE2D-1945)

Lines whose lower level is a J=1 level connected to the ground state by a resonance transition (the
noble gases' 1s2 = ``ns_2[1/2]*_1`` and 1s4 = ``ns_2[3/2]*_1``; Hg 6p 3P1; Cd 5p 1P1 and 3P1) are
broadened by resonance (self-) broadening, which scales with the lamp's fill pressure.  They are also
shifted to the blue.  ``BROAD`` is set on whole families, as the effect is a property of the lower
level and the lamp, not of the individual line.

* `lineFamilies` names the family of each line in a `ReferenceLineSet` (or of descriptions and
  transitions).
* `flagBroadFamilies` sets ``BROAD`` on the members of the given families, and clears it on the other
  lines with known transitions.
* `familyWidthExcess` compares each family's Lorentzian width (FWHM, pixels, per fibre) with the same
  lamp's non-family lines of similar brightness, and applies the flagging rule (>= 3 sigma and >= 2 GHz).
  Comparing with much fainter lines is biased, as the fitted Lorentzian depends on brightness; the b
  arm (~100 GHz/pixel) can't measure these widths.
* `readLsfLines` and `familyShifts` measure the families' centroid shifts from the LSF fit's per-line
  table (``LSFLINES``).

Transitions are NIST tokens, ``lower->upper`` with blends joined by ``;`` (see
`pfs.drp.stella.lineLists.nist`); a line belongs to a family if any component's lower level does.
"""
import numpy as np
import pandas as pd

from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineStatus

__all__ = ["FAMILIES", "lowerLevels", "lineFamilies", "flagBroadFamilies", "familyWidthExcess",
           "readLsfLines", "familyShifts"]

# species -> {lower level token: family name}
FAMILIES = {
    "NeI": {"3s_2[1/2]*_1": "1s2", "3s_2[3/2]*_1": "1s4"},
    "ArI": {"4s_2[1/2]*_1": "1s2", "4s_2[3/2]*_1": "1s4"},
    "KrI": {"5s_2[1/2]*_1": "1s2", "5s_2[3/2]*_1": "1s4"},
    "XeI": {"6s_2[1/2]*_1": "1s2", "6s_2[3/2]*_1": "1s4"},
    "HgI": {"6p_3P*_1": "6p3P1"},
    "CdI": {"5p_1P*_1": "5p1P1", "5p_3P*_1": "5p3P1"},
}
C_GHZ_NM = 299792458.0   # speed of light in GHz nm: dnu (GHz) = C_GHZ_NM*dlambda/lambda**2 (nm)


def _robustStd(values):
    """Gaussian-equivalent robust standard deviation (0.741*IQR) of the finite values"""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    return 0.741*np.subtract(*np.percentile(values, [75, 25]))


def lowerLevels(transition):
    """The lower levels of the components of a transition string

    Parameters
    ----------
    transition : `str`
        NIST transition token(s), ``lower->upper`` joined by ``;``.

    Returns
    -------
    lowers : `list` of `str`
        Lower level of each component (``[]`` for ``UNKNOWN`` or empty).
    """
    if not transition or transition == "UNKNOWN":
        return []
    return [tt.split("->")[0] for tt in transition.split(";") if "->" in tt]


def lineFamilies(lines, transitions=None):
    """Name the resonance family of each line

    Parameters
    ----------
    lines : `ReferenceLineSet` or iterable of `str`
        Lines, or their descriptions (species, e.g. ``ArI``) if ``transitions`` is given.
    transitions : iterable of `str`, optional
        Transitions, if ``lines`` are descriptions.

    Returns
    -------
    families : `numpy.ndarray` of `str`
        Family name (e.g. ``1s2``), or ``""`` if the line isn't in a family.
    """
    if transitions is None:
        descriptions, transitions = lines.data.description, lines.data.transition
    else:
        descriptions = lines
    result = []
    for desc, tr in zip(descriptions, transitions):
        families = FAMILIES.get(desc, {})
        result.append(next((families[lo] for lo in lowerLevels(tr) if lo in families), ""))
    return np.array(result, dtype=object)


def flagBroadFamilies(lines, families):
    """Set BROAD on every member of the given families, and clear it on other lines

    A line is a member if any component of its transition has the family's lower level.  BROAD on
    lines with ``UNKNOWN`` transitions is left as is (we can't tell whether they're members).

    Parameters
    ----------
    lines : `ReferenceLineSet`
        Lines (not modified).
    families : iterable of `str`
        Families to flag: names (e.g. ``1s2``, applied to every species that has that family) or lower
        level tokens (e.g. ``5s_2[1/2]*_1``).  If empty, BROAD is cleared on all lines with known
        transitions.

    Returns
    -------
    newLines : `ReferenceLineSet`
        Copy of ``lines`` with the flags changed, sorted.
    counts : `dict`
        ``before`` and ``after`` (number of BROAD lines), ``set`` and ``cleared`` (number changed), and
        ``clearedWavelengths``.
    """
    broad = int(ReferenceLineStatus.BROAD)
    families = set(families)
    data = lines.data.copy()
    names = lineFamilies(data.description, data.transition)
    member = np.array([name in families or any(lo in families for lo in lowerLevels(tr))
                       for name, tr in zip(names, data.transition)], dtype=bool)
    unknown = np.array([not lowerLevels(tr) for tr in data.transition], dtype=bool)
    old = (data.status.to_numpy() & broad) != 0
    setBroad = member & ~old
    clearBroad = old & ~member & ~unknown
    status = data.status.to_numpy().copy()
    status[setBroad] |= broad
    status[clearBroad] &= ~broad
    data["status"] = status
    newLines = ReferenceLineSet(data)
    newLines.sort()
    counts = dict(before=int(old.sum()), after=int((member | (old & unknown)).sum()), set=int(setBroad.sum()),
                  cleared=int(clearBroad.sum()), clearedWavelengths=data.wavelength.to_numpy()[clearBroad])
    return newLines, counts


def familyWidthExcess(fits, *, minFibres=4, brightness=10.0, minReference=3, nSigma=3.0, minGHz=2.0):
    """Excess Lorentzian width of each resonance family over the same lamp's other lines

    Each line's median Lorentzian FWHM over fibres is compared with the median of the same lamp's,
    arm's and species' non-family lines whose median peak is within a factor ``brightness`` of the
    family's.

    Parameters
    ----------
    fits : `pandas.DataFrame`
        Per-line, per-fibre fits, with columns ``lamp`` (e.g. ``Ar``, ``Kr_IIS``), ``arm``,
        ``description``, ``transition``, ``wavelength`` (nm), ``lorentz`` (FWHM, pixels), ``peak`` and
        ``dispersion`` (nm/pixel).  E.g. `readLsfLines` (with ``lamp`` added) of a fit with the
        Lorentzian enabled.
    minFibres : `int`
        Minimum number of fibres for a line to be used.
    brightness : `float`
        Reference lines' peaks must be within this factor of the family's median peak.
    minReference : `int`
        Minimum number of reference lines to measure an excess.
    nSigma, minGHz : `float`
        Flagging rule: a family is ``broad`` if its excess is >= ``nSigma`` sigma and >= ``minGHz``.

    Returns
    -------
    excess : `pandas.DataFrame`
        One row per (lamp, arm, species, family): ``nFamily``, ``nReference``, ``excessPix``, ``err``,
        ``significance``, ``excessGHz``, ``broad``, and ``lines`` (each member's excess in GHz).
    """
    fits = fits.copy()
    fits["family"] = lineFamilies(fits.description, fits.transition)
    fits["ghz"] = C_GHZ_NM*fits.dispersion/fits.wavelength**2
    perLine = fits.groupby(["lamp", "arm", "wavelength"]).agg(
        species=("description", "first"), family=("family", "first"), lorentz=("lorentz", "median"),
        peak=("peak", "median"), num=("lorentz", "size"), ghz=("ghz", "median")).reset_index()
    perLine = perLine[perLine.num >= minFibres]
    rows = []
    columns = ["lamp", "arm", "species", "family"]
    for (lamp, arm, species, family), fam in perLine[perLine.family != ""].groupby(columns):
        peak = fam.peak.median()
        ref = perLine[(perLine.lamp == lamp) & (perLine.arm == arm) & (perLine.species == species)
                      & (perLine.family == "") & (perLine.peak > peak/brightness)
                      & (perLine.peak < peak*brightness)]
        row = dict(lamp=lamp, arm=arm, species=species, family=family, nFamily=len(fam), nReference=len(ref),
                   excessPix=np.nan, err=np.nan, significance=np.nan, excessGHz=np.nan, broad=False, lines="")
        if len(ref) >= minReference:
            excess = fam.lorentz - ref.lorentz.median()
            famErr = 1.25*_robustStd(excess)/np.sqrt(len(fam)) if len(fam) > 2 else _robustStd(ref.lorentz)
            err = np.hypot(1.25*_robustStd(ref.lorentz)/np.sqrt(len(ref)), famErr)
            excessGHz = (excess*fam.ghz).median()
            row.update(excessPix=excess.median(), err=err, significance=excess.median()/err,
                       excessGHz=excessGHz,
                       broad=bool(excess.median()/err >= nSigma and excessGHz >= minGHz),
                       lines=" ".join(f"{wl:.2f}:{ee*gg:+.1f}" for wl, ee, gg in
                                      zip(fam.wavelength, excess, fam.ghz)))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns + ["nFamily", "nReference", "excessPix", "err", "significance",
                                                 "excessGHz", "broad", "lines"])


def _fitsString(value):
    """A FITS string table entry (possibly an array of characters) as a `str`"""
    return value if isinstance(value, str) else "".join(np.asarray(value).astype(str).ravel())


def readLsfLines(filenames):
    """Read the LSF fit's per-line tables (``LSFLINES``) into a `pandas.DataFrame`

    Parameters
    ----------
    filenames : iterable of `str`
        ``lsfLines`` FITS files (one per detector).

    Returns
    -------
    lsfLines : `pandas.DataFrame`
        The numeric columns, plus ``description`` and ``transition`` (strings) and ``file``.
    """
    import astropy.io.fits

    tables = []
    for filename in filenames:
        with astropy.io.fits.open(filename) as fd:
            table = fd["LSFLINES"].data
            data = {}
            for name in table.columns.names:
                if name in ("description", "transition"):
                    strings = fd[f"LSFLINES_{name.upper()}"].data
                    lookup = {nn: _fitsString(ss) for nn, ss in zip(strings["number"], strings["string"])}
                    data[name] = [lookup.get(nn, "") for nn in np.asarray(table[name])]
                else:
                    data[name] = np.asarray(table[name]).astype(float)
        df = pd.DataFrame(data)
        df["file"] = filename
        tables.append(df)
    return pd.concat(tables, ignore_index=True)


def familyShifts(lsfLines, lamps, *, badFlags=0x1 | 0x2 | 0x4 | 0x20, minFibres=20, minReference=3,
                 referenceLamp="Ne"):
    """Centroid shifts of the resonance families relative to the same lamp's other lines

    The residual of each line in each fibre is ``pixel + dataCentroid - refPixel`` (``dataCentroid`` is
    the data's first moment about ``pixel``; ``refPixel`` is the detectorMap position of the catalogue
    wavelength).  Per-fibre offsets common to all lamps are removed, and each line's median over fibres
    is compared with the median of the same lamp's non-family lines.

    Parameters
    ----------
    lsfLines : `pandas.DataFrame`
        Output of `readLsfLines`.
    lamps : `dict` [`int`, `str`]
        Lamp of each visit to use (e.g. ``{148314: "Ar"}``).
    badFlags : `int`
        Lines with any of these ``flag`` bits are ignored (default: SATURATED, SATURATED_ELSEWHERE,
        EXCLUDED_STATUS and REJECTED_CENTROID).
    minFibres : `int`
        Minimum number of fibres for a line to be used.
    minReference : `int`
        Minimum number of non-family lines for a lamp's offset to be reported.
    referenceLamp : `str`
        Lamp offsets are relative to this lamp's non-family lines.

    Returns
    -------
    offsets : `pandas.DataFrame`
        Per lamp: ``num``, ``offset`` and ``err`` (pixels, non-family lines relative to
        ``referenceLamp``; +ve is redder).
    shifts : `pandas.DataFrame`
        Per (lamp, family): ``num``, ``shift`` and ``err`` (pixels; family minus non-family lines),
        and ``shiftPm`` (picometres).
    """
    data = lsfLines[lsfLines.visit.isin(list(lamps))].copy()
    good = ((data.flag.astype(int) & badFlags) == 0)
    good &= np.isfinite(data.dataCentroid) & np.isfinite(data.refPixel)
    data = data[good]
    data["lamp"] = data.visit.astype(int).map(lamps)
    data["residual"] = data.pixel + data.dataCentroid - data.refPixel
    data["residual"] -= data.groupby(["file", "fiberId"]).residual.transform("median")
    perLine = data.groupby(["lamp", "wavelength"]).agg(
        residual=("residual", "median"), num=("residual", "size"), description=("description", "first"),
        transition=("transition", "first"), dispersion=("dispersion", "median")).reset_index()
    perLine = perLine[perLine.num >= minFibres]
    perLine["family"] = lineFamilies(perLine.description, perLine.transition)

    def errorOfMedian(values):
        return 1.25*_robustStd(values)/np.sqrt(len(values))

    nonFamily = perLine[perLine.family == ""]
    zero = nonFamily[nonFamily.lamp == referenceLamp].residual.median()
    offsets = []
    for lamp, ref in nonFamily.groupby("lamp"):
        if len(ref) >= minReference:
            offsets.append(dict(lamp=lamp, num=len(ref), offset=ref.residual.median() - zero,
                                err=errorOfMedian(ref.residual)))
    shifts = []
    for (lamp, family), fam in perLine[perLine.family != ""].groupby(["lamp", "family"]):
        ref = nonFamily[nonFamily.lamp == lamp]
        if len(ref) == 0:
            continue
        shift = fam.residual.median() - ref.residual.median()
        famErr = errorOfMedian(fam.residual) if len(fam) > 2 else _robustStd(ref.residual)
        err = np.hypot(errorOfMedian(ref.residual), famErr)
        shifts.append(dict(lamp=lamp, family=family, num=len(fam), shift=shift, err=err,
                           shiftPm=shift*fam.dispersion.median()*1e3))
    return (pd.DataFrame(offsets, columns=["lamp", "num", "offset", "err"]),
            pd.DataFrame(shifts, columns=["lamp", "family", "num", "shift", "err", "shiftPm"]))
