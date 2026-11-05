"""Transitions and line lists from the NIST Atomic Spectra Database (ASD)

* `fetchNistLines` / `fetchNistLevels` download NIST's tab-separated tables, with vacuum wavelengths
  (``show_av=3``, the PFS convention), for one species at a time: a query for several species
  (e.g. ``Cd I;Hg I``) is rejected by NIST.  NIST reports errors as HTML pages, so the download is
  checked for that.
* `addTransitions` fills in the ``transition`` of the lines in a `ReferenceLineSet` by matching them,
  by species and wavelength, to a NIST dump.
* `buildLineList` builds a line list for one species from a NIST dump, merging components that PFS
  can't resolve.
* `findTransitions` lists allowed transitions between NIST energy levels near a wavelength (Ritz
  wavelengths), to identify lines missing from NIST's line list (e.g. Cd I 1039.748, 5s5p 1P1 - 5s6s 1S0).

Transition tokens are ``<valence>_<term>_<J>-><valence>_<term>_<J>`` (lower -> upper), where valence is
the last ``.``-separated part of NIST's configuration; NIST blends join their tokens with ``;``.
"""
import csv
import re
import urllib.parse
import urllib.request
from collections import OrderedDict
from fractions import Fraction

import numpy as np
import pandas as pd

from pfs.drp.stella.referenceLine import ReferenceLineSet, ReferenceLineSource

__all__ = ["ROMAN", "parseSpecies", "speciesInLines", "nistLinesUrl", "nistLevelsUrl", "fetchNistLines",
           "fetchNistLevels", "loadNist", "transitionToken", "addTransitions", "buildLineList",
           "readLevels", "findTransitions"]

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9}
SPECIES_RE = re.compile(r"^([A-Z][a-z]?)(I{1,3}|IV|VI{0,3}|IX)$")
# Dumps for a single species have no element/sp_num columns; take the species from the filename:
# nist_<El>.dat (stage I) or nist_<El><stage>.dat (e.g. nist_ArII.dat)
FILENAME_RE = re.compile(r"nist_([A-Z][a-z]?)(I{1,3}|IV|VI{0,3}|IX)?\.dat$")
NIST_LINES_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
NIST_LEVELS_URL = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"


def parseSpecies(description):
    """Return ``(element, stage)`` for a line-list description such as ``"ArI"``, or `None`"""
    m = SPECIES_RE.match(description)
    return (m.group(1), ROMAN[m.group(2)]) if m else None


def speciesInLines(lines):
    """Return the species used by ``lines`` and their wavelength range

    Returns
    -------
    species : `set` of (`str`, `int`)
        ``(element, stage)`` pairs.
    lamMin, lamMax : `float`
        Wavelength range (nm).
    """
    species = {parseSpecies(dd) for dd in set(lines.data.description)} - {None}
    return species, float(lines.data.wavelength.min()), float(lines.data.wavelength.max())


def _speciesString(element, stage):
    roman = {v: k for k, v in ROMAN.items()}
    return f"{element} {roman[stage]}"


def nistLinesUrl(element, stage, lamMin, lamMax):
    """Return the NIST ASD lines query for one species (vacuum wavelengths, tab-separated, all lines)"""
    params = [
        ("spectra", _speciesString(element, stage)), ("limits_type", "0"),
        ("low_w", f"{lamMin:.1f}"), ("upp_w", f"{lamMax:.1f}"), ("unit", "1"), ("submit", "Retrieve Data"),
        ("de", "0"), ("format", "3"),     # tab-separated
        ("line_out", "0"), ("en_unit", "0"), ("output", "0"),     # all results, not paginated
        ("bibrefs", "1"), ("page_size", "15"),    # NB: NIST rejects page_size=0 ("Invalid page size")
        ("show_obs_wl", "1"), ("show_calc_wl", "1"), ("unc_out", "1"), ("order_out", "0"),
        ("max_low_enrg", ""), ("show_av", "3"),   # 3 = vacuum wavelengths everywhere (PFS convention)
        ("max_upp_enrg", ""), ("tsb_value", "0"), ("min_str", ""), ("A_out", "0"), ("intens_out", "on"),
        ("max_str", ""), ("allowed_out", "1"), ("forbid_out", "1"), ("min_accur", ""), ("min_intens", ""),
        ("conf_out", "on"), ("term_out", "on"), ("enrg_out", "on"), ("J_out", "on"),
    ]
    return NIST_LINES_URL + "?" + urllib.parse.urlencode(params)


def nistLevelsUrl(element, stage):
    """Return the NIST ASD energy-levels query for one species (tab-separated)"""
    params = [("spectrum", _speciesString(element, stage)), ("units", "0"), ("format", "3"), ("output", "0"),
              ("page_size", "15"), ("multiplet_ordered", "0"), ("conf_out", "on"), ("term_out", "on"),
              ("level_out", "on"), ("j_out", "on"), ("submit", "Retrieve Data")]
    return NIST_LEVELS_URL + "?" + urllib.parse.urlencode(params)


def _fetch(url, path):
    # NIST's front end rejects Python's default User-Agent with HTTP 403 (curl's is accepted)
    headers = {"User-Agent": "pfs.drp.stella.lineLists (curl-compatible)"}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("latin-1")
    if text.lstrip().startswith("<") or "Error Message" in text[:5000]:
        message = re.sub(r"<[^>]+>", " ", text)
        message = re.search(r"Error Message:\s*(.*?)\s{2,}", message)
        reason = message.group(1) if message else "unknown"
        raise RuntimeError(f"NIST returned an error page ({reason}): {url}")
    with open(path, "w") as fd:
        fd.write(text)
    return path


def fetchNistLines(element, stage, lamMin, lamMax, path):
    """Download NIST's lines for one species to ``path`` (name it ``nist_<El>[<stage>].dat``)"""
    return _fetch(nistLinesUrl(element, stage, lamMin, lamMax), path)


def fetchNistLevels(element, stage, path):
    """Download NIST's energy levels for one species to ``path``"""
    return _fetch(nistLevelsUrl(element, stage), path)


def _parseWavelength(value, unc):
    """Return (wavelength, uncertainty), or None if value is empty

    Strips NIST's flag characters (e.g. the "+" in "347.0794+").  If NIST gave no uncertainty, use one unit
    in the last quoted decimal place.
    """
    value = re.sub(r"[^0-9.]", "", value or "")
    if not value:
        return None
    if not unc:
        nDecimal = len(value.partition(".")[2])
        unc = 10**-nDecimal if nDecimal > 0 else 0.0005
    return float(value), float(unc)


def loadNist(filenames):
    """Read NIST line dumps

    Parameters
    ----------
    filenames : iterable of `str`
        Tab-separated NIST lines dumps.  A dump for several species has ``element`` and ``sp_num`` columns;
        a single-species dump must be named ``nist_<El>[<stage>].dat``.

    Returns
    -------
    bySpecies : `dict` [(`str`, `int`), `list` [`dict`]]
        NIST's rows for each ``(element, stage)``, each annotated with ``_wl`` and ``_unc`` (nm).
    """
    bySpecies = {}
    for filename in filenames:
        m = FILENAME_RE.search(filename)
        defaultKey = (m.group(1), ROMAN[m.group(2)] if m.group(2) else 1) if m else None
        with open(filename) as fd:
            for r in csv.DictReader(fd, delimiter="\t"):
                r = {k: (v or "").strip('"') for k, v in r.items() if k}
                wlUnc = (_parseWavelength(r.get("obs_wl_vac(nm)"), r.get("unc_obs_wl"))
                         or _parseWavelength(r.get("ritz_wl_vac(nm)"), r.get("unc_ritz_wl")))
                if wlUnc is None:
                    continue
                r["_wl"], r["_unc"] = wlUnc
                if "element" in r:
                    key = (r["element"], int(r["sp_num"]))
                elif defaultKey is not None:
                    key = defaultKey
                else:
                    raise RuntimeError(f"{filename} has no element/sp_num columns and its name "
                                       "isn't of the form nist_<El>[<stage>].dat")
                bySpecies.setdefault(key, []).append(r)
    return bySpecies


def transitionToken(row):
    """Return the transition token ``<valence>_<term>_<J>-><valence>_<term>_<J>`` for a NIST row"""
    lo = f"{row['conf_i'].split('.')[-1]}_{row['term_i']}_{row['J_i']}"
    hi = f"{row['conf_k'].split('.')[-1]}_{row['term_k']}_{row['J_k']}"
    return f"{lo}->{hi}"


def addTransitions(lines, bySpecies):
    """Fill in the transitions of ``lines`` from NIST

    Lines whose transition is already known (not ``UNKNOWN``) are left alone, so this is safe to rerun
    with more NIST data.  A line matches NIST rows within ``max(0.0003, 5*uncertainty)`` nm (NIST's own
    uncertainties nearby); rows at the same wavelength as the closest match (within 0.0003 nm) are a
    blend, and their tokens are joined with ``;``.  Lines NIST lists without a classification stay
    ``UNKNOWN``.  Matched lines get `ReferenceLineSource.NIST`.

    Returns
    -------
    newLines : `ReferenceLineSet`
    counts : `dict` [`str`, `int`]
        ``matched``, ``blended``, ``unmatched``, ``unclassified``, ``noSpecies``, ``skipped``.
    """
    data = lines.data.copy()
    transition = data.transition.to_numpy().copy()
    source = data.source.to_numpy().copy()
    counts = dict(matched=0, blended=0, unmatched=0, unclassified=0, noSpecies=0, skipped=0)
    for i, (lam, desc) in enumerate(zip(data.wavelength.to_numpy(), data.description.to_numpy())):
        if transition[i] != "UNKNOWN":
            counts["skipped"] += 1
            continue
        candidates = bySpecies.get(parseSpecies(desc))
        if candidates is None:
            counts["noSpecies"] += 1
            continue
        nearbyUnc = [r["_unc"] for r in candidates if abs(r["_wl"] - lam) < 0.05]
        tol = max(0.0003, 5*max(nearbyUnc, default=0.0003))
        cands = [r for r in candidates if abs(r["_wl"] - lam) < tol]
        if not cands:
            counts["unmatched"] += 1
            continue
        cands = [r for r in cands if r.get("conf_i") and r.get("conf_k")]   # unclassified lines -> UNKNOWN
        if not cands:
            counts["unclassified"] += 1
            continue
        # A blend is components at the same wavelength as the closest match; with large NIST uncertainties
        # tol could otherwise pull in a resolved neighbour
        best = min(cands, key=lambda r: abs(r["_wl"] - lam))
        blend = [r for r in cands if abs(r["_wl"] - best["_wl"]) <= 0.0003]
        tokens = list(dict.fromkeys(transitionToken(r) for r in blend))
        transition[i] = ";".join(tokens)
        source[i] |= int(ReferenceLineSource.NIST)
        counts["matched" if len(tokens) == 1 else "blended"] += 1
    data["transition"] = transition
    data["source"] = source
    newLines = type(lines)(data)
    newLines.sort()
    return newLines, counts


def _gWeight(J):
    try:
        return 2*float(Fraction(J)) + 1
    except (ValueError, ZeroDivisionError):
        return np.nan


def buildLineList(filename, description, *, minFrac=0.0, lamMin=370.0, lamMax=1270.0, gap=0.06):
    """Build a line list for one species from a NIST lines dump

    Components closer than ``gap`` nm (unresolved by PFS) are merged into one line at the
    ``g_k*A_ki``-weighted mean Ritz wavelength, with their transitions combined
    (e.g. ``2p_3P*_0/1/2->3d_3D_1/2/3``; ``;`` between different term pairs).  Only groups that NIST
    gives an intensity for are kept, and of those only lines with intensity >= ``minFrac`` times the
    strongest.  Intensities are NIST's (measure them from PFS data afterwards).

    Returns
    -------
    lines : `ReferenceLineSet`
    """
    rows = []
    with open(filename) as fd:
        for r in csv.DictReader(fd, delimiter="\t"):
            r = {k: (v or "").strip('"') for k, v in r.items() if k}
            ritz = re.sub(r"[^0-9.]", "", r.get("ritz_wl_vac(nm)", ""))
            obs = re.sub(r"[^0-9.]", "", r.get("obs_wl_vac(nm)", ""))
            if not (ritz or obs):
                continue
            lam = float(ritz or obs)
            if not lamMin < lam < lamMax:
                continue
            A = float(r["Aki(s^-1)"]) if r.get("Aki(s^-1)") else np.nan
            intensity = re.sub(r"[^0-9.]", "", r.get("intens", ""))
            rows.append(dict(lam=lam, gA=_gWeight(r.get("J_k", ""))*A,
                             intensity=float(intensity) if intensity else np.nan,
                             lo=(r.get("conf_i", "").split(".")[-1], r.get("term_i", "")),
                             hi=(r.get("conf_k", "").split(".")[-1], r.get("term_k", "")),
                             Ji=r.get("J_i", ""), Jk=r.get("J_k", "")))
    rows.sort(key=lambda r: r["lam"])
    groups = []
    for r in rows:
        if groups and r["lam"] - groups[-1][-1]["lam"] < gap:
            groups[-1].append(r)
        else:
            groups.append([r])
    out = []
    for g in groups:
        intensities = [r["intensity"] for r in g if np.isfinite(r["intensity"])]
        if not intensities:
            continue
        lam = np.array([r["lam"] for r in g])
        weight = np.nan_to_num(np.array([r["gA"] for r in g]))
        center = np.sum(lam*weight)/np.sum(weight) if np.sum(weight) > 0 else lam.mean()
        terms = OrderedDict()
        for r in sorted(g, key=lambda r: -np.nan_to_num(r["gA"])):
            jLo, jHi = terms.setdefault((r["lo"], r["hi"]), (set(), set()))
            jLo.add(r["Ji"])
            jHi.add(r["Jk"])
        token = ";".join(f"{lo[0]}_{lo[1]}_{'/'.join(sorted(ji))}->{hi[0]}_{hi[1]}_{'/'.join(sorted(jk))}"
                         for (lo, hi), (ji, jk) in terms.items())
        if "__->__" in token or not token:
            token = "UNKNOWN"   # NIST lists the line, but unclassified
        out.append(dict(description=description, wavelength=round(center, 5), intensity=max(intensities),
                        status=0, transition=token, source=int(ReferenceLineSource.NIST),
                        intensityErr=np.nan))
    columns = ["description", "wavelength", "intensity", "status", "transition", "source", "intensityErr"]
    df = pd.DataFrame(out, columns=columns)
    if len(df) > 0:
        df = df[df.intensity >= minFrac*df.intensity.max()].reset_index(drop=True)
    lines = ReferenceLineSet(df)
    lines.sort()
    return lines


def readLevels(filename):
    """Read a NIST energy-levels dump

    Returns
    -------
    levels : `list` of `dict`
        ``energy`` (cm^-1), ``valence``, ``term``, ``J`` (`list` of `fractions.Fraction`; NIST sometimes
        gives several, e.g. "2,3,4") and ``Jstr``.
    """
    levels = []
    with open(filename) as fd:
        for r in csv.DictReader(fd, delimiter="\t"):
            r = {k: (v or "").strip('"') for k, v in r.items() if k}
            try:
                energy = float(r["Level (cm-1)"].strip("[]()?+"))
                J = [Fraction(j) for j in r["J"].split(",")]
            except (KeyError, ValueError, ZeroDivisionError):
                continue
            levels.append(dict(energy=energy, valence=r["Configuration"].split(".")[-1], term=r["Term"], J=J,
                               Jstr=r["J"]))
    return levels


def findTransitions(levels, wavelength, tolerance=0.5):
    """List electric-dipole-allowed transitions with Ritz (vacuum) wavelengths near ``wavelength``

    Allowed means a change of parity (``*`` in the term) and ``|dJ| <= 1`` (not 0 -> 0).

    Returns
    -------
    transitions : `pandas.DataFrame`
        ``ritz`` (nm), ``offset`` (ritz - wavelength) and the transition token, sorted by ``|offset|``.
    """
    rows = []
    for lo in levels:
        for hi in levels:
            dE = hi["energy"] - lo["energy"]
            if dE <= 0:
                continue
            ritz = 1e7/dE
            if abs(ritz - wavelength) > tolerance or ("*" in lo["term"]) == ("*" in hi["term"]):
                continue
            if not any(abs(a - b) <= 1 and not (a == 0 and b == 0) for a in lo["J"] for b in hi["J"]):
                continue
            rows.append(dict(ritz=ritz, offset=ritz - wavelength,
                             transition=f"{lo['valence']}_{lo['term']}_{lo['Jstr']}->"
                                        f"{hi['valence']}_{hi['term']}_{hi['Jstr']}"))
    df = pd.DataFrame(rows, columns=["ritz", "offset", "transition"])
    return df.reindex(df.offset.abs().sort_values().index).reset_index(drop=True)
