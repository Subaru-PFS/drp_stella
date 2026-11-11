"""Lamp exposures from the raw headers, and the HgCd lamp's warm-up (PIPE2D-1969)

* `scanLampHeaders` lists the lamps lit in each visit according to the raw headers (``getLamps``), with
  the registry's ``lamps`` field alongside.  The registry is wrong for some visits (~205, mostly 2026
  IIS lamps), so select lamp exposures with this table rather than the registry.
* `lampSeries` groups exposures into series (a gap of more than ``gap`` seconds starts a new one).
* `hgcdRatios` and `warmupGrowth` measure the HgCd lamp's warm-up: the ratio of Cd to Hg line fluxes
  (b arm), relative to the first exposure of each series, against the time since that exposure.
  Ratios don't depend on the exposure time, which is imprecise for HgCd: its exposures are timed by
  the shutter (``W_CLHGCT``).

The HgCd lamp is turned on and left until a photodiode sees the Cd (200-300 s); the voltage is then
lowered and the exposures taken.  The Cd lines keep brightening relative to Hg over the following
minutes: x1.12 at 2 min and x1.20 at 4 min before 2026-07, x1.23 and x1.37 since.
"""
import multiprocessing
import time

import numpy as np
import pandas as pd

__all__ = ["scanLampHeaders", "lampMismatches", "hasLamp", "lampSeries", "HG_LINES", "CD_LINES",
           "integratedLineFluxes", "hgcdRatios", "warmupGrowth", "warmupSummary", "plotWarmup"]

# Isolated b-arm lines (vacuum nm)
HG_LINES = (404.7708, 435.956, 546.22675, 577.121, 579.22757)
CD_LINES = (467.94587, 480.1254, 508.72393)
# Ratios measured: (numerator, denominator); the first is the warm-up diagnostic
RATIOS = ((508.72393, 435.956), (480.1254, 435.956), (467.94587, 435.956), (508.72393, 546.22675),
          (404.7708, 435.956))

_butler = None   # per-process butler for scanLampHeaders


def _initScan(repo, collection):
    global _butler
    from lsst.daf.butler import Butler
    _butler = Butler(repo, collections=[collection])


def _scanOne(args):
    from lsst.obs.pfs.utils import getLamps
    visit, refId = args
    try:
        metadata = _butler.get(_butler.get_dataset(refId).makeComponentRef("metadata"))
        return visit, ",".join(sorted(getLamps(metadata))), metadata.get("DATA-TYP")
    except Exception as exc:
        return visit, f"ERROR {type(exc).__name__}", None


def scanLampHeaders(repo="/work/datastore", collection="PFS/raw/sps", *, since=0, visits=None,
                    nProcesses=64, log=None):
    """List the lamps lit in each visit, according to the raw headers

    One raw per visit is read.  Biases and darks with no lamp in the registry are skipped.

    Parameters
    ----------
    repo : `str`
        Butler repository.
    collection : `str`
        Collection with the raws.
    since : `int`
        Only visits with ``day_obs`` on or after this (YYYYMMDD).
    visits : iterable of `int`, optional
        Only these visits.
    nProcesses : `int`
        Number of processes (~10 min for all ~84k visits with 64).
    log : callable, optional
        Called with progress messages (e.g. ``print``).

    Returns
    -------
    scan : `pandas.DataFrame`
        Columns ``visit``, ``day_obs``, ``obsType``, ``group``, ``registry`` (the registry's lamps),
        ``header`` (``getLamps(header)``, comma-separated and sorted, or ``ERROR ...``) and ``dataType``.
    """
    from lsst.daf.butler import Butler

    butler = Butler(repo, collections=[collection])
    where = "instrument='PFS'"
    if visits is not None:
        where += f" AND visit IN ({','.join(str(int(vv)) for vv in visits)})"
    if since > 0:
        where += f" AND visit.day_obs >= {int(since)}"
    records = {rec.id: rec for rec in butler.registry.queryDimensionRecords("visit", where=where)
               if rec.day_obs >= since and not (rec.observation_type in ("bias", "dark") and not rec.lamps)}
    start = time.time()
    refs = {}
    for ref in butler.registry.queryDatasets("raw", collections=collection, where=where):
        visit = ref.dataId["visit"]
        if visit in records and visit not in refs:
            refs[visit] = ref.id
    if nProcesses > 1:
        context = multiprocessing.get_context("spawn")
        with context.Pool(nProcesses, initializer=_initScan, initargs=(repo, collection)) as pool:
            results = pool.map(_scanOne, list(refs.items()), chunksize=50)
    else:
        _initScan(repo, collection)
        results = [_scanOne(item) for item in refs.items()]
    rows = []
    for visit, header, dataType in results:
        rec = records[visit]
        rows.append(dict(visit=visit, day_obs=rec.day_obs, obsType=rec.observation_type,
                         group=(rec.group_name or "")[:30], registry=rec.lamps or "", header=header,
                         dataType=dataType))
    columns = ["visit", "day_obs", "obsType", "group", "registry", "header", "dataType"]
    scan = pd.DataFrame(rows, columns=columns).sort_values("visit").reset_index(drop=True)
    if log is not None:
        log(f"{len(scan)} visits scanned in {time.time() - start:.0f} s")
    return scan


def lampMismatches(scan):
    """Visits whose registry lamps disagree with the headers (excluding unreadable headers)"""
    return scan[(scan.registry != scan.header) & ~scan.header.str.startswith("ERROR")]


def hasLamp(scan, lamp, exclude=("Quartz", "Quartz_eng")):
    """Select the visits in which a lamp is lit, according to the headers

    Parameters
    ----------
    scan : `pandas.DataFrame`
        Output of `scanLampHeaders`.
    lamp : `str`
        Lamp name (e.g. ``HgCd``, ``Ar_eng``).
    exclude : iterable of `str`
        Reject visits in which any of these lamps is also lit (exact names).

    Returns
    -------
    select : `pandas.Series` of `bool`
        Visits with the lamp lit.
    """
    lit = scan.header.str.split(",")
    exclude = set(exclude)
    return lit.apply(lambda lamps: lamp in lamps and not exclude.intersection(lamps))


def lampSeries(butler, visits, gap=600.0):
    """Group exposures into series, separated by gaps of more than ``gap`` seconds

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler, for the visits' timespans.
    visits : iterable of `int`
        Visits.
    gap : `float`
        A gap (seconds, start to start) longer than this starts a new series.

    Returns
    -------
    series : `pandas.DataFrame`
        Sorted by start time: ``visit``, ``start`` (`datetime`), ``series`` (index), ``tSeries`` (seconds
        since the series' first exposure) and ``idx`` (index of the exposure within the series).
    """
    visits = [int(vv) for vv in visits]
    records = {rec.id: rec for rec in butler.registry.queryDimensionRecords(
        "visit", where=f"instrument='PFS' AND visit IN ({','.join(map(str, visits))})")}
    data = pd.DataFrame(dict(visit=visits, start=[records[vv].timespan.begin.datetime for vv in visits]))
    data = data.sort_values("start").reset_index(drop=True)
    data["series"] = np.cumsum(data.start.diff().dt.total_seconds().fillna(np.inf) > gap)
    data["tSeries"] = (data.start - data.groupby("series").start.transform("min")).dt.total_seconds()
    data["idx"] = data.groupby("series").cumcount()
    return data


def integratedLineFluxes(wavelength, flux, mask, badMask, lines, halfWidth=0.6, sideband=(1.0, 2.0)):
    """Fluxes of isolated lines: the sum over +-halfWidth nm minus the median of a sideband continuum

    Parameters
    ----------
    wavelength, flux, mask : `numpy.ndarray`
        One fibre's spectrum.
    badMask : `int`
        Lines with any of these mask bits in the core are rejected (NaN).
    lines : iterable of `float`
        Line wavelengths (nm).
    halfWidth : `float`
        Half-width of the core (nm).
    sideband : (`float`, `float`)
        Inner and outer distances of the continuum sidebands (nm).

    Returns
    -------
    fluxes : `dict` [`float`, `float`]
        Flux of each line (NaN if it couldn't be measured).
    """
    fluxes = {}
    for wl in lines:
        distance = np.abs(wavelength - wl)
        core = distance < halfWidth
        side = (distance > sideband[0]) & (distance < sideband[1])
        if (core.sum() < 5 or side.sum() < 5 or np.any(mask[core] & badMask)
                or not np.all(np.isfinite(flux[core]))):
            fluxes[wl] = np.nan
        else:
            fluxes[wl] = np.nansum(flux[core] - np.nanmedian(flux[side]))
    return fluxes


def hgcdRatios(butler, visits, arm="b", spectrographs=(1, 2, 3, 4), minFibres=10, log=None):
    """Measure Cd/Hg line ratios in HgCd exposures

    For each exposure's pfsArms, the line fluxes are measured in each fibre (`integratedLineFluxes`,
    rejecting saturated lines), and each ratio is the median over fibres of the per-fibre ratios.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler with the pfsArms (reduce the b arm first: isr, cosmicray, reduceExposure).
    visits : iterable of `int`
        HgCd visits; those without pfsArms are skipped.
    arm : `str`
        Arm (the lines in `HG_LINES` and `CD_LINES` are in b).
    spectrographs : iterable of `int`
        Spectrographs.
    minFibres : `int`
        Minimum number of fibres for a ratio.
    log : callable, optional
        Called with progress messages.

    Returns
    -------
    ratios : `pandas.DataFrame`
        One row per (visit, spectrograph): ``expTime``, the ratios (e.g. ``509/436``), and the median
        line fluxes per second (e.g. ``I436``; per the imprecise shutter time).
    """
    visits = [int(vv) for vv in visits]
    have = {ref.dataId["visit"] for ref in butler.registry.queryDatasets(
        "pfsArm", where=f"instrument='PFS' AND arm='{arm}' AND visit IN ({','.join(map(str, visits))})")}
    rows = []
    for visit in sorted(have):
        for spectrograph in spectrographs:
            try:
                pfsArm = butler.get("pfsArm", visit=visit, arm=arm, spectrograph=spectrograph)
            except Exception:
                continue
            sat = pfsArm.flags["SAT"]
            fluxes = pd.DataFrame([integratedLineFluxes(pfsArm.wavelength[ii], pfsArm.flux[ii],
                                                        pfsArm.mask[ii], sat, HG_LINES + CD_LINES)
                                   for ii in range(len(pfsArm)) if np.nanmedian(pfsArm.norm[ii]) > 0])
            row = dict(visit=visit, spectrograph=spectrograph, expTime=float(pfsArm.identity.expTime))
            for num, den in RATIOS:
                ratio = fluxes[num]/fluxes[den]
                ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                row[f"{num:.0f}/{den:.0f}"] = np.median(ratio) if len(ratio) > minFibres else np.nan
            for wl in HG_LINES + CD_LINES:
                row[f"I{wl:.0f}"] = np.nanmedian(fluxes[wl])/max(row["expTime"], 1e-3)
            rows.append(row)
    if log is not None:
        log(f"{len(rows)} rows ({len(have)} visits)")
    return pd.DataFrame(rows)


def warmupGrowth(series, ratios, *, ratio="509/436", minLight=0.05, maxErr=0.01):
    """The growth of a line ratio within each series of exposures

    Parameters
    ----------
    series : `pandas.DataFrame`
        Output of `lampSeries` (with any extra columns, e.g. ``day_obs``).
    ratios : `pandas.DataFrame`
        Output of `hgcdRatios`.
    ratio : `str`
        Ratio to use (default: Cd 508.7/Hg 436.0).
    minLight : `float`
        Exposures whose median Hg 436 flux is below this fraction of the median over all exposures are
        dropped (the lamp didn't light).
    maxErr : `float`
        Series in which any exposure's relative ratio has an error above this are marked not ``clean``
        (the spectrographs disagree).

    Returns
    -------
    growth : `pandas.DataFrame`
        One row per exposure in series of at least two: ``series``, ``idx``, ``day_obs``, ``t``
        (minutes since the first exposure), ``rel`` and ``relErr`` (the ratio relative to the first
        exposure; the mean and its error over spectrographs), ``hg`` (Hg 404.7/436.0, relative), ``R``
        and ``Rerr`` (the ratio itself), ``nSeries`` and ``clean``.  Errors come from the scatter between
        spectrographs.
    """
    columns = [cc for cc in ("series", "idx", "tSeries", "day_obs") if cc in series.columns]
    data = ratios.join(series.set_index("visit")[columns], on="visit", how="inner")
    lit = data.groupby("visit").I436.median() > minLight*data.I436.median()
    data = data[data.visit.map(lit)].copy()
    for name in (ratio, "405/436"):
        first = data[data.idx == 0].set_index(["series", "spectrograph"])[name]
        data[f"rel{name}"] = [value/first.get((ss, sp), np.nan) for value, ss, sp in
                              zip(data[name], data.series, data.spectrograph)]

    def mean(values):
        return values[np.isfinite(values)].mean()

    def error(values):
        num = np.isfinite(values).sum()
        return values[np.isfinite(values)].std(ddof=1)/np.sqrt(num) if num > 1 else np.nan

    aggregate = dict(t=("tSeries", "first"), rel=(f"rel{ratio}", mean), relErr=(f"rel{ratio}", error),
                     hg=("rel405/436", mean), R=(ratio, mean), Rerr=(ratio, error))
    if "day_obs" in data.columns:
        aggregate["day_obs"] = ("day_obs", "first")
    growth = data.groupby(["series", "idx"]).agg(**aggregate).reset_index()
    growth["t"] /= 60
    growth["nSeries"] = growth.groupby("series").idx.transform("size")
    growth = growth[growth.nSeries >= 2].copy()
    noisy = growth[growth.idx > 0].groupby("series").relErr.max() > maxErr
    growth["clean"] = ~growth.series.map(noisy).fillna(False).astype(bool)
    return growth.reset_index(drop=True)


def warmupSummary(growth, *, epochs=((0, 20260701), (20260701, 99999999)),
                  exposures=((1, (1.5, 2.5)), (2, (3.5, 4.5)))):
    """Mean growth at given exposures within series, for clean series, by epoch

    Parameters
    ----------
    growth : `pandas.DataFrame`
        Output of `warmupGrowth`.
    epochs : iterable of (`int`, `int`)
        ``day_obs`` ranges [start, end).
    exposures : iterable of (`int`, (`float`, `float`))
        Exposure index within the series, and the range of minutes since the first exposure.

    Returns
    -------
    summary : `pandas.DataFrame`
        ``epochStart``, ``epochEnd``, ``idx``, ``t`` (median minutes), ``growth`` (mean), ``err``,
        ``scatter`` and ``num``.
    """
    rows = []
    for start, end in epochs:
        for idx, (tMin, tMax) in exposures:
            select = growth[growth.clean & growth.day_obs.between(start, end - 1) & (growth.idx == idx)
                            & growth.t.between(tMin, tMax)]
            if len(select) > 1:
                scatter = select.rel.std(ddof=1)
                rows.append(dict(epochStart=start, epochEnd=end, idx=idx, t=select.t.median(),
                                 growth=select.rel.mean(), err=scatter/np.sqrt(len(select)), scatter=scatter,
                                 num=len(select)))
    columns = ["epochStart", "epochEnd", "idx", "t", "growth", "err", "scatter", "num"]
    return pd.DataFrame(rows, columns=columns)


def plotWarmup(growth, split=20260701, fig=None):
    """Plot the warm-up: within series; growth at the third exposure, and the first exposure's ratio,
    against date

    Parameters
    ----------
    growth : `pandas.DataFrame`
        Output of `warmupGrowth`.
    split : `int`
        ``day_obs`` separating the two epochs plotted in different colours.
    fig : `matplotlib.figure.Figure`, optional
        Figure to use.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The figure.
    """
    import matplotlib.pyplot as plt

    blue, orange = "#2a78d6", "#eb6834"
    growth = growth.copy()
    growth["date"] = pd.to_datetime(growth.day_obs.astype(str))
    growth["late"] = growth.day_obs >= split
    if fig is None:
        fig = plt.figure(figsize=(16, 4.8), layout="constrained")
    axes = fig.subplots(1, 3)
    for late, color in ((False, blue), (True, orange)):
        for _, ss in growth[growth.clean & (growth.late == late)].groupby("series"):
            axes[0].errorbar(ss.t, ss.rel, yerr=ss.relErr.fillna(0), fmt="-o", color=color, ms=4, lw=1,
                             alpha=0.6, capsize=2)
        for clean, face in ((True, color), (False, "white")):
            ss = growth[(growth.idx == 2) & (growth.late == late) & (growth.clean == clean)]
            axes[1].errorbar(ss.date, ss.rel, yerr=ss.relErr.fillna(0), fmt="o", color=color, mfc=face,
                             capsize=2)
            ss = growth[(growth.idx == 0) & (growth.late == late) & (growth.clean == clean)]
            axes[2].errorbar(ss.date, ss.R, yerr=ss.Rerr.fillna(0), fmt="o", color=color, mfc=face, capsize=2)
    axes[0].set(xlabel="min since first exposure", ylabel="Cd 508.7/Hg 436.0 rel. to first",
                title="Within series (clean)")
    axes[1].set(xlabel="date", ylabel="growth at 3rd exposure", title="Growth by ~4 min (open: noisy)",
                ylim=(0.9, 1.5))
    axes[2].set(xlabel="date", ylabel="Cd/Hg in first exposure", title="First exposure")
    for ax in axes[1:]:
        ax.tick_params(axis="x", rotation=30)
    return fig
