"""Tools for transferring datasets out of butler repos"""

import contextlib
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from collections import deque
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from lsst.daf.butler import (
    Butler,
    CalibrationLookupError,
    CollectionType,
    DataCoordinate,
    DatasetRef,
    DatasetType,
    EmptyQueryResultError,
    Timespan,
)
from lsst.daf.butler.registry._registry import CollectionArgType
from lsst.pipe.base import QuantumGraph
from lsst.resources import ResourcePath
from .gen3 import butlerImport, getDataCoordinate
from .utils.logging import getLogger

__all__ = (
    "transferButlerFiles",
    "butlerTransfer",
    "mirrorCollectionChains",
    "extractOverallInputRefs",
    "butlerTransferForGraph",
    "butlerTransferCalib",
)


def transferButlerFiles(
    repo: str,
    collections: CollectionArgType,
    datasetTypes: Iterable[str],
    dataId: Optional[Dict[str, Any]] = None,
    directory: str = ".",
    transfer: str = "symlink",
) -> List[ResourcePath]:
    """Transfer dataset files from a butler repo to a local directory

    Parameters
    ----------
    repo : `str`
        URI to the location of the repo or URI to a config file describing the
        repo and its location.
    collections : collection expression
        Data collections to search.
    datasetTypes : iterable of `str`
        Names of the dataset types to transfer.
    dataId : `dict` mapping `str` to `int`/`str`, optional
        Data identifier to select the datasets.
    directory : `str`
        Directory to which the files will be transferred.
    transfer : `str`
        Transfer mode to use, as accepted by `Butler.retrieveArtifacts` (e.g.,
        "auto", "copy", "link", "symlink", "hardlink", "relsymlink").

    Returns
    -------
    targets : `list` of `lsst.resources.ResourcePath`
        URIs of the transferred files.
    """
    log = getLogger("pfs.transferButlerFiles")
    butler = Butler(repo, collections=collections, writeable=False)
    dataCoord = getDataCoordinate(dataId, butler.dimensions) if dataId else {}
    targets = []
    for datasetType in datasetTypes:
        ref = butler.find_dataset(datasetType, dataCoord, collections=collections)
        if ref is None:
            raise LookupError(f"No {datasetType!r} dataset found for dataId={dataId} in {collections}")
        # Transferred one dataset type at a time, since retrieveArtifacts does not
        # preserve the correspondence between input refs and output targets.
        newTargets = butler.retrieveArtifacts([ref], directory, transfer=transfer, preserve_path=False)
        for target in newTargets:
            log.info("Transferred %s to %s", datasetType, target)
        targets.extend(newTargets)
    return targets


def butlerTransfer(
    sourceRepo: str,
    targetRepo: str,
    collections: CollectionArgType,
    datasetTypes: Iterable[str] = (),
    dataIds: Optional[Iterable[Dict[str, Union[int, str]]]] = None,
    where: str = "",
    transfer: str = "copy",
    resolveCalib: bool = False,
    clobber: bool = False,
) -> int:
    """Pull datasets from a remote butler repo into a local repo

    This transfers both the registry records and the dataset files for
    ordinary (non-calibration) datasets, so that ``pipetask`` can
    subsequently run against ``targetRepo`` alone, without needing any
    further access to ``sourceRepo``. Already-present datasets are not
    re-transferred.

    For ``CALIBRATION`` collections, use `butlerTransferCalib` instead, so
    that certification timespans are preserved.

    Parameters
    ----------
    sourceRepo : `str`
        URI to the location of the source (remote) repo, or URI to a config
        file describing it.
    targetRepo : `str`
        URI to the location of the target repo.
    collections : collection expression
        Data collections to search in the source repo.
    datasetTypes : iterable of `str`, optional
        Names of the dataset types to pull. If empty, all dataset types are
        considered. Required (and may not be empty) if ``resolveCalib`` is
        `True`.
    dataIds : iterable of `dict` [`str`: `int` or `str`], optional
        Data identifiers to select the datasets to pull. If not provided,
        all datasets of the given ``datasetTypes`` (subject to ``where``)
        are considered. Required if ``resolveCalib`` is `True`.
    where : `str`, optional
        SQL WHERE clause to restrict the datasets pulled. Not supported
        together with ``resolveCalib``.
    transfer : `str`, optional
        Transfer mode to use for the file artifacts (e.g., "auto", "copy",
        "link", "symlink", "hardlink").
    resolveCalib : `bool`, optional
        Instead of querying ``collections`` directly, resolve the single
        calibration dataset of each of ``datasetTypes`` that is valid (per
        its ``CALIBRATION`` collection certification timespan) for each of
        ``dataIds`` -- the same per-exposure calibration lookup ``pipetask``
        itself performs, without needing a pipeline or quantum graph. Each
        dataId must include a temporal dimension (``exposure`` or
        ``visit``).
    clobber : `bool`, optional
        Overwrite files that already exist in the target repo's datastore but
        are not registered there? If `False`, such datasets are not
        transferred (and a warning is issued).

    Returns
    -------
    num : `int`
        Number of datasets transferred.
    """
    log = getLogger("pfs.butlerTransfer")
    source = Butler.from_config(sourceRepo, writeable=False)
    types = list(datasetTypes)
    refs: set[DatasetRef] = set()
    calibAssociations: list[Tuple[str, DatasetRef, Timespan]] = []
    if resolveCalib:
        if where:
            raise ValueError("'where' is not supported together with resolveCalib=True")
        if not types:
            raise ValueError("'datasetTypes' must be provided when resolveCalib=True")
        if not dataIds:
            raise ValueError("'dataIds' must be provided when resolveCalib=True")
        for ident in dataIds:
            coord = getDataCoordinate(ident, source.dimensions)
            timespan = _timespanFromDataId(source, coord)
            for datasetType in types:
                datasetTypeObj = source.get_dataset_type(datasetType)
                constraint = coord.subset(coord.dimensions & datasetTypeObj.dimensions)
                ref = source.find_dataset(
                    datasetTypeObj, constraint, collections=collections, timespan=timespan
                )
                if ref is None:
                    log.warning("No valid %s found for dataId=%s at %s.", datasetType, ident, timespan)
                    continue
                refs.add(ref)
                association = _findCalibAssociation(source, datasetTypeObj, collections, ref, timespan)
                if association is None:
                    log.warning(
                        "Unable to determine the certifying CALIBRATION collection for %s; "
                        "it will be transferred but not re-certified locally.",
                        ref,
                    )
                    continue
                calibCollection, calibTimespan = association
                calibAssociations.append((calibCollection, ref, calibTimespan))
    else:
        if not types:
            types = [dt.name for dt in source.registry.queryDatasetTypes(...)]
        coords = [getDataCoordinate(ident, source.dimensions) for ident in dataIds] if dataIds else [None]
        for datasetType in types:
            for coord in coords:
                try:
                    refs.update(
                        source.query_datasets(
                            datasetType,
                            collections=collections,
                            data_id=coord,
                            where=where,
                            limit=None,
                        )
                    )
                except EmptyQueryResultError:
                    log.debug("No %s datasets found for dataId=%s.", datasetType, coord)
                    continue
                except CalibrationLookupError:
                    log.warning(
                        "Unable to resolve calibration dataset type %s for dataId=%s without a "
                        "temporal constraint; skipping. Use --resolve-calib or --calib for "
                        "calibrations.",
                        datasetType,
                        coord,
                    )
                    continue
    if not refs:
        log.warning("No datasets found to pull.")
        return 0
    target = Butler.from_config(targetRepo, writeable=True)
    mirrorCollectionChains(source, target, collections)
    num = _transferRefs(log, source, target, refs, transfer, clobber)
    for calibCollection, ref, calibTimespan in calibAssociations:
        if target.get_dataset(ref.id) is None:
            # Skipped by _transferRefs (e.g., unregistered file already in the target)
            log.warning("Not certifying %s in %s because it is not in the target.", ref, calibCollection)
            continue
        _certifyIfNeeded(log, target, calibCollection, ref, calibTimespan)
    return num


def _timespanFromDataId(source: Butler, coord: DataCoordinate) -> Timespan:
    """Derive the observation timespan of a dataId

    Parameters
    ----------
    source : `Butler`
        Butler to query for dimension records.
    coord : `DataCoordinate`
        Data coordinate, including a temporal dimension (``exposure`` or
        ``visit``).

    Returns
    -------
    timespan : `Timespan`
        Observation timespan.

    Raises
    ------
    ValueError
        If neither an ``exposure`` nor a ``visit`` dimension is present (or
        resolvable) in ``coord``.
    """
    for dimension in ("exposure", "visit"):
        if dimension not in coord.dimensions.names:
            continue
        records = list(source.registry.queryDimensionRecords(dimension, dataId=coord))
        if records:
            return records[0].timespan
    raise ValueError(f"Unable to determine a timespan from dataId={coord}; need 'exposure' or 'visit'")


def _transferRefs(
    log: logging.Logger,
    source: Butler,
    target: Butler,
    refs: Iterable[DatasetRef],
    transfer: str,
    clobber: bool = False,
) -> int:
    """Transfer a set of resolved dataset references from one butler to
    another

    Parameters
    ----------
    log : `logging.Logger`
        Logger to use for progress messages.
    source : `Butler`
        Read-only butler to pull datasets from.
    target : `Butler`
        Writeable butler to pull datasets into.
    refs : iterable of `DatasetRef`
        Dataset references to transfer. Datasets already present in the
        target datastore (by UUID) are skipped.
    transfer : `str`
        Transfer mode to use for the file artifacts.
    clobber : `bool`, optional
        Overwrite files that already exist in the target datastore (but are
        not registered there)? If `False`, such datasets are skipped.

    Returns
    -------
    num : `int`
        Number of datasets actually transferred (excluding those skipped
        because they already exist in the target).
    """
    known = target._datastore.knows_these(set(refs))
    numPresent = sum(known.values())
    refs = {ref for ref, isKnown in known.items() if not isKnown}
    if numPresent > 0:
        log.info("Skipping %d datasets already present in the target.", numPresent)
    targetFiles = _getTargetFiles(source, target, refs) if refs else {}
    if not clobber and targetFiles:
        existing = {ref: [uri for uri, _ in files if uri.exists()] for ref, files in targetFiles.items()}
        existing = {ref: uris for ref, uris in existing.items() if uris}
        if existing:
            for ref, uris in existing.items():
                log.warning(
                    "Not transferring %s because file(s) already exist in the target: %s",
                    ref,
                    ", ".join(str(uri) for uri in uris),
                )
            log.warning(
                "Skipping %d datasets with unregistered files already in the target; "
                "use clobber to overwrite.",
                len(existing),
            )
            refs -= existing.keys()
    if not refs:
        log.info("No datasets to pull.")
        return 0
    log.info("Pulling %d datasets...", len(refs))
    files = [(uri.ospath, size) for ref in refs for uri, size in targetFiles.get(ref, [])]
    monitor = _TransferMonitor(log, files) if files and "link" not in transfer else contextlib.nullcontext()
    with monitor:
        imported = target.transfer_from(
            source, refs, transfer=transfer, register_dataset_types=True, transfer_dimensions=True
        )
    log.info("Pulled %d datasets.", len(imported))
    return len(imported)


def _getTargetFiles(
    source: Butler, target: Butler, refs: Iterable[DatasetRef]
) -> dict[DatasetRef, list[tuple[ResourcePath, int]]]:
    """Determine where the files of datasets will be written in the target

    This mirrors the calculation of target locations in
    `FileDatastore.transfer_from`, so that we can check for existing files
    before the transfer (`Butler.transfer_from` skips datasets that are
    already registered in the target datastore, but for unregistered datasets
    it overwrites any file already present at the target location, e.g., left
    over from an interrupted transfer), and monitor progress during it.

    Parameters
    ----------
    source : `Butler`
        Butler from which datasets will be transferred.
    target : `Butler`
        Butler into which datasets will be transferred.
    refs : iterable of `DatasetRef`
        Datasets to be transferred, none of which are already registered in
        the target datastore.

    Returns
    -------
    targetFiles : `dict` [`DatasetRef`, `list` [`tuple`]]
        For each dataset, a list of the target URI and expected size (bytes;
        negative if unknown) of each of its files. Files that are not copied
        into the target datastore (absolute URIs) are not included.

    Notes
    -----
    There is no public API for this, so this relies on daf_butler internals
    (``Butler._datastore``, ``Butler._file_transfer_source``,
    ``retrieve_file_transfer_records`` and ``FileDatastore.locationFactory``)
    and on replicating the location calculation in
    ``FileDatastore.transfer_from``. Changes to daf_butler may break it: the
    likely symptom is an exception (e.g., `AttributeError`), but if the
    location calculation changes, the computed paths could be wrong, so that
    existing files are not detected (and are overwritten) and progress is
    not reported.
    """
    from lsst.daf.butler.datastores.file_datastore.transfer import retrieve_file_transfer_records

    unknown = {ref.id: ref for ref in refs}
    if not unknown:
        return {}
    # For a chained datastore, use the first file datastore.
    datastore = target._datastore
    datastores = [
        ds for ds in getattr(datastore, "datastores", [datastore]) if hasattr(ds, "locationFactory")
    ]
    if not datastores:
        return {}
    locationFactory = datastores[0].locationFactory
    records = retrieve_file_transfer_records(source._file_transfer_source, unknown.values(), {})
    targetFiles: dict[DatasetRef, list[tuple[ResourcePath, int]]] = {}
    for datasetId, transferRecords in records.items():
        for record in transferRecords:
            location = record.file_info.file_location(locationFactory)
            if location.pathInStore.isabs():
                continue  # Not copied into the datastore
            targetFiles.setdefault(unknown[datasetId], []).append((location.uri, record.file_info.file_size))
    return targetFiles


def _formatBytes(num: float) -> str:
    """Format a number of bytes for humans

    Parameters
    ----------
    num : `float`
        Number of bytes.

    Returns
    -------
    text : `str`
        Formatted number of bytes, with units.
    """
    for unit in ("B", "kB", "MB", "GB"):
        if abs(num) < 1000:
            return f"{num:.1f} {unit}" if unit != "B" else f"{num:.0f} {unit}"
        num /= 1000
    return f"{num:.1f} TB"


def _formatDuration(seconds: float) -> str:
    """Format a duration for humans

    Parameters
    ----------
    seconds : `float`
        Duration, in seconds.

    Returns
    -------
    text : `str`
        Formatted duration.
    """
    seconds = int(round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


class _TransferMonitor:
    """Monitor and report the progress of a file transfer

    A background thread periodically scans the target directories and
    reports the number of bytes that have arrived, the transfer speed and an
    estimate of the time remaining. This is used as a context manager around
    the transfer.

    If ``stream`` is a terminal, the report is a single status line that is
    updated in place (and cleared whenever a log message is emitted, so the
    two don't collide); otherwise, the report is logged.

    Files are copied into a temporary file in the target directory and then
    renamed (see ``lsst.resources.FileResourcePath.transfer_from``), so the
    bytes transferred include those in any new file that has appeared in the
    target directories, not just the target files. Files that existed before
    the transfer started (identified by their inode) are ignored.

    Parameters
    ----------
    log : `logging.Logger`
        Logger to use for progress messages.
    files : iterable of `tuple` [`str`, `int`]
        Target path and expected size (bytes; negative if unknown) of each
        file to be transferred.
    interval : `float`, optional
        Interval between progress reports (seconds). Defaults to 1 second for
        a status line, or 30 seconds for logging.
    window : `float`, optional
        Time window over which to measure the current speed (seconds).
    stream : file-like, optional
        Stream for the status line. Defaults to ``sys.stderr``.
    """

    def __init__(
        self,
        log: logging.Logger,
        files: Iterable[tuple[str, int]],
        interval: float | None = None,
        window: float = 5.0,
        stream: Any = None,
    ):
        self.log = log
        self.stream = stream if stream is not None else sys.stderr
        try:
            self.isTerminal = self.stream.isatty()
        except Exception:
            self.isTerminal = False
        self.interval = interval if interval is not None else (1.0 if self.isTerminal else 30.0)
        self.window = window
        sizes = dict(files)
        self.targets = set(sizes.keys())
        self.numFiles = len(sizes)
        self.totalBytes = sum(sizes.values()) if all(size >= 0 for size in sizes.values()) else None
        self.directories = {os.path.dirname(path) for path in self.targets}
        self.preexisting = {path: inode for path, inode, _ in self._scan()}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="transferMonitor", daemon=True)
        self._startTime = 0.0
        self._lock = threading.Lock()  # Protects the status line
        self._lineShown = False  # Is the status line currently displayed?
        self._filter = _ClearStatusFilter(self)
        self._handlers: list[logging.Handler] = []

    def _scan(self) -> Iterator[tuple[str, int, int]]:
        """Scan the target directories

        Yields
        ------
        path : `str`
            Path of a file.
        inode : `int`
            Inode of the file.
        size : `int`
            Size of the file (bytes).
        """
        for directory in self.directories:
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            yield entry.path, entry.inode(), entry.stat(follow_symlinks=False).st_size
                        except FileNotFoundError:
                            continue  # Temporary file was renamed or removed
            except FileNotFoundError:
                continue  # Directory not yet created

    def _measure(self) -> tuple[int, int]:
        """Measure the progress of the transfer

        Returns
        -------
        numBytes : `int`
            Number of bytes transferred.
        numDone : `int`
            Number of target files completed.
        """
        numBytes = 0
        numDone = 0
        for path, inode, size in self._scan():
            if self.preexisting.get(path) == inode:
                continue
            numBytes += size
            if path in self.targets:
                numDone += 1
        return numBytes, numDone

    def _report(self, numBytes: int, numDone: int, elapsed: float, speed: float) -> None:
        """Report progress

        Parameters
        ----------
        numBytes : `int`
            Number of bytes transferred.
        numDone : `int`
            Number of target files completed.
        elapsed : `float`
            Time since the start of the transfer (seconds).
        speed : `float`
            Current transfer speed (bytes/second).
        """
        if self.totalBytes:
            fraction = min(numBytes / self.totalBytes, 1.0)
            progress = f"{_formatBytes(numBytes)}/{_formatBytes(self.totalBytes)} ({100*fraction:.0f}%)"
            average = numBytes / elapsed if elapsed > 0 else 0.0
            remaining = self.totalBytes - numBytes
            eta = f", ETA {_formatDuration(remaining/average)}" if average > 0 and remaining > 0 else ""
        else:
            progress = _formatBytes(numBytes)
            eta = ""
        text = f"Transferred {progress}, {numDone}/{self.numFiles} files, {_formatBytes(speed)}/s{eta}"
        if not self.isTerminal:
            self.log.info("%s", text)
            return
        # Truncate to avoid wrapping, which would defeat the carriage return
        width = shutil.get_terminal_size().columns - 1
        with self._lock:
            self.stream.write("\r" + text[:width] + "\x1b[K")
            self.stream.flush()
            self._lineShown = True

    def clearStatus(self) -> None:
        """Clear the status line, if it is displayed"""
        with self._lock:
            if self._lineShown:
                self.stream.write("\r\x1b[K")
                self.stream.flush()
                self._lineShown = False

    def _run(self) -> None:
        """Periodically report progress until stopped"""
        samples: deque[tuple[float, int]] = deque([(self._startTime, 0)])  # (time, bytes)
        while not self._stop.wait(self.interval):
            try:
                numBytes, numDone = self._measure()
                now = time.monotonic()
                # Measure the speed over the last 'window' seconds
                while len(samples) > 1 and now - samples[1][0] >= self.window:
                    samples.popleft()
                startTime, startBytes = samples[0]
                speed = (numBytes - startBytes) / (now - startTime) if now > startTime else 0.0
                samples.append((now, numBytes))
                self._report(numBytes, numDone, now - self._startTime, max(speed, 0.0))
            except Exception as exc:  # Never let monitoring interfere with the transfer
                self.log.debug("Unable to report transfer progress: %s", exc)

    def __enter__(self) -> "_TransferMonitor":
        if self.isTerminal:
            # Clear the status line before any log message is emitted
            loggers = [logging.getLogger()] + [
                ll for ll in logging.Logger.manager.loggerDict.values() if isinstance(ll, logging.Logger)
            ]
            handlers = {hh for ll in loggers for hh in ll.handlers}
            self._handlers = [hh for hh in handlers if self._filter not in hh.filters]
            for handler in self._handlers:
                handler.addFilter(self._filter)
        self._startTime = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, excType, excValue, traceback) -> None:
        self._stop.set()
        self._thread.join()
        self.clearStatus()
        for handler in self._handlers:
            handler.removeFilter(self._filter)
        self._handlers = []
        if excType is None:
            elapsed = time.monotonic() - self._startTime
            numBytes, _ = self._measure()
            speed = f" ({_formatBytes(numBytes/elapsed)}/s)" if elapsed > 0 else ""
            self.log.info(
                "Transferred %s in %d files in %s%s",
                _formatBytes(numBytes),
                self.numFiles,
                _formatDuration(elapsed),
                speed,
            )


class _ClearStatusFilter(logging.Filter):
    """Logging filter that clears a transfer status line before each message

    This doesn't filter anything; it's a hook that runs just before a
    handler emits a log record.

    Parameters
    ----------
    monitor : `_TransferMonitor`
        Monitor whose status line should be cleared.
    """

    def __init__(self, monitor: _TransferMonitor):
        super().__init__()
        self.monitor = monitor

    def filter(self, record: logging.LogRecord) -> bool:
        self.monitor.clearStatus()
        return True


def mirrorCollectionChains(source: Butler, target: Butler, collections: CollectionArgType) -> None:
    """Recursively mirror collection structure from one butler to another

    For each name in ``collections``, registers the same named collection on
    ``target`` with the same type as on ``source``. For any that are
    ``CHAINED``, also registers the same ordered list of child collections,
    recursing into any of those children that are themselves ``CHAINED`` --
    regardless of whether those children currently contain any datasets on
    ``target``. This is pure registry metadata (no file copying), so it's
    safe and cheap to call unconditionally. Already-registered collections
    (with a matching type) are left alone, except that the children of a
    ``CHAINED`` collection are always reset to match ``source``, replacing any
    local changes to that chain.

    This does not transfer any dataset content or ``CALIBRATION``
    certifications -- just the collection names and ``CHAINED`` structure
    needed for ``target`` to be searched the same way as ``source``.

    Fetches the full descendant tree of ``collections`` from ``source`` in a
    single bulk query (``Butler.collections.query_info`` with
    ``flatten_chains=True``, which uses a recursive CTE under the hood)
    rather than walking it one collection at a time, since each individual
    lookup is a separate round trip to the (likely remote) source registry.

    Parameters
    ----------
    source : `Butler`
        Butler to mirror collection structure from.
    target : `Butler`
        Butler to mirror collection structure into.
    collections : collection expression
        Names of collections whose structure (and full descendant tree, for
        any that are ``CHAINED``) should be mirrored. Must be exact names,
        not glob expressions.
    """
    if isinstance(collections, str):
        collections = [collections]
    collections = list(collections)
    infos = {
        info.name: info
        for info in source.collections.query_info(
            collections, flatten_chains=True, include_chains=True, include_doc=True
        )
    }
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        info = infos[name]
        if info.type is CollectionType.CHAINED:
            for child in info.children:
                visit(child)
            target.collections.register(name, CollectionType.CHAINED, doc=info.doc)
            target.registry.setCollectionChain(name, list(info.children))
        else:
            target.collections.register(name, info.type, doc=info.doc)

    for name in collections:
        visit(name)


def _findCalibAssociation(
    source: Butler,
    datasetType: DatasetType,
    collections: CollectionArgType,
    ref: DatasetRef,
    timespan: Timespan,
) -> Optional[Tuple[str, Timespan]]:
    """Determine the real certifying CALIBRATION collection and validity
    timespan for a dataset resolved via a point-in-time calibration lookup

    Parameters
    ----------
    source : `Butler`
        Butler to query for dataset associations.
    datasetType : `DatasetType`
        Dataset type of ``ref``.
    collections : collection expression
        Collections ``ref`` was resolved from (searched, possibly
        ``CHAINED``).
    ref : `DatasetRef`
        The resolved dataset reference.
    timespan : `Timespan`
        The point-in-time timespan ``ref`` was resolved for.

    Returns
    -------
    result : `tuple` of (`str`, `Timespan`), or `None`
        The real certifying collection name and its full validity timespan,
        or `None` if no matching ``CALIBRATION`` association was found (e.g.
        ``collections`` did not actually contain a ``CALIBRATION``
        collection for this dataset).
    """
    for association in source.registry.queryDatasetAssociations(
        datasetType,
        collections=collections,
        collectionTypes={CollectionType.CALIBRATION},
        flattenChains=True,
    ):
        # Use overlap (not containment), matching the calibration lookup
        if (
            association.ref.id == ref.id
            and association.timespan is not None
            and association.timespan.overlaps(timespan)
        ):
            return association.collection, association.timespan
    return None


def _certifyIfNeeded(
    log: logging.Logger, target: Butler, collection: str, ref: DatasetRef, timespan: Timespan
) -> None:
    """Certify a dataset reference into a CALIBRATION collection, unless it
    is already certified there with the same timespan

    Parameters
    ----------
    log : `logging.Logger`
        Logger to use for progress messages.
    target : `Butler`
        Writeable butler to certify into.
    collection : `str`
        ``CALIBRATION`` collection to certify ``ref`` into. Must already be
        registered (e.g. via `mirrorCollectionChains`).
    ref : `DatasetRef`
        Dataset reference to certify. Must already exist in ``target``'s
        datastore/registry (e.g. via `_transferRefs`).
    timespan : `Timespan`
        Validity timespan to certify with.
    """
    for association in target.registry.queryDatasetAssociations(
        ref.datasetType, collections=collection, collectionTypes={CollectionType.CALIBRATION}
    ):
        if association.ref.id == ref.id and association.timespan == timespan:
            log.debug("%s already certified in %s with timespan=%s.", ref, collection, timespan)
            return
    log.info("Certifying %s in %s with timespan=%s.", ref, collection, timespan)
    target.registry.certify(collection, [ref], timespan)


def extractOverallInputRefs(qgraph: QuantumGraph) -> set[DatasetRef]:
    """Extract the overall input dataset references of a quantum graph

    These are the datasets that the graph consumes but does not produce
    itself (raw data, calibrations, reference catalogs, init-inputs, etc.):
    the ones that must already exist for ``pipetask run`` to execute the
    graph. Intermediate products (produced by one quantum and consumed by
    another within the same graph) are deliberately excluded, since those
    will be created locally during execution.

    Parameters
    ----------
    qgraph : `lsst.pipe.base.QuantumGraph`
        Quantum graph to inspect.

    Returns
    -------
    refs : `set` of `DatasetRef`
        The overall input dataset references.
    """
    overallTypes = {name for name, _ in qgraph.pipeline_graph.iter_overall_inputs()}
    refs: set[DatasetRef] = set()
    for node in qgraph:
        for datasetType, typeRefs in node.quantum.inputs.items():
            if datasetType.name in overallTypes:
                refs.update(typeRefs)
    for taskDef in qgraph.iterTaskGraph():
        for ref in qgraph.get_init_input_refs(taskDef.label):
            if ref.datasetType.name in overallTypes:
                refs.add(ref)
    return refs


def butlerTransferForGraph(
    sourceRepo: str,
    targetRepo: str,
    graphPath: str,
    transfer: str = "copy",
    clobber: bool = False,
) -> int:
    """Pull exactly the overall input datasets of a quantum graph from a
    remote butler repo into a local repo

    This is the precise counterpart to `butlerTransfer`: instead of
    guessing at collections/dataset types/data IDs, build the quantum graph
    for the pipeline and data selection you want to run, e.g.::

        pipetask qgraph -b <remote-view-config> -i <collections> -o <run> \\
            -p <pipeline> -d "<query>" -q <graphPath>

    against the remote-view config, then pull exactly the inputs it needs
    with this function, and finally execute the pipeline locally by reusing
    the same graph (so dataset UUIDs line up with what was just pulled)::

        pipetask run -b <local-repo> -g <graphPath>

    Parameters
    ----------
    sourceRepo : `str`
        URI to the location of the source (remote-view) repo, or URI to a
        config file describing it.
    targetRepo : `str`
        URI to the location of the target repo.
    graphPath : `str`
        Path to a quantum graph file (as produced by ``pipetask qgraph``),
        built against ``sourceRepo``.
    transfer : `str`, optional
        Transfer mode to use for the file artifacts.
    clobber : `bool`, optional
        Overwrite files that already exist in the target repo's datastore but
        are not registered there? If `False`, such datasets are not
        transferred (and a warning is issued).

    Returns
    -------
    num : `int`
        Number of datasets transferred.
    """
    log = getLogger("pfs.butlerTransferForGraph")
    qgraph = QuantumGraph.loadUri(graphPath)
    refs = extractOverallInputRefs(qgraph)
    if not refs:
        log.warning("No overall input datasets found in graph.")
        return 0
    source = Butler.from_config(sourceRepo, writeable=False)
    target = Butler.from_config(targetRepo, writeable=True)
    return _transferRefs(log, source, target, refs, transfer, clobber)


def butlerTransferCalib(
    sourceRepo: str,
    targetRepo: str,
    collections: CollectionArgType,
    datasetTypes: Iterable[str] = (),
    transfer: str = "copy",
    clobber: bool = False,
) -> None:
    """Pull calibration datasets from a remote butler repo into a local
    repo, preserving their certification timespans

    This uses ``butler export-calibs`` (via the source repo) followed by
    `butlerImport` (into the target repo), which is the established pattern
    for moving ``CALIBRATION`` collection datasets between repos while
    keeping their certifications intact.

    Parameters
    ----------
    sourceRepo : `str`
        URI to the location of the source (remote) repo, or URI to a config
        file describing it.
    targetRepo : `str`
        URI to the location of the target repo.
    collections : collection expression
        ``CALIBRATION`` (or ``CHAINED`` of such) collections to pull from.
    datasetTypes : iterable of `str`, optional
        Names of the calibration dataset types to pull. If empty, all
        calibration dataset types are considered.
    transfer : `str`, optional
        Transfer mode to use for the file artifacts.
    clobber : `bool`, optional
        Clobber datasets that already exist (by UUID) in the target repo?
    """
    from lsst.daf.butler.script.exportCalibs import exportCalibs

    log = getLogger("pfs.butlerTransferCalib")
    with tempfile.TemporaryDirectory() as tempDir:
        # exportCalibs refuses to write into an existing directory
        directory = os.path.join(tempDir, "export")
        log.info("Exporting calibs from %s to %s...", sourceRepo, directory)
        exportCalibs(sourceRepo, directory, collections, tuple(datasetTypes), transfer)
        log.info("Importing calibs into %s...", targetRepo)
        butlerImport(targetRepo, directory, transfer=transfer, clobber=clobber)
