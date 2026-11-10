"""Tools for transferring datasets out of butler repos"""

import logging
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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
    if not clobber and refs:
        existing = _findExistingFiles(source, target, refs)
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
    imported = target.transfer_from(
        source, refs, transfer=transfer, register_dataset_types=True, transfer_dimensions=True
    )
    log.info("Pulled %d datasets.", len(imported))
    return len(imported)


def _findExistingFiles(
    source: Butler, target: Butler, refs: Iterable[DatasetRef]
) -> dict[DatasetRef, list[ResourcePath]]:
    """Find datasets whose files would clobber existing files in the target

    `Butler.transfer_from` skips datasets that are already registered in the
    target datastore, but for unregistered datasets it overwrites any file
    already present at the target location (e.g., left over from an
    interrupted transfer, or put there by some other means). This identifies
    those datasets so they can be skipped.

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
    existing : `dict` [`DatasetRef`, `list` [`ResourcePath`]]
        Datasets whose target file(s) already exist, along with the URIs of
        those files.
    """
    from lsst.daf.butler.datastores.file_datastore.transfer import retrieve_file_transfer_records

    datastore = target._datastore
    unknown = {ref.id: ref for ref in refs}
    if not unknown:
        return {}
    # Handle chained datastores by checking each constituent file datastore.
    datastores = [
        ds for ds in getattr(datastore, "datastores", [datastore]) if hasattr(ds, "locationFactory")
    ]
    records = retrieve_file_transfer_records(source._file_transfer_source, unknown.values(), {})
    existing: dict[DatasetRef, list[ResourcePath]] = {}
    for datasetId, transferRecords in records.items():
        for record in transferRecords:
            for ds in datastores:
                location = record.file_info.file_location(ds.locationFactory)
                if location.pathInStore.isabs():
                    continue  # Not copied into the datastore
                if location.uri.exists():
                    existing.setdefault(unknown[datasetId], []).append(location.uri)
    return existing


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
