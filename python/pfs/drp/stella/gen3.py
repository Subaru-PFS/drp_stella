"""Helper functions for the LSST Gen3 middleware"""

import functools
import os
from collections.abc import Sequence
from glob import glob
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
from astropy.time import Time

from lsst.daf.butler import (
    Butler,
    CollectionType,
    DataCoordinate,
    DatasetRef,
    DatasetType,
    DimensionGraph,
    DimensionUniverse,
    FileDataset,
    Registry,
    Timespan,
)
from lsst.obs.base.formatters.fitsGeneric import FitsGenericFormatter
from lsst.pipe.base import Instrument
from lsst.pipe.base import QuantumContext
from lsst.pipe.base.connections import InputQuantizedConnection
from lsst.resources import ResourcePath
from lsst.obs.pfs.formatters import DetectorMapFormatter
from pfs.datamodel.target import Target
from pfs.datamodel.utils import calculatePfsVisitHash
from pfs.drp.stella.datamodel import PfsConfig
from .utils.logging import getLogger

__all__ = ("DatasetRefList", "zipDatasetRefs", "readDatasetRefs", "targetFromDataId")


class DatasetRefList(Sequence):
    """List of `DatasetRef` that also allows lookups by `DataCoordinate`

    This is intended as a substitute for ``List[DatasetRef]`` that provides the
    feature that many users will want, viz., retrieval by ``dataId``. This will
    make it easier to iterate over aligned sets of dataset references.

    Parameters
    ----------
    datasetType : `DatasetType`
        The dataset type.
    dimensions : `DimensionGraph`
        The set of dimensions associated with this ``datasetType``.
    """

    def __init__(self, datasetType: DatasetType, dimensions: DimensionGraph):
        self.datasetType = datasetType
        self.dimensions = dimensions
        self._byIndex: List[DatasetRef] = []
        self._byCoordinate: Dict[DataCoordinate, DatasetRef] = {}

    @classmethod
    def fromList(cls, dataRefs: List[DatasetRef], ignoreDuplicates: bool = False) -> "DatasetRefList":
        """Construct from an iterable collection of dataset references

        Parameters
        ----------
        dataRefs : list of `DatasetRef`
            List of dataset references from which to construct.
        ignoreDuplicates : `bool`, optional
            Silently drop duplicate entries?

        Returns
        -------
        self : `DatasetRefList`
            Constructed list of dataset references.

        Raises
        ------
        RuntimeError
            If the ``datasetType`` or ``dimensions`` don't match.
        KeyError
            If ``ignoreDuplicates=False`` and a dataset reference with matching
            coordinates has already been incorporated into this container.
        """
        if len(dataRefs) == 0:
            raise RuntimeError("No data references in list")
        first = dataRefs[0]
        self = cls(first.datasetType, first.dimensions)
        self.add(first)
        self.extend(dataRefs[1:], ignoreDuplicates)
        return self

    def extend(self, other: Iterable[DatasetRef], ignoreDuplicates: bool = False) -> None:
        """Extend from an iterable collection of dataset references

        Parameters
        ----------
        other : iterable of `DatasetRef`
            List of dataset references with which to extend this container.
        ignoreDuplicates : `bool`, optional
            Silently drop duplicate entries?

        Raises
        ------
        RuntimeError
            If the ``datasetType`` or ``dimensions`` don't match.
        KeyError
            If ``ignoreDuplicates=False`` and a dataset reference with matching
            coordinates has already been incorporated into this container.
        """
        for ref in other:
            self.add(ref, ignoreDuplicates)

    def add(self, dataRef: DatasetRef, ignoreDuplicates: bool = False) -> None:
        """Add a dataset reference

        Parameters
        ----------
        dataRef : `DatasetRef`
            Dataset reference to add to this container.
        ignoreDuplicates : `bool`, optional
            Silently drop duplicate entries?

        Raises
        ------
        RuntimeError
            If the ``datasetType`` or ``dimensions`` don't match.
        KeyError
            If ``ignoreDuplicates=False`` and a dataset reference with matching
            coordinates has already been incorporated into this container.
        """
        if dataRef.datasetType != self.datasetType:
            raise RuntimeError(f"Dataset type mismatch: {dataRef.datasetType} vs {self.datasetType}")
        if dataRef.dimensions != self.dimensions:
            raise RuntimeError(f"Dimension mismatch: {dataRef.dimensions} vs {self.dimensions}")
        dataId = dataRef.dataId
        if dataId in self._byCoordinate:
            if ignoreDuplicates:
                return
            raise KeyError(f"Duplicate coordinate: {dataId}")
        self._byIndex.append(dataRef)
        self._byCoordinate[dataId] = dataRef

    def __getitem__(self, indexOrCoord: Union[int, DataCoordinate]) -> DatasetRef:
        """Get by integer index or dataId"""
        if isinstance(indexOrCoord, DataCoordinate):
            return self._byCoordinate[indexOrCoord]
        return self._byIndex[indexOrCoord]

    def __len__(self) -> int:
        """Length"""
        return len(self._byIndex)

    def __iter__(self) -> Iterator["DatasetRefList"]:
        """Iterator over the list"""
        return iter(self._byIndex)

    def __contains__(self, dataId: Union[DatasetRef, DataCoordinate]) -> bool:
        """Is this dataId in the container?"""
        if isinstance(dataId, DatasetRef):
            dataId = dataId.dataId
        return dataId in self._byCoordinate

    def byIndex(self, index: int) -> DatasetRef:
        """Retrieve by integer index, as a list"""
        return self._byIndex[index]

    def byCoordinate(self, dataId: DataCoordinate, allowMissing: bool = False) -> DatasetRef:
        """Retrieve by dataId, as a dict

        Parameters
        ----------
        coord : `DataCoordinate`
            Data coordinate.
        allowMissing : `bool`, optional
            Allow the ``coord`` to be missing?

        Returns
        -------
        result : `DatasetRef` or `None`
             If the ``coord`` is present in the container, the indexed data
             reference is returned. Otherwise, if ``allowMissing``, ``None``
             is returned. Otherwise, an exception is raised.

        Raises
        ------
        KeyError
            If the ``coord`` is not present in the container and
            ``allowMissing`` is ``False``.
        """
        if allowMissing:
            return self._byCoordinate.get(dataId, None)
        return self._byCoordinate[dataId]

    def byExtendedCoordinate(self, dataId: DataCoordinate, allowMissing: bool = False) -> DatasetRef:
        """Retrieve by dataId, as a dict, where the dataId may cover more
        dimensions

        We subset the coordinate before lookup.

        Parameters
        ----------
        dataId : `DataCoordinate`
            Data coordinate.
        allowMissing : `bool`, optional
            Allow the ``coord`` to be missing?

        Returns
        -------
        result : `DatasetRef` or `None`
             If the appropriate subset of ``coord`` is present in the container,
             the indexed data reference is returned. Otherwise, if
             ``allowMissing``, ``None`` is returned. Otherwise, an exception is
             raised.

        Raises
        ------
        KeyError
            If the ``coord`` is not present in the container and
            ``allowMissing`` is ``False``.
        """
        return self.byCoordinate(dataId.subset(self.dimensions), allowMissing=allowMissing)

    def keys(self) -> List[DataCoordinate]:
        """Return the list of data coordinates in the container"""
        return self._byCoordinate.keys()


def zipDatasetRefs(*refLists: DatasetRefList, allowMissing: bool = False) -> Iterator[Tuple[DatasetRef, ...]]:
    """Provide zip iteration over separate lists of `DatasetRef`

    Each input list should contain references for a single ``datasetType``,
    with differing ``dataId`` and the same dimensions ``dataId.graph``.

    For ease of calculation, the set of dimensions (``DatasetRef.dataId.graph``)
    of all of the references must be a subset of the set of dimensions of one of
    the lists. You cannot have mutually exclusive sets of dimensions.

    We iterate over the `DataCoordinate`s, and ensure that the `DatasetRef`s are
    matched in the appropriate dimensions.

    For example, if ``refs1`` consists of references with coordinates:
    - ``(exposure=1, detector=1)``
    - ``(exposure=1, detector=2)``
    - ``(exposure=2, detector=1)``

    and ``refs2`` has references with coordinates:
    - ``(exposure=1)``
    - ``(exposure=2)``

    Then ``zipDatasetRefs(refs1, refs2)`` would produce references with the
    coordinates:
    - ``(exposure=1, detector=1), (exposure=1)``
    - ``(exposure=1, detector=2), (exposure=1)``
    - ``(exposure=1, detector=1), (exposure=2)``

    If ``allowMissing=True`` were provided, then ``(None), (exposure=2)``
    would also be included.

    No particular order of iteration is specified.

    Parameters
    ----------
    *refLists : `DatasetRefList`
        Lists of dataset references to iterate over.
    allMissing : `bool`
        Allow an iteration element that doesn't have all products available?
        In this case, the missing product would have the value ``None``.

    Yields
    ------
    refs : `tuple` of `DatasetRef`
        One `DatasetRef` from each of the ``refLists``, with matching
        coordinates.
    """
    if len(refLists) == 0:
        raise RuntimeError("No lists provided.")
    if len(refLists) == 1:
        return iter(refLists[0])

    # Identify the largest set of dimensions
    refDimensions = [refs.dimensions if refs is not None else None for refs in refLists]
    mostDimensions = max(refDimensions, key=lambda dims: len(dims) if dims is not None else 0)
    # All other sets of dimensions should be a subset of that.
    # This assumption makes the calculation MUCH easier,
    # and it should usually be true in the cases we care about.
    # We don't have to solve the general problem.
    # I'm not even sure what the general solution would look like (let alone
    # how to get there), so users wanting to zip-iterate over mis-aligned sets
    # of dimensions need to craft their own solution that does what they
    # expect.
    for dims in refDimensions:
        if dims is None:
            continue
        if not dims.issubset(mostDimensions):
            raise RuntimeError(f"Graph {dims} is not a subset of {mostDimensions}")

    # Knowing the largest set of dimensions, we can calculate the largest set
    # of coordinates (dataIds).
    mostIndices = [ii for ii, dims in enumerate(refDimensions) if dims is not None and dims == mostDimensions]
    allCoordinates = functools.reduce(
        lambda coords, index: coords.union(set(refLists[index].keys())),
        mostIndices[1:],
        set(refLists[mostIndices[0]].keys()),
    )

    # Iterate over the largest set of coordinates, providing for each an
    # appropriate reference from each list.
    for dataId in allCoordinates:
        try:
            result = tuple(
                refs.byExtendedCoordinate(dataId, allowMissing=allowMissing) if refs is not None else None
                for refs in refLists
            )
        except KeyError:
            assert not allowMissing, "Should only get KeyError if allowMissing=False"
            continue
        yield tuple(result)


def readDatasetRefs(
    butler: QuantumContext,
    inputRefs: InputQuantizedConnection,
    *names: str,
    allowMissing: bool = False,
) -> Dict[str, List[Any]]:
    """Performs a coordinated read of datasets

    Each input list should contain data for a single dataset, with
    differing ``dataId`` and the same dimensions ``dataId.graph``.

    For ease of calculation, the set of dimensions (``DatasetRef.dataId.graph``)
    of all of the references must be a subset of the set of dimensions of one of
    the lists. You cannot have mutually exclusive sets of dimensions.

    We iterate over the `DataCoordinate`s, and ensure that the `DatasetRef`s are
    matched in the appropriate dimensions.

    For example, if ``refs1`` consists of references with coordinates:
    - ``(exposure=1, detector=1)``
    - ``(exposure=1, detector=2)``
    - ``(exposure=2, detector=1)``

    and ``refs2`` has references with coordinates:
    - ``(exposure=1)``
    - ``(exposure=2)``

    Then ``readDatasetRefs(refs1, refs2)`` would produce data with the
    coordinates:
    - ``(exposure=1, detector=1), (exposure=1)``
    - ``(exposure=1, detector=2), (exposure=1)``
    - ``(exposure=1, detector=1), (exposure=2)``

    If ``allowMissing=True`` were provided, then ``(None), (exposure=2)``
    would also be included.

    All datasets read are removed from the ``inputRefs`` (so it can be used for
    reading other data).

    No particular order of iteration is specified.

    Parameters
    ----------
    butler : `QuantumContext`
        Data butler for a particular quantum.
    inputRefs : `InputQuantizedConnection`
        Dataset references; modified.
    *names : `str`
        Names of datasets to read. If empty, all datasets in the ``inputRefs``
        will be read.
    allowMissing : `bool`, optional
        Allow an iteration element that doesn't have all products available?
        In this case, the missing product would have the value ``None``.

    Returns
    -------
    dataLists : `SimpleNamespace`
        Struct with attributes corresponding to the input ``names``, which
        are coordinated lists of data read by the butler.
    """
    if not names:
        names = inputRefs.keys()
    refLists = [DatasetRefList.fromList(getattr(inputRefs, key)) for key in names]
    for key in names:
        delattr(inputRefs, key)
    dataLists: Dict[str, List[Any]] = {key: [] for key in names}
    for refs in zipDatasetRefs(*refLists, allowMissing=allowMissing):
        for key, rr in zip(names, refs):
            dataLists[key].append(butler.get(rr) if rr is not None else None)
    return SimpleNamespace(**dataLists)


def targetFromDataId(dataId: Union[DataCoordinate, Dict[str, Union[int, str]]]) -> Target:
    """Construct a `Target` from a ``dataId``

    Parameters
    ----------
    dataId : `DataCoordinate` or `dict` [`str`: `int` or `str`]
        Data identifier; contains ``catId``, ``tract``, ``patch``, ``objId``
        keys.

    Returns
    -------
    target : `Target`
        Constructed target.
    """
    return Target(dataId["catId"], dataId["tract"], dataId["patch"], dataId["objId"])


class NoResultsError(ValueError):
    """Exception raised when no results are found"""
    pass


def addPfsConfigRecords(
    registry: Registry,
    pfsConfig: PfsConfig,
    instrument: str,
    update: bool = True,
) -> None:
    """Add records for a pfsConfig to the butler registry

    This is needed in order to associate an ``exposure`` with the various
    ``catId``.

    Parameters
    ----------
    registry : `Registry`
        Butler registry.
    pfsConfig : `PfsConfig`
        PFS fiber configuration.
    instrument : `str`
        Name of instrument.
    update : `bool`, optional
        Update the record if it exists? Otherwise an exception will be generated
        if the record exists.

    Raises
    ------
    lsst.daf.butler.ConflictingDefinitionError
        Raised if the record exists in the database (according to primary key
        lookup) but is inconsistent with the given one.
    """
    exposure = pfsConfig.visit
    pfsDesignId = pfsConfig.pfsDesignId
    dataId = [
        rr
        for rr in registry.queryDimensionRecords(
            "exposure", dataId=dict(instrument=instrument, exposure=exposure)
        )
    ]
    if len(dataId) == 0:
        raise NoResultsError(f"No exposure records found for instrument={instrument}, exposure={exposure}")
    assert len(dataId) == 1
    dataId = dataId[0]
    if dataId.pfs_design_id != pfsDesignId:
        raise RuntimeError(
            "pfsDesignId mismatch between exposure in registry (%016x) and pfsConfig (%016x)",
            dataId.pfs_design_id,
            pfsDesignId,
        )

    for catId in np.unique(pfsConfig.catId):
        catId = int(catId)
        registry.syncDimensionData("cat_id", dict(instrument=instrument, id=catId), update=update)
        registry.syncDimensionData(
            "pfsConfig", dict(instrument=instrument, cat_id=catId, exposure=exposure), update=update
        )


def ingestPfsConfig(
    repo: str,
    instrument: str,
    run: str,
    pathList: Iterable[str],
    transfer: Optional[str] = "auto",
    update: bool = True,
) -> None:
    """Ingest a pfsConfig into the datastore

    Parameters
    ----------
    repo : `str`
        URI string of the Butler repo to use.
    instrument : `str`
        Instrument name or fully-qualified class name as a string.
    run : `str`
        The run in which the files should be ingested.
    pathList : iterable of `str`
        Paths/globs of pfsConfig files to ingest.
    transfer : `str`, optional
        Transfer mode to use for ingest. If not `None`, must be one of 'auto',
        'move', 'copy', 'direct', 'split', 'hardlink', 'relsymlink' or
        'symlink'.
    update : bool, optional
        Update the record if it exists? Otherwise an exception will be generated
        if the record exists.
    """
    log = getLogger("pfs.ingestPfsConfig")
    butler = Butler(repo, run=run)
    registry = butler.registry
    instrumentName = Instrument.from_string(instrument, registry).getName()
    datasetType = butler.registry.getDatasetType("pfsConfig")
    cwd = os.getcwd()

    datasets = []
    with registry.transaction():
        for path in pathList:
            for filename in glob(path):
                pfsConfig = PfsConfig.readFits(filename)
                exposure = pfsConfig.visit
                pfsDesignId = pfsConfig.pfsDesignId
                dataId = dict(instrument=instrumentName, exposure=exposure, pfs_design_id=pfsDesignId)
                ref = DatasetRef(datasetType, dataId, run)
                uri = ResourcePath(path, root=cwd, forceAbsolute=True)

                log.info("Registering %s ...", filename)
                try:
                    addPfsConfigRecords(registry, pfsConfig, instrumentName, update=update)
                except NoResultsError as exc:
                    log.warn(str(exc))
                    continue

                datasets.append(FileDataset(path=uri, refs=[ref], formatter=FitsGenericFormatter))

    log.info("Ingesting files...")
    butler.ingest(*datasets, transfer=transfer)


def certifyDetectorMaps(
    repo: str,
    instrument: str,
    datasetType: str,
    collection: str,
    target: str,
    timespan: Timespan,
    transfer: Optional[str] = "copy",
) -> None:
    """Ingest and certify detectorMaps.

    Building PFS detectorMaps frequently uses the calib detectorMap as part of
    the process of fitting a new detectorMap (e.g., for the centroiding
    starting point, and the slit offsets). In order to avoid cycles when
    building the detectorMap, we write the new detectorMap as a separate
    dataset type from the calib detectorMap dataset type.

    This function copies the new detectorMaps as calib detectorMaps, and then
    certifies them for new use as calibs.

    Parameters
    ----------
    repo : `str`
        URI string of the Butler repo to use.
    instrument : `str`
        Instrument name or fully-qualified class name as a string.
    datasetType : `str`
        Dataset type of the detectorMaps to certify.
    collection : `str`
        The collection containing the files to certify.
    target : `str`
        The collection to which the certified detectorMaps will be added.
    timespan : `Timespan`
        Timespan to use for certification.
    transfer : `str`, optional
        Transfer mode to use for ingest. If not `None`, must be one of 'auto',
        'move', 'copy', 'direct', 'split', 'hardlink', 'relsymlink' or
        'symlink'.
    """
    log = getLogger("pfs.certifyDetectorMaps")
    run = collection + "/certify"
    butler = Butler(repo, run=run)
    registry = butler.registry
    instrumentName = Instrument.from_string(instrument, registry).getName()

    fromType = registry.getDatasetType(datasetType)
    toType = registry.getDatasetType("detectorMap_calib")
    dimensions = registry.dimensions.extract(["instrument"])

    query = list(registry.queryDatasets(
        fromType, collections=collection, dimensions=dimensions, instrument=instrumentName
    ))
    if not query:
        log.warn("No detectorMaps found.")
        return
    datasets = []
    for ref in query:
        log.info("Ingesting %s ...", ref.dataId)
        uri = butler.getURI(ref)
        dataId = dict(
            instrument=instrumentName, arm=ref.dataId["arm"], spectrograph=ref.dataId["spectrograph"]
        )
        new = DatasetRef(toType, dataId, run)
        datasets.append(FileDataset(path=uri, refs=[new], formatter=DetectorMapFormatter))

    with registry.transaction():
        butler.ingest(*datasets, transfer=transfer)

        log.info("Certifying newly-ingested detectorMaps...")
        query = list(registry.queryDatasets(toType, collections=run))
        if not datasets:
            raise RuntimeError("Unable to find newly-ingested detectorMaps!")
        registry.registerCollection(target, type=CollectionType.CALIBRATION)
        registry.certify(target, query, timespan)


def defineFiberProfilesInputs(
    repo: str,
    instrument: str,
    name: str,
    bright: List[List[int]],
    dark: List[List[int]],
    update: bool = False,
):
    """Define inputs to a fiberProfiles run

    Parameters
    ----------
    repo : `str`
        URI string of the Butler repo to use.
    instrument : `str`
        Instrument name or fully-qualified class name as a string.
    name : `str`
        Symbolic name of the fiberProfiles run.
    bright : `list` of `list` of `int`]
        Bright exposure IDs for each group.
    dark : `list` of `list` of `int`]
        Dark exposure IDs for each group. May be empty; otherwise, the length
        of this list must match the length of the ``bright`` list.
    update : `bool`, optional
        Update the record if it exists? Otherwise an exception will be generated
        if the record exists.
    """
    log = getLogger("pfs.defineFiberProfilesInputs")
    run = instrument + "/fiberProfilesInputs"
    butler = Butler(repo, run=run)
    registry = butler.registry
    instrumentName = Instrument.from_string(instrument, registry).getName()

    if len(bright) != len(dark) and any(len(dd) > 0 for dd in dark):
        raise RuntimeError(f"Length of bright ({len(bright)}) and dark ({len(dark)}) lists do not match")
    numGroups = len(bright)

    datasetType = DatasetType(
        "profiles_exposures",
        ("instrument", "profiles_run", "profiles_group"),
        "StructuredDataDict",
        universe=registry.dimensions,
    )
    registry.registerDatasetType(datasetType)

    with registry.transaction():
        log.info("Registering run %s ...", name)
        registry.syncDimensionData(
            "profiles_run", dict(instrument=instrumentName, run=name), update=update
        )
        for group in range(numGroups):
            log.info("Registering group %d ...", group)
            groupId = dict(instrument=instrumentName, profiles_run=name, profiles_group=group)
            registry.syncDimensionData("profiles_group", groupId, update=update)
            log.info("Registering exposures for group %d: %s", group, bright[group])
            darkList = dark[group] if dark else []
            for exposure in bright[group] + darkList:
                registry.syncDimensionData(
                    "profiles_exposures", dict(exposure=exposure, **groupId), update=update
                )
            # Write a file identifying the bright and dark exposures for this
            # group. This might be a bit of a hack, but it is a simple way to
            # provide the information to the pipeline. It's better than
            # putting the information in the registry database, since that's
            # more difficult to access.
            butler.put(dict(bright=bright[group], dark=darkList), "profiles_exposures", **groupId)


def decertifyCalibrations(
    repo: str,
    collection: str,
    datasetType: str,
    timespanBegin: Optional[str],
    timespanEnd: Optional[str],
    dataIds: Optional[Iterable[Dict[str, Union[int, str]]]] = None,
) -> None:
    """Decertify a calibration dataset specified by its timespan

    Parameters
    ----------
    repo : `str`
        URI string of the Butler repo to use.
    collection : `str`
        Collection containing the datasets to decertify.
    datasetType : `str`
        Dataset type to decertify.
    timespanBegin : `str` or `None`
        Beginning timespan.
    timespanEnd : `str` or `None`
        Ending timespan.
    dataIds : iterable of `dict` [`str`: `int` or `str`], optional
        Data identifiers to decertify. If not provided, all datasets in the
        collection matching the timespan will be decertified.
    """
    timespan = Timespan(
        begin=Time(timespanBegin, scale="tai") if timespanBegin is not None else None,
        end=Time(timespanEnd, scale="tai") if timespanEnd is not None else None,
    )
    butler = Butler(repo, writeable=True)

    butler.registry.decertify(
        collection,
        datasetType,
        timespan,
        dataIds=[getDataCoordinate(ident, butler.dimensions) for ident in dataIds] if dataIds else None,
    )


def certifyCalibrations(
    repo: str,
    inputCollection: str,
    outputCollection: str,
    datasetType: str,
    beginDate: str | None,
    endDate: str | None,
    searchAllInputs: bool = False,
    dataIds: Optional[Iterable[Dict[str, Union[int, str]]]] = None,
) -> None:
    """Certify a set of calibrations with a validity range.

    Parameters
    ----------
    repo : `str`
        URI to the location of the repo or URI to a config file describing the
        repo and its location.
    inputCollection : `str`
       Data collection to pull calibrations from.  Usually an existing
        `~CollectionType.RUN` or `~CollectionType.CHAINED` collection, and may
        _not_ be a `~CollectionType.CALIBRATION` collection or a nonexistent
        collection.
    outputCollection : `str`
        Data collection to store final calibrations.  If it already exists, it
        must be a `~CollectionType.CALIBRATION` collection.  If not, a new
        `~CollectionType.CALIBRATION` collection with this name will be
        registered.
    datasetType : `str`
        Name of the dataset type to certify.
    beginDate : `str`, optional
        ISO-8601 date (TAI) this calibration should start being used.
    endDate : `str`, optional
        ISO-8601 date (TAI) this calibration should stop being used.
    searchAllInputs : `bool`, optional
        Search all children of the inputCollection if it is a CHAINED
        collection, instead of just the most recent one.
    dataIds : iterable of `dict` [`str`: `int` or `str`], optional
        Data identifiers to certify. If not provided, all datasets in the
        collection will be certified.
    """
    butler = Butler(repo, writeable=True, without_datastore=True)
    registry = butler.registry
    timespan = Timespan(
        begin=Time(beginDate, scale="tai") if beginDate is not None else None,
        end=Time(endDate, scale="tai") if endDate is not None else None,
    )
    if not searchAllInputs and registry.getCollectionType(inputCollection) is CollectionType.CHAINED:
        inputCollection = next(iter(registry.getCollectionChain(inputCollection)))

    if dataIds:
        refs = set()
        for ident in dataIds:
            coord = getDataCoordinate(ident, butler.dimensions)
            newRefs = registry.queryDatasets(datasetType, collections=[inputCollection], dataId=coord)
            refs.update(newRefs)
    else:
        refs = set(registry.queryDatasets(datasetType, collections=[inputCollection]))
    if not refs:
        raise RuntimeError(f"No inputs found for dataset {datasetType} in {inputCollection}.")
    registry.registerCollection(outputCollection, type=CollectionType.CALIBRATION)
    registry.certify(outputCollection, refs, timespan)


def defineCombination(
    repo: str,
    instrument: str,
    name: str,
    where: Optional[str] = None,
    exposureList: Optional[List[int]] = None,
    update: bool = False,
):
    """Define a combination of exposures

    Parameters
    ----------
    repo : `str`
        URI string of the Butler repo to use.
    instrument : `str`
        Instrument name or fully-qualified class name as a string.
    name : `str`
        Name of the combination. This is a symbolic name that can be used to
        refer to this combination in the future.
    where : `str`, optional
        SQL WHERE clause to use for selecting exposures.
    exposureList : list of `int`, optional
        List of exposure numbers to include in the combination. If the ``where``
        clause is provided, this list is used to further restrict the selection.
    update : `bool`, optional
        Update the record if it exists? Otherwise an exception will be generated
        if the record exists.

    Returns
    -------
    visitHash : `int`
        Hash of the exposures in the combination.
    exposureList : list of `int`
        List of exposure numbers in the combination.
    """
    if not where and not exposureList:
        raise ValueError("Must provide at least one of 'where' and 'exposureList'")
    bind = None
    if exposureList:
        add = "exposure IN (exposureList)"
        bind = dict(exposureList=exposureList)
        if where is None:
            where = add
        else:
            where = f"({where}) AND {add}"

    log = getLogger("pfs.defineCombination")
    butler = Butler(repo, writeable=True)
    registry = butler.registry
    instrumentName = Instrument.from_string(instrument, registry).getName()

    query = registry.queryDimensionRecords(
        "exposure", where=where, bind=bind, instrument=instrumentName
    )
    if not query:
        log.warn("No data found.")
        return

    exposureList = sorted([ref.dataId["exposure"] for ref in query])
    visitHash = calculatePfsVisitHash(exposureList)
    log.info(
        "Defining combination %s with pfsVisitHash=%016x for exposures: %s", name, visitHash, exposureList
    )

    with registry.transaction():
        registry.syncDimensionData(
            "combination", dict(instrument=instrument, name=name, pfs_visit_hash=visitHash), update=update
        )
        for exposure in exposureList:
            registry.syncDimensionData(
                "combination_join",
                dict(instrument=instrument, combination=name, exposure=exposure),
                update=update,
            )

    return visitHash, exposureList


def cleanRun(
    repo: str,
    collections: str,
    datasetTypes: Iterable[str],
    dataIds: Optional[Iterable[Dict[str, Union[int, str]]]] = None,
):
    """Clean a run by deleting all datasets of specified types in a collection

    Parameters
    ----------
    repo : `str`
        URI string of the Butler repo to use.
    collections : `str`
        Glob for collections to clean.
    datasetTypes : list of `str`
        Dataset types to delete.
    dataIds : iterable of `dict` [`str`: `int` or `str`], optional
        Data identifiers to delete. If not provided, all datasets of the
        specified types in the collection will be deleted.
    """
    log = getLogger("pfs.cleanRun")
    butler = Butler(repo, writeable=True)
    for coll in butler.registry.queryCollections(
        collectionTypes=CollectionType.RUN,
        expression=collections,
        includeChains=True,
    ):
        for dst in datasetTypes:
            if dataIds:
                refs = []
                for ident in dataIds:
                    coord = getDataCoordinate(ident, butler.dimensions)
                    refs.extend(butler.registry.queryDatasets(dst, collections=coll, dataId=coord))
            else:
                refs = list(butler.registry.queryDatasets(dst, collections=coll))
            if not refs:
                log.debug("No datasets found for %s in %s", dst, coll)
                continue
            log.info("Cleaning %d %s datasets in %s", len(refs), dst, coll)
            butler.pruneDatasets(refs, disassociate=True, unstore=True, purge=True)


def getDataCoordinate(dataId: dict[str, Any], universe: DimensionUniverse) -> DataCoordinate:
    """Get a `DataCoordinate` from a dictionary

    Parameters
    ----------
    dataId : `dict` [`str`: `str`]
        Data identifier dictionary.
    universe : `DimensionUniverse`
        Dimension universe.

    Returns
    -------
    coord : `DataCoordinate`
        Constructed data coordinate.
    """
    coord = {}
    for key, value in dataId.items():
        dtype = universe[key].primaryKey.dtype().python_type
        coord[key] = dtype(value)
    return DataCoordinate.standardize(coord, universe=universe)
