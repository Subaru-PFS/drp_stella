from typing import List
from lsst.daf.butler import DataCoordinate, DatasetRef, DatasetType, Registry

__all__ = ("lookupDetectorMap", "lookupFiberNorms")


def lookupDetectorMap(
    datasetType: str | DatasetType, registry: Registry, dataId: DataCoordinate, collections: List[str]
) -> List[DatasetRef]:
    """Look up a detectorMap

    This is a lookup function for a PrerequisiteConnection that finds
    detectorMap for a given dataId.

    Parameters
    ----------
    datasetType : `str` or `lsst.daf.butler.DatasetType`
        The dataset type to look up.
    registry : `lsst.daf.butler.Registry`
        The butler registry.
    dataId : `lsst.daf.butler.DataCoordinate`
        The data identifier.
    collections : `list` of `str`
        The collections to search.

    Returns
    -------
    refs : `list` of `lsst.daf.butler.DatasetRef`
        The references to the bias or dark frame.
    """
    if isinstance(datasetType, str):
        datasetType: DatasetType = registry.getDatasetType(datasetType)

    component: str | None = None
    if datasetType.isComponent():
        parent, component = datasetType.nameAndComponent()
    else:
        parent = datasetType.name

    if "visit" not in dataId or dataId.timespan is None:
        # We need to provide the entire set of available detectorMaps for the join
        result = list(registry.queryDatasets(parent, dataId=dataId, collections=collections))
    else:
        result = [registry.findDataset(parent, dataId, collections=collections, timespan=dataId.timespan)]
    if component is not None:
        result = [ref.makeComponentRef(component) for ref in result]
    return result


def lookupFiberNorms(
    datasetType: str, registry: Registry, dataId: DataCoordinate, collections: List[str]
) -> List[DatasetRef]:
    """Look up a fiberNorms

    This is a lookup function for a PrerequisiteConnection that finds fiberNorms
    for a given dataId.

    Parameters
    ----------
    datasetType : `str`
        The dataset type to look up.
    registry : `lsst.daf.butler.Registry`
        The butler registry.
    dataId : `lsst.daf.butler.DataCoordinate`
        The data identifier.
    collections : `list` of `str`
        The collections to search.

    Returns
    -------
    refs : `list` of `lsst.daf.butler.DatasetRef`
        The references to the bias or dark frame.
    """
    if "visit" not in dataId or dataId.timespan is None:
        # We need to provide the entire set of available fiberNorms for the join
        result = registry.queryDatasets(datasetType, collections=collections)
        return [ref for ref in result]
    if "arm" in dataId:
        # We know exactly what we want
        return [registry.findDataset(datasetType, dataId, collections=collections, timespan=dataId.timespan)]

    refList = []
    for arm in "brnm":
        try:
            ref = registry.findDataset(
                "fiberNorms_calib", dataId, arm=arm, collections=collections, timespan=dataId.timespan
            )
        except Exception:
            continue
        if ref is not None:
            refList.append(ref)
    return refList
