"""Tools for transferring datasets out of butler repos"""

from typing import Any, Dict, Iterable, List, Optional

from lsst.daf.butler import Butler
from lsst.daf.butler.registry._registry import CollectionArgType
from lsst.resources import ResourcePath
from .gen3 import getDataCoordinate
from .utils.logging import getLogger

__all__ = ("transferButlerFiles",)


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
