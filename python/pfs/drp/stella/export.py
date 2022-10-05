import asyncio
import logging
import os
from argparse import ArgumentParser
from functools import partial, wraps
from typing import Any, Callable, Coroutine, Dict, Optional, Protocol, TypeVar, cast

from lsst.daf.butler import Butler, DataCoordinate, DatasetRef, DimensionRecord, Registry
from lsst.resources import ResourcePath
from lsst.resources.file import FileResourcePath
from typing_extensions import ParamSpec

from .datamodel.pfsTargetSpectra import PfsTargetSpectra
from .lsf import LsfDict

_LOG = logging.getLogger(__name__)

# Filename templates for copying products directly
COPY_TEMPLATES = dict(
    pfsConfig="pfsConfig/%(day_obs)s/pfsConfig-0x%(pfs_design_id)016x-%(obs_id)06d.fits",
    calexp="images/%(day_obs)s/%(obs_id)06d/calexp-%(obs_id)06d-%(arm)1s%(spectrograph)1d.fits",
    detectorMap="images/%(day_obs)s/%(obs_id)06d/detectorMap-%(obs_id)06d-%(arm)1s%(spectrograph)1d.fits",
    pfsArm="pfsArm/%(day_obs)s/%(obs_id)06d/pfsArm-%(obs_id)06d-%(arm)1s%(spectrograph)1d.fits",
    pfsArmLsf="pfsArm/%(day_obs)s/%(obs_id)06d/pfsArmLsf-%(obs_id)06d-%(arm)1s%(spectrograph)1d.pickle",
    pfsMerged="pfsMerged/%(day_obs)s/pfsMerged-%(obs_id)06d.fits",
    pfsMergedLsf="pfsMerged/%(day_obs)s/pfsMergedLsf-%(obs_id)06d.pickle",
)

# Filename templates for conglomerated spectra to be split
SPLIT_TEMPLATES = dict(
    pfsCalibrated=(
        "pfsSingle/%(catId)05d/%(tract)05d/%(patch)s/"
        "pfsSingle-%(catId)05d-%(tract)05d-%(patch)s-%(objId)016x-%(obs_id)06d.fits"
    ),
    pfsCoadd=(
        "pfsObject/%(catId)05d/%(tract)05d/%(patch)s/"
        "pfsObject-%(catId)05d-%(tract)05d-%(patch)s-%(objId)016x-"
        "%(nVisit)03d-0x%(pfsVisitHash)016x.fits"
    ),
)

# Filename templates for LSFs of conglomerated spectra to be split
LSF_TEMPLATES = dict(
    pfsCalibratedLsf=(
        "pfsSingle/%(catId)05d/%(tract)05d/%(patch)s/"
        "pfsSingleLsf-%(catId)05d-%(tract)05d-%(patch)s-%(objId)016x-%(obs_id)06d.pickle"
    ),
    pfsCoaddLsf=(
        "pfsObject/%(catId)05d/%(tract)05d/%(patch)s/"
        "pfsObjectLsf-%(catId)05d-%(tract)05d-%(patch)s-%(objId)016x-"
        "%(nVisit)03d-0x%(pfsVisitHash)016x.pickle"
    ),
)

# Types to support `wrap_async` decorator
Params = ParamSpec("Params")
ReturnType = TypeVar("ReturnType")


def wrap_async(func: Callable[Params, ReturnType]):
    """Decorator to convert function to ``async``

    Runs function under an asyncio event loop, which allows waiting to be done
    asynchronously.

    Based on https://stackoverflow.com/a/50450553/834250 .
    Typing help based on
    https://rednafi.github.io/reflections/static-typing-python-decorators.html

    Parameters
    ----------
    func : callable
        Function being wrapped.

    Returns
    -------
    wrapped_func : callable
        ``async`` version of function.
    """

    @wraps(func)
    async def run(
        *args: Params.args,
        **kwargs: Params.kwargs,
    ) -> Coroutine[None, Any, ReturnType]:
        loop = cast(Optional[asyncio.AbstractEventLoop], kwargs.get("async_loop", None))
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        executor = kwargs.get("async_executor", None)
        functor = cast(Callable[..., Coroutine[None, Any, ReturnType]], partial(func, *args, **kwargs))
        return await loop.run_in_executor(executor, functor)

    return run


def getRecord(
    dataId: DataCoordinate, dimension: str, registry: Optional[Registry] = None,
) -> Optional[DimensionRecord]:
    """Get record for a particular dimension

    Sometimes records are directly available from the ``dataId``, and sometimes
    we need an extra registry query.

    Parameters
    ----------
    dataId : `DataCoordinate`
        Data identifier.
    dimension : `str`
        Name of dimension.
    registry : `Registry`, optional
        Butler registry.

    Returns
    -------
    record : `DimensionRecord` or None
        Record for dimension of interest, or `None` if it's not available.
    """
    if dataId.hasRecords():
        return dataId.records.get(dimension, None)
    if registry is None:
        return None
    records = registry.queryDimensionRecords(dimension, dataId=dataId)
    if records.count() != 1:
        return None
    return next(iter(records))


def getDataId(registry: Registry, dataId: DataCoordinate) -> Dict[str, Any]:
    """Get Gen2-stye data identifier

    We want a set of key,value pairs that we can use to determine the filename.
    Gen3 puts the interesting keys in a few different records, so we gather
    those together.

    Parameters
    ----------
    registry : `Registry`
        Butler registry.
    dataId : `DataCoordinate`
        Gen3 data identifier.

    Returns
    -------
    result : `dict`
        Gen2-style data identifier.
    """
    result = dict(zip(dataId.full.names, dataId.full.values()))
    exposure = getRecord(dataId, "exposure", registry)
    if exposure:
        for key, dtype in dict(
            pfs_design_id=int,
            obs_id=int,
            observation_type=str,
            observation_reason=str,
            day_obs=int,
            science_program=str,
            target_name=str,
        ).items():
            result[key] = dtype(getattr(exposure, key))
    arm = getRecord(dataId, "arm")
    if arm:
        result["arm"] = arm.name
    spectrograph = getRecord(dataId, "spectrograph")
    if spectrograph:
        result["spectrograph"] = spectrograph.num
    return result


def prepareWrite(target: ResourcePath, clobber: bool = False) -> bool:
    """Prepare to write a file

    We don't write the file, but we check whether the file exists already, and
    create the containing directory.

    Parameters
    ----------
    target : `ResourcePath`
        Filename that will be written.
    clobber : `bool`, optional
        Clobber file if it already exists?

    Returns
    -------
    write : `bool`
        The file should be written.

    Raises
    ------
    `RuntimeError`
        If we are unable to create the containing directory.
    """
    if target.exists():
        if not clobber:
            _LOG.warning("Target file %s already exists, and not configured to clobber", target)
            return False
        target.remove()
        return True
    try:
        target.dirname().mkdir()
    except OSError as exc:
        # Silently ignore mkdir failures due to race conditions
        if not os.path.isdir(target.dirname().ospath):
            raise RuntimeError(f"Failed to create directory {target.dirname()}") from exc
    return True


@wrap_async
def transferDataset(
    butler: Butler,
    ref: DatasetRef,
    template: str,
    mode: str,
):
    """Copy dataset to new location

    Parameters
    ----------
    butler : `Butler`
        Data butler; holds the original dataset.
    ref : `DatasetRef`
        Reference to dataset of interest.
    template : `str`
        Filename template for the copy.
    mode : `str`
        Copy mode; typically one of "link" or "copy".
    """
    source = butler.getURI(ref)
    dataId = getDataId(butler.registry, ref.dataId)
    target = ResourcePath(template % dataId)
    target.transfer_from(source, mode)


class FitsWriteable(Protocol):
    """How to identify an object that we can write to FITS"""

    def writeFits(self, filename: str):
        pass


@wrap_async
def writeObject(obj: FitsWriteable, target: ResourcePath, clobber: bool):
    """Write object to FITS file

    Parameters
    ----------
    obj : has ``writeFits`` method
        Object that can be written as FITS.
    target : `ResourcePath`
        Target filename.
    clobber : `bool`
        Clobber file if it already exists?
    """
    if prepareWrite(target, clobber):
        obj.writeFits(target.ospath)


async def splitSpectra(
    butler: Butler,
    spectraRef: DatasetRef,
    lsfDataset: str,
    spectrumTemplate: str,
    lsfTemplate: str,
    dataId: Dict[str, Any],
    clobber: bool,
):
    """Split conglomeration of spectra into individual files

    Parameters
    ----------
    butler : `Butler`
        Data butler.
    spectraRef : `DatasetRef`
        Reference to dataset containing spectra.
    lsfDataset : `str`
        Name of dataset containing LSFs.
    spectrumTemplate : `str`
        Filename template for each spectrum.
    lsfTemplate : `str`
        Filename template for each LSF.
    dataId : `dict`
        Gen2-style data identifier.
    clobber : `bool`
        Clobber file if it already exists?
    """
    spectra: PfsTargetSpectra = butler.get(spectraRef)
    lsf: LsfDict = butler.get(lsfDataset, spectraRef.dataId)
    dataId = getDataId(butler.registry, spectraRef.dataId)
    for target in spectra:
        identity = dataId.copy() if dataId else {}
        identity.update(spectra[target].getIdentity())
        await writeObject(spectra[target], ResourcePath(spectrumTemplate % identity), clobber)
        await writeObject(lsf[target], ResourcePath(lsfTemplate % identity), clobber)


async def runAsync(
    butler: Butler,
    base: str,
    exposures: Optional[str] = None,
    objects: Optional[str] = None,
    mode: str = "link",
    clobber: bool = False,
):
    """Run dataset export asynchronously

    This allows multiple copies to happen simultaneously.

    Parameters
    ----------
    butler : `Butler`
        Data butler.
    base : `str`
        Base directory for output.
    exposures : `str`, optional
        Query to apply for exposure-based datasets.
    objects : `str`, optional
        Query to apply for object-based datasets.
    mode : `str`, optional
        Mode for copying data, typically "copy" or "link".
    clobber : `bool`, optional
        Clobber file if it already exists?
    """
    for dataset in COPY_TEMPLATES:
        ref: DatasetRef
        for ref in butler.registry.queryDatasets(dataset, where=exposures, findFirst=True):
            await transferDataset(butler, ref, os.path.join(base, COPY_TEMPLATES[dataset]), mode)
    for spectraDataset in SPLIT_TEMPLATES:
        lsfDataset = spectraDataset + "Lsf"
        spectraRef: DatasetRef
        for spectraRef in butler.registry.queryDatasets(spectraDataset, where=objects, findFirst=True):
            await splitSpectra(
                butler,
                spectraRef,
                lsfDataset,
                os.path.join(base, SPLIT_TEMPLATES[spectraDataset]),
                os.path.join(base, LSF_TEMPLATES[lsfDataset]),
                getDataId(butler.registry, spectraRef.dataId),
                clobber,
            )


def run(*args, **kwargs):
    """Run dataset export

    Parameters
    ----------
    butler : `Butler`
        Data butler.
    base : `str`
        Base directory for output.
    exposures : `str`, optional
        Query to apply for exposure-based datasets.
    objects : `str`, optional
        Query to apply for object-based datasets.
    mode : `str`, optional
        Mode for copying data, typically "copy" or "link".
    clobber : `bool`, optional
        Clobber file if it already exists?
    """
    asyncio.run(runAsync(*args, **kwargs))


def main():
    """Command-line entry point for dataset export"""
    parser = ArgumentParser(description="Export PFS DRP2D products for downstream users")
    parser.add_argument(
        "-i", "--input", required=True, help="Comma-separated names of the input collection(s)"
    )
    parser.add_argument(
        "-b", "--butler-config", required=True, help="Location of the gen3 butler/registry config file"
    )
    parser.add_argument("--exposures", help="Data selection expression for exposures")
    parser.add_argument("--objects", help="Data selection expression for objects")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        default="link",
        choices=set(FileResourcePath.transferModes) - set("move"),
        help="Mode for transferring existing files",
    )
    parser.add_argument("--clobber", default=False, action="store_true", help="Clobber existing files?")
    args = parser.parse_args()

    butler = Butler(args.butler_config, collections=args.input.split(","))
    run(butler, args.output, args.exposures, args.objects, args.mode, args.clobber)
