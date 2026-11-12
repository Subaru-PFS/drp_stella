"""Generic harness for running a `lsst.pipe.base.Task` on local files

This allows a `Task` (or `PipelineTask`) to be configured and run from the
command line, without any butler, for simple debugging and testing away from
the full pipeline framework. See ``bin.src/runTask.py`` for the command-line
entry point.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from matplotlib.figure import Figure

from lsst.afw.detection import Psf
from lsst.afw.image import ExposureF
from lsst.geom import Box2I, Extent2I, Point2I

__all__ = (
    "TYPE_REGISTRY",
    "OUTPUT_WRITERS",
    "resolveClass",
    "readData",
    "parseDataSpec",
    "readDataSpecs",
    "writeResult",
    "parseOutputSpec",
    "writeOutputs",
    "parseLogLevel",
    "configureLogging",
    "loadConfig",
    "applyExtraData",
    "runTask",
)

#: Maps a short type name (as used on the command line) to either the
#: fully-qualified dotted path of a class with a ``readFits`` classmethod, or
#: a callable that takes a filename and returns the loaded object. Add new
#: entries here as new data types are needed. The ``readFits`` method must
#: accept an arbitrary filename (e.g., `~pfs.drp.stella.FiberTraceSet` doesn't,
#: as it parses the filename for the data identifier).
TYPE_REGISTRY: Dict[str, Union[str, Callable[[str], Any]]] = {
    "PfsArm": "pfs.drp.stella.PfsArm",
    "PfsConfig": "pfs.drp.stella.PfsConfig",
    "DetectorMap": "pfs.drp.stella.DetectorMap",
    "ArcLineSet": "pfs.drp.stella.ArcLineSet",
    "FiberProfileSet": "pfs.drp.stella.FiberProfileSet",
    "Exposure": "lsst.afw.image.ExposureF",
}

_DATA_SPEC_RE = re.compile(r"^(?P<name>\w+):(?P<type>\w+)=(?P<value>.+)$")
_OUTPUT_SPEC_RE = re.compile(r"^(?P<name>\w+):(?P<filename>.+)$")


def _writeFigure(obj: Any, filename: str) -> None:
    """Write a `matplotlib.figure.Figure` to a file"""
    obj.savefig(filename)


def _writePsf(obj: Any, filename: str) -> None:
    """Write an `lsst.afw.detection.Psf` to a file

    `Psf` has no ``writeFits`` of its own (it's normally persisted only as a
    component of an `~lsst.afw.image.Exposure`), so we wrap it in a minimal
    dummy exposure purely to carry it.
    """
    dummy = ExposureF(Box2I(Point2I(0, 0), Extent2I(16, 16)))
    dummy.setPsf(obj)
    dummy.writeFits(filename)


#: Special-cased (type, writer) pairs, checked via `isinstance` (so
#: subclasses match too) before falling back to calling ``obj.writeFits()``.
#: Add new entries here for types that can't just be ``writeFits``-ed.
OUTPUT_WRITERS: List[Tuple[type, Callable[[Any, str], None]]] = [
    (Figure, _writeFigure),
    (Psf, _writePsf),
]


def resolveClass(dottedPath: str) -> type:
    """Import and return the class named by a fully-qualified dotted path

    Parameters
    ----------
    dottedPath : `str`
        Fully-qualified dotted path to a class, e.g.
        ``"pfs.drp.stella.centroidSolar.CentroidSolarTask"``.

    Returns
    -------
    cls : `type`
        The resolved class.
    """
    moduleName, className = dottedPath.rsplit(".", 1)
    module = importlib.import_module(moduleName)
    return getattr(module, className)


def readData(typeName: str, path: str) -> Any:
    """Read a file into an object of the nominated type

    Parameters
    ----------
    typeName : `str`
        Short type name, a key in `TYPE_REGISTRY`.
    path : `str`
        Path to the file to read.

    Returns
    -------
    data : object
        The object read from ``path``.
    """
    if typeName not in TYPE_REGISTRY:
        known = ", ".join(sorted(TYPE_REGISTRY))
        raise KeyError(f"Unrecognized type {typeName!r}; known types are: {known}")
    loader = TYPE_REGISTRY[typeName]
    if isinstance(loader, str):
        return resolveClass(loader).readFits(path)
    return loader(path)


def parseDataSpec(spec: str) -> Tuple[str, str, str]:
    """Parse a ``name:type=value`` command-line data specification

    Parameters
    ----------
    spec : `str`
        Specification of the form ``name:type=value``.

    Returns
    -------
    name : `str`
        Name of the variable to read the data into.
    typeName : `str`
        Short type name, a key in `TYPE_REGISTRY`.
    value : `str`
        Path to the file to read.
    """
    match = _DATA_SPEC_RE.match(spec)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid data specification {spec!r}; expected name:type=value")
    return match["name"], match["type"], match["value"]


def readDataSpecs(specs: Iterable[str]) -> Dict[str, Any]:
    """Parse and read a list of ``name:type=value`` data specifications

    Parameters
    ----------
    specs : iterable of `str`
        Specifications of the form ``name:type=value``.

    Returns
    -------
    data : `dict`
        Mapping of name to the object read from the corresponding file.
    """
    data: Dict[str, Any] = {}
    for spec in specs:
        name, typeName, value = parseDataSpec(spec)
        if name in data:
            raise ValueError(f"Duplicate data name: {name!r}")
        data[name] = readData(typeName, value)
    return data


def writeResult(obj: Any, filename: str) -> None:
    """Write an object to a file, how exactly depending on its type

    Checks `OUTPUT_WRITERS` first (for types needing special handling, e.g.
    `~matplotlib.figure.Figure` or `~lsst.afw.detection.Psf`), then falls
    back to calling ``obj.writeFits(filename)``, which works for most
    LSST/PFS data products (`~lsst.afw.image.Exposure`,
    `~lsst.afw.table.SourceCatalog`, `~pfs.drp.stella.DetectorMap`, etc.).

    Parameters
    ----------
    obj : object
        The object to write.
    filename : `str`
        Path to write it to.

    Raises
    ------
    TypeError
        If ``obj``'s type isn't in `OUTPUT_WRITERS` and it has no
        ``writeFits`` method.
    """
    for cls, writer in OUTPUT_WRITERS:
        if isinstance(obj, cls):
            writer(obj, filename)
            return
    if hasattr(obj, "writeFits"):
        obj.writeFits(filename)
        return
    raise TypeError(
        f"Don't know how to write an object of type {type(obj).__name__!r} to a file; "
        "add an entry to OUTPUT_WRITERS in pfs.drp.stella.harness"
    )


def parseOutputSpec(spec: str) -> Tuple[str, str]:
    """Parse a ``name:filename`` command-line output specification

    Parameters
    ----------
    spec : `str`
        Specification of the form ``name:filename``.

    Returns
    -------
    name : `str`
        Name of the attribute of the task's result to write.
    filename : `str`
        Path to write it to.
    """
    match = _OUTPUT_SPEC_RE.match(spec)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid output specification {spec!r}; expected name:filename")
    return match["name"], match["filename"]


def writeOutputs(result: Any, outputSpecs: Iterable[str]) -> None:
    """Write named attributes of a task's result to files

    All the requested attributes are checked before any are written.

    Parameters
    ----------
    result : object
        The task's result (typically an `lsst.pipe.base.Struct`); each
        output is read off it by attribute name.
    outputSpecs : iterable of `str`
        Output specifications of the form ``name:filename``.

    Raises
    ------
    AttributeError
        If ``result`` has no attribute of the requested name.
    """
    outputs = [parseOutputSpec(spec) for spec in outputSpecs]
    missing = [name for name, _ in outputs if not hasattr(result, name)]
    if missing:
        raise AttributeError(f"Task result has no attribute(s) {missing} to write")
    for name, filename in outputs:
        writeResult(getattr(result, name), filename)


def parseLogLevel(spec: str) -> Tuple[Optional[str], int]:
    """Parse a ``[name=]LEVEL`` log level specification

    Parameters
    ----------
    spec : `str`
        Log level specification: either ``LEVEL`` (for the root logger) or
        ``name=LEVEL`` (for the logger named ``name``). The level name is
        case-insensitive.

    Returns
    -------
    name : `str` or `None`
        Name of the logger, or `None` for the root logger.
    level : `int`
        Log level.
    """
    name, levelName = spec.split("=", 1) if "=" in spec else (None, spec)
    level = logging.getLevelName(levelName.upper())
    if not isinstance(level, int):
        raise argparse.ArgumentTypeError(f"Invalid log level in {spec!r}")
    return name, level


def configureLogging(levels: Iterable[str]) -> logging.Logger:
    """Configure the root logger and any named child loggers

    The root logger is set to ``INFO`` unless a level is provided for it.

    Parameters
    ----------
    levels : iterable of `str`
        Log level specifications. Each is either ``LEVEL`` (sets the root
        logger's level) or ``name=LEVEL`` (sets the level of the logger
        named ``name``). The level names are case-insensitive.

    Returns
    -------
    logger : `logging.Logger`
        The root logger, with a console handler attached.
    """
    parsed = [parseLogLevel(spec) for spec in levels]  # Check them all before changing anything
    logger = logging.getLogger()
    if not any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    ):
        logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    for name, level in parsed:
        (logger if name is None else logger.getChild(name)).setLevel(level)
    return logger


def loadConfig(taskClass: type, configFile: Optional[str]) -> Any:
    """Instantiate a task's configuration, optionally loading overrides

    The configuration is validated and frozen, as in the pipeline framework.

    Parameters
    ----------
    taskClass : `type`
        The `~lsst.pipe.base.Task` subclass to configure.
    configFile : `str`, optional
        Path to a configuration file with overrides, of the form used by
        `lsst.pex.config.Config.load` (e.g. ``config.someField = 123``).

    Returns
    -------
    config : `lsst.pex.config.Config`
        The task's configuration.
    """
    config = taskClass.ConfigClass()
    if configFile:
        config.load(configFile)
    config.validate()
    config.freeze()
    return config


def applyExtraData(extraFile: Optional[str], data: Dict[str, Any]) -> None:
    """Execute a python file that may add or modify entries in ``data``

    This allows supplying keyword arguments for the task's ``run`` method
    that don't come from a file (e.g., a plain string, or an object
    constructed in code), by having the file assign into the ``data`` dict
    directly. For example, a file containing::

        from lsst.afw.image import VisitInfo
        data["arm"] = "b"
        data["visitInfo"] = VisitInfo()

    Templates for such files are provided in ``examples/runTask`` in this
    package; copy one and edit it as necessary.

    The file is executed with ``__file__`` set to its path, and ``__name__``
    set to ``"__runTask_extra__"``.

    Parameters
    ----------
    extraFile : `str`, optional
        Path to a python file to execute. If `None`, nothing is done.
    data : `dict`
        Mapping of name to data, as built up by `readDataSpecs`. Modified
        in-place by the executed file via the ``data`` name bound in its
        namespace.
    """
    if not extraFile:
        return
    with open(extraFile) as fd:
        code = compile(fd.read(), extraFile, "exec")
    exec(code, {"data": data, "__file__": extraFile, "__name__": "__runTask_extra__"})


def runTask(
    taskClassPath: str,
    dataSpecs: Iterable[str],
    configFile: Optional[str] = None,
    logLevels: Iterable[str] = (),
    extraFile: Optional[str] = None,
    outputSpecs: Iterable[str] = (),
) -> Any:
    """Configure and run a task on local data

    Parameters
    ----------
    taskClassPath : `str`
        Fully-qualified dotted path to the `~lsst.pipe.base.Task` subclass
        to run.
    dataSpecs : iterable of `str`
        Data specifications of the form ``name:type=value``, giving the
        keyword arguments to pass to the task's ``run`` method.
    configFile : `str`, optional
        Path to a configuration file with overrides.
    logLevels : iterable of `str`
        Log level specifications; see `configureLogging`.
    extraFile : `str`, optional
        Path to a python file that may add or modify entries in the data
        passed to the task's ``run`` method; see `applyExtraData`.
    outputSpecs : iterable of `str`, optional
        Output specifications of the form ``name:filename``, naming
        attributes of the task's result to write to files; see
        `writeOutputs`.

    Returns
    -------
    result
        Whatever the task's ``run`` method returns.
    """
    outputSpecs = list(outputSpecs)
    for spec in outputSpecs:
        parseOutputSpec(spec)  # Check the syntax before doing anything expensive
    logger = configureLogging(logLevels)
    taskClass = resolveClass(taskClassPath)
    config = loadConfig(taskClass, configFile)
    data = readDataSpecs(dataSpecs)
    applyExtraData(extraFile, data)
    task = taskClass(config=config, log=logger)
    result = task.run(**data)
    writeOutputs(result, outputSpecs)
    return result
