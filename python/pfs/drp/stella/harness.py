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
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

__all__ = (
    "TYPE_REGISTRY",
    "resolveClass",
    "readData",
    "parseDataSpec",
    "readDataSpecs",
    "configureLogging",
    "loadConfig",
    "applyExtraData",
    "runTask",
)

#: Maps a short type name (as used on the command line) to either the
#: fully-qualified dotted path of a class with a ``readFits`` classmethod, or
#: a callable that takes a filename and returns the loaded object. Add new
#: entries here as new data types are needed.
TYPE_REGISTRY: Dict[str, Union[str, Callable[[str], Any]]] = {
    "PfsArm": "pfs.drp.stella.PfsArm",
    "PfsConfig": "pfs.drp.stella.PfsConfig",
    "DetectorMap": "pfs.drp.stella.DetectorMap",
    "ArcLineSet": "pfs.drp.stella.ArcLineSet",
    "FiberProfileSet": "pfs.drp.stella.FiberProfileSet",
    "FiberTraceSet": "pfs.drp.stella.FiberTraceSet",
}

_DATA_SPEC_RE = re.compile(r"^(?P<name>\w+):(?P<type>\w+)=(?P<value>.+)$")


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


def configureLogging(levels: Iterable[str]) -> logging.Logger:
    """Configure the root logger and any named child loggers

    Parameters
    ----------
    levels : iterable of `str`
        Log level specifications. Each is either ``LEVEL`` (sets the root
        logger's level) or ``name=LEVEL`` (sets the level of the logger
        named ``name``).

    Returns
    -------
    logger : `logging.Logger`
        The root logger, with a `~logging.StreamHandler` attached.
    """
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    for level in levels:
        if "=" in level:
            name, levelName = level.split("=", 1)
            logger.getChild(name).setLevel(levelName)
        else:
            logger.setLevel(level)
    return logger


def loadConfig(taskClass: type, configFile: Optional[str]) -> Any:
    """Instantiate a task's configuration, optionally loading overrides

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
    exec(code, {"data": data})


def runTask(
    taskClassPath: str,
    dataSpecs: Iterable[str],
    configFile: Optional[str] = None,
    logLevels: Iterable[str] = (),
    extraFile: Optional[str] = None,
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

    Returns
    -------
    result
        Whatever the task's ``run`` method returns.
    """
    logger = configureLogging(logLevels)
    taskClass = resolveClass(taskClassPath)
    config = loadConfig(taskClass, configFile)
    data = readDataSpecs(dataSpecs)
    applyExtraData(extraFile, data)
    task = taskClass(config=config, log=logger)
    return task.run(**data)
