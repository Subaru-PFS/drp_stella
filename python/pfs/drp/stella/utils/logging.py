import logging

__all__ = ("getLogger",)


class _ConsoleHandler(logging.StreamHandler):
    """Console handler added by `getLogger`

    A distinct class, so we can recognise whether a logger already has one.
    """

    pass


def getLogger(name, level=logging.INFO):
    """Return a logger object

    Configures the logger object so it actually works to log output to the
    console. Calling this multiple times for the same ``name`` does not add
    multiple console handlers (which would duplicate every message), but
    does update the level.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : int
        Level of the logger.

    Returns
    -------
    logger: logging.Logger
        Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, _ConsoleHandler):
            handler.setLevel(level)
            return logger
    consoleHandler = _ConsoleHandler()
    consoleHandler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger
