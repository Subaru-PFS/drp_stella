import logging

__all__ = ("getLogger",)


def getLogger(name, level=logging.INFO):
    """Return a logger object

    Configures the logger object so it actually works to log output to the
    console.

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
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    return logger
