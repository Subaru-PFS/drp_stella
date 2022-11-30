import os
import pickle

__all__ = ["writeExtraData"]


def writeExtraData(path: str, **kwargs) -> None:
    """Pickle extra data for debugging.

    Parameters
    ----------
    path : `str`
        Output file name. Its directory will be created if not exists.
    **kwargs : `Dict[str, Any]`
        Extra data to pickle.
    """
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(kwargs, f)
