from typing import Any

import lsst.pipe.base

__all__ = ("Struct",)


class Struct(lsst.pipe.base.Struct):
    """Version of Struct that doesn't cause mypy errors on attribute lookup

    A temporary solution until DM-34696 is fixed/merged.
    """
    def __getattr__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any):
        return super().__setattr__(name, value)
