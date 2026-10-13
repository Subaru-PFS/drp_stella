import importlib
import importlib.util

# Modules whose names this package used to re-export. They pull in matplotlib
# and opdb access, so they are imported only when one of their names is asked
# for (e.g. ``from pfs.drp.stella.utils import addPfsCursor`` in a notebook),
# never by the pipeline.
_LAZY_MODULES = ("asinhNorm", "display", "fiberProfiles", "guiders", "math", "plotting", "sunss", "sysUtils")


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # ``from pfs.drp.stella.utils import debugging`` lands here first; import
    # just that submodule rather than searching the others.
    if importlib.util.find_spec(f"{__name__}.{name}") is not None:
        return importlib.import_module(f"{__name__}.{name}")
    for moduleName in _LAZY_MODULES:
        module = importlib.import_module(f"{__name__}.{moduleName}")
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
