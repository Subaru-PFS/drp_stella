from types import ModuleType
import sys

import lsst.daf
import lsst.pipe.base

class Bang:
    """Class that blows up in your face if you attempt to use it"""
    def __new__(*args, **kwargs):
        raise RuntimeError("Attempt to use Gen2 middleware")


class Boom(Bang):
    """Class that blows up in your face if you attempt to use or subclass it"""
    def __init_subclass__(*args, **kwargs):
        raise RuntimeError("Attempt to use Gen2 middleware")


def kaboom(*args, **kwargs):
    """Function that blows up in your face if you attempt to use it"""
    raise RuntimeError("Attempt to use Gen2 middleware")


lsst.pipe.base.ArgumentParser = Bang
lsst.pipe.base.CmdLineTask = Boom
lsst.pipe.base.TaskRunner = Bang

# This is a dummy module that simply exists so that it can be imported.
# If it's actually used, it will break.
dafPersistence = ModuleType("persistence")
dafPersistence.Butler = Boom
dafPersistence.ButlerDataRef = Boom
dafPersistence.NoResults = Boom

lsst.daf.persistence = dafPersistence
sys.modules["lsst.daf.persistence"] = dafPersistence
