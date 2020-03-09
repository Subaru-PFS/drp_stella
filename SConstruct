# -*- python -*-
import os
from lsst.sconsUtils import scripts
scripts.BasicSConstruct(
    "drp_stella",
    versionModuleName="python/pfs/%s/version.py",
    subDirList=[path for path in os.listdir(".") if os.path.isdir(path) and not path.startswith(".")] +
        ["bin"],
)
