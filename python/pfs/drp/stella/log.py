import os
import copy_reg

import lsst.pex.logging as pexLog

def pickleLog(log):
    """Pickle a log

    Drop the log on the floor; we'll get a new default.
    Assumes that we're always just using the default log,
    and not modifying it.
    """
    return pexLog.getDefaultLog, tuple()

copy_reg.pickle(pexLog.Log, pickleLog)
copy_reg.pickle(pexLog.ScreenLog, pickleLog)

def jobLog(job):
    """Add a job-specific log destination"""
    if job is None or job == "None":
        return
    machine = os.uname()[1].split(".")[0]
    pexLog.getDefaultLog().addDestination(job + ".%s.%d" % (machine, os.getpid()))

