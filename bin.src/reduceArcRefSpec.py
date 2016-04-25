#!/usr/bin/env python
#/Users/azuri/stella-git/drp_stella/bin.src/reduceArc.py '/Volumes/My Passport/data/spectra/pfs/PFS' --id visit=4 filter='PFS-R' spectrograph=2 site='F' category='A' --refSpec '/Users/azuri/stella-git/obs_subaru/pfs/lineLists/refCdHgKrNeXe_red.fits' --lineList '/Users/azuri/stella-git/obs_subaru/pfs/lineLists/CdHgKrNeXe_red.fits' --loglevel 'info'
from pfs.drp.stella.reduceArcRefSpecTask import ReduceArcRefSpecTask
ReduceArcRefSpecTask.parseAndRun()
