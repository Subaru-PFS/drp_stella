#!/usr/bin/env python
#/Users/azuri/stella-git/drp_stella/bin.src/reduceDark.py '/Volumes/My Passport/data/spectra/pfs/PFS' --calib '/Volumes/My Passport/data/spectra/pfs/PFS/CALIB' --detrendId calibVersion='dark' site='S' category='A' --id field='DARK' dateObs='2015-12-22' spectrograph=2 site='S' category='A' filter='PFS-M' visit=7291..7294 --loglevel info --clobber-config  -c isr.doBias=True --do-exec
from pfs.drp.stella.detrends import DarkTask
DarkTask.parseAndSubmit()
