#!/usr/bin/env python
#%run /Users/azuri/stella-git/drp_stella/bin.src/reduceFlat.py '/Volumes/My Passport/data/spectra/pfs/PFS' --calib '/Volumes/My Passport/data/spectra/pfs/PFS/CALIB' --detrendId calibVersion=flat site=F category=A --do-exec --id field=FLAT dateObs=2016-01-12 spectrograph=2 site=F category=A filter=PFS-R --loglevel 'info' --config isr.doBias='False' isr.doDark='False' --clobber-config
from lsst.obs.pfs.detrends import PfsFlatCombineTask
PfsFlatCombineTask.parseAndSubmit()