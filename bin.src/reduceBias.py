#!/usr/bin/env python
#/Users/azuri/stella-git/drp_stella/bin.src/reduceBias.py '/Volumes/My Passport/data/spectra/pfs/PFS' --calib '/Volumes/My Passport/data/spectra/pfs/PFS/CALIB' --detrendId calibVersion=bias site=S category=A --do-exec --id field=BIAS dateObs=2015-12-22 spectrograph=2 site=S category=A filter=PFS-M
from pfs.drp.stella.detrends import BiasTask
BiasTask.parseAndSubmit()
