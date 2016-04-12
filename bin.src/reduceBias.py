#!/usr/bin/env python
#/Users/azuri/stella-git/drp_stella/bin.src/reduceBias.py '/Volumes/My Passport/data/spectra/pfs/PFS' --calib '/Volumes/My Passport/data/spectra/pfs/PFS/CALIB' --detrendId calibVersion=bias site=S category=A --do-exec --id field=BIAS dateObs=2015-12-22 spectrograph=2 site=S category=A filter=PFS-M
#/Users/azuri/stella-git/drp_stella/bin.src/reduceBias.py '/Users/azuri/spectra/pfs/PFS' --calib '/Users/azuri/spectra/pfs/PFS/CALIB' --calibId calibVersion=bias arm=m --do-exec --id field=BIAS dateObs=2015-12-22 spectrograph=2 site=S category=A filter=m --nodes=1 --procs=1
#from pfs.drp.stella.detrends import BiasTask
#import lsst.obs.pfs; lsst.obs.pfs.PfsMapper
from lsst.pipe.drivers.constructCalibs import BiasTask
BiasTask.parseAndSubmit()
