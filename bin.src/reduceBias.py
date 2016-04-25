#!/usr/bin/env python
#/Users/azuri/stella-git/drp_stella/bin.src/reduceBias.py '/Users/azuri/spectra/pfs/PFS' --calib '/Users/azuri/spectra/pfs/PFS/CALIB' --output '/Users/azuri/spectra/pfs/PFS/CALIB/combined' --calibId calibVersion=bias arm=m calibDate=2015-12-22 spectrograph=2 --do-exec --id field=BIAS dateObs=2015-12-22 spectrograph=2 site=S category=A filter=m arm=m taiObs=2015-12-22 --nodes=1 --procs=1
from lsst.pipe.drivers.constructCalibs import BiasTask
BiasTask.parseAndSubmit()
