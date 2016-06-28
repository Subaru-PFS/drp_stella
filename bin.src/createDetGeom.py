#!/usr/bin/env python
#createRefSpec.py '/Users/azuri/spectra/pfs/PFS' --id visit=4 filter='r' spectrograph=2 site='F' category='A' -C ../obs_pfs/config/pfs/createRefSpec.py --output "/Users/azuri/spectra/pfs/PFS" --clobber-config
from pfs.drp.stella.createDetGeom import CreateDetGeomTask
task = CreateDetGeomTask()
task.run()
