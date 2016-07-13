#!/usr/bin/env python
# example:
# createRefSpec.py '/Users/azuri/spectra/pfs/PFS' --id visit=4 filter='r' spectrograph=2 site='F' category='A' --output '/Users/azuri/spectra/pfs/PFS' --lineList='/Users/azuri/stella-git/obs_pfs/pfs/lineLists/CdHgKrNeXe_red.fits' --output='Users/azuri/stella-git/obs_pfs/pfs/arcSpectra/refCdHgKrNeXe_red.fits' --clobber-config
from pfs.drp.stella.createRefSpec import CreateRefSpecTask
CreateRefSpecTask.parseAndRun()
