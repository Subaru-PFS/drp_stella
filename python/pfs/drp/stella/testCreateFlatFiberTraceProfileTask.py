#!/usr/bin/env python

import lsst.afw.image                                 as afwImage
import pfs.drp.stella.createFlatFiberTraceProfileTask as cfftpTask
import pfs.drp.stella.findAndTraceAperturesTask       as fataTask

exp = afwImage.ExposureF("/home/azuri/spectra/pfs/2014-10-14/IR-23-0-sampledFlatx2-nonoise.fits")
myFindTask = fataTask.FindAndTraceAperturesTask()
fts = myFindTask.run(exp)
myExtractTask = cfftpTask.CreateFlatFiberTraceProfileTask()
myExtractTask.run(fts, 0)
