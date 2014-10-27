#!/usr/bin/env python

import lsst.afw.image                                 as afwImage
import pfs.drp.stella.createFlatFiberTraceProfileTask as cfftpTask
import pfs.drp.stella.findAndTraceAperturesTask       as fataTask
import pfs.drp.stella.extractFromProfileTask          as efpTask

expFlat = afwImage.ExposureF("/home/azuri/spectra/pfs/2014-10-14/IR-23-0-sampledFlatx2-nonoise.fits")
expSpec = afwImage.ExposureF("/home/azuri/spectra/pfs/2014-10-14/IR-23-0-sampledFlatx2.fits")

"""Find and trace apertures"""
myFindTask = fataTask.FindAndTraceAperturesTask()
fts = myFindTask.run(expFlat)

"""Calculate spatial profiles"""
myCalcProfTask = cfftpTask.CreateFlatFiberTraceProfileTask()
myCalcProfTask.run(fts)
myExtractFromProfileTask = efpTask.ExtractFromProfileTask()
myExtractFromProfileTask.run(fts,expSpec,[0])
