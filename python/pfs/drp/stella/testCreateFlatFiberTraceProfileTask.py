#!/usr/bin/env python

import lsst.afw.image                                 as afwImage
import pfs.drp.stella.createFlatFiberTraceProfileTask as cfftpTask
import pfs.drp.stella.findAndTraceAperturesTask       as fataTask

import numpy as np

exp = afwImage.ExposureF("IR-23-0-sampledFlatx2-nonoise.fits")
myFindTask = fataTask.FindAndTraceAperturesTask()
fts = myFindTask.run(exp)
myExtractTask = cfftpTask.CreateFlatFiberTraceProfileTask()

"""Extract all apertures"""
myExtractTask.run(fts)

"""Extract aperture 0"""
iAperturesToExtract = [0]#np.arange(0,2,1,int)
myExtractTask.run(fts, iAperturesToExtract)

"""Extract apertures 0 and 1"""
iAperturesToExtract = range(2)
myExtractTask.run(fts, iAperturesToExtract)
