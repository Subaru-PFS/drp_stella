import argparse
import numpy
import lsst.daf.persistence as dafPersist
import lsst.afw.image as afwImage

flatFile = '/tigress/HSC/PFS/sim/2015-12-14/Red-Lam1Flat-0-23.fz'
flat = afwImage.ImageF(flatFile)
flatExp = afwImage.makeExposure(afwImage.makeMaskedImage(flat))

# make a butler and specify your dataId
butler = dafPersist.Butler('/tigress/HSC/PFS/2015-11-20/')
dataId = {'calibDate': '2015-11-20', 'calibVersion': 'flat', 'visit': 5445, 'ccd': 2, 'arm': 1}

butler.put(flatExp, "flat", dataId)    


