import lsst.daf.persistence as dafPersist
# <snip> #
dataDir = "/data/Subaru/HSC/rerun/myrerun"
butler = dafPersist.Butler(dataDir)

dataId = {'visit': 1234, 'ccd': 56}
bias = butler.get('bias', dataId)

