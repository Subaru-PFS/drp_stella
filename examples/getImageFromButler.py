import lsst.daf.persistence as dafPersist

butler = dafPersist.Butler('/tigress/HSC/PFS/2015-11-20/')
dataId = {'visit': '5441', 'ccd': 2}
im = butler.get('raw', dataId)
