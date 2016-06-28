import os
import sys
import argparse

import lsst.pex.config                  as pexConfig
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
from lsst.pipe.base import Task

class CreateDetGeomConfig(pexConfig.Config):
    """Configuration for creating the PFS detector geometry files"""
    outDir = pexConfig.Field( doc = "Name of output directory", dtype=str, default="/Users/azuri/stella-git/obs_pfs/pfs/camera/")

class CreateDetGeomTask(Task):
    """Task to create the detector geometry files for PFS"""
    ConfigClass = CreateDetGeomConfig
    _DefaultName = "createDetGeomTask"

    def __init__(self, *args, **kwargs):
        super(CreateDetGeomTask, self).__init__(*args, **kwargs)

    def run(self):
        gain = (1.24, 1.24, 1.27, 1.18, 1.26, 1.20, 1.24, 1.26)
        rdnoise = (3.61, 3.78, 3.18, 2.95, 3.19, 3.80, 4.51, 3.18)
        amp = 0
        for iArm in range(3):
            for iCCD in range(4):
                detector = afwTable.AmpInfoCatalog(afwTable.AmpInfoTable.makeMinimalSchema())

                for iAmp in range(8):

                    x0 = iAmp * 552
                    x1 = ((iAmp + 1) * 552)-1
                    print 'iAmp = ',iAmp,': x0 = ',x0,', x1 = ',x1
                    y0 = 0
                    y1 = 4299
                    AmpA = detector.addNew()
                    print 'AmpA = ',type(AmpA)
                    print 'dir(AmpA) = ',dir(AmpA)
                    AmpA.setName(str(amp))

                    bb1 = afwGeom.Point2I(iAmp * 512, 0)
                    bb2 = afwGeom.Point2I((512*(iAmp+1))-1, 4173)
                    print 'iAmp = ',iAmp,': bb1 = ',bb1
                    print 'iAmp = ',iAmp,': bb2 = ',bb2
                    AmpA.setBBox(afwGeom.Box2I(bb1, bb2))

                    AmpA.setGain(gain[iAmp])
                    print 'gain[',iAmp,'] = ',gain[iAmp]
                    print 'rdnoise[',iAmp,'] = ',rdnoise[iAmp]
                    AmpA.setReadNoise(rdnoise[iAmp])
                    AmpA.setSaturation(60000)

                    rbb1 = afwGeom.Point2I(x0, y0)
                    rbb2 = afwGeom.Point2I(x1, y1)
                    print 'iAmp = ',iAmp,': rbb1 = ',rbb1
                    print 'iAmp = ',iAmp,': rbb2 = ',rbb2
                    AmpA.setRawBBox(afwGeom.Box2I(rbb1, rbb2))

                    rdbb1 = afwGeom.Point2I(x0+9, 49)
                    rdbb2 = afwGeom.Point2I(x1-31, y1-77)
                    print 'iAmp = ',iAmp,': rdbb1 = ',rdbb1
                    print 'iAmp = ',iAmp,': rdbb2 = ',rdbb2
                    AmpA.setRawDataBBox(afwGeom.Box2I(rdbb1, rdbb2))

                    if iAmp in [1,3,5,7]:
                        AmpA.setRawFlipX(True)
                    else:
                        AmpA.setRawFlipX(False)
                    AmpA.setRawFlipY(False)
                    AmpA.setRawXYOffset(afwGeom.Extent2I(0, 0))

                    rhobb1 = afwGeom.Point2I(x0+522, 49)
                    rhobb2 = afwGeom.Point2I(x1-1, y1-77)
                    print 'iAmp = ',iAmp,': rhobb1 = ',rhobb1
                    print 'iAmp = ',iAmp,': rhobb2 = ',rhobb2
                    AmpA.setRawHorizontalOverscanBBox(afwGeom.Box2I(rhobb1, rhobb2))

                    rvobb1 = afwGeom.Point2I(x0+9, y1-76)
                    rvobb2 = afwGeom.Point2I(x1-31, y1)
                    print 'iAmp = ',iAmp,': rvobb1 = ',rvobb1
                    print 'iAmp = ',iAmp,': rvobb2 = ',rvobb2
                    AmpA.setRawVerticalOverscanBBox(afwGeom.Box2I(rvobb1, rvobb2))

                    rpbb1 = afwGeom.Point2I(x0+9, 0)
                    rpbb2 = afwGeom.Point2I(x1-31, 48)
                    print 'iAmp = ',iAmp,': rpbb1 = ',rpbb1
                    print 'iAmp = ',iAmp,': rpbb2 = ',rpbb2
                    AmpA.setRawPrescanBBox(afwGeom.Box2I(rpbb1, rpbb2))

                    AmpA.setHasRawInfo(True)
                    AmpA.setReadoutCorner(afwTable.LL)

                    amp = amp+1

                sArm = ' '
                if iArm == 0:
                    sArm = 'b'
                elif iArm == 1:
                    sArm = 'r'
                elif iArm == 2:
                    sArm = 'n'
                else: 
                    sArm = 'm'
                detector.writeFits(self.config.outDir+"/"+sArm+"_"+str(iCCD+1)+".fits")

