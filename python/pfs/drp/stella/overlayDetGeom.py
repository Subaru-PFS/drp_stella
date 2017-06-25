#!/usr/bin/env python
import argparse

import matplotlib.pyplot as pyplot
import numpy

import lsst.afw.cameraGeom.utils as camGeomUtils
import lsst.afw.table
import lsst.daf.persistence as dafPersist

def main(rootDir, visit, arm, ccd, fitsFileName):

    # make a butler and specify your dataId
    butler = dafPersist.Butler(rootDir)
    dataId = {'visit': visit, 'arm': arm, 'ccd': ccd}

    # get the exposure from the butler
    exposure = butler.get('raw', dataId)
    print 'dir(dataId) = ',dir(dataId)

    # get the maskedImage from the exposure, and the image from the mimg
    mimg = exposure.getMaskedImage()
    img = mimg.getImage()

    # convert to a numpy ndarray
    nimg = img.getArray()

    # stretch it with arcsinh and make a png with pyplot
    pyplot.imshow(numpy.arcsinh(nimg), cmap='gray')
    pyplot.gcf().savefig("test.png")

    ampInfoCatalog = lsst.afw.table.AmpInfoCatalog(lsst.afw.table.AmpInfoTable.makeMinimalSchema())

    # Create instance of ampInfoCatalog by reading fitsFileName
    amps = ampInfoCatalog.readFits(fitsFileName)

    # Get CCD image
    print 'dir(exposure) = ',dir(exposure)
    det = exposure.getDetector()
    print 'det.getBBox() = ',det.getBBox()
    butlerImage = camGeomUtils.ButlerImage(butler, "raw", visit=visit, arm=arm, isTrimmed=True, verbose=True)
    print 'butlerImage = ',butlerImage
    print 'dir(butlerImage) = ',dir(butlerImage)
    print 'butlerImage.isTrimmed = ',butlerImage.isTrimmed
    camGeomUtils.showCcd(det, imageSource=butlerImage, frame=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Root directory of data repository")
    parser.add_argument("visit", type=int, help="Visit to show")
    parser.add_argument("arm", type=int, help="Spectrograph arm to show [0,1,2]")
    parser.add_argument("ccd", type=int, help="CCD to show")
    parser.add_argument("fitsFileName", help="Name of fits file containing the detector geometry")
    args = parser.parse_args()

    main(args.root, args.visit, args.arm, args.ccd, args.fitsFileName)
