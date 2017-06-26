#!/usr/bin/env python
import numpy as np
import pyfits

import lsst.afw.image as afwImage
import pfs.drp.stella as drpStella

def extractFromProfile(flatfilename, flatprofilename, specfilename):
    # --- create FiberTraceFunctionFindingControl
    ftffc = drpStella.FiberTraceFunctionFindingControl()
    ftffc.fiberTraceFunctionControl.interpolation = "POLYNOMIAL"
    ftffc.fiberTraceFunctionControl.order = 5
    ftffc.fiberTraceFunctionControl.xLow = -4.2
    ftffc.fiberTraceFunctionControl.xHigh = 4.2
    ftffc.apertureFWHM = 3.2
    ftffc.signalThreshold = 120.
    ftffc.nTermsGaussFit = 3
    ftffc.saturationLevel = 65500.

    # --- create FiberTraceExtractionControl
    ftec = drpStella.FiberTraceExtractionControl()
    ftec.wingSmoothFactor = 2.
    ftec.overSample = 15
    ftec.lambdaSF = 1. / ftec.overSample
    ftec.maxIterSF = 10
    ftec.swathWidth = 500

    # Create a afwImage::MaskedImageF from the flat fits file
    mif = afwImage.MaskedImageF(flatfilename)
    print("mif created")

    # Create a FiberTraceSet given a fits file name
    msi = drpStella.MaskedSpectrographImageF(mif)
    print("msi created")

    # Trace fibers
    msi.findAndTraceApertures(ftffc, 0, mif.getHeight(), 10)
    print("msi.findAndTraceApertures finished")

    # --- sort traces by xCenters
    msi.getFiberTraceSet().sortTracesByXCenter();

    # --- create Fiber Trace for object exposure
    mis = afwImage.MaskedImageF(specfilename)
    msis = drpStella.MaskedSpectrographImageF(mis)
    ft = drpStella.FiberTraceF(mis)
    ft.setFiberTraceFunction(msi.getFiberTraceSet().getFiberTrace(0).getFiberTraceFunction())
    ft.calculateXCenters()
    ft.createTrace()
    ft.getImage().writeFits(specfilename.replace(".fits", "_trace0.fits"))

    # ---  read profile and extract fiber trace 0
    ft.setFiberTraceExtractionControl(ftec)
    profile = afwImage.ImageF(flatprofilename)
    ft.setProfile(profile)
    ft.extractFromProfile()

    # --- add FiberTrace to FiberTraceSet
    msis.getFiberTraceSet().addFiberTrace(ft)

    # --- get reconstructed 2D spectrum and write to fits file
    reconstructed = msis.getFiberTraceSet().getFiberTrace(0).getReconstructed2DSpectrum()
    print("got reconstructed 2D spectrum: reconstructed.getArray() = ", reconstructed.getArray())
    specfilename_rec = specfilename.replace(".fits", "_trace0_recFromProf.fits")
    reconstructed.writeFits(specfilename_rec)
    print("reconstructed spectrum written to ",specfilename_rec)

    # --- subtract reconstructed spectrum from input Flat spectrum
    imMinusRec = msis.getFiberTraceSet().getFiberTrace(0).getImage().getArray() - reconstructed.getArray()
    specfilename_MinusRec = specfilename.replace(".fits", "_trace0-recFromProf.fits")
    imMinusRecIm = afwImage.ImageF(imMinusRec)
    print("got difference of original image and reconstructed 2D spectrum: imMinusRecIm.getArray() = ", imMinusRecIm.getArray())
    imMinusRecIm.writeFits(specfilename_MinusRec)
    print("difference of original and reconstructed spectrum written to ",specfilename_MinusRec)

    # --- write extracted 1D spectrum (from MkSlitFunc) to fits file
    spec = msis.getFiberTraceSet().getFiberTrace(0).getSpectrum()
    np_spec = np.fromiter(spec, dtype=np.float)
    specfilename_spec = specfilename.replace(".fits", "_trace0_specFromProf.fits")
    pyfits.writeto(specfilename_spec,np_spec,clobber=True)

    return msis;
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main(argv=None):
    if argv is None:
      import sys
      argv = sys.argv[1:]
    if isinstance(argv, basestring):
      import shlex
      argv = shlex.split(argv)

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--display", '-d', default=False, action="store_true", help="Activate display?")
    parser.add_argument("--verbose", '-v', type=int, default=0, help="Verbosity level")
    parser.add_argument("--flatfilename", '-f', type=str, help="fits file name of flat exposure")
    parser.add_argument("--flatprofilename", '-p', type=str, help="fits file name of flat exposure profile")
    parser.add_argument("--specfilename", '-s', type=str, help="fits file name of object exposure")
    args = parser.parse_args(argv)
    display = args.display
    verbose = args.verbose
    flatFileName = args.flatfilename
    flatProfileName = args.flatprofilename
    specFileName = args.specfilename
    return extractFromProfile(flatFileName, flatProfileName, specFileName)

if __name__ == "__main__":
    main()
