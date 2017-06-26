#!/usr/bin/env python
import numpy as np
import pyfits

import lsst.afw.image as afwImage
import pfs.drp.stella as drpStella

def createFlatFiberTraceProfiles(filename):
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
    ftffc.minLength = 3000
    ftffc.maxLendth = 4096
    ftffc.nLost = 10

    # --- create FiberTraceExtractionControl
    ftec = drpStella.FiberTraceExtractionControl()
    ftec.xCorProf = 0
    ftec.wingSmoothFactor = 0.
    ftec.overSample = 30
    ftec.lambdaSF = 5000. / ftec.overSample
    ftec.maxIterSF = 10
    ftec.swathWidth = 500
    ftec.telluric = "NONE"
    ftec.maxIterSky = 1

    # --- Create a FiberTraceSet given a flat-field fits file name
    mif = afwImage.MaskedImageF(filename)
    print("mif created")

    # --- Trace fibers
    fts = drpStella.findAndTraceAperturesF(mif, ftffc)
    print("findAndTraceApertures finished")

    # --- sort traces by xCenters
    fts.sortTracesByXCenter();

    # --- write trace 0 to fits file
    for i in range(0,fts.size()):
        filename_trace = filename.replace(".fits", "_trace"+str(i)+".fits")
        fts.getFiberTrace(i).getImage().writeFits(filename_trace)

    # ---  create profile and extract fiber trace 0
    fts.setFiberTraceExtractionControl(ftec)
    fts.extractAllTraces()

    # --- get profile and write profile to fits file
    for i in range(0,fts.size()):
        profile = fts.getFiberTrace(i).getProfile()
        print("got profile: profile.getArray() = ",profile.getArray())
        print("writing profile "+str(i)+" to fits file")
        filename_flatprof = filename.replace(".fits", "_trace"+str(i)+"_prof.fits")
        profile.writeFits(filename_flatprof)
        print("profile written to ",filename_flatprof)

    # --- get reconstructed 2D spectrum and write to fits file
        reconstructed = fts.getFiberTrace(i).getReconstructed2DSpectrum()
        print("got reconstructed 2D spectrum: reconstructed.getArray() = ", reconstructed.getArray())
        filename_flatrec = filename.replace(".fits", "_trace"+str(i)+"_rec.fits")
        reconstructed.writeFits(filename_flatrec)
        print("reconstructed spectrum written to ",filename_flatrec)

    # --- subtract reconstructed spectrum from input spectrum
        imMinusRec = fts.getFiberTrace(i).getImage().getArray() - reconstructed.getArray()
        filename_flatMinusRec = filename.replace(".fits", "_trace"+str(i)+"-rec.fits")
        imMinusRecIm = afwImage.ImageF(imMinusRec)
        print("got difference of original image and reconstructed 2D spectrum: imMinusRecIm.getArray() = ", imMinusRecIm.getArray())
        imMinusRecIm.writeFits(filename_flatMinusRec)
        print("difference of original and reconstructed spectrum written to ",filename_flatMinusRec)

    # --- write extracted 1D spectrum (from MkSlitFunc) to fits file
        spec = fts.getFiberTrace(i).getSpectrum()
        np_spec = np.fromiter(spec, dtype=np.float)
        filename_spec = filename.replace(".fits", "_trace"+str(i)+"_spec.fits")
        pyfits.writeto(filename_spec,np_spec,clobber=True)

    # --- extract 1D spectrum from profile
        fts.getFiberTrace(i).extractFromProfile()

    # --- get reconstructed 2D spectrum and write to fits file
        reconstructed = fts.getFiberTrace(i).getReconstructed2DSpectrum()
        print("got reconstructed 2D spectrum: reconstructed.getArray() = ", reconstructed.getArray())
        filename_flatrec = filename.replace(".fits", "_trace"+str(i)+"_recFromProf.fits")
        reconstructed.writeFits(filename_flatrec)
        print("reconstructed spectrum written to ",filename_flatrec)

    # --- subtract reconstructed spectrum from input spectrum
        imMinusRec = fts.getFiberTrace(i).getImage().getArray() - reconstructed.getArray()
        filename_flatMinusRec = filename.replace(".fits", "_trace"+str(i)+"-recFromProf.fits")
        imMinusRecIm = afwImage.ImageF(imMinusRec)
        print("got difference of original image and reconstructed 2D spectrum: imMinusRecIm.getArray() = ", imMinusRecIm.getArray())
        imMinusRecIm.writeFits(filename_flatMinusRec)
        print("difference of original and reconstructed spectrum written to ",filename_flatMinusRec)

    # --- write extracted 1D spectrum (from extractFromProfile) to fits file
        speca = fts.getFiberTrace(i).getSpectrum()
        np_speca = np.fromiter(speca, dtype=np.float)
        filename_specFromProf = filename.replace(".fits", "_trace"+str(i)+"_specFromProf.fits")
        pyfits.writeto(filename_specFromProf,np_speca,clobber=True)

    return fts;
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
    parser.add_argument("--filename", '-f', type=str, help="fits file name of flat exposure")
    args = parser.parse_args(argv)
    display = args.display
    verbose = args.verbose
    fileName = args.filename
    return createFlatFiberTraceProfiles(fileName)

if __name__ == "__main__":
    main()
