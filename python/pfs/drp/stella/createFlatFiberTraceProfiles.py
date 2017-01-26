#!/usr/bin/env python
"""
Tests for measuring things

Run with:
   python FiberTrace.py
or
   python
   >>> import FiberTrace; FiberTrace.run()
"""

#import unittest
import numpy as np
#import lsst.utils.tests as tests
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import pfs.drp.stella as drpStella
#import pfs.drp.stella.math as drpStellaMath
import pyfits

try:
    type(display)
except NameError:
    display = False

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

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
    ftec.overSample = 30
    ftec.swathWidth = 500
    ftec.telluric = "NONE"
    
    """Create a FiberTraceSet given a flat-field fits file name"""
    mif = afwImage.MaskedImageF(filename)
    print("mif created")
        
    """Trace fibers"""
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
   
    return fts;#.getFiberTraceSet();
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main(argv=None):
# --- start with <msi=createFlatFiberTraceProfiles.main('-f="/home/azuri/spectra/pfs/IR-23-0-centerFlatx2.fits"')>
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
    