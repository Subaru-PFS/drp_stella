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

def createFlatFiberTraceProfile(filename):
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
    ftffc.maxLength = 4096
    ftffc.nLost = 10

    # --- create FiberTraceExtractionControl
    ftec = drpStella.FiberTraceExtractionControl()
    ftec.overSample = 30
    ftec.swathWidth = 500
    ftec.telluric = "NONE"
    
    """Create a FiberTraceSet given a flat-field fits file name"""
#    filename = "/home/azuri/spectra/pfs/2014-10-14/IR-23-0-sampledFlatx2-nonoise.fits"
    mif = afwImage.MaskedImageF(filename)
    print("mif created")
        
    """Trace fibers"""
    fts = drpStella.findAndTraceAperturesF(mif, ftffc)
    print("findAndTraceApertures finished")
    
    # --- sort traces by xCenters
    fts.sortTracesByXCenter();
    
    # --- write trace 0 to fits file
    filename_trace = filename.replace(".fits", "_trace0.fits")
    fts.getFiberTrace(0).getImage().writeFits(filename_trace)

    # ---  create profile and extract fiber trace 0
    fts.setFiberTraceExtractionControl(ftec)
#    print("fts.getFiberTrace(0).getFiberTraceExtractionControl().overSample = ",fts.getFiberTrace(0).getFiberTraceExtractionControl().overSample)
#    return fts;
    fts.getFiberTrace(0).MkSlitFunc()
    
    # --- get profile and write profile to fits file
    profile = fts.getFiberTrace(0).getProfile()
    print("got profile: profile.getArray() = ",profile.getArray())
    print("writing profile to fits file")
    filename_flatprof = filename.replace(".fits", "_trace0_prof.fits")
    profile.writeFits(filename_flatprof)
    print("profile written to ",filename_flatprof)
   
    # --- get reconstructed 2D spectrum and write to fits file
    reconstructed = fts.getFiberTrace(0).getReconstructed2DSpectrum()
    print("got reconstructed 2D spectrum: reconstructed.getArray() = ", reconstructed.getArray())
    filename_flatrec = filename.replace(".fits", "_trace0_rec.fits")
    reconstructed.writeFits(filename_flatrec)
    print("reconstructed spectrum written to ",filename_flatrec)
    
    # --- subtract reconstructed spectrum from input spectrum
    imMinusRec = fts.getFiberTrace(0).getImage().getArray() - reconstructed.getArray()
    filename_flatMinusRec = filename.replace(".fits", "_trace0-rec.fits")
    imMinusRecIm = afwImage.ImageF(imMinusRec)
    print("got difference of original image and reconstructed 2D spectrum: imMinusRecIm.getArray() = ", imMinusRecIm.getArray())
    imMinusRecIm.writeFits(filename_flatMinusRec)
    print("difference of original and reconstructed spectrum written to ",filename_flatMinusRec)
    
    # --- write extracted 1D spectrum (from MkSlitFunc) to fits file
    spec = fts.getFiberTrace(0).getSpectrum()
    np_spec = np.fromiter(spec, dtype=np.float)
    filename_spec = filename.replace(".fits", "_trace0_spec.fits")
    pyfits.writeto(filename_spec,np_spec,clobber=True)
    
    # --- extract 1D spectrum from profile
    fts.getFiberTrace(0).extractFromProfile()

    # --- get reconstructed 2D spectrum and write to fits file
    reconstructed = fts.getFiberTrace(0).getReconstructed2DSpectrum()
    print("got reconstructed 2D spectrum: reconstructed.getArray() = ", reconstructed.getArray())
    filename_flatrec = filename.replace(".fits", "_trace0_recFromProf.fits")
    reconstructed.writeFits(filename_flatrec)
    print("reconstructed spectrum written to ",filename_flatrec)
    
    # --- subtract reconstructed spectrum from input spectrum
    imMinusRec = fts.getFiberTrace(0).getImage().getArray() - reconstructed.getArray()
    filename_flatMinusRec = filename.replace(".fits", "_trace0-recFromProf.fits")
    imMinusRecIm = afwImage.ImageF(imMinusRec)
    print("got difference of original image and reconstructed 2D spectrum: imMinusRecIm.getArray() = ", imMinusRecIm.getArray())
    imMinusRecIm.writeFits(filename_flatMinusRec)
    print("difference of original and reconstructed spectrum written to ",filename_flatMinusRec)
    
    # --- write extracted 1D spectrum (from extractFromProfile) to fits file
    speca = fts.getFiberTrace(0).getSpectrum()
    np_speca = np.fromiter(speca, dtype=np.float)
    filename_specFromProf = filename.replace(".fits", "_trace0_specFromProf.fits")
    pyfits.writeto(filename_specFromProf,np_speca,clobber=True)

    return fts;#.getFiberTraceSet();
    
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main(argv=None):
# --- start with <msi=createFlatFiberTraceProfile.main('-f="/home/azuri/spectra/pfs/IR-23-0-centerFlatx2.fits"')>
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
    return createFlatFiberTraceProfile(fileName)
  
if __name__ == "__main__":
    main()
    