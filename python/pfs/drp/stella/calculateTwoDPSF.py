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
import pyfits

try:
    type(display)
except NameError:
    display = False

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

def calculateTwoDPSF(flatfilename, specfilename):
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
#    ftec.xCorProf = 20
    ftec.wingSmoothFactor = 2.
    ftec.overSample = 15
    ftec.lambdaSF = 1. / ftec.overSample
    ftec.maxIterSF = 10
    ftec.swathWidth = 500

    # --- create twoDPSFControl
    tdpsfc = drpStella.TwoDPSFControl()
    tdpsfc.signalThreshold = 500.
    tdpsfc.nTermsGaussFit = 3
    tdpsfc.nKrigingPointsX = 25
    tdpsfc.nKrigingPointsY = 25

    """Create a afwImage::MaskedImageF from the flat fits file"""
    mif = afwImage.MaskedImageF(flatfilename)
    print("mif created")

    """Trace fibers"""
    fts = drpStella.findAndTraceAperturesF(mif, ftffc)
    print("findAndTraceApertures finished")

    # --- sort traces by xCenters
    fts.sortTracesByXCenter();
    fts.setTwoDPSFControl(tdpsfc)

    # --- create FiberTraceSet for object exposure
    mis = afwImage.MaskedImageF(specfilename)
    traces = fts.getTraces()
    for i in range(0,fts.size()) :
        trace = fts.getFiberTrace(i)
        trace.createTrace(mis)

    trace = fts.getFiberTrace(0)
    trace.calculate2dPSFPerBin()

    return fts;

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main(argv=None):
# --- start with <msi=calculateTwoDPSF.main('-f="/home/azuri/spectra/pfs/IR-23-0-centerSkyx2.fits" -p="/home/azuri/spectra/pfs/IR-23-0-centerFlatx2.fits_trace0_prof.fits')>
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
    parser.add_argument("--specfilename", '-s', type=str, help="fits file name of object exposure")
    args = parser.parse_args(argv)
    display = args.display
    verbose = args.verbose
    flatFileName = args.flatfilename
    specFileName = args.specfilename
    return calculateTwoDPSF(flatFileName, specFileName)

if __name__ == "__main__":
    main()
