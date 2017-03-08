#!/usr/bin/env python
import numpy as np
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import pfs.drp.stella as drpStella

try:
    type(display)
except NameError:
    display = False

import lsst.afw.display.ds9 as ds9
import lsst.afw.display.utils as displayUtils

def createFlatFiberTraceProfileSES(filename):
    # --- create FiberTraceFunctionFindingControl
    ftffc = drpStella.FiberTraceFunctionFindingControl()
    ftffc.fiberTraceFunctionControl.interpolation = "POLYNOMIAL"
    ftffc.fiberTraceFunctionControl.order = 5
    ftffc.fiberTraceFunctionControl.xLow = -4.2
    ftffc.fiberTraceFunctionControl.xHigh = 4.2
    ftffc.apertureFWHM = 2.5
    ftffc.signalThreshold = 100.
    ftffc.nTermsGaussFit = 3
    ftffc.saturationLevel = 65500.
    ftffc.minLength = 100
    ftffc.maxLendth = 300
    ftffc.nLost = 10

    # --- create FiberTraceExtractionControl
    ftec = drpStella.FiberTraceExtractionControl()
    ftec.xCorProf = 0#100
    ftec.wingSmoothFactor = 2.
    ftec.lambdaSF = .1
    ftec.maxIterSF = 8

    # --- Create a FiberTraceSet given a flat-field fits file name
    mif = afwImage.MaskedImageF(filename)
    print("mif created")

    fts = drpStella.findAndTraceApertures(mif, ftffc)
    print("findAndTraceApertures finished")

    # --- write trace 0 to fits file
    filename_trace = filename + '_trace20.fits'
    fts.getFiberTrace(20).getImage().writeFits(filename_trace)

    # ---  create profile and extract fiber trace 0
    fts.getFiberTrace(20).setFiberTraceExtractionControl(ftec)
    fts.getFiberTrace(20).MkSlitFunc()
    return msi;

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
    return createFlatFiberTraceProfileSES("/home/azuri/spectra/SES/test/combinedFlat.fits")#fileName)

if __name__ == "__main__":
    main()
