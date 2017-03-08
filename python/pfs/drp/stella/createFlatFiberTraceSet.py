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

def createFlatFiberTraceSet(filename):
    ftffc = drpStella.FiberTraceFunctionFindingControl()
    ftffc.fiberTraceFunctionControl.interpolation = "POLYNOMIAL"
    ftffc.fiberTraceFunctionControl.order = 4
    ftffc.fiberTraceFunctionControl.xLow = -4.2
    ftffc.fiberTraceFunctionControl.xHigh = 4.2
    ftffc.apertureFWHM = 3.2
    ftffc.signalThreshold = 100.
    ftffc.nTermsGaussFit = 3
    ftffc.saturationLevel = 65500.

    # Create a FiberTraceSet given a flat-field fits file name
    mif = afwImage.MaskedImageF(filename)
    print("mif created")

    # Trace fibers
    msi = drpStella.MaskedSpectrographImageF(mif)
    print("msi created")

    msi.findAndTraceApertures(ftffc, 0, mif.getHeight(), 10)
    print("msi.findAndTraceApertures finished")

    return msi;
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main(argv=None):
# --- start with <msi=createFlatFiberTraceSet.main('-f="/home/azuri/spectra/pfs/IR-23-0-centerFlatx2.fits"')>
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
    return createFlatFiberTraceSet(fileName)

if __name__ == "__main__":
    main()
