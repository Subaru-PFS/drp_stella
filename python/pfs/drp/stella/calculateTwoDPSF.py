#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
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
    ftffc.fiberTraceFunctionControl.xLow = -5.0#-4.2
    ftffc.fiberTraceFunctionControl.xHigh = 5.0#4.2
    ftffc.apertureFWHM = 3.2
    ftffc.signalThreshold = 120.
    ftffc.nTermsGaussFit = 3
    ftffc.saturationLevel = 65500.

    # --- create FiberTraceProfileFittingControl
    ftec = drpStella.FiberTraceProfileFittingControl()
    ftec.profileInterpolation = "SPLINE3"
    ftec.telluric = "NONE"
    ftec.wingSmoothFactor = 2.
    ftec.overSample = 15
    ftec.lambdaSF = 1. / ftec.overSample
    ftec.maxIterSF = 10
    ftec.maxIterSky = 1
    ftec.maxIterSig = 2
    ftec.swathWidth = 500

    # --- create twoDPSFControl
    tdpsfc = drpStella.TwoDPSFControl()
    tdpsfc.signalThreshold = 500.
    tdpsfc.nTermsGaussFit = 3
    tdpsfc.nKnotsX = 80
    tdpsfc.nKnotsY = 80
    tdpsfc.smooth = 3500000.

    bias = pyfits.getdata(flatfilename, 3)

    # --- Create a afwImage::MaskedImageF from the flat fits file
    mif = afwImage.MaskedImageF(flatfilename)
    bias = afwImage.ImageF(flatfilename, 3)
    mif[:] -= bias
    print("mif created")

    # --- Trace fibers
    ftffcp = ftffc.getPointer()
    fts = drpStella.findAndTraceAperturesF(mif, ftffcp)
    print("findAndTraceApertures finished")

    # --- sort traces by xCenters
    fts.sortTracesByXCenter()
    tdpsfcp = tdpsfc.getPointer()
    ftecp = ftec.getPointer()
    fts.setFiberTraceProfileFittingControl(ftecp)
    fts.calcProfileAllTraces()

    # --- create FiberTraceSet for object exposure
    bias = pyfits.getdata(specfilename, 3)
    mis = afwImage.MaskedImageF(specfilename)
    bias = afwImage.ImageF(specfilename, 3)
    mis[:] -= bias
    for i in range(0,5):#fts.size()) :
        trace = fts.getFiberTrace(i)
        trace.setITrace(i)
        trace.createTrace(mis)
        spectrum = trace.extractFromProfile()
        if i == 5:
            return fts
        if i != 5:
            psfset = drpStella.calculate2dPSFPerBinF(trace, spectrum, tdpsfcp)
            print "trace ",i,": psfset.size() = ",psfset.size()
            print "trace ",i," done"

        if False:
            for j in range(0,len(psfset)) :
                psfa = psfset[j]

                xvec = psfa.getImagePSF_XRelativeToCenter()
                yvec = psfa.getImagePSF_YRelativeToCenter()
                zvec = psfa.getImagePSF_ZNormalized()
                wvec = psfa.getImagePSF_Weight()

                fig, ax = plt.subplots(figsize=[10,10])
                ax.scatter(xvec, yvec, c=zvec)#, alpha=0.5)
                ax.set_xlabel(r'x', fontsize=20)
                ax.set_ylabel(r'y', fontsize=20)
                ax.set_title('extracted PSFs')

                ax.grid(True)
                fig.tight_layout()

                plt.show()

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
