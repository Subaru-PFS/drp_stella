#!/usr/bin/env python

#USAGE: exp = lsst.afw.image.ExposureF("/home/azuri/spectra/pfs/2014-10-14/IR-23-0-sampledFlatx2-nonoise.fits")
#       myTask = findAndTraceAperturesTask.FindAndTraceAperturesTask()
#       fts = myTask.run(exp)

#import os
#import math
#import numpy

#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab

import lsst.afw.geom                    as afwGeom
import lsst.afw.image                   as afwImage
import lsst.pex.config                  as pexConfig
import pfs.drp.stella as drpStella
from lsst.pipe.base import Task

class FindAndTraceAperturesConfig(pexConfig.Config):
      interpolation = pexConfig.Field(
          doc = "Interpolation schemes (CHEBYSHEV, LEGENDRE, CUBIC, LINEAR, POLYNOMIAL[only one implemented atm])",
          dtype = str,
          default = "POLYNOMIAL")
      order = pexConfig.Field(
          doc = "Polynomial order",
          dtype = int,
          default = 3,
          check = lambda x : x >= 0)
      xLow = pexConfig.Field(
          doc = "Lower (left) limit of aperture relative to center position of trace in x (< 0.)",
          dtype = float,
          default = -4.,
          check = lambda x : x < 0.)
      xHigh = pexConfig.Field(
          doc = "Upper (right) limit of aperture relative to center position of trace in x",
          dtype = float,
          default = 4.,
          check = lambda x : x > 0.)
      apertureFWHM = pexConfig.Field(
          doc = "FWHM of an assumed Gaussian spatial profile for tracing the spectra",
          dtype = float,
          default = 2.5,
          check = lambda x : x > 0.)
      signalThreshold = pexConfig.Field(
          doc = "Signal below this threshold is assumed zero for tracing the spectra",
          dtype = float,
          default = 120.,
          check = lambda x : x >= 0.)
      nTermsGaussFit = pexConfig.Field(
          doc = "1 to look for maximum only without GaussFit; 3 to fit Gaussian; 4 to fit Gaussian plus constant background, 5 to fit Gaussian plus linear term (sloped backgfound)",
          dtype = int,
          default = 3,
          check = lambda x : x > 0)
      saturationLevel = pexConfig.Field(
          doc = "CCD saturation level",
          dtype = float,
          default = 65000.,
          check = lambda x : x > 0.)
      minLength = pexConfig.Field(
          doc = "Minimum aperture length to count as found FiberTrace",
          dtype = int,
          default = 3000,
          check = lambda x : x >= 0)
      maxLength = pexConfig.Field(
          doc = "Maximum aperture length to count as found FiberTrace",
          dtype = int,
          default = 4096,
          check = lambda x : x >= 0)
      nLost = pexConfig.Field(
          doc = "Number of consecutive times the trace is lost before aborting the trace",
          dtype = int,
          default = 10,
          check = lambda x : x >= 0)

class FindAndTraceAperturesTask(Task):
    ConfigClass = FindAndTraceAperturesConfig
    _DefaultName = "findAndTraceApertures"

    def __init__(self, *args, **kwargs):
        super(FindAndTraceAperturesTask, self).__init__(*args, **kwargs)

        """create FiberTraceFunctionFindingControl"""
        self.ftffc = drpStella.FiberTraceFunctionFindingControl()
        ftffc = self.ftffc
        ftffc.fiberTraceFunctionControl.interpolation = self.config.interpolation
        ftffc.fiberTraceFunctionControl.order = self.config.order
        ftffc.fiberTraceFunctionControl.xLow = self.config.xLow
        ftffc.fiberTraceFunctionControl.xHigh = self.config.xHigh
        ftffc.apertureFWHM = self.config.apertureFWHM
        ftffc.signalThreshold = self.config.signalThreshold
        ftffc.nTermsGaussFit = self.config.nTermsGaussFit
        ftffc.saturationLevel = self.config.saturationLevel
        ftffc.minLength = self.config.minLength
        ftffc.maxLength = self.config.maxLength
        ftffc.nLost = self.config.nLost

    def findAndTraceApertures(self, inExposure):
    
        """Create a FiberTraceSet given a flat-field exposure"""
        inMaskedImage = inExposure.getMaskedImage()
        print("inMaskedImage created")
        
        """Trace fibers"""
        fts = drpStella.findAndTraceAperturesF(inMaskedImage, self.ftffc)
        return fts

    def run(self, inExposure):
        """Find and trace fiber traces

        This method is the top-level for running the automatic finding and tracing of fiber traces on the CCD image
        as a stand-alone BatchPoolTask.
        
        This method returns a FiberTraceSet
        """
        
        fiberTraceSet = self.findAndTraceApertures(inExposure)
        
        return fiberTraceSet
 