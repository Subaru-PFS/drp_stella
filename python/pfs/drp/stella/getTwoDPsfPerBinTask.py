#!/usr/bin/env python

#DESCRIPTION: If inExposure != NULL the input inFiberTraceWithProfile will be changed, meaning that inFiberTraceWithProfile._trace
#             will contain the trace of the Exposure to be extracted

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

class GetTwoDPsfPerBinConfig(pexConfig.Config):
    signalThreshold = pexConfig.Field(
                doc = "Minimum signal above continuum to count as emission line",
              dtype = float,
            default = 1000.,
              check = lambda x : x > 0.)
    
    swathWidth = pexConfig.Field(
           doc = "Size of individual extraction swaths",
         dtype = int,
       default = 500,
         check = lambda x : x > 100)
    
    xFWHM = pexConfig.Field(
      doc = "FWHM of an assumed Gaussian PSF perpendicular to the dispersion direction in pixels",
    dtype = float,
    default = 2.5,
      check = lambda x : x > 1.)
    
    yFWHM = pexConfig.Field(
      doc = "FWHM of an assumed Gaussian PSF in the dispersion direction in pixels",
    dtype = float,
    default = 2.5,
      check = lambda x : x > 1.)
    
    nTermsGaussFit = pexConfig.Field(
               doc = "3 to fit Gaussian; 4 to fit Gaussian plus constant (sky), profile must be at least 5 pixels wide; 5 to fit Gaussian plus linear term (sloped sky), profile must be at least 6 pixels wide",
             dtype = int,
           default = 5,
             check = lambda x : x < 6)
    
    saturationLevel = pexConfig.Field(
                doc = "CCD saturation level",
              dtype = float,
            default = 65000.,
              check = lambda x : x > 0.)
              
    nKnotsX = pexConfig.Field(
        doc = "Number of interpolation knots in X direction",
      dtype = int,
    default = 75,
      check = lambda x : x > 1)
      
    nKnotsY = pexConfig.Field(
        doc = "Number of interpolation knots in Y direction",
      dtype = int,
    default = 75,
      check = lambda x : x > 1)
      
    smooth = pexConfig.Field(
       doc = "Smoothing factor for bidirectional spline interpolation",
     dtype = float,
    default = 35000.,
      check = lambda x : x > 0.)

class GetTwoDPsfPerBinTask(Task):
    ConfigClass = GetTwoDPsfPerBinConfig
    _DefaultName = "getTwoDPsfPerBin"

    def __init__(self, *args, **kwargs):
        super(GetTwoDPsfPerBinTask, self).__init__(*args, **kwargs)

        """create TwoDPSFControl"""
        self.tdpsfc = drpStella.TwoDPSFControl()
        tdpsfc = self.tdpsfc
        tdpsfc.signalThreshold = self.config.signalThreshold
        tdpsfc.swathWidth = self.config.swathWidth
        tdpsfc.xFWHM = self.config.xFWHM
        tdpsfc.yFWHM = self.config.yFWHM
        tdpsfc.saturationLevel = self.config.saturationLevel
        tdpsfc.nKnotsX = self.config.nKnotsX
        tdpsfc.nKnotsY = self.config.nKnotsY
        tdpsfc.smooth = self.config.smooth

    def getTwoDPsfPerBin(self, inFiberTraceSet, inSpectrumSet, inTraceNumbers):
    
        try:
            tdpsfcp = self.tdpsfc.getPointer()
            if inTraceNumbers[0] == -1:
                traceNumbers = range(inFiberTraceSet.size())
            else:
                traceNumbers = inTraceNumbers
            print "inTraceNumbers = ", inTraceNumbers
            print "traceNumbers = ", traceNumbers
            psfSets = drpStella.PSFSetVectorF()

            if inSpectrumSet.size() != inFiberTraceSet.size():
                print "inSpectrumSet.size(=", inSpectrumSet.size(), ") != inFiberTraceSet.size(=", inFiberTraceSet.size(),")"
                raise Exception("inSpectrumSet.size(=", inSpectrumSet.size(), ") != inFiberTraceSet.size(=", inFiberTraceSet.size(),")")

            """Create PSFSet"""
            for i in traceNumbers:
                if i < 0:
                    raise Exception("i < 0")
                elif i >= inFiberTraceSet.size():
                    raise Exception("i >= inFiberTraceSet.size()")

                trace = inFiberTraceSet.getFiberTrace(i)
                spectrum = inSpectrumSet.getSpectrum(i)

                psfSet = drpStella.calculate2dPSFPerBinF(trace, spectrum, tdpsfcp)
#                print "psfSet = ", psfSet
#                print "psfSet.size() = ",psfSet.size()
#                print "psfSet.getPSFs() = ",psfSet.getPSFs()
#                print "psfSet.getPSFs()[0] = ", psfSet.getPSFs()[0]
#                print "psfSets.size() = ",psfSets.size()
                psfSets.push_back(psfSet)
#                print "psfSets.size() = ",psfSets.size()
#                print "psfSets[0] = ",psfSets[0]
                print "trace ",i," done"

            print "getTwoDPsfPerBinTask::getTwoDPsfPerBin finished"
            return psfSets
        except Exception as e:
            print e

    def run(self, inFiberTraceSet, inSpectrumSet, inTraceNumbers=[-1]):
        """Create one 2DPSF per bin of size swathWidth and return a set of 2DPSFs

        This method is the top-level for running the automatic extraction of the 2D PSFs from the inFiberTraceSet as a stand-alone BatchPoolTask.
        
        This method returns a SpectrumSet
        """
        
        psfSets = self.getTwoDPsfPerBin(inFiberTraceSet, inSpectrumSet, inTraceNumbers)
        print "self.run finished"
        return psfSets
