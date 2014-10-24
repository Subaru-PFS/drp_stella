#!/usr/bin/env python

#USAGE: exp = lsst.afw.image.ExposureF("/home/azuri/spectra/pfs/2014-10-14/IR-23-0-sampledFlatx2-nonoise.fits")
#       myFindTask = findAndTraceAperturesTask.FindAndTraceAperturesTask()
#       fts = myFindTask.run(exp)
#       myExtractTask = createFlatFiberTraceProfileTask.CreateFlatFiberTraceProfileTask()
#       myExtractTask.run(fts)

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
import pyfits
from lsst.pipe.base import Task

class CreateFlatFiberTraceProfileConfig(pexConfig.Config):
        profileInterpolation = pexConfig.Field(
            doc = "Method for determining the spatial profile, [PISKUNOV, SPLINE3], default: PISKUNOV",
            dtype = str,
            default = "PISKUNOV")
        ccdReadOutNoise = pexConfig.Field(
            doc = "CCD readout noise",
            dtype = float,
            default = 1.,
            check = lambda x : x > 0.)
        swathWidth = pexConfig.Field(
            doc = "Size of individual extraction swaths",
            dtype = int,
            default = 500,
            check = lambda x : x > 10)
        telluric = pexConfig.Field(
            doc = "Method for determining the background (+sky in case of slit spectra, default: NONE)",
            dtype = str,
            default = "NONE")
        overSample = pexConfig.Field(
            doc = "Oversampling factor for the determination of the spatial profile (default: 10)",
            dtype = int,
            default = 30,
            check = lambda x : x > 0)
        maxIterSF = pexConfig.Field(
            doc = "Maximum number of iterations for the determination of the spatial profile (default: 8)",
            dtype = int,
            default = 8,
            check = lambda x : x > 0)
        maxIterSky = pexConfig.Field(
            doc = "Maximum number of iterations for the determination of the (constant) background/sky (default: 10)",
            dtype = int,
            default = 10,
            check = lambda x : x >= 0)
        maxIterSig = pexConfig.Field(
            doc = "Maximum number of iterations for masking bad pixels and CCD defects (default: 2)",
            dtype = int,
            default = 2,
            check = lambda x : x > 0)
        lambdaSF = pexConfig.Field(
            doc = "Lambda smoothing factor for spatial profile (default: 1. / overSample)",
            dtype = float,
            default = 17000.,
            check = lambda x : x > 0.)
        lambdaSP = pexConfig.Field(
            doc = "Lambda smoothing factor for spectrum (default: 0)",
            dtype = float,
            default = 0.,
            check = lambda x : x >= 0)
        wingSmoothFactor = pexConfig.Field(
            doc = "Lambda smoothing factor to remove possible oscillation of the wings of the spatial profile (default: 0.)",
            dtype = float,
            default = 0.,
            check = lambda x : x >= 0)

class CreateFlatFiberTraceProfileTask(Task):
    ConfigClass = CreateFlatFiberTraceProfileConfig
    _DefaultName = "createFlatFiberTraceProfileTask"

    def __init__(self, *args, **kwargs):
        super(CreateFlatFiberTraceProfileTask, self).__init__(*args, **kwargs)
#        self.makeSubtask("isr")
#        self.schema = afwTable.SourceTable.makeMinimalSchema()
#        self.makeSubtask("detection", schema=self.schema)
#        self.makeSubtask("measurement", schema=self.schema)
#        self.starSelector = self.config.starSelector.apply()
#        self.candidateKey = self.schema.addField(
#            "calib.psf.candidate", type="Flag",
#            doc=("Flag set if the source was a candidate for PSF determination, "
#                 "as determined by the '%s' star selector.") % self.config.starSelector.name
#        )

    def createFlatFiberTraceProfile(self, inFiberTraceSet, inTraceNumber):
        # --- create FiberTraceFunctionFindingControl
        fiberTraceExtractionControl = drpStella.FiberTraceExtractionControl()
        fiberTraceExtractionControl.profileInterpolation = self.config.profileInterpolation
        fiberTraceExtractionControl.ccdReadOutNoise = self.config.ccdReadOutNoise
        fiberTraceExtractionControl.swathWidth = self.config.swathWidth
        fiberTraceExtractionControl.telluric = self.config.telluric
        fiberTraceExtractionControl.overSample = self.config.overSample
        fiberTraceExtractionControl.maxIterSF = self.config.maxIterSF
        fiberTraceExtractionControl.maxIterSky = self.config.maxIterSky
        fiberTraceExtractionControl.maxIterSig = self.config.maxIterSig
        fiberTraceExtractionControl.lambdaSF = self.config.lambdaSF
        fiberTraceExtractionControl.lambdaSP = self.config.lambdaSP
        fiberTraceExtractionControl.wingSmoothFactor = self.config.wingSmoothFactor
    
        """Create a FiberTraceSet given a flat-field fits file name"""
        fiberTraceSet = inFiberTraceSet
        
        """Calculate spatial profile and extract"""
        fiberTraceSet.sortTracesByXCenter()
        fiberTraceSet.setFiberTraceExtractionControl(fiberTraceExtractionControl)
        fiberTraceSet.extractTraceNumber(inTraceNumber)
        return

    def run(self, inFiberTraceSet, inTraceNumber):
        """Calculate spatial profile and extract FiberTrace number inTraceNumber to 1D

        This method changes the input FiberTraceSet and returns void
        """
        
        self.createFlatFiberTraceProfile(inFiberTraceSet, inTraceNumber)
        
        return
 