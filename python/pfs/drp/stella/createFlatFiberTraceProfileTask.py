#!/usr/bin/env python

#USAGE: exp = lsst.afw.image.ExposureF("/home/azuri/spectra/pfs/2014-10-14/IR-23-0-sampledFlatx2-nonoise.fits")
#       myFindTask = findAndTraceAperturesTask.FindAndTraceAperturesTask()
#       fts = myFindTask.run(exp)
#       myExtractTask = createFlatFiberTraceProfileTask.CreateFlatFiberTraceProfileTask()
#       myExtractTask.run(fts)

import lsst.pex.config                  as pexConfig
import pfs.drp.stella as drpStella
from lsst.pipe.base import Task

class CreateFlatFiberTraceProfileConfig(pexConfig.Config):
        profileInterpolation = pexConfig.Field(
            doc = "Method for determining the spatial profile, [PISKUNOV, SPLINE3], default: SPLINE3",
            dtype = str,
            default = "SPLINE3")
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
            default = 10,
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

    def createFlatFiberTraceProfile(self, inFiberTraceSet, inTraceNumbers):
        # --- create FiberTraceProfileFittingControl
        fiberTraceProfileFittingControl = drpStella.FiberTraceProfileFittingControl()
        fiberTraceProfileFittingControl.profileInterpolation = self.config.profileInterpolation
        fiberTraceProfileFittingControl.swathWidth = self.config.swathWidth
        fiberTraceProfileFittingControl.telluric = self.config.telluric
        fiberTraceProfileFittingControl.overSample = self.config.overSample
        fiberTraceProfileFittingControl.maxIterSF = self.config.maxIterSF
        fiberTraceProfileFittingControl.maxIterSky = self.config.maxIterSky
        fiberTraceProfileFittingControl.maxIterSig = self.config.maxIterSig
        fiberTraceProfileFittingControl.lambdaSF = self.config.lambdaSF
        fiberTraceProfileFittingControl.lambdaSP = self.config.lambdaSP
        fiberTraceProfileFittingControl.wingSmoothFactor = self.config.wingSmoothFactor
        
        """Calculate spatial profile"""
        inFiberTraceSet.setFiberTraceProfileFittingControl(fiberTraceProfileFittingControl)
        if inTraceNumbers[0] == -1 :
            inFiberTraceSet.calcProfileAllTraces()
        else :
#            spectrumSet = drpStella.SpectrumSet()
            for i in inTraceNumbers :
                inFiberTraceSet.getFiberTrace(i).calcProfile()
        return inFiberTraceSet

    def run(self, inFiberTraceSet, inTraceNumbers=[-1]):
        """Calculate spatial profile and extract FiberTrace number inTraceNumber to 1D

        This method changes the input FiberTraceSet and returns void
        """
        
        spectrumSet = self.createFlatFiberTraceProfile(inFiberTraceSet, inTraceNumbers)
        
        return spectrumSet
 