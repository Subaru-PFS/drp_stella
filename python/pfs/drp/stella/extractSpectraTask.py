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

class ExtractSpectraConfig(pexConfig.Config):
      saturationLevel = pexConfig.Field(
          doc = "CCD saturation level",
          dtype = float,
          default = 65000.,
          check = lambda x : x > 0.)

class ExtractSpectraTask(Task):
    ConfigClass = ExtractSpectraConfig
    _DefaultName = "extractSpectra"

    def __init__(self, *args, **kwargs):
        super(ExtractSpectraTask, self).__init__(*args, **kwargs)

#        """create FiberTraceFunctionFindingControl"""
#        self.ftffc = drpStella.FiberTraceFunctionFindingControl()
#        ftffc = self.ftffc
#        ftffc.saturationLevel = self.config.saturationLevel

    def extractSpectra(self, inExposure, inFiberTraceSetWithProfiles, inTraceNumbers):
    
#        try:
        if inTraceNumbers[0] == -1:
            traceNumbers = range(inFiberTraceSetWithProfiles.size())
        else:
            traceNumbers = inTraceNumbers
        print "inTraceNumbers = ", inTraceNumbers
        print "traceNumbers = ", traceNumbers

        spectrumSet = drpStella.SpectrumSetF()
        print spectrumSet
        print spectrumSet.size()

        """Create FiberTraces for inExposure and store them in inFiberTraceSetWithProfile"""
        if inExposure != None:
            inMaskedImage = inExposure.getMaskedImage()
            print("inMaskedImage created")

        """Create traces and extract spectrum"""
        for i in traceNumbers:
            if i < 0:
                raise Exception("i < 0")
            elif i >= inFiberTraceSetWithProfiles.size():
                raise Exception("i >= inFiberTraceSetWithProfiles.size()")

            trace = inFiberTraceSetWithProfiles.getFiberTrace(i)
            if trace.isProfileSet() == False:
                raise Exception("profile not set")

            """Create trace from inMaskedImage"""
            if inExposure != None:
                trace.setITrace(i)
                trace.createTrace(inMaskedImage)

            """Extract spectrum from profile"""
            spectrum = trace.extractFromProfile()
            print spectrum
            spectrumSet.addSpectrum(spectrum)
            print "extractSpectraTask::extractSpectra: spectrum ",i,"added to spectrumSet"

        print "extractSpectraTask::extractSpectra: returning spectrumSet of size ",spectrumSet.size()
        return spectrumSet
#        except Exception as e:
#            print e

    def run(self, inExposure, inFiberTraceSetWithProfiles, inTraceNumbers=[-1]):
        """Create traces from inExposure and extract spectra from profiles in inFiberTraceSetWithProfiles

        This method is the top-level for running the automatic 1D extraction of the fiber traces on the Exposure
        of the object spectra as a stand-alone BatchPoolTask.
        
        This method returns a SpectrumSet
        """
        
        spectrumSet = self.extractSpectra(inExposure, inFiberTraceSetWithProfiles, inTraceNumbers)
        print "self.extractSpectra finished, returning spectrumSet of size ",spectrumSet.size()
        return spectrumSet
