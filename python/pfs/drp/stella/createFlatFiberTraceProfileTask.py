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
        maxIterSig = pexConfig.Field(
            doc = "Maximum number of iterations for masking bad pixels and CCD defects (default: 2)",
            dtype = int,
            default = 2,
            check = lambda x : x > 0)
        lowerSigma = pexConfig.Field(
            dtype = float,
            doc = "lower sigma rejection threshold if maxIterSig > 0 (default: 3.)",
            default = 3.,
            check = lambda x : x >= 0 )
        upperSigma = pexConfig.Field(
            dtype = float,
            doc = "upper sigma rejection threshold if maxIterSig > 0 (default: 3.)",
            default = 3.,
            check = lambda x : x >= 0 )

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

    def createFlatFiberTraceProfile(self, inFiberTraceSet, inTraceNumbers):
        # --- create FiberTraceProfileFittingControl
        self.fiberTraceProfileFittingControl.swathWidth = self.config.swathWidth
        self.fiberTraceProfileFittingControl.telluric = self.config.telluric
        self.fiberTraceProfileFittingControl.overSample = self.config.overSample
        self.fiberTraceProfileFittingControl.maxIterSig = self.config.maxIterSig
        self.fiberTraceProfileFittingControl.lowerSigma = self.config.lowerSigma
        self.fiberTraceProfileFittingControl.upperSigma = self.config.upperSigma

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
