#!/usr/bin/env python
import lsst.afw.geom                    as afwGeom
import lsst.afw.image                   as afwImage
import lsst.pex.config                  as pexConfig
import pfs.drp.stella as drpStella
from lsst.pipe.base import Task

class ExtractFromProfileConfig(pexConfig.Config):
        maxIterSig = pexConfig.Field(
            doc = "Maximum number of iterations for masking bad pixels and CCD defects (default: 2)",
            dtype = int,
            default = 2,
            check = lambda x : x > 0)

class ExtractFromProfileTask(Task):
    ConfigClass = ExtractFromProfileConfig
    _DefaultName = "extractFromProfileTask"

    def __init__(self, *args, **kwargs):
        super(ExtractFromProfileTask, self).__init__(*args, **kwargs)

    def extractFromProfile(self, inFlatFiberTraceSet, inExposure, inTraceNumbers):
        # --- create FiberTraceFunctionFindingControl
        fiberTraceExtractionControl = drpStella.FiberTraceExtractionControl()
        fiberTraceExtractionControl.maxIterSig = self.config.maxIterSig

        """Create a FiberTraceSet given a flat-field fits file name"""
        specFiberTraceSet = inFlatFiberTraceSet
        if inTraceNumbers[0] == -1:
            for i in specFiberTraceSet:
                i.createTrace(inExposure.getMaskedImage())
                i.extractFromProfile()
        else:
            for i in inTraceNumbers:
                specFiberTraceSet.getFiberTrace(i).createTrace(inExposure.getMaskedImage())
                specFiberTraceSet.getFiberTrace(i).extractFromProfile()

        return specFiberTraceSet

    def run(self, inFlatFiberTraceSet, inExposure, inTraceNumbers=[-1]):
        """Calculate spatial profile and extract FiberTrace number inTraceNumber to 1D

        This method changes the input FiberTraceSet and returns void
        """

        self.extractFromProfile(inFlatFiberTraceSet, inExposure, inTraceNumbers)

        return
