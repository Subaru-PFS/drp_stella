import os
import sys
import argparse

import pfs.drp.stella.findAndTraceAperturesTask as fataTask
import pfs.drp.stella.createFlatFiberTraceProfileTask as cfftpTask
import pfs.drp.stella.extractSpectraTask as esTask
import pfs.drp.stella as drpStella
import lsst.daf.persistence as dafPersist
import lsst.afw.image as afwImage
from lsst.pex.config import Config, ConfigField, ConfigurableField, Field, ListField
from lsst.pipe.base import Task, Struct, TaskRunner, ArgumentParser, CmdLineTask
import lsst.daf.persistence.butler as lsstButler
import numpy as np
from astropy.io import fits as pyfits

class ReduceArcConfig(Config):
    """Configuration for reducing arc images"""
    #directoryRoot = Field( dtype = str, default="", doc = "root directory for butler" )
    #flatVisit = Field( dtype = int, default = 0, doc = "visit number of flat exposure for tracing the fiber traces" )
    #arcVisit = Field( dtype = int, default = 0, doc = "visit number of arc exposure to extract and calibrate" )
    #filter = Field( dtype = str, default="None", doc = "key for filter name in exposure/calib registries")
    #spectrograph = Field( dtype = int, default = 1, doc = "spectrograph number (1-4)" )
    #site = Field( dtype = str, default = "S", doc = "site (J: JHU, L: LAM, X: Subaru offline, I: IPMU, A: ASIAA, S: Summit, P: Princeton, F: simulation (fake))" )
    #category = Field( dtype = str, default = "A", doc = "data category (A: science, B: UTR, C: Meterology, D: AG (auto guider))")
    #refSpec = Field( dtype = str, default = " ", doc = "name of reference spectrum fits file")
    #lineList = Field( dtype = str, default = " ", doc = "name of lineList fits file")
    function = Field( doc = "Function for fitting the dispersion", dtype=str, default="POLYNOMIAL" );
    order = Field( doc = "Fitting function order", dtype=int, default = 5 );
    searchRadius = Field( doc = "Radius in pixels relative to line list to search for emission line peak", dtype = int, default = 2 );
    fwhm = Field( doc = "FWHM of emission lines", dtype=float, default = 2.6 );
    radiusXCor = Field( doc = "Radius in pixels in which to cross correlate a spectrum relative to the reference spectrum", dtype = int, default = 50 );
    lengthPieces = Field( doc = "Length of pieces of spectrum to match to reference spectrum by stretching and shifting", dtype = int, default = 500 );
    nCalcs = Field( doc = "Number of iterations > spectrumLength / lengthPieces, e.g. spectrum length is 3800 pixels, <lengthPieces> = 500, <nCalcs> = 15: run 1: pixels 0-499, run 2: 249-749,...", dtype = int, default = 15 );
    stretchMinLength = Field( doc = "Minimum length to stretched pieces to (< lengthPieces)", dtype = int, default = 460 );
    stretchMaxLength = Field( doc = "Maximum length to stretched pieces to (> lengthPieces)", dtype = int, default = 540 );
    nStretches = Field( doc = "Number of stretches between <stretchMinLength> and <stretchMaxLength>", dtype = int, default = 80 );

class ReduceArcIdAction(argparse.Action):
    """Split name=value pairs and put the result in a dict"""
    def __call__(self, parser, namespace, values, option_string):
        output = getattr(namespace, self.dest, {})
        for nameValue in values:
            name, sep, valueStr = nameValue.partition("=")
            if not valueStr:
                parser.error("%s value %s must be in form name=value" % (option_string, nameValue))
            output[name] = valueStr
        setattr(namespace, self.dest, output)

class ReduceArcArgumentParser(ArgumentParser):
    """Add a --flatId argument to the argument parser"""
    def __init__(self, *args, **kwargs):
        super(ReduceArcArgumentParser, self).__init__(*args, **kwargs)
        #self.calibName = calibName
        self.add_id_argument("--id", datasetType="postISRCCD",
                             help="input identifiers, e.g., --id visit=123 ccd=4")
#        self.add_id_argument("--flatId", nargs="*", action=ReduceArcIdAction, default={},
        self.add_argument("--flatId", nargs="*", action=ReduceArcIdAction, default={},
                          help="identifiers for detrend, e.g., --detrendId version=1",
                          metavar="KEY=VALUE1[^VALUE2[^VALUE3...]")#, datasetType='postISRCCD',
                          #help="identifiers for flat, e.g., --flatId visit=123 spectrograph=2 filter='PFS-R'")
        self.add_argument("--refSpec", help='directory and name of reference spectrum')
        self.add_argument("--lineList", help='directory and name of line list')
#    def parse_args(self, *args, **kwargs):
#        namespace = super(ReduceArcArgumentParser, self).parse_args(*args, **kwargs)
#        keys = namespace.butler.getKeys('postISRCCD')
#        parsed = {}
#        for name, value in namespace.flatId.items():
#            if not name in keys:
#                self.error("%s is not a relevant flat identifier key (%s)" % (name, keys))
#            parsed[name] = keys[name](value)
#        namespace.flatId = parsed

#        return namespace

class ReduceArcTaskRunner(TaskRunner):
    """Get parsed values into the ReduceArcTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [dict(arcId=parsedCmd.id, butler=parsedCmd.butler, flatId=parsedCmd.flatId, refSpec=parsedCmd.refSpec, lineList=parsedCmd.lineList)]
#        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, flatId=parsedCmd.flatId, refSpec=parsedCmd.refSpec, lineList=parsedCmd.lineList)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
#        if self.doRaise:
        result = task.run(**args)
#        else:
#            try:
#                result = task.run(**args)
#            except Exception, e:
#                task.log.fatal("Failed: %s" % e)
#                traceback.print_exc(file=sys.stderr)

#        if self.doReturnResults:
#            return Struct(
#                args = args,
#                metadata = task.metadata,
#                result = result,
#            )

class ReduceArcTask(CmdLineTask):
    """Task to reduce Arc images"""
    ConfigClass = ReduceArcConfig
    RunnerClass = ReduceArcTaskRunner
    _DefaultName = "reduceArcTask"

    def __init__(self, *args, **kwargs):
        super(ReduceArcTask, self).__init__(self, *args, **kwargs)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        #doBatch = kwargs.pop("doBatch", False)
        return ReduceArcArgumentParser(name=cls._DefaultName, *args, **kwargs)

    def run(self, arcId, butler, flatId, refSpec, lineList):
        print 'arcId = ',arcId
        print 'flatId = ',flatId
        print 'butler = ',butler
        print 'refSpec = ',refSpec
        print 'lineList = ',lineList
        #outputId=<{'category': 'A', 'site': 'S', 'filter': 'PFS-M', 'calibDate': '2015-12-21', 'ccd': 5, 'calibVersion': 'dark'}>
        flat = getDataRef(cache.butler, flatId)

        """ find and trace fiber traces """
        print 'tracing flat fiber traces'
        myFindTask = fataTask.FindAndTraceAperturesTask()
        fts = myFindTask.run(flat)
        print fts.size(),' traces found'

        """ calculate spatial profiles """
        print 'calculating spatial profiles'
        myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
        myProfileTask.run(flatFiberTraceSet)
        
        """ optimally extract arc spectra """
        print 'extracting arc spectra'
        arc = getDataRef(cache.butler, arcId)

        myExtractTask = esTask.ExtractSpectraTask()
        aperturesToExtract = [-1]
        spectrumSetFromProfile = myExtractTask.run(arc, fts, aperturesToExtract)

        """ read line list """
        hdulist = pyfits.open(lineList)
        tbdata = hdulist[1].data
        lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float64')
        lineListArr[:,0] = tbdata.field(0)
        lineListArr[:,1] = tbdata.field(1)

        """ read reference Spectrum """
        hdulist = pyfits.open(refSpec)
        tbdata = hdulist[1].data
        refSpecArr = np.ndarray(shape=(len(tbdata)), dtype='float64')
        refSpecArr[:] = tbdata.field(0)

        dispCorControl = drpStella.DispCorControl()
        dispCorControl.function = self.config.function
        dispCorControl.order = self.config.order
        dispCorControl.searchRadius = self.config.searchRadius
        dispCorControl.fwhm = self.config.fwhm
        dispCorControl.radiusXCor = self.config.radiusXCor
        dispCorControl.lengthPieces = self.config.lengthPieces
        dispCorControl.nCalcs = self.config.nCalcs
        dispCorControl.stretchMinLength = self.config.stretchMinLength
        dispCorControl.stretchMaxLength = self.config.stretchMaxLength
        dispCorControl.nStretches = self.config.nStretches

        for i in range(spectrumSetFromProfile.size()):
            spec = spectrumSetFromProfile.getSpectrum(i)
            specSpec = spec.getSpectrum()
            result = drpStella.stretchAndCrossCorrelateSpecFD(specSpec, refSpecArr, lineListArr, dispCorControl)
            print result.lineList
            spec.identify(result.lineList, dispCorControl)
            print spec.getDispCoeffs()
            print spec.getDispRms()
            print spec.getWavelength()

def getDataRef(butler, dataId, datasetType="postISRCCD"):
    """Construct a dataRef from a butler and data identifier"""
    dataRefList = [ref for ref in butler.subset(datasetType, **dataId)]
    self.log.info('getDataRef: dataId = %s' % dataId)
    camera = dataRefList[0].get("camera")
    self.log.info('getDataRef: dataRefList[0].get("camera") = %s' % camera)
    self.log.info('getDataRef: dataRefList[0] = %s' % dataRefList[0])
    dataRef = dataRefList[0]
    assert len(dataRefList) == 1
    return dataRefList[0]
