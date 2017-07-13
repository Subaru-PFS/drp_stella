import os

import matplotlib.pyplot as plt
import numpy as np

from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, TaskRunner, ArgumentParser, CmdLineTask
from lsst.utils import getPackageDir
import pfs.drp.stella as drpStella
import pfs.drp.stella.extractSpectraTask as esTask
from pfs.drp.stella.reduceArcTask import ReduceArcTask, ReduceArcTaskRunner
from pfs.drp.stella.utils import makeFiberTraceSet, readLineListFile
from pfs.drp.stella.utils import readReferenceSpectrum, writePfsArm

class ReduceArcRefSpecConfig(Config):
    """Configuration for reducing arc images"""
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
    refSpec = Field( doc = "reference spectrum including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/arcSpectra/refSpec_CdHgKrNeXe_red.fits"));
    lineList = Field( doc = "reference line list including path", dtype = str, default=os.path.join(getPackageDir("obs_pfs"), "pfs/lineLists/CdHgKrNeXe_red.fits"));

class ReduceArcRefSpecTaskRunner(ReduceArcTaskRunner):
    """Get parsed values into the ReduceArcTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(parsedCmd.id.refList,
                 dict(butler=parsedCmd.butler, refSpec=parsedCmd.refSpec, lineList=parsedCmd.lineList))]

class ReduceArcRefSpecTask(ReduceArcTask):
    """Task to reduce Arc images"""
    ConfigClass = ReduceArcRefSpecConfig
    RunnerClass = ReduceArcRefSpecTaskRunner
    _DefaultName = "reduceArcRefSpecTask"

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="raw",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
        parser.add_argument("--refSpec", help='directory and name of reference spectrum')
        parser.add_argument("--lineList", help='directory and name of line list')
        return parser

    def run(self, expRefList, butler, refSpec=None, lineList=None, immediate=True):
        if refSpec == None:
            refSpec = self.config.refSpec
        if lineList == None:
            lineList = self.config.lineList
        self.log.debug('expRefList = %s' % expRefList)
        self.log.debug('len(expRefList) = %d' % len(expRefList))
        self.log.debug('refSpec = %s' % refSpec)
        self.log.debug('lineList = %s' % lineList)

        if len(expRefList) == 0:
            raise RuntimeError("Unable to find exposure reference")

        # read line list
        lineListArr = readLineListFile(lineList)

        # read reference Spectrum
        refSpecArr = readReferenceSpectrum(refSpec)
        self.log.debug('len(refSpecArr) = %d' % len(refSpecArr))

        for arcRef in expRefList:
            self.log.debug('arcRef.dataId = %s' % arcRef.dataId)
            self.log.debug('arcRef = %s' % arcRef)
            self.log.debug('type(arcRef) = %s' % type(arcRef))

            # construct fiberTraceSet from pfsFiberTrace
            try:
                fiberTrace = arcRef.get('fibertrace', immediate=True)
            except Exception, e:
                raise RuntimeError("Unable to load fibertrace for %s from %s: %s" %
                                   (arcRef.dataId, arcRef.get('fibertrace_filename')[0], e))
            flatFiberTraceSet = makeFiberTraceSet(fiberTrace)

            self.log.debug('flatFiberTraceSet.size() = %d' % flatFiberTraceSet.size())

            arcExp = arcRef.get("arc", immediate=True)
            self.log.debug('arcExp = %s' % arcExp)
            self.log.debug('type(arcExp) = %s' % type(arcExp))

            # optimally extract arc spectra
            self.log.info('extracting arc spectra')

            myExtractTask = esTask.ExtractSpectraTask()
            aperturesToExtract = [-1]
            spectrumSetFromProfile = myExtractTask.run(arcExp, flatFiberTraceSet, aperturesToExtract)

            refSpec = spectrumSetFromProfile.getSpectrum(int(spectrumSetFromProfile.size() / 2))
            ref = refSpec.getSpectrum()
            self.log.debug('ref.shape = %d' % ref.shape)

            dispCorControl = drpStella.DispCorControl()
            dispCorControl.fittingFunction = self.config.function
            dispCorControl.order = self.config.order
            dispCorControl.searchRadius = self.config.searchRadius
            dispCorControl.fwhm = self.config.fwhm
            dispCorControl.radiusXCor = self.config.radiusXCor
            dispCorControl.lengthPieces = self.config.lengthPieces
            dispCorControl.nCalcs = self.config.nCalcs
            dispCorControl.stretchMinLength = self.config.stretchMinLength
            dispCorControl.stretchMaxLength = self.config.stretchMaxLength
            dispCorControl.nStretches = self.config.nStretches
            self.log.debug('dispCorControl.fittingFunction = %s' % dispCorControl.fittingFunction)
            self.log.debug('dispCorControl.order = %d' % dispCorControl.order)
            self.log.debug('dispCorControl.searchRadius = %d' % dispCorControl.searchRadius)
            self.log.debug('dispCorControl.fwhm = %g' % dispCorControl.fwhm)
            self.log.debug('dispCorControl.radiusXCor = %d' % dispCorControl.radiusXCor)
            self.log.debug('dispCorControl.lengthPieces = %d' % dispCorControl.lengthPieces)
            self.log.debug('dispCorControl.nCalcs = %d' % dispCorControl.nCalcs)
            self.log.debug('dispCorControl.stretchMinLength = %d' % dispCorControl.stretchMinLength)
            self.log.debug('dispCorControl.stretchMaxLength = %d' % dispCorControl.stretchMaxLength)
            self.log.debug('dispCorControl.nStretches = %d' % dispCorControl.nStretches)

            for i in range(spectrumSetFromProfile.size()):
                spec = spectrumSetFromProfile.getSpectrum(i)
                specSpec = spectrumSetFromProfile.getSpectrum(i).getSpectrum()
                self.log.debug('calibrating spectrum %d: xCenter = %f' % (i,flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().xCenter))
                self.log.debug('specSpec.shape = %d' % specSpec.shape)
                self.log.debug('lineListArr.shape = [%d,%d]' % (lineListArr.shape[0], lineListArr.shape[1]))
                self.log.debug('type(specSpec) = %s: <%s>' % (type(specSpec),type(specSpec[0])))
                self.log.debug('type(refSpecArr) = %s: <%s>' % (type(refSpecArr),type(refSpecArr[0])))
                self.log.debug('type(lineListArr) = %s: <%s>' % (type(lineListArr),type(lineListArr[0][0])))
                result = drpStella.stretchAndCrossCorrelateSpec(specSpec, refSpecArr, lineListArr, dispCorControl)

                self.log.debug('type(result.lineList) = %s: <%s>: <%s>' % (type(result.lineList),type(result.lineList[0]),type(result.lineList[0][0])))
                self.log.debug('type(spectrumSetFromProfile.getSpectrum(i)) = %s: <%s>: <%s>' % (type(spectrumSetFromProfile.getSpectrum(i)),type(spectrumSetFromProfile.getSpectrum(i).getSpectrum()),type(spectrumSetFromProfile.getSpectrum(i).getSpectrum()[0])))
                for j in range(result.lineList.shape[0]):
                    self.log.debug('result.lineList[%d][*] = %f, %f' % (j,result.lineList[j][0],result.lineList[j][1]))

                spec.identifyF(drpStella.createLineListFromWLenPix(result.lineList),
                               dispCorControl)

                # set Spectrum in spectrumSetFromProfile because it is not
                # identitical anymore to the original object
                if spectrumSetFromProfile.setSpectrum(i, spec ):
                    self.log.debug('setSpectrum for spectrumSetFromProfile[%d] done' % i)
                else:
                    self.log.warn('setSpectrum for spectrumSetFromProfile[%d] failed' % i)
                for j in range(specSpec.shape[0]):
                    self.log.debug('spectrum %d: spec.getWavelength()[%d] = %f' % (i,j,spec.getWavelength()[j]))



            writePfsArm(butler, arcExp, spectrumSetFromProfile, arcRef.dataId)

        return spectrumSetFromProfile
