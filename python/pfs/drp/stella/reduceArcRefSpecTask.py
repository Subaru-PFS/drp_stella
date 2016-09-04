#/Users/azuri/stella-git/drp_stella/bin.src/reduceArc.py '/Volumes/My Passport/data/spectra/pfs/PFS' --id visit=4 filter='PFS-R' spectrograph=2 site='F' category='A' --refSpec '/Users/azuri/stella-git/obs_subaru/pfs/lineLists/refCdHgKrNeXe_red.fits' --lineList '/Users/azuri/stella-git/obs_subaru/pfs/lineLists/CdHgKrNeXe_red.fits' --loglevel 'info'
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
import matplotlib.pyplot as plt

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
    refSpec = Field( doc = "reference reference spectrum including path", dtype = str, default="/Users/azuri/stella-git/obs_subaru/pfs/lineLists/refCdHgKrNeXe_red.fits");
    lineList = Field( doc = "reference line list including path", dtype = str, default="/Users/azuri/stella-git/obs_subaru/pfs/lineLists/CdHgKrNeXe_red.fits");

class ReduceArcRefSpecTaskRunner(TaskRunner):
    """Get parsed values into the ReduceArcTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        print 'ReduceArcTask.getTargetList: kwargs = ',kwargs
        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, refSpec=parsedCmd.refSpec, lineList=parsedCmd.lineList)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            self.log.info('ReduceArcTask.__call__: args = %s' % args)
            result = task.run(**args)
        else:
            try:
                result = task.run(**args)
            except Exception, e:
                task.log.fatal("Failed: %s" % e)

        if self.doReturnResults:
            return Struct(
                args = args,
                metadata = task.metadata,
                result = result,
            )

class ReduceArcRefSpecTask(CmdLineTask):
    """Task to reduce Arc images"""
    ConfigClass = ReduceArcRefSpecConfig
    RunnerClass = ReduceArcRefSpecTaskRunner
    _DefaultName = "reduceArcRefSpecTask"

    def __init__(self, *args, **kwargs):
        super(ReduceArcRefSpecTask, self).__init__(*args, **kwargs)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="postISRCCD",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
        parser.add_argument("--refSpec", help='directory and name of reference spectrum')
        parser.add_argument("--lineList", help='directory and name of line list')
        return parser# ReduceArcArgumentParser(name=cls._DefaultName, *args, **kwargs)

    def run(self, expRefList, butler, refSpec=None, lineList=None, immediate=True):
        if refSpec == None:
            refSpec = self.config.refSpec
        if lineList == None:
            lineList = self.config.lineList
        self.log.info('expRefList = %s' % expRefList)
        self.log.info('len(expRefList) = %d' % len(expRefList))
        self.log.info('refSpec = %s' % refSpec)
        self.log.info('lineList = %s' % lineList)

        arcRef = expRefList[0]
        self.log.info('arcRef.dataId = %s' % arcRef.dataId)
        self.log.info('arcRef = %s' % arcRef)
        self.log.info('type(arcRef) = %s' % type(arcRef))

	try:
            flatExposure = arcRef.get('flat', immediate=immediate)
        except Exception, e:
            raise RuntimeError("Unable to retrieve flat for %s: %s" % (arcRef.dataId, e))
        
        arcExp = arcRef.get("postISRCCD", immediate=True)
        self.log.info('arcExp = %s' % arcExp)
        self.log.info('type(arcExp) = %s' % type(arcExp))

        """ find and trace fiber traces """
        print 'tracing flat fiber traces'
        myFindTask = fataTask.FindAndTraceAperturesTask()
        flatFiberTraceSet = myFindTask.run(flatExposure)
        print flatFiberTraceSet.size(),' traces found'

        """ calculate spatial profiles """
        print 'calculating spatial profiles'
        myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
        myProfileTask.run(flatFiberTraceSet)
        
        """ optimally extract arc spectra """
        print 'extracting arc spectra'

        myExtractTask = esTask.ExtractSpectraTask()
        aperturesToExtract = [-1]
        spectrumSetFromProfile = myExtractTask.run(arcExp, flatFiberTraceSet, aperturesToExtract)

        """ read line list """
        hdulist = pyfits.open(lineList)
        tbdata = hdulist[1].data
        lineListArr = np.ndarray(shape=(len(tbdata),2), dtype='float32')
        lineListArr[:,0] = tbdata.field(0)
        lineListArr[:,1] = tbdata.field(1)

        """ read reference Spectrum """
        hdulist = pyfits.open(refSpec)
        tbdata = hdulist[1].data
        refSpecArr = np.ndarray(shape=(len(tbdata)), dtype='float32')
        refSpecArr[:] = tbdata.field(0)
        self.log.info('refSpecArr.shape = %d' % refSpecArr.shape)
        
        refSpec = spectrumSetFromProfile.getSpectrum(int(spectrumSetFromProfile.size() / 2))
        ref = refSpec.getSpectrum()
        self.log.info('ref.shape = %d' % ref.shape)
        
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(ref,'-+')
            ax.plot(refSpecArr,'-+')
            plt.show()
            plt.close(fig)
            fig.clf()
        
        if ref.shape != refSpecArr.shape:
            raise("ref.shape != refSpecArr.shape")

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
        self.log.info('dispCorControl.fittingFunction = %s' % dispCorControl.fittingFunction)
        self.log.info('dispCorControl.order = %d' % dispCorControl.order)
        self.log.info('dispCorControl.searchRadius = %d' % dispCorControl.searchRadius)
        self.log.info('dispCorControl.fwhm = %g' % dispCorControl.fwhm)
        self.log.info('dispCorControl.radiusXCor = %d' % dispCorControl.radiusXCor)
        self.log.info('dispCorControl.lengthPieces = %d' % dispCorControl.lengthPieces)
        self.log.info('dispCorControl.nCalcs = %d' % dispCorControl.nCalcs)
        self.log.info('dispCorControl.stretchMinLength = %d' % dispCorControl.stretchMinLength)
        self.log.info('dispCorControl.stretchMaxLength = %d' % dispCorControl.stretchMaxLength)
        self.log.info('dispCorControl.nStretches = %d' % dispCorControl.nStretches)

        for i in range(spectrumSetFromProfile.size()):
            spec = spectrumSetFromProfile.getSpectrum(i)
            specSpec = spectrumSetFromProfile.getSpectrum(i).getSpectrum()
            print 'calibrating spectrum ',i,': xCenter = ',flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().xCenter
            self.log.info('specSpec.shape = %d' % specSpec.shape)
            self.log.info('lineListArr.shape = [%d,%d]' % (lineListArr.shape[0], lineListArr.shape[1]))
            self.log.info('type(specSpec) = %s: <%s>' % (type(specSpec),type(specSpec[0])))
            self.log.info('type(refSpecArr) = %s: <%s>' % (type(refSpecArr),type(refSpecArr[0])))
            self.log.info('type(lineListArr) = %s: <%s>' % (type(lineListArr),type(lineListArr[0][0])))
            result = drpStella.stretchAndCrossCorrelateSpecFF(specSpec, refSpecArr, lineListArr, dispCorControl)
            #self.log.info("result.lineList = %g" % result.lineList)
            self.log.info('type(result.lineList) = %s: <%s>: <%s>' % (type(result.lineList),type(result.lineList[0]),type(result.lineList[0][0])))
            self.log.info('type(spectrumSetFromProfile.getSpectrum(i)) = %s: <%s>: <%s>' % (type(spectrumSetFromProfile.getSpectrum(i)),type(spectrumSetFromProfile.getSpectrum(i).getSpectrum()),type(spectrumSetFromProfile.getSpectrum(i).getSpectrum()[0])))
            for j in range(result.lineList.shape[0]):
                print 'result.lineList[',j,'][*] = ',result.lineList[j][0],' ',result.lineList[j][1]
            spec.identifyF(result.lineList, dispCorControl, 8)
            if spectrumSetFromProfile.setSpectrum(i, spec ):
                print 'setSpectrum for spectrumSetFromProfile[',i,'] done'
            else:
                print 'setSpectrum for spectrumSetFromProfile[',i,'] failed'
            for j in range(specSpec.shape[0]):
                self.log.info('spectrum %d: spec.getWavelength()[%d] = %f' % (i,j,spec.getWavelength()[j]))

        if False:
            xPixMinMax = np.ndarray(2, dtype='float32')
            xPixMinMax[0] = 1000.
            xPixMinMax[1] = 1600.
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for i in range(spectrumSetFromProfile.size()):
                ax.plot(spectrumSetFromProfile.getSpectrum(i).getSpectrum(),'-+')
                plt.xlim(xPixMinMax[0],xPixMinMax[1])
                plt.ylim(0,25000)
            plt.xlabel('Pixel')
            plt.ylabel('Flux [ADUs]')
            plt.show()
            plt.close(fig)
            fig.clf()
            
        if False:
            xMinMax = drpStella.poly(xPixMinMax, spec.getDispCoeffs())
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for i in range(spectrumSetFromProfile.size()):
                ax.plot(spectrumSetFromProfile.getSpectrum(i).getWavelength(),spectrumSetFromProfile.getSpectrum(i).getSpectrum(),'-+')
                plt.xlim(xMinMax[0],xMinMax[1])
                plt.ylim(0,25000)
            plt.xlabel('Wavelength [Angstroems]')
            plt.ylabel('Flux [ADUs]')
            plt.show()
            plt.close(fig)
            fig.clf()
        
        print 'writing SpectrumSet object'
        print 'arcRef.dataId = ',arcRef.dataId
        butler.put(spectrumSetFromProfile, 'spArm', arcRef.dataId )
