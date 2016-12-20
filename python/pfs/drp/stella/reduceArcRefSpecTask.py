#/Users/azuri/stella-git/drp_stella/bin.src/reduceArc.py '/Volumes/My Passport/data/spectra/pfs/PFS' --id visit=4 filter='PFS-R' spectrograph=2 site='F' category='A' --refSpec '/Users/azuri/stella-git/obs_subaru/pfs/lineLists/refCdHgKrNeXe_red.fits' --lineList '/Users/azuri/stella-git/obs_subaru/pfs/lineLists/CdHgKrNeXe_red.fits' --loglevel 'info'
from astropy.io import fits as pyfits
from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, TaskRunner, ArgumentParser, CmdLineTask
from lsst.utils import getPackageDir
import matplotlib.pyplot as plt
import numpy as np
import os
from pfs.drp.stella.datamodelIO import spectrumSetToPfsArm, PfsArmIO
import pfs.drp.stella.extractSpectraTask as esTask
import pfs.drp.stella as drpStella
from pfs.drp.stella.utils import makeFiberTraceSet
import sys
import traceback

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
    percentageOfLinesForCheck = Field( doc = "Hold back this percentage of lines in the line list for check", dtype=int, default=10)

class ReduceArcRefSpecTaskRunner(TaskRunner):
    """Get parsed values into the ReduceArcTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, refSpec=parsedCmd.refSpec, lineList=parsedCmd.lineList)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            self.log.debug('ReduceArcTask.__call__: args = %s' % args)
            result = task.run(**args)
        else:
            try:
                result = task.run(**args)
            except Exception, e:
                task.log.fatal("Failed: %s" % e)
                traceback.print_exc(file=sys.stderr)

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
        self.log.debug('expRefList = %s' % expRefList)
        self.log.debug('len(expRefList) = %d' % len(expRefList))
        self.log.debug('refSpec = %s' % refSpec)
        self.log.debug('lineList = %s' % lineList)

        for arcRef in expRefList:
            self.log.debug('arcRef.dataId = %s' % arcRef.dataId)
            self.log.debug('arcRef = %s' % arcRef)
            self.log.debug('type(arcRef) = %s' % type(arcRef))

            try:
                fiberTrace = arcRef.get('fiberTrace', immediate=True)
            except Exception, e:
                raise RuntimeError("Unable to retrieve fiberTrace for %s: %s" % (arcRef.dataId, e))

            arcExp = arcRef.get("postISRCCD", immediate=True)
            self.log.debug('arcExp = %s' % arcExp)
            self.log.debug('type(arcExp) = %s' % type(arcExp))

            """ construct fiberTraceSet from pfsFiberTrace """
            flatFiberTraceSet = makeFiberTraceSet(fiberTrace)

            """ optimally extract arc spectra """
            self.log.debug('extracting arc spectra')

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
            self.log.debug('refSpecArr.shape = %d' % refSpecArr.shape)

            refSpec = spectrumSetFromProfile.getSpectrum(int(spectrumSetFromProfile.size() / 2))
            ref = refSpec.getSpectrum()
            self.log.debug('ref.shape = %d' % ref.shape)

            if False:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(ref,'-+')
                ax.plot(refSpecArr,'-+')
                plt.show()
                plt.close(fig)
                fig.clf()

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
                self.log.debug('calibrating spectrum %d: xCenter = %f' % (i, flatFiberTraceSet.getFiberTrace(i).getFiberTraceFunction().xCenter))
                self.log.debug('specSpec.shape = %d' % specSpec.shape)
                self.log.debug('lineListArr.shape = [%d,%d]' % (lineListArr.shape[0], lineListArr.shape[1]))
                self.log.debug('type(specSpec) = %s: <%s>' % (type(specSpec),type(specSpec[0])))
                self.log.debug('type(refSpecArr) = %s: <%s>' % (type(refSpecArr),type(refSpecArr[0])))
                self.log.debug('type(lineListArr) = %s: <%s>' % (type(lineListArr),type(lineListArr[0][0])))
                result = drpStella.stretchAndCrossCorrelateSpecFF(specSpec, refSpecArr, lineListArr, dispCorControl)
                self.log.debug("result.lineList = %s" % np.array_str(result.lineList))
                self.log.debug('type(result.lineList) = %s: <%s>: <%s>' % (type(result.lineList),type(result.lineList[0]),type(result.lineList[0][0])))
                self.log.debug('type(spectrumSetFromProfile.getSpectrum(i)) = %s: <%s>: <%s>' % (type(spectrumSetFromProfile.getSpectrum(i)),type(spectrumSetFromProfile.getSpectrum(i).getSpectrum()),type(spectrumSetFromProfile.getSpectrum(i).getSpectrum()[0])))
                spec.identifyF(result.lineList, dispCorControl, int(lineListArr.shape[0] * self.config.percentageOfLinesForCheck / 100.))
                for j in range(specSpec.shape[0]):
                    self.log.debug('spectrum %d: spec.getWavelength()[%d] = %f' % (i,j,spec.getWavelength()[j]))
                self.log.debug("FiberTrace %d: spec.getDispCoeffs() = %s" % (i,np.array_str(spec.getDispCoeffs())))
                self.log.info("FiberTrace %d: spec.getDispRms() = %f" % (i, spec.getDispRms()))
                self.log.info("FiberTrace %d: spec.getDispRmsCheck() = %f" % (i, spec.getDispRmsCheck()))

            #
            # Do the I/O using a trampoline object PfsArmIO (to avoid adding butler-related details
            # to the datamodel product)
            #
            # This is a bit messy as we need to include the pfsConfig file in the pfsArm file
            #
            dataId = arcRef.dataId

            md = arcExp.getMetadata().toDict()
            key = "PFSCONFIGID"
            if key in md:
                pfsConfigId = md[key]
            else:
                self.log.info('No pfsConfigId is present in postISRCCD file for dataId %s' %
                              str(dataId.items()))
                pfsConfigId = 0x0

            pfsConfig = butler.get("pfsConfig", pfsConfigId=pfsConfigId, dateObs=dataId["dateObs"])

            pfsArm = spectrumSetToPfsArm(pfsConfig, spectrumSetFromProfile,
                                         dataId["visit"], dataId["spectrograph"], dataId["arm"])
            butler.put(PfsArmIO(pfsArm), 'pfsArm', dataId)

        return spectrumSetFromProfile
