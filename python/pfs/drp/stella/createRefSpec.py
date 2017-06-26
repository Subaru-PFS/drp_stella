import os

from astropy.io import fits as pyfits
import numpy as np

from lsst.pex.config import Config, Field
from lsst.pipe.base import Struct, TaskRunner, ArgumentParser, CmdLineTask
import lsst.utils
import pfs.drp.stella as drpStella
import pfs.drp.stella.createFlatFiberTraceProfileTask as cfftpTask
import pfs.drp.stella.extractSpectraTask as esTask
import pfs.drp.stella.findAndTraceAperturesTask as fataTask

class CreateRefSpecConfig(Config):
    """Configuration for creating the reference spectrum"""
    lineList = Field( doc = "Reference line list including path", dtype = str, default=os.path.join(lsst.utils.getPackageDir('obs_pfs'),"pfs/lineLists/CdHgKrNeXe_red.fits"))
    output = Field( doc = "Name of output file", dtype=str, default=os.path.join(lsst.utils.getPackageDir('obs_pfs'),"pfs/arcSpectra/refCdHgKrNeXe_red.fits"))

class CreateRefSpecTaskRunner(TaskRunner):
    """Get parsed values into the CreateRefSpecTask.run"""
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, lineList=parsedCmd.lineList)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        if self.doRaise:
            self.log.info('CreateRefSpecTask.__call__: args = %s' % args)
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

class CreateRefSpecTask(CmdLineTask):
    """Task to reduce Arc images"""
    ConfigClass = CreateRefSpecConfig
    RunnerClass = CreateRefSpecTaskRunner
    _DefaultName = "createRefSpecTask"

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self,*args, **kwargs)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", datasetType="postISRCCD",
                               help="input identifiers, e.g., --id visit=123 ccd=4")
        parser.add_argument("--lineList", help='directory and name of line list')
        return parser

    def run(self, expRefList, butler, lineList=None, immediate=True):
        print 'expRefList = ',expRefList
        print 'expRefList[0] = ',expRefList[0]
        arcRef = expRefList[0]
        try:
            arcExp = arcRef.get("postISRCCD", immediate=immediate)
        except Exception, e:
            raise RuntimeError("Unable to retrieve %s: %s" % (arcRef.dataId, e))

	try:
            flatExposure = arcRef.get('flat', immediate=immediate)
        except Exception, e:
            raise RuntimeError("Unable to retrieve flat for %s: %s" % (arcRef.dataId, e))

        # find and trace apertures
        print 'tracing apertures of flatfield'
        myFindTask = fataTask.FindAndTraceAperturesTask()
        flatFiberTraceSet = myFindTask.run(flatExposure)
        print flatFiberTraceSet.size(),' traces found'

        # calculate spatial profiles
        print 'calculating spatial profiles'
        myProfileTask = cfftpTask.CreateFlatFiberTraceProfileTask()
        myProfileTask.run(flatFiberTraceSet)

        # optimally extract arc spectra
        print 'extracting arc spectra'

        # Extract all apertures
        myExtractTask = esTask.ExtractSpectraTask()
        aperturesToExtract = [-1]
        spectrumSetFromProfile = myExtractTask.run(arcExp, flatFiberTraceSet, aperturesToExtract)

        refSpec = spectrumSetFromProfile.getSpectrum(int(spectrumSetFromProfile.size() / 2))
        ref = refSpec.getSpectrum()

        hdulist = pyfits.open(self.config.lineList)
        tbdata = hdulist[1].data

        print len(tbdata)
        lineList = np.ndarray(shape=(len(tbdata),2), dtype='float64')

        lineList[:,0] = tbdata.field(0)
        lineList[:,1] = tbdata.field(1)

        dispCorControl = drpStella.DispCorControl()
        refSpec.identifyD(lineList, dispCorControl)
        print refSpec.getDispCoeffs()
        print refSpec.getDispRms()
        print refSpec.getWavelength()
        print 'ref = ',ref.shape,': ',ref
        col1 = pyfits.Column(name='flux', format='E', array=ref)
        cols = pyfits.ColDefs([col1])
        tbhdu = pyfits.BinTableHDU.from_columns(cols)
        tbhdu.writeto(self.config.output)
