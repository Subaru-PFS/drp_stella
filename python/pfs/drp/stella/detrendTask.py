#python /Users/azuri/stella-git/drp_stella/python/pfs/drp/stella/demo.py '/Volumes/My Passport/data/spectra/pfs/PFS' --calib='/Volumes/My Passport/data/spectra/pfs/PFS/CALIB' --id site='S' category='A' filter='PFS-M' spectrograph=2 dateObs='2015-12-22' -C myConfig.py

from lsst.pex.config import Config, ConfigurableField
from lsst.pipe.base import CmdLineTask
from lsst.obs.subaru.isr import SubaruIsrTask

class DetrendConfig(Config):
    isr = ConfigurableField(target=SubaruIsrTask, doc="Instrumental signature removal")

class DetrendTask(CmdLineTask):
    _DefaultName = "detrend"
    ConfigClass = DetrendConfig

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")

    def run(self, dataRef):
        #print 'type(exp) = ',type(exp)
        exp = self.isr.runDataRef(dataRef).exposure  # Should do ISR and CCD assembly
        #exp = self.isr.runDataRef(dataRef)  # Should do ISR and CCD assembly
        dataRef.put(exp, "calexp")

    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None

#if __name__ == "__main__":
#    DetrendTask.parseAndRun()

