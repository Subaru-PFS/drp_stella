from lsst.obs.subaru.isr import SubaruIsrTask
from lsst.pex.config import Config, ConfigurableField
from lsst.pipe.base import CmdLineTask

class DemoConfig(Config):
    isr = ConfigurableField(target=SubaruIsrTask, doc="Instrumental signature removal")

class DemoTask(CmdLineTask):
    _DefaultName = "demo"
    ConfigClass = DemoConfig

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

if __name__ == "__main__":
    DemoTask.parseAndRun()
