import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import pfs.drp.stella as drpStella

class AddDummyHeaderKeywordsCmdLineConfig(pexConfig.Config):
    """!Configuration for AddDummyHeaderKeywordsCmdLineTask     """
    
class AddDummyHeaderKeywordsCmdLineTask(pipeBase.CmdLineTask):
    ConfigClass = AddDummyHeaderKeywordsCmdLineConfig
    _DefaultName = "addDummyHeaderKeywordsTask"
    
    def __init__(self, *args, **kwargs):
        pipeBase.CmdLineTask.__init__(self, *args, **kwargs)
        
    def run(self, fileList):
        addDummyHeaderKeywordsTask = drpStella.AddDummyHeaderKeywordsTask()
        return addDummyHeaderKeywordsTask.run(filelist)

    def _getConfigName(self):
        return None
    
    def _getMetadataName(self):
        return None
        
        