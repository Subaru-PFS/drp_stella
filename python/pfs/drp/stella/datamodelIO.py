import os
import re
import numpy as np
from pfs.datamodel.pfsArm import PfsArm
from pfs.datamodel.pfsConfig import PfsConfig

class PfsConfigIO(object):
    """A class to perform butler-based I/O for pfsConfig"""

    def __init__(self, pfsConfig):
        self._pfsConfig = pfsConfig

    @staticmethod
    def readFits(pathName, hdu=None, flags=None):
        dirName, fileName = os.path.split(pathName)

        match = re.search(r"pfsConfig-(0x[0-9a-f]+)\.fits$", fileName)
        if not match:
            raise RuntimeError("Unable to extract pfsConfigId from \"%s\"" % pathName)
        pfsConfigId = int(match.group(1), 16)
        
        pfsConfig = PfsConfig(pfsConfigId)
        pfsConfig.read(dirName=dirName)

        return pfsConfig

    def writeFits(self, pathName, flags=None):
        dirName, fileName = os.path.split(pathName)
        self._pfsConfig.write(dirName=dirName, fileName=fileName)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def spectrumSetToPfsArm(pfsConfig, spectrumSet, visit, spectrograph, arm):
    pfsArm = PfsArm(visit, spectrograph, arm, pfsConfigId=pfsConfig.pfsConfigId, pfsConfig=pfsConfig)
    
    pfsArm.flux = spectrumSet.getAllFluxes().T
    pfsArm.covar = spectrumSet.getAllCovars().T
    pfsArm.mask = spectrumSet.getAllMasks().T
    pfsArm.lam = spectrumSet.getAllWavelengths().T
    pfsArm.lam[pfsArm.lam == 0] = np.nan
    pfsArm.lam /= 10.0                  # convert from AA to nm
    pfsArm.sky = spectrumSet.getAllSkies().T

    return pfsArm

class PfsArmIO(object):
    """A class to perform butler-based I/O for pfsArm

    N.b. there's no readFits as we use pfsArm_bypass() (because it's passed a dataId)
    """

    def __init__(self, pfsArm):
        self._pfsArm = pfsArm

    def writeFits(self, pathName, flags=None):
        dirName, fileName = os.path.split(pathName)
        self._pfsArm.write(dirName=dirName, fileName=fileName)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PfsFiberTraceIO(object):
    """A class to perform butler-based I/O for pfsFiberTrace

    N.b. there's no readFits as we use pfsFiberTrace_bypass() (because it's passed a dataId)
    """

    def __init__(self, pfsFiberTrace):
        self._pfsFiberTrace = pfsFiberTrace

    def writeFits(self, pathName, flags=None):
        dirName, fileName = os.path.split(pathName)
        self._pfsFiberTrace.write(dirName=dirName, fileName=fileName)
