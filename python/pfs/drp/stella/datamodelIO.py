import os
import re

import numpy as np

from pfs.datamodel.pfsArm import PfsArm
from pfs.datamodel.pfsConfig import PfsConfig
from pfs.datamodel.pfsFiberTrace import PfsFiberTrace

class PfsConfigIO(object):
    """A class to perform butler-based I/O for pfsConfig"""

    def __init__(self, pfsConfig):
        self._pfsConfig = pfsConfig

    @staticmethod
    def readFits(pathName, hdu=None, flags=None):
        dirName, fileName = os.path.split(pathName)

        match = re.search(r"^pfsConfig-(0x[0-9a-f]+)\.fits$", fileName)
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

    pfsArm.flux = spectrumSet.getAllFluxes()
    pfsArm.covar = spectrumSet.getAllCovars()
    pfsArm.mask = spectrumSet.getAllMasks()
    pfsArm.lam = spectrumSet.getAllWavelengths()
    pfsArm.lam[pfsArm.lam == 0] = np.nan
    pfsArm.sky = np.zeros_like(pfsArm.flux)

    return pfsArm

class PfsArmIO(object):
    """A class to perform butler-based I/O for pfsArm

    N.b. there's no readFits as we use pfsArm_bypass() (because it's passed a dataId); Stop Press!
    this isn't true as recent versions of the butler silently ignore the _bypass function if the
    file doesn't exist; so for now there *is* a readFits that parses the filename.  Ughh.
    """

    def __init__(self, pfsArm):
        self._pfsArm = pfsArm

    def writeFits(self, pathName, flags=None):
        dirName, fileName = os.path.split(pathName)
        self._pfsArm.write(dirName=dirName, fileName=fileName)

    @staticmethod
    def readFits(pathName, hdu=None, flags=None):
        """Read a PfsArm object from pathName

        Note that we need to parse the filename -- see comment at top of class
        """
        dirName, fileName = os.path.split(pathName)

        pfsArmRE = r"^pfsArm-(\d{6})-([brnm])(\d)\.fits"
        mat = re.search(pfsArmRE, fileName)
        if not mat:
            #
            # See if our RE matches the datamodel
            #
            v, a, s = 666, 'r', 1            
            exampleName = PfsArm.fileNameFormat % (v, a, s)
            mat = re.search(pfsArmRE, exampleName)
            if not mat:
                # if doesn't match
                raise RuntimeError("filename %s doesn't match pfsArm pattern, "
                                   "which doesn't match PfsArm.fileNameFormat" % fileName)

            raise RuntimeError("filename %s doesn't match pfsArm pattern" % fileName)
        #
        # All is well; proceed to read the file
        #
        visit, arm, spectrograph = mat.groups()
        visit = int(visit)
        spectrograph = int(spectrograph)
        
        pfsArm = PfsArm(visit, spectrograph, arm)
        try:
            pfsArm.read(dirName=dirName, setPfsConfig=False)
        except Exception as e:
            pass
        else:
            return pfsArm

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class PfsFiberTraceIO(object):
    """A class to perform butler-based I/O for pfsFiberTrace
    """

    def __init__(self, pfsFiberTrace, metadata=None):
        self._pfsFiberTrace = pfsFiberTrace
        self._metadata = metadata

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    @staticmethod
    def readFits(pathName, hdu=None, flags=None):
        dirName, fileName = os.path.split(pathName)

        match = re.search(r"^pfsFiberTrace-(\d{4}-\d{2}-\d{2})-0-([brmn])([1-4])\.fits", fileName)
        if not match:
            raise RuntimeError("Unable to extract pfsConfigId from \"%s\"" % pathName)

        dateObs, arm, spectrograph = match.groups()
        spectrograph = int(spectrograph)
        pfsFiberTrace = PfsFiberTrace(dateObs, spectrograph, arm)
        pfsFiberTrace.read(dirName=dirName)

        return pfsFiberTrace

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    def writeFits(self, pathName, flags=None):
        dirName, fileName = os.path.split(pathName)
        self._pfsFiberTrace.write(dirName=dirName, fileName=fileName, metadata=self._metadata)
