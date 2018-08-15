import os
import re

from pfs.datamodel.pfsConfig import PfsConfig


class PfsConfigIO(PfsConfig):
    """A class to perform butler-based I/O for pfsConfig"""
    fileNameRegex = r"^pfsConfig-(0x[0-9a-f]+)\.fits$"

    @classmethod
    def readFits(cls, pathName, hdu=None, flags=None):
        dirName, fileName = os.path.split(pathName)
        matches = re.search(cls.fileNameRegex, fileName)
        if not matches:
            raise RuntimeError("Unable to parse filename: %s" % (fileName,))
        pfsConfigId = int(matches.group(1), 16)

        pfsConfig = cls(pfsConfigId)
        pfsConfig.read(dirName=dirName)
        return pfsConfig

    def writeFits(self, pathName, flags=None):
        dirName, fileName = os.path.split(pathName)
        self.write(dirName=dirName, fileName=fileName)
