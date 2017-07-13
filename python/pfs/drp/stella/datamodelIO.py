#
# LSST Data Management System
#
# Copyright 2008-2017  AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""
This module describes IO procedures according to the PFS datamodel.

@author Andreas Ritter, Princeton University
"""
from __future__ import absolute_import, division, print_function
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
        self._pfsFiberTrace.write(dirName=dirName, fileName=fileName)
