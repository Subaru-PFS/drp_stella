#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program. If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
"""
This module contains python math function related to pfs.drp.stella

@author Andreas Ritter, Princeton University
"""
import numpy as np

import lsst.afw.image as afwImage


def getMeanStdXBins(x, y, binWidthX):
    """
    @brief Calculate the mean and standard deviations for bins in x
    This function returns the mean and standard deviation of the y values in
    bins in x with bin width <binWidthX>
    @param x in: np.ndarray of x values
           NOTE that the array must be sorted in X.
    @param y in: np.ndarray of y values, same shape as x
    @param binWidthX in: width of center distance bins
    @return array of shape[nBins,2]: the X for start and end of the bin,
            array of shape[nBins,1]: the mean of the y values in each bin
            array of shape[nBins,1]: the standard deviation of the y values in each bin
    """
    nBins = int((x[x.shape[0]-1] - x[0]) / binWidthX) + 1
    binRanges = np.ndarray(shape=(nBins, 2), dtype=np.float32)
    binRanges[:,:] = 0.
    mean = np.ndarray(shape=(nBins), dtype=np.float32)
    std = np.ndarray(shape=(nBins), dtype=np.float32)
    binRange = [x[0], x[0]+binWidthX]
    nPix = 0
    for iBin in range(mean.shape[0]):
        iPix = nPix
        while ((iPix < x.shape[0])
               and
               (x[iPix if iPix < x.shape[0]
                        else x.shape[0] - 1] < binRange[1])):
            iPix += 1
        binRanges[iBin, :] = [binRange[0], binRange[1]]
        if iPix > nPix:
            mean[iBin] = np.mean(y[nPix:iPix])
            std[iBin] = np.std(y[nPix:iPix])
        nPix = iPix
        binRange[0] = binRange[1]
        binRange[1] += binWidthX
    return binRanges, mean, std
