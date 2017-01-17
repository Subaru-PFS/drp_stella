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
#import matplotlib.pyplot as plt
import numpy as np
import pfs.drp.stella as drpStella

def getDistTraceProfRec(fiberTraceSet):
    """
    @brief This function returns the center distances (sorted), the original
    values, the spatial profile, and the reconstructed values (from spatial
    profile and extracted spectrum) for each pixel in each FiberTrace in
    fiberTraceSet as a numpy array of shape(sum(FiberTraceHeights)*FiberTraceWidth,4)
    output[:,0]: distance of each pixel to the FiberTrace center (sorted)
    output[:,1]: original pixel values
    output[:,2]: spatial profile
    output[:,3]: reconstructed pixel values from spatial profile and spectrum
    @param fiberTraceSet: FiberTraceSet for which to perform the operation, profile calculated
    @return ouput array with 4 columns, sorted by the first column (center distance)
    """
    sumHeights = 0
    width = fiberTraceSet.getFiberTrace(0).getWidth()
    for iTrace in range(fiberTraceSet.size()):
        ft = fiberTraceSet.getFiberTrace(iTrace)
        sumHeights += ft.getHeight()
    distanceFromCenterRatioLength = (sumHeights
                                     * fiberTraceSet.getFiberTrace(0).getWidth())
    distanceFromCenterRatio = np.ndarray(shape=(distanceFromCenterRatioLength,
                                                4),
                                         dtype=np.float32)

    iRow = 0
    for iTrace in range(fiberTraceSet.size()):
        ft = fiberTraceSet.getFiberTrace(iTrace)
        if not ft.isProfileSet():
            raise RuntimeError("ERROR: iTrace = ",iTrace,": profile not set")
        spec = ft.extractFromProfile()
        orig = ft.getImage().getArray()
        rec = ft.getReconstructed2DSpectrum(spec).getArray()
        prof = ft.getProfile().getArray()
        xCenters = ft.getXCenters()

        ftfc = ft.getFiberTraceFunction().fiberTraceFunctionControl
        minCenMax = drpStella.calcMinCenMax(xCenters,
                                            ftfc.xHigh,
                                            ftfc.xLow,
                                            ftfc.nPixCutLeft,
                                            ftfc.nPixCutRight)
        for row in range(ft.getHeight()):
            for col in range(ft.getWidth()):
                distanceFromCenterRatio[iRow * width + col, :] = [(minCenMax[row, 0]
                                                                   + col
                                                                   - xCenters[row]
                                                                   + 0.5),
                                                                  orig[row, col],
                                                                  prof[row, col],
                                                                  rec[row, col]]
            iRow += 1
    distanceFromCenterRatio = distanceFromCenterRatio[distanceFromCenterRatio[:,0].argsort()]
    return distanceFromCenterRatio

def getMeanStdXBins(x, y, binWidthX):
    """
    @brief Calculate the mean and standard deviations for bins in x
    This function returns the mean and standard deviation of the y values in
    bins in x with bin width <binWidthX>
    @param x in: np.ndarray of x values
           NOTE that the array must be sorted in X.
    @param y in: np.ndarray of y values, same shape as x
    @param binWidthX in: width of center distance bins
    @return array of shape[nBins,2,2] containing:
            [:,0,:]: the X for start and end of the bin,
            [:,1,:]: the mean and standard deviation of the y values in each bin
    """
    binMeanStdDevs = np.ndarray(
                       shape=(int((x[x.shape[0]-1] - x[0]) / binWidthX) + 1,
                              2,
                              2),
                       dtype=np.float32)
    binRange = [x[0], x[0]+binWidthX]
    nPix = 0
    for iBin in range(binMeanStdDevs.shape[0]):
        iPix = nPix
        binMeanStdDevs[iBin,:,:] = 0.
        while ((iPix < x.shape[0])
               and
               (x[iPix if iPix < x.shape[0]
                        else x.shape[0] - 1] < binRange[1])):
            iPix += 1
        binMeanStdDevs[iBin, 0, :] = [binRange[0], binRange[1]]
        if iPix > nPix:
            binMeanStdDevs[iBin, 1, :] = [np.mean(y[nPix:iPix]), np.std(y[nPix:iPix])]
        nPix = iPix
        binRange[0] = binRange[1]
        binRange[1] += binWidthX
    return binMeanStdDevs
