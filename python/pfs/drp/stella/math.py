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
from scipy.optimize import curve_fit

import lsst.log as log
import pfs.drp.stella as drpStella

def gaussFunc(x, *p0):
    """
    Gaussian Function
    @param x : x values (array like)
    @param p0 : ['strength', 'mu', 'sima']
    """
    logger = log.Log.getLogger("gaussFunc")
    strength, mu, sigma = p0
    logger.trace('strength = %f, mu = %f, sigma = %f' % (strength, mu, sigma))
    logger.trace('x = %s' % (np.array_str(np.array(x))))
    if sigma > 0.0:
        return strength * np.exp(-(np.array(x) - mu) ** 2 / (2. * sigma ** 2))
    logger.warn('sigma <= 0.0')
    return x * 0.0

def gauss(x, gaussCoeffs):
    """
    Gaussian Function
    @param x : x values (array like)
    @param gaussCoeffs : ['strength', 'mu', 'sima']
    """
    logger = log.Log.getLogger("gauss")
    logger.trace('x = %s' % (np.array_str(np.array(x))))
    logger.trace('gaussCoeffs = [strength = %f, mu = %f, sigma = %f'
                 % (gaussCoeffs.strength, gaussCoeffs.mu, gaussCoeffs.sigma))
    p = [gaussCoeffs.strength, gaussCoeffs.mu, gaussCoeffs.sigma]
    logger.trace('p = %s' % (np.array_str(np.array(p))))
    return gaussFunc(x, *p)

def gaussFit(x, y, guess):
    """
    Fit Gauss curve to x and y
    @param x : independent values
    @param y : dependent values
    @param guess : GaussCoeffs, initial guess for the fitting coefficients (strength, mu, and sigma above)
    @return : [strength, mu, sigma, eStrength, eMu, eSigma]
    """
    logger = log.Log.getLogger("gaussFit")
    p0 = [guess.strength, guess.mu, guess.sigma]
    failed = False
    try:
        coeffs, varCoeffs = curve_fit(gaussFunc, x, y, p0=p0)
    except:
        failed = True
    if failed or (True in np.isinf(varCoeffs)):
        coeffs = [0.0, 0.0, 0.0]
        varCoeffs = [coeffs, coeffs, coeffs]
    logger.trace('coeffs: strength = %f, mu = %f, sigma = %f'
                 % (coeffs[0], coeffs[1], coeffs[2]))
    return [coeffs[0], coeffs[1], coeffs[2], varCoeffs[0][0], varCoeffs[1][1], varCoeffs[2][2]]

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
        orig = ft.getTrace().getImage().getArray()
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
                distanceFromCenterRatio[iRow * width + col, :] = [
                    (minCenMax[row, 0] + col - xCenters[row]),
                    orig[row, col],
                    prof[row, col],
                    rec[row, col]
                ]
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

def makeArtificialSpectrum(lambdaPix, lines, lambdaMin=0.0, lambdaMax=0.0, fwhm=1.0):
    """
    Create and return an artificial spectrum
    @param lambdaPix : predicted wavelengths for the spectrum
    @param lines : array of NistLines
    @param lambdaMin : minimum wavelength of spectrum, if 0 take min(lambdaPix)
    @param lambdaMax : maximum wavelength of spectrum, if 0 take max(lambdaPix)
    @param fwhm : FWHM of the artificial emission lines in pixels
    @return calculated flux
    """
    logger = log.Log.getLogger("makeArtificialSpectrum")
    logger.debug('len(lines) = %d',len(lines))
    logger.debug('lambdaPix = %d: %s' % (len(lambdaPix),
                                         np.array_str(lambdaPix)))
    logger.trace('lambdaMin = %f, lambdaMax = %f' % (lambdaMin, lambdaMax))
    xWidth = int(2. * fwhm)

    if lambdaMin == 0.0:
        lambdaMin = np.min(lambdaPix)
    if lambdaMax == 0.0:
        lambdaMax = np.max(lambdaPix)
    logger.trace('lambdaMin = %f, lambdaMax = %f' % (lambdaMin, lambdaMax))

    lambdaLines = [line.laboratoryWavelength for line in lines]
    logger.trace('len(lambdaLines) = %d',len(lambdaLines))
    strengthLines = [line.predictedStrength for line in lines]

    calculatedFlux = np.ndarray(shape=lambdaPix.shape[0], dtype=np.float32)
    calculatedFlux[:] = 0.

    for k in range(len(lambdaLines)):
        logger.trace('lambdaLines[%d] = %f' % (k, lambdaLines[k]))
        if lambdaLines[k] > lambdaMin and lambdaLines[k] < lambdaMax:
            logger.debug('lambdaLines[%d] = %f, lambdaPix[0] = %f'
                % (k, lambdaLines[k], lambdaPix[0]))
            dist = abs(lambdaLines[k] - lambdaPix)
            linePos = np.argmin(dist)
            logger.debug('linePos for line %d is at %f' % (k, linePos))
            x = np.linspace(-1 * xWidth, xWidth, (2 * xWidth) + 1)
            logger.debug('x = %s, strengthLines[%d] = %f'
                % (np.array_str(x), k, strengthLines[k]))
            gaussCoeff = drpStella.GaussCoeffs()
            gaussCoeff.strength, gaussCoeff.mu, gaussCoeff.sigma = [strengthLines[k],
                                                                    0.0,
                                                                    fwhm / 1.1774]
            logger.trace('k=%d: gaussCoeff: strength = %f, mu = %f, sigma = %f' %
                         (k, gaussCoeff.strength, gaussCoeff.mu, gaussCoeff.sigma))
            gaussian = gauss(x, gaussCoeff)
            logger.trace('k=%d: gaussian = %s' % (k, np.array_str(gaussian)))
            if (linePos - xWidth >= 0
                and linePos + xWidth < len(calculatedFlux)):
                calculatedFlux[linePos - xWidth:linePos + xWidth + 1] += gaussian
            if np.isnan(np.min(calculatedFlux)):
                raise RuntimeError("calculatedFlux contains NaNs")

    logger.debug('calculatedFlux = %d: ' % (len(calculatedFlux)))
    for iF in range(len(calculatedFlux)):
        logger.trace("%d %f "% (iF, calculatedFlux[iF]))
    return calculatedFlux
