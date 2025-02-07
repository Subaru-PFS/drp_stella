#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
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
Measure intra-CCD crosstalk coefficients.
"""

__all__ = ["MeasureCrosstalkConfig", "MeasureCrosstalkTask"]


import numpy as np

from lsst.afw.math import LeastSquares
from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import Task, Struct

from lsst.ip.isr.crosstalk import CrosstalkCalib

from .images import getIndices
from .selectFibers import SelectFibersTask


class MeasureCrosstalkConfig(Config):
    """Configuration for MeasureCrosstalkTask"""
    doSubtractBackground = Field(dtype=bool, default=True, doc="Subtract background?")
    backgroundMask = ListField(dtype=str, default=["BAD", "SAT", "CR", "NO_DATA", "CROSSTALK"])
    traceRadius = Field(dtype=float, default=24.0, doc="Half-width of trace to exclude from target")
    crosstalkRadius = Field(dtype=float, default=6.0, doc="Half-width of trace to exclude for crosstalk")
    badMask = ListField(dtype=str, default=["SAT", "BAD", "CR", "NO_DATA"], doc="Mask planes to ignore")
    equationThreshold = Field(dtype=float, default=0.0,
                              doc="Threshold on ratio of eigenvalue to sum of eigenvalues for solving matrix")
    imageThreshold = Field(dtype=float, default=10000, doc="Threshold for source pixels")
    selectFibers = ConfigurableField(target=SelectFibersTask, doc="Select fibers for masking")
    excessNoise = Field(dtype=float, default=7.0, doc="Excess noise to add in calculation")

    def setDefaults(self):
        Config.setDefaults(self)
        # Fiber selection
        self.selectFibers.fiberStatus = ["GOOD", "BROKENFIBER"]
        self.selectFibers.targetType = ["SCIENCE", "SKY", "DCB"]


class MeasureCrosstalkTask(Task):
    """Measure intra-CCD crosstalk

    Crosstalk coefficients are logged, and optionally written to a file.

    This Task behaves in a scatter-gather fashion:
    * Scatter: get ratios for each CCD.
    * Gather: combine ratios to produce crosstalk coefficients.
    """
    ConfigClass = MeasureCrosstalkConfig
    _DefaultName = "measureCrosstalk"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.makeSubtask("selectFibers")
        self.makeSubtask("subtractBackground")

    def run(self, exposure, detectorMap, pfsConfig):
        """Extract and return cross talk ratios for an exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image data to measure crosstalk ratios from.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Fiber configuration.

        Returns
        -------
        ratios : `list` of `list` of `numpy.ndarray`
            A matrix of pixel arrays.
        """
        self.maskFiberTraces(exposure.mask, detectorMap, pfsConfig, exposure.getDetector())

        if self.config.doSubtractBackground:
            from pfs.drp.stella.traces import medianFilterColumns
            badMask = self.config.subtractBackground.ignoredPixelMask
            badMask.remove("CROSSTALK")
            badBitMask = exposure.mask.getPlaneBitMask(self.config.subtractBackground.ignoredPixelMask)
            mask = (exposure.mask.array & badBitMask) != 0
            image = exposure.image.array
            median = medianFilterColumns(image.T.copy(), mask.T.copy(), 7).T
            exposure.image.array[:] -= median

        return Struct(equation=self.buildEquation(exposure), exposure=exposure)

    def maskFiberTraces(self, mask, detectorMap, pfsConfig, detector):
        """Mask fiber traces as ``FIBERTRACE``

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask image.
        detectorMap : `pfs.drp.stella.DetectorMap`
            Mapping of fiberId,wavelength to x,y.
        pfsConfig : `pfs.datamodel.PfsConfig`
            Fiber configuration.
        """
        mask.addMaskPlane("FIBERTRACE")
        mask.addMaskPlane("CROSSTALK")
        fiberTraceMask = mask.getPlaneBitMask("FIBERTRACE")
        crosstalkMask = mask.getPlaneBitMask("CROSSTALK")

        pfsConfig = pfsConfig.select(fiberId=detectorMap.fiberId)
        pfsConfig = self.selectFibers.run(pfsConfig)

        xx, yy = getIndices(mask.getBBox())
        fiberTrace = np.zeros_like(mask.array, dtype=bool)
        aggressor = np.zeros_like(mask.array, dtype=bool)
        for fiberId in pfsConfig.fiberId:
            xCenter = detectorMap.getXCenter(fiberId)
            distance = xx - xCenter[:, np.newaxis]
            fiberTrace |= np.abs(distance) < self.config.traceRadius
            aggressor |= np.abs(distance) < self.config.crosstalkRadius

        mask.array[fiberTrace] |= fiberTraceMask
        mask.array[aggressor] |= mask.getPlaneBitMask("CROSSTALK")

        # Propagate the traces to the crosstalk victims
        crosstalk = np.zeros_like(mask.array, dtype=bool)
        for victim in detector:
            box = victim.getBBox()
            ct = crosstalk[box.getMinY():box.getMaxY()+1, box.getMinX():box.getMaxX()+1]
            for amp in detector:
                if amp == victim:
                    continue
                subMask = CrosstalkCalib.extractAmp(mask, amp, victim, True)
                ct |= (subMask.array & crosstalkMask) != 0
        mask &= ~crosstalkMask
        mask.array[crosstalk] |= crosstalkMask

    def makeSources(self, image, detector, refAmp):
        """Make a list of source amplifiers

        We extract each of the amplifiers and align them.

        Parameters
        ----------
        image : `lsst.afw.image.Image`
            Image to extract sources from.
        detector : `lsst.afw.cameraGeom.Detector`
            Detector geometry.
        refAmp : `lsst.afw.cameraGeom.Amplifier`
            Reference amplifier.

        Returns
        -------
        sources : `list` of `numpy.ndarray`
            List of source images.
        """
        return [
            CrosstalkCalib.extractAmp(
                image, amp, refAmp, True
            ).array.astype(float) for amp in detector if amp != refAmp
        ]

    def buildEquation(self, exposure):
        """Build least-squares equation for each amp

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image data to measure crosstalk ratios from.

        Returns
        -------
        matrices : `numpy.ndarray` of `float`, shape ``(nAmp, nAmp - 1, nAmp - 1)``
            Least-squares matrices for each amplifier.
        vectors : `numpy.ndarray` of `float`, shape ``(nAmp, nAmp - 1)``
            Least-squares vectors for each amplifier.
        """
        detector = exposure.getDetector()
        numAmps = len(detector)

        badBitMask = exposure.mask.getPlaneBitMask(self.config.badMask)
        bad = ((exposure.mask.array & badBitMask) != 0)
        fiberTrace = (exposure.mask.array & exposure.mask.getPlaneBitMask("FIBERTRACE")) != 0

        # These are the traces that form the sources for the crosstalk
        source = exposure.image.clone()
        use = ~bad & fiberTrace & (exposure.image.array > self.config.imageThreshold)
        source.array[~use] = 0.0

        # The target contains the crosstalk we're trying to model and subtract
        target = exposure.image.clone()
        variance = exposure.variance.clone()
        target.array[bad | fiberTrace] = 0.0

        # One matrix+vector for each amplifier
        numSources = numAmps - 1
        matrices = np.zeros((numAmps, numSources, numSources), dtype=float)
        vectors = np.zeros((numAmps, numSources), dtype=float)
        excessVar = self.config.excessNoise**2
        for index, amp in enumerate(detector):
            sourceAmps = self.makeSources(source, detector, amp)
            targetAmp = CrosstalkCalib.extractAmp(target, amp, amp, True).array.astype(float)
            var = CrosstalkCalib.extractAmp(variance, amp, amp, True).array.astype(float) + excessVar
            select = (targetAmp != 0)

            for ii in range(numSources):
                vectors[index, ii] = np.sum((sourceAmps[ii]*targetAmp/var)[select])  # model dot data
                matrices[index, ii, ii] = np.sum((sourceAmps[ii]**2/var)[select])
                for jj in range(ii + 1, numSources - 1):
                    modelDotModel = np.sum((sourceAmps[ii]*sourceAmps[jj]/var)[select])
                    matrices[index, ii, jj] = modelDotModel
                    matrices[index, jj, ii] = modelDotModel

        return Struct(matrices=matrices, vectors=vectors)

    def reduce(self, equationList):
        """Combine equations to produce crosstalk coefficients

        Parameters
        ----------
        equationList : `list` of `lsst.pipe.base.Struct`
            A list of ``matrices`` and ``vectors``, from ``buildEquation``.

        Returns
        -------
        coeff : `numpy.ndarray`
            Crosstalk coefficients.
        """
        numAmps = None
        numSources = None
        for eqn in equationList:
            if eqn is None:
                continue

            if numAmps is None:
                numAmps, numSources = eqn.vectors.shape
                assert numSources == numAmps - 1

            assert eqn.matrices.shape == (numAmps, numSources, numSources)
            assert eqn.vectors.shape == (numAmps, numSources)

        if numAmps is None:
            raise RuntimeError("Unable to measure crosstalk signal for any amplifier")

        # Accumulate the matrices
        matrices = np.zeros((numAmps, numSources, numSources), dtype=float)
        vectors = np.zeros((numAmps, numSources))
        for eqn in equationList:
            matrices += eqn.matrices
            vectors += eqn.vectors

        # Solve the equations
        crosstalk = np.zeros((numAmps, numSources), dtype=float)
        for ii in range(numAmps):
            scale = np.mean(np.diagonal(matrices[ii]))
            matrices[ii] /= scale
            vectors[ii] /= scale
            self.log.info("Amp %s matrix: %s", ii, matrices[ii])
            self.log.info("Amp %s vector: %s", ii, vectors[ii])
            equation = LeastSquares.fromNormalEquations(matrices[ii], vectors[ii])
            eigen = equation.getDiagnostic(equation.NORMAL_EIGENSYSTEM)
            self.log.info("Amp %s eigenvalues: %s", ii, eigen)
            if self.config.equationThreshold > 0:
                equation.setThreshold(self.config.equationThreshold*eigen.sum()/eigen.max())
            solution = equation.getSolution()
            self.log.info("Amp %s solution: %s", ii, solution)
            crosstalk[ii] = solution

        self.log.info("Coefficients:\n%s\n", np.array2string(crosstalk, separator=", "))

        lq, med, uq = np.percentile(crosstalk[crosstalk != 0.0], (25.0, 50.0, 75.0))
        rms = 0.741*(uq - lq)
        numSignificant = (np.abs(crosstalk[crosstalk != 0.0] - med) > 2*rms).sum()
        self.log.info("Coefficients = %g +/- %g --> %d 2-sigma significant values", med, rms, numSignificant)
        return crosstalk
