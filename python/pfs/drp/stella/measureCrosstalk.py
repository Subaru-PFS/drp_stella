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
from lsst.daf.persistence.butlerExceptions import NoResults
from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import CmdLineTask, Struct, TaskRunner
from lsst.meas.algorithms.subtractBackground import SubtractBackgroundTask

from lsst.ip.isr.crosstalk import extractAmp, writeCrosstalkCoeffs
from lsst.ip.isr.isrTask import IsrTask

from .buildFiberTraces import BuildFiberTracesTask

from lsst.afw.image import MaskX


class MeasureCrosstalkConfig(Config):
    """Configuration for MeasureCrosstalkTask"""
    isr = ConfigurableField(target=IsrTask, doc="Instrument signature removal")
    subtractBackground = ConfigurableField(target=SubtractBackgroundTask, doc="Subtract background")
    buildFiberTraces = ConfigurableField(target=BuildFiberTracesTask, doc="Build fiber traces")
    traceRadius = Field(dtype=float, default=30.0, doc="Half-width of trace to exclude from target")
    doRerunIsr = Field(dtype=bool, default=True, doc="Rerun the ISR, even if postISRCCD files are available")
    badMask = ListField(dtype=str, default=["SAT", "BAD", "INTRP"], doc="Mask planes to ignore")
    equationThreshold = Field(dtype=float, default=0.05,
                              doc="Threshold on ratio of eigenvalue to sum of eigenvalues for solving matrix")
    imageThreshold = Field(dtype=float, default=50.0, doc="Threshold on trace-masked image to toss CRs")

    def setDefaults(self):
        Config.setDefaults(self)
        self.isr.doOverscan = True
        self.isr.doBias = True
        self.isr.doDark = True
        self.isr.doFlat = False
        self.isr.doWrite = False
        self.isr.doLinearize = False
        self.isr.doBrighterFatter = False
        self.isr.doAddDistortionModel = False
        self.isr.doCrosstalk = False
        self.isr.doCrosstalkBeforeAssemble = False
        self.isr.overscanFitType = "AKIMA_SPLINE"
        self.isr.overscanOrder = 30
        # Turn off all kinds of interpolation: we don't want the images corrected
        self.isr.growSaturationFootprintSize = 0  # We want the saturation spillover: it's good signal
        self.isr.doDefect = False
        self.isr.doWidenSaturationTrails = False
        self.isr.doSaturationInterpolation = False
        self.isr.maskListToInterpolate = []
        # Background subtraction
        self.subtractBackground.statisticsProperty = "MEDIAN"
        self.subtractBackground.binSize = 512
        self.subtractBackground.useApprox = False
        MaskX.addMaskPlane("FIBERTRACE")
        self.subtractBackground.ignoredPixelMask.append("FIBERTRACE")
        self.buildFiberTraces.doBlindFind = True  # Traces shift around in a dithered flat sequence


class MeasureCrosstalkRunner(TaskRunner):
    def __call__(self, *args, **kwargs):
        """Remove the dataRef from the result

        Otherwise, running threaded yields:

            sqlite3.ProgrammingError: SQLite objects created in a thread can
            only be used in that same thread.
        """
        result = TaskRunner.__call__(self, *args, **kwargs)
        if hasattr(result, "dataRef"):
            del result.dataRef
        return result


class MeasureCrosstalkTask(CmdLineTask):
    """Measure intra-CCD crosstalk

    Crosstalk coefficients are logged, and optionally written to a file.

    This Task behaves in a scatter-gather fashion:
    * Scatter: get ratios for each CCD.
    * Gather: combine ratios to produce crosstalk coefficients.
    """
    ConfigClass = MeasureCrosstalkConfig
    RunnerClass = MeasureCrosstalkRunner
    _DefaultName = "measureCrosstalk"

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("subtractBackground")
        self.makeSubtask("buildFiberTraces")

    @classmethod
    def _makeArgumentParser(cls):
        parser = super(MeasureCrosstalkTask, cls)._makeArgumentParser()
        parser.add_argument("--crosstalkName",
                            help="Name for this set of crosstalk coefficients", default="Unknown")
        parser.add_argument("--outputFileName",
                            help="Name of yaml file to which to write crosstalk coefficients")
        parser.add_argument("--dump-data", dest="dumpData",
                            help="Name of pickle file to which to write gathered data")
        return parser

    @classmethod
    def parseAndRun(cls, *args, **kwargs):
        """Implement scatter/gather

        The gathered data may be dumped to a pickle file if ``--dump-data` is
        specified.

        Returns
        -------
        coeff : `numpy.ndarray`
            Crosstalk coefficients.
        coeffErr : `numpy.ndarray`
            Crosstalk coefficient errors.
        coeffNum : `numpy.ndarray`
            Number of pixels used for crosstalk measurement.
        """
        kwargs["doReturnResults"] = True
        results = super(MeasureCrosstalkTask, cls).parseAndRun(*args, **kwargs)
        task = cls(config=results.parsedCmd.config, log=results.parsedCmd.log)
        resultList = [rr.result for rr in results.resultList]
        if results.parsedCmd.dumpData:
            import pickle
            pickle.dump(resultList, open(results.parsedCmd.dumpData, "wb"))
        coeff = task.reduce(resultList)

        outputFileName = results.parsedCmd.outputFileName
        if outputFileName is not None:
            butler = results.parsedCmd.butler
            dataId = results.parsedCmd.id.idList[0]
            dataId["detector"] = butler.queryMetadata("raw", ["detector"], dataId)[0]

            det = butler.get('raw', dataId).getDetector()
            writeCrosstalkCoeffs(outputFileName, coeff, det=det,
                                 crosstalkName=results.parsedCmd.crosstalkName, indent=2)

        return Struct(
            coeff=coeff,
        )

    def runDataRef(self, dataRef):
        """Get crosstalk ratios for CCD

        Parameters
        ----------
        dataRef : `lsst.daf.peristence.ButlerDataRef`
            Data reference for CCD.

        Returns
        -------
        ratios : `list` of `list` of `numpy.ndarray`
            A matrix of pixel arrays.
        """
        exposure = None
        if not self.config.doRerunIsr:
            try:
                exposure = dataRef.get("postISRCCD")
            except NoResults:
                pass

        if exposure is None:
            exposure = self.isr.runDataRef(dataRef).exposure

        return self.run(exposure)

    def run(self, exposure):
        """Extract and return cross talk ratios for an exposure

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Image data to measure crosstalk ratios from.

        Returns
        -------
        ratios : `list` of `list` of `numpy.ndarray`
            A matrix of pixel arrays.
        """
        traces = self.buildFiberTraces.buildFiberTraces(exposure.maskedImage)
        self.maskFiberTraces(exposure.mask, traces)
        self.subtractBackground.run(exposure)
        return self.buildEquation(exposure)

    def maskFiberTraces(self, mask, traces):
        """Mask fiber traces as ``FIBERTRACE``

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Mask image.
        traces : `lsst.pipe.base.Struct`
            Output of `pfs.drp.stella.BuildFiberTracesTask`.
        """
        select = np.zeros_like(mask.array, dtype=bool)
        columns = np.arange(mask.getWidth(), dtype=int)
        rows = np.arange(mask.getHeight(), dtype=int)
        xx, yy = np.meshgrid(columns, rows)
        for cen in traces.centers:
            distance = xx - cen(rows)[:, np.newaxis]
            select[yy, xx] |= np.abs(distance) < self.config.traceRadius

        mask.array[select] |= 2**mask.addMaskPlane("FIBERTRACE")

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
        refCorner = detector[0].getReadoutCorner()

        badBitMask = exposure.mask.getPlaneBitMask(self.config.badMask)
        bad = (exposure.mask.array & badBitMask) != 0
        fiberTrace = (exposure.mask.array & exposure.mask.getPlaneBitMask("FIBERTRACE")) != 0

        # These are the traces that form the sources for the crosstalk
        source = exposure.image.clone()
        source.array[bad] = 0.0
        source.array[~fiberTrace] = 0.0
        sourceAmps = [extractAmp(source, amp, refCorner, isTrimmed=True).array.astype(float) for
                      amp in detector]

        # The target contains the crosstalk we're trying to model and subtract
        target = exposure.image.clone()
        target.array[bad] = 0.0
        target.array[fiberTrace] = 0.0
        target.array[target.array > self.config.imageThreshold] = 0.0  # Poor man's CR rejection

        # One matrix+vector for each amplifier
        matrices = np.zeros((numAmps, numAmps - 1, numAmps - 1), dtype=float)
        vectors = np.zeros((numAmps, numAmps - 1), dtype=float)
        for index, amp in enumerate(detector):
            targetAmp = extractAmp(target, amp, refCorner, isTrimmed=True).array.astype(float)
            select = targetAmp != 0
            sources = [ss for ii, ss in enumerate(sourceAmps) if ii != index]
            for ii in range(numAmps - 1):
                vectors[index, ii] = np.sum((sources[ii]*targetAmp)[select])  # model dot data
                matrices[index, ii, ii] = np.sum((sources[ii]**2)[select])
                for jj in range(ii + 1, numAmps - 1):
                    modelDotModel = np.sum((sources[ii]*sources[jj])[select])
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
        for eqn in equationList:
            if eqn is None:
                continue

            if numAmps is None:
                numAmps = len(eqn.matrices)

            assert eqn.matrices.shape == (numAmps, numAmps - 1, numAmps - 1)
            assert eqn.vectors.shape == (numAmps, numAmps - 1)

        if numAmps is None:
            raise RuntimeError("Unable to measure crosstalk signal for any amplifier")

        # Accumulate the matrices
        matrices = np.zeros((numAmps, numAmps - 1, numAmps - 1), dtype=float)
        vectors = np.zeros((numAmps, numAmps - 1))
        for eqn in equationList:
            matrices += eqn.matrices
            vectors += eqn.vectors

        # Solve the equations
        indices = np.arange(numAmps, dtype=int)
        crosstalk = np.zeros((numAmps, numAmps), dtype=float)
        for ii in range(numAmps):
            scale = np.mean(np.diagonal(matrices[ii]))
            matrices[ii] /= scale
            vectors[ii] /= scale
            equation = LeastSquares.fromNormalEquations(matrices[ii], vectors[ii])
            eigen = equation.getDiagnostic(equation.NORMAL_EIGENSYSTEM)
            equation.setThreshold(self.config.equationThreshold*eigen.sum()/eigen.max())
            crosstalk[ii][indices != ii] = equation.getSolution()

        self.log.info("Coefficients:\n%s\n", crosstalk)

        lq, med, uq = np.percentile(crosstalk[crosstalk != 0.0], (25.0, 50.0, 75.0))
        rms = 0.741*(uq - lq)
        numSignificant = (np.abs(crosstalk[crosstalk != 0.0] - med) > 2*rms).sum()
        self.log.info("Coefficients = %g +/- %g --> %d 2-sigma significant values", med, rms, numSignificant)
        return crosstalk

    def _getConfigName(self):
        """Disable config output"""
        return None

    def _getMetadataName(self):
        """Disable metdata output"""
        return None


if __name__ == "__main__":
    MeasureCrosstalkTask.parseAndRun()
