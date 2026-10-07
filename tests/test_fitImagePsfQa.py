import types

import numpy as np
from matplotlib.figure import Figure

import lsst.utils.tests
import lsst.afw.image
from lsst.afw.image import VisitInfo

from pfs.datamodel import FiberStatus
from pfs.drp.stella import ReferenceLine, ReferenceLineSet, ReferenceLineStatus, ReferenceLineSource
from pfs.drp.stella.pipelines.fitImagePsf import FitImagePsfTask, FitImagePsfConfig
from pfs.drp.stella.pipelines.fitImagePsfQa import FitImagePsfQaTask, FitImagePsfQaConfig
from pfs.drp.stella.synthetic import (
    SyntheticConfig,
    makeSyntheticDetectorMap,
    makeSyntheticPfsConfig,
    makeSpectrumImage,
    addNoiseToImage,
)
from pfs.drp.stella.tests.utils import runTests
from pfs.drp.stella.utils.psf import fwhmToSigma

display = None


class FakeReadLineList:
    """Stand-in for `ReadLineListTask`, returning a fixed set of reference lines"""

    def __init__(self, lines):
        self.lines = lines
        self.config = types.SimpleNamespace(exclusionRadius=0.0)

    def run(self, detectorMap=None, metadata=None):
        return self.lines


class FitImagePsfQaTestCase(lsst.utils.tests.TestCase):
    def testMultipleExposures(self):
        """Fit a PSF (as in test_fitImagePsf.py) and then run the QA task on
        its outputs, checking that it produces sane figures and residuals.
        """
        rng = np.random.RandomState(12345)
        fwhm = 3.21
        flux = 5.0e5
        numVisits = 4
        numLines = 15

        synthConfig = SyntheticConfig()
        synthConfig.width = 1200
        synthConfig.separation = 40.0
        synthConfig.slope = 0.0
        synthConfig.fwhm = fwhm

        detMap = makeSyntheticDetectorMap(synthConfig)
        fiberId = detMap.fiberId

        lineRows = np.linspace(0, synthConfig.height - 1, numLines + 2)[1:-1]
        sigma = fwhmToSigma(fwhm)
        yy = np.arange(synthConfig.height, dtype=np.float64)
        norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
        spectrum = np.zeros(synthConfig.height, dtype=np.float32)
        for row in lineRows:
            spectrum += np.exp(-0.5 * ((yy - row) / sigma) ** 2)
        spectrum *= flux * norm

        midFiber = fiberId[len(fiberId) // 2]
        referenceLines = ReferenceLineSet.fromRows(
            [
                ReferenceLine(
                    description="Simulated",
                    wavelength=detMap.getWavelength(midFiber, row),
                    intensity=flux,
                    status=ReferenceLineStatus.GOOD,
                    transition="UNKNOWN",
                    source=ReferenceLineSource.NONE,
                )
                for row in lineRows
            ]
        )

        fitConfig = FitImagePsfConfig()
        fitConfig.centroidLines.fwhm = fwhm
        fitConfig.centroidLines.doSubtractContinuum = False
        fitConfig.centroidLines.doSubtractTraces = False
        fitConfig.reserve.fraction = 0.2
        fitTask = FitImagePsfTask(config=fitConfig)
        fitTask.readLineList = FakeReadLineList(referenceLines)

        exposures = []
        pfsConfigs = []
        for visit in range(numVisits):
            litFiberId = fiberId[visit::numVisits]
            indices = np.isin(fiberId, litFiberId)

            image = makeSpectrumImage(
                spectrum,
                synthConfig.dims,
                synthConfig.traceCenters[indices],
                synthConfig.traceOffset,
                synthConfig.fwhm,
            )
            addNoiseToImage(image, synthConfig.gain, synthConfig.readnoise, rng)

            exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(image))
            exposure.mask.set(0)
            exposure.variance.set(synthConfig.readnoise)
            exposure.getInfo().setVisitInfo(VisitInfo(id=1000 + visit))

            pfsConfig = makeSyntheticPfsConfig(synthConfig, pfsDesignId=1, visit=1000 + visit, rng=rng)
            pfsConfig.fiberStatus[:] = int(FiberStatus.BLACKSPOT)
            pfsConfig.fiberStatus[np.isin(pfsConfig.fiberId, litFiberId)] = int(FiberStatus.GOOD)

            exposures.append(exposure)
            pfsConfigs.append(pfsConfig)

        fitResult = fitTask.run(exposures, pfsConfigs, detMap, arm="r", spectrograph=1)
        self.assertEqual(len(fitResult.adjustedDetectorMaps), numVisits)

        qaConfig = FitImagePsfQaConfig()
        qaConfig.centroidLines.fwhm = fwhm
        qaConfig.centroidLines.doSubtractContinuum = False
        qaConfig.centroidLines.doSubtractTraces = False
        qaConfig.reserve.fraction = 0.2
        qaTask = FitImagePsfQaTask(config=qaConfig)
        qaTask.readLineList = FakeReadLineList(referenceLines)

        qaResult = qaTask.run(exposures, pfsConfigs, fitResult.adjustedDetectorMaps, fitResult.psf)

        self.assertIsInstance(qaResult.qualityPlot, Figure)
        self.assertIsInstance(qaResult.spatialPlot, Figure)
        self.assertEqual(len(qaResult.residuals), numVisits)
        for residual in qaResult.residuals:
            self.assertEqual(residual.getBBox(), exposures[0].getBBox())
            # The brightest lines should be substantially knocked down by subtraction
            self.assertLess(np.nanmax(np.abs(residual.image.array)), 0.5 * flux)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    runTests(globals())
