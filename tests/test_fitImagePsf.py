import types

import numpy as np

import lsst.utils.tests
import lsst.afw.image
from lsst.afw.image import VisitInfo

from pfs.datamodel import FiberStatus
from pfs.drp.stella import ReferenceLine, ReferenceLineSet, ReferenceLineStatus, ReferenceLineSource
from pfs.drp.stella.pipelines.fitImagePsf import FitImagePsfTask, FitImagePsfConfig
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
    """Stand-in for `ReadLineListTask`, returning a fixed set of reference lines

    Avoids a dependency on real lamp-header parsing/line-list files, which
    `ReadLineListTask` normally needs: here we simply supply the lines that
    the synthetic arc images were built to contain.
    """

    def __init__(self, lines):
        self.lines = lines
        # FitImagePsfTask.run()/adjustDetectorMapForVisit() read and temporarily override
        # config.exclusionRadius; a plain mutable namespace is enough to support that here.
        self.config = types.SimpleNamespace(exclusionRadius=0.0)

    def run(self, detectorMap=None, metadata=None):
        return self.lines


class FitImagePsfTestCase(lsst.utils.tests.TestCase):
    def testMultipleExposures(self):
        """Fit a PSF from a series of exposures, each with a different 1-in-4
        subset of fibers lit, and check that the combined candidate list spans
        all fibers and that the recovered PSF matches the (Gaussian) truth.
        """
        rng = np.random.RandomState(12345)
        fwhm = 3.21
        flux = 5.0e5
        numVisits = 4
        numLines = 15

        synthConfig = SyntheticConfig()
        synthConfig.width = 1200
        synthConfig.separation = 40.0
        synthConfig.slope = 0.0  # keep traces straight, so edge fibers don't wander into the PSF-stamp margin
        synthConfig.fwhm = fwhm

        detMap = makeSyntheticDetectorMap(synthConfig)
        fiberId = detMap.fiberId  # same order as synthConfig.traceCenters

        # Build the (shared) spectrum: a fixed set of lines, same for every fiber/visit
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

        config = FitImagePsfConfig()
        config.centroidLines.fwhm = fwhm
        config.centroidLines.doSubtractContinuum = False
        config.centroidLines.doSubtractTraces = False
        config.reserve.fraction = 0.2
        task = FitImagePsfTask(config=config)
        task.readLineList = FakeReadLineList(referenceLines)

        exposures = []
        pfsConfigs = []
        litFiberSets = []
        for visit in range(numVisits):
            litFiberId = fiberId[visit::numVisits]
            litFiberSets.append(set(litFiberId.tolist()))
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

        result = task.run(exposures, pfsConfigs, detMap, arm="r", spectrograph=1)

        # All fibers (from all 4 visits) should have contributed candidates
        seenFiberId = set(int(ff) for ff in result.usedCatalog["fiberId"])
        self.assertEqual(seenFiberId, set.union(*litFiberSets))
        self.assertEqual(seenFiberId, set(int(ff) for ff in fiberId))

        # Some sources should have been reserved for validation, and excluded from the determiner
        numUsed = len(result.usedCatalog)
        self.assertGreater(numUsed, 0)
        numReserved = int(np.sum(result.usedCatalog["calib_psf_reserved"]))
        self.assertGreater(numReserved, 0)
        self.assertLess(numReserved, numUsed)

        # Recovered PSF shape should match the (Gaussian) ground truth
        shape = result.psf.computeShape(result.psf.getAveragePosition())
        self.assertFloatsAlmostEqual(shape.getIxx(), sigma**2, rtol=0.2)
        self.assertFloatsAlmostEqual(shape.getIyy(), sigma**2, rtol=0.2)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    runTests(globals())
