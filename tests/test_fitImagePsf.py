import types

import numpy as np

import lsst.utils.tests
import lsst.afw.image
import lsst.geom
from lsst.afw.image import VisitInfo
from lsst.afw.table import SourceCatalog
from lsst.pipe.base import Struct

from pfs.datamodel import FiberStatus
from pfs.drp.stella import ReferenceLine, ReferenceLineSet, ReferenceLineStatus, ReferenceLineSource
from pfs.drp.stella.fitImagePsf import FitImagePsfTask, FitImagePsfConfig, maskNeighboringFibers
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
        # FitImagePsfTask.run() reads config.exclusionRadius (to compute a sufficient value); a plain
        # mutable namespace is enough to support that here.
        self.config = types.SimpleNamespace(exclusionRadius=0.0)

    def run(self, detectorMap=None, metadata=None, exclusionRadius=None):
        return self.lines


class FakeCandidate:
    """Stand-in for `lsst.meas.algorithms.PsfCandidate`

    Provides only what `maskNeighboringFibers` uses.
    """

    def __init__(self, maskedImage, xCenter, yCenter):
        self.maskedImage = maskedImage
        self.xCenter = xCenter
        self.yCenter = yCenter

    def getMaskedImage(self):
        return self.maskedImage

    def getXCenter(self):
        return self.xCenter

    def getYCenter(self):
        return self.yCenter


def makeLineSpectrum(height, lineRows, fwhm, flux):
    """Make a spectrum of Gaussian lines

    Parameters
    ----------
    height : `int`
        Length of the spectrum (pixels).
    lineRows : `numpy.ndarray` of `float`
        Rows of the lines.
    fwhm : `float`
        Full width at half maximum of the lines (pixels).
    flux : `float`
        Flux of each line.

    Returns
    -------
    spectrum : `numpy.ndarray` of `float`
        Spectrum.
    """
    sigma = fwhmToSigma(fwhm)
    yy = np.arange(height, dtype=np.float64)
    spectrum = np.zeros(height, dtype=np.float32)
    for row in lineRows:
        spectrum += np.exp(-0.5 * ((yy - row) / sigma) ** 2)
    return spectrum * flux / (sigma * np.sqrt(2 * np.pi))


def makeReferenceLines(detMap, lineRows, flux):
    """Make reference lines at the nominated rows of the middle fiber"""
    midFiber = detMap.fiberId[len(detMap.fiberId) // 2]
    return ReferenceLineSet.fromRows(
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


def makeArcExposure(synthConfig, detMap, spectrum, litIndices, visit, rng):
    """Make an arc exposure with a subset of fibers lit

    Parameters
    ----------
    synthConfig : `pfs.drp.stella.synthetic.SyntheticConfig`
        Configuration for synthetic data.
    detMap : `pfs.drp.stella.DetectorMap`
        Synthetic detectorMap, with fibers in the same order as
        ``synthConfig.traceCenters``.
    spectrum : `numpy.ndarray` of `float`
        Spectrum of each lit fiber.
    litIndices : `numpy.ndarray` of `bool`
        Which fibers (in the order of ``detMap.fiberId``) are lit.
    visit : `int`
        Visit identifier.
    rng : `numpy.random.RandomState`
        Random number generator, for the noise and the ``pfsConfig``.

    Returns
    -------
    exposure : `lsst.afw.image.Exposure`
        Arc exposure.
    pfsConfig : `pfs.datamodel.PfsConfig`
        Top-end configuration, with the unlit fibers marked ``BLACKSPOT``.
    """
    image = makeSpectrumImage(
        spectrum,
        synthConfig.dims,
        synthConfig.traceCenters[litIndices],
        synthConfig.traceOffset,
        synthConfig.fwhm,
    )
    addNoiseToImage(image, synthConfig.gain, synthConfig.readnoise, rng)
    exposure = lsst.afw.image.makeExposure(lsst.afw.image.makeMaskedImage(image))
    exposure.mask.set(0)
    exposure.variance.set(synthConfig.readnoise)
    exposure.getInfo().setVisitInfo(VisitInfo(id=visit))

    pfsConfig = makeSyntheticPfsConfig(synthConfig, pfsDesignId=1, visit=visit, rng=rng)
    pfsConfig.fiberStatus[:] = int(FiberStatus.BLACKSPOT)
    pfsConfig.fiberStatus[np.isin(pfsConfig.fiberId, detMap.fiberId[litIndices])] = int(FiberStatus.GOOD)
    return exposure, pfsConfig


class MaskNeighboringFibersTestCase(lsst.utils.tests.TestCase):
    """Tests of `maskNeighboringFibers`"""

    def setUp(self):
        self.fwhm = 3.0
        self.synthConfig = SyntheticConfig()
        self.synthConfig.width = 200
        self.synthConfig.height = 200
        self.synthConfig.separation = 7.0  # Similar to the real fiber pitch
        self.synthConfig.slope = 0.0
        self.synthConfig.fwhm = self.fwhm
        self.detMap = makeSyntheticDetectorMap(self.synthConfig)
        self.lineRow = 100.0
        self.lines = makeReferenceLines(self.detMap, [self.lineRow], 1.0)
        self.stampSize = 17
        self.halfWidth = 3.0
        self.halfHeight = 5.0
        self.threshold = 3.0
        self.targetIndex = len(self.detMap.fiberId) // 2
        self.targetFiberId = int(self.detMap.fiberId[self.targetIndex])

    def makeCandidate(self, neighborAmplitude, targetAmplitude=1000.0):
        """Make a candidate stamp for the target fiber's line

        The target's line has the nominated amplitude. The lines of the
        neighboring fibers (on either side, 7 pixels away: centers are
        outside the region within ``halfWidth`` of the stamp edge) have the
        nominated amplitude, on a noise level of 1.
        """
        xTarget = self.detMap.getXCenter(self.targetFiberId, self.lineRow)
        yTarget = self.detMap.findPoint(self.targetFiberId, self.lines.wavelength[0]).getY()
        half = self.stampSize // 2
        bbox = lsst.geom.Box2I(
            lsst.geom.Point2I(int(round(xTarget)) - half, int(round(yTarget)) - half),
            lsst.geom.Extent2I(self.stampSize, self.stampSize),
        )
        stamp = lsst.afw.image.MaskedImageF(bbox)
        stamp.mask.set(0)
        stamp.variance.set(1.0)
        xx, yy = np.meshgrid(
            np.arange(bbox.getMinX(), bbox.getMaxX() + 1), np.arange(bbox.getMinY(), bbox.getMaxY() + 1)
        )
        sigma = fwhmToSigma(self.fwhm)
        array = np.zeros(xx.shape, dtype=np.float32)
        for offset, amplitude in ((0, targetAmplitude), (-1, neighborAmplitude), (1, neighborAmplitude)):
            fiberId = int(self.detMap.fiberId[self.targetIndex + offset])
            point = self.detMap.findPoint(fiberId, self.lines.wavelength[0])
            array += amplitude * np.exp(
                -0.5 * ((xx - point.getX()) ** 2 + (yy - point.getY()) ** 2) / sigma**2
            )
        stamp.image.array[:] = array
        return FakeCandidate(stamp, xTarget, yTarget)

    def runMasking(self, candidate):
        maskNeighboringFibers(
            [candidate],
            {"fiberId": np.array([self.targetFiberId])},
            self.detMap,
            self.lines,
            self.synthConfig.height,
            self.halfWidth,
            self.halfHeight,
            self.threshold,
        )
        return candidate.getMaskedImage().mask.array & candidate.getMaskedImage().mask.getPlaneBitMask("BAD")

    def testLitNeighbors(self):
        """Lit neighbors near the stamp edge are masked, but not the target"""
        bad = self.runMasking(self.makeCandidate(1000.0))
        half = self.stampSize // 2
        # Columns of the neighbors' lines (7 pixels either side of the target) are masked...
        self.assertTrue(np.all(bad[half, half + 7 :] != 0))
        self.assertTrue(np.all(bad[half, : half - 7 + 1] != 0))
        # ... but not the target's core
        self.assertTrue(np.all(bad[:, half - 2 : half + 3] == 0))
        # ... nor rows beyond halfHeight of the neighbors' lines
        self.assertTrue(np.all(bad[: half - int(self.halfHeight)] == 0))
        self.assertTrue(np.all(bad[half + int(self.halfHeight) + 1 :] == 0))

    def testDarkNeighbors(self):
        """Neighbors without flux are not masked

        The target is bright, so its own wings are significant within the
        neighbors' boxes; they must not be mistaken for the neighbors' flux.
        """
        bad = self.runMasking(self.makeCandidate(0.0, targetAmplitude=1.0e5))
        self.assertTrue(np.all(bad == 0))

    def testLeakingNeighbors(self):
        """Faint flux leaking through dark neighbors is masked"""
        bad = self.runMasking(self.makeCandidate(10.0))  # S/N=10 at the peak, above threshold
        half = self.stampSize // 2
        self.assertTrue(np.all(bad[half, half + 7 :] != 0))
        self.assertTrue(np.all(bad[:, half - 2 : half + 3] == 0))


class DenseExposureTestCase(lsst.utils.tests.TestCase):
    """Tests of the handling of dense exposures in `FitImagePsfTask`"""

    def setUp(self):
        self.synthConfig = SyntheticConfig()
        self.synthConfig.width = 600
        self.synthConfig.height = 1000
        self.synthConfig.separation = 10.0  # Less than the stamp size: dense if all fibers are lit
        self.synthConfig.slope = 0.0
        self.synthConfig.fwhm = 3.21
        self.detMap = makeSyntheticDetectorMap(self.synthConfig)

    def makeTask(self):
        config = FitImagePsfConfig()
        config.centroidLines.fwhm = self.synthConfig.fwhm
        config.centroidLines.doSubtractContinuum = False
        config.centroidLines.doSubtractTraces = False
        return FitImagePsfTask(config=config)

    def testSeparation(self):
        """Test measureLitFiberSeparation"""
        task = self.makeTask()
        rng = np.random.RandomState(12345)
        pfsConfig = makeSyntheticPfsConfig(self.synthConfig, pfsDesignId=1, visit=1, rng=rng)
        fiberId = self.detMap.fiberId
        separation = self.synthConfig.separation

        pfsConfig.fiberStatus[:] = int(FiberStatus.GOOD)
        self.assertFloatsAlmostEqual(task.measureLitFiberSeparation(pfsConfig, self.detMap), separation)
        self.assertFloatsAlmostEqual(task.measureLitFiberSeparation(None, self.detMap), separation)

        pfsConfig.fiberStatus[:] = int(FiberStatus.BLACKSPOT)
        pfsConfig.fiberStatus[np.isin(pfsConfig.fiberId, fiberId[::4])] = int(FiberStatus.GOOD)
        self.assertFloatsAlmostEqual(task.measureLitFiberSeparation(pfsConfig, self.detMap), 4 * separation)

        pfsConfig.fiberStatus[:] = int(FiberStatus.BLACKSPOT)
        pfsConfig.fiberStatus[pfsConfig.fiberId == fiberId[len(fiberId) // 2]] = int(FiberStatus.GOOD)
        self.assertEqual(task.measureLitFiberSeparation(pfsConfig, self.detMap), np.inf)

    def makeGroup(self, task, ids, isDense):
        """Make a group of candidates

        The candidates are stand-ins: just their IDs, which are also the IDs
        of the catalog records, so we can check that they remain aligned.
        """
        catalog = SourceCatalog(task.centroidLines.schema)
        for ii in ids:
            catalog.addNew().setId(int(ii))
        return Struct(candidates=list(ids), catalog=catalog, isDense=isDense)

    def testLimitDenseCandidates(self):
        """Test limitDenseCandidates"""
        task = self.makeTask()
        task.config.denseCandidateRatio = 0.5

        def makeGroups():
            return [
                self.makeGroup(task, range(0, 100), False),
                self.makeGroup(task, range(1000, 1300), True),
                self.makeGroup(task, range(2000, 2100), True),
            ]

        groups = makeGroups()
        task.limitDenseCandidates(groups, seed=1)
        self.assertEqual(groups[0].candidates, list(range(0, 100)))  # Sparse is unchanged
        self.assertEqual(len(groups[1].candidates) + len(groups[2].candidates), 50)
        for group in groups:
            self.assertEqual(group.candidates, [int(ii) for ii in group.catalog["id"]])
        for group, (start, stop) in zip(groups[1:], ((1000, 1300), (2000, 2100))):
            self.assertTrue(all(start <= ii < stop for ii in group.candidates))

        # No limit
        task.config.denseCandidateRatio = None
        groups = makeGroups()
        task.limitDenseCandidates(groups, seed=1)
        self.assertEqual([len(group.candidates) for group in groups], [100, 300, 100])

        # Only dense, or only sparse
        task.config.denseCandidateRatio = 0.5
        for isDense in (True, False):
            groups = [
                self.makeGroup(task, range(0, 100), isDense),
                self.makeGroup(task, range(200, 400), isDense),
            ]
            task.limitDenseCandidates(groups, seed=1)
            self.assertEqual([len(group.candidates) for group in groups], [100, 200])

    def testRun(self):
        """Fit a PSF from a sparse and a dense exposure

        The sparse exposure has every 4th fiber lit, and the dense exposure
        has all fibers lit (with neighboring fibers' lines in each stamp). The
        dense candidates should be limited relative to the sparse candidates,
        the neighbors' lines should be masked in the dense stamps, and the
        PSF should match the (Gaussian) truth.
        """
        rng = np.random.RandomState(12345)
        flux = 5.0e5
        lineRows = np.linspace(0, self.synthConfig.height - 1, 12)[1:-1]
        spectrum = makeLineSpectrum(self.synthConfig.height, lineRows, self.synthConfig.fwhm, flux)

        task = self.makeTask()
        task.readLineList = FakeReadLineList(makeReferenceLines(self.detMap, lineRows, flux))

        numFibers = len(self.detMap.fiberId)
        sparseIndices = np.zeros(numFibers, dtype=bool)
        sparseIndices[::4] = True
        denseIndices = np.ones(numFibers, dtype=bool)
        sparseExposure, sparsePfsConfig = makeArcExposure(
            self.synthConfig, self.detMap, spectrum, sparseIndices, 1000, rng
        )
        denseExposure, densePfsConfig = makeArcExposure(
            self.synthConfig, self.detMap, spectrum, denseIndices, 1001, rng
        )

        result = task.run(
            [sparseExposure, denseExposure], [sparsePfsConfig, densePfsConfig], self.detMap, arm="r"
        )

        sparseFiberId = set(int(ff) for ff in self.detMap.fiberId[sparseIndices])
        isSparse = np.array([int(ff) in sparseFiberId for ff in result.usedCatalog["fiberId"]])
        # The sparse fibers are also lit in the dense exposure, so distinguish by the stamp masking
        badBit = denseExposure.mask.getPlaneBitMask("BAD")
        masked = np.array(
            [np.any(cand.getMaskedImage().mask.array & badBit) for cand in result.allCandidates]
        )
        numSparseOnlyFibers = np.sum(isSparse & ~masked)
        self.assertGreater(numSparseOnlyFibers, 0)
        # Candidates from the dense exposure have their neighbors masked
        numDense = np.sum(masked)
        self.assertGreater(numDense, 0)
        self.assertEqual(len(result.allCandidates), numSparseOnlyFibers + numDense)
        self.assertLessEqual(numDense, task.config.denseCandidateRatio * numSparseOnlyFibers)

        sigma = fwhmToSigma(self.synthConfig.fwhm)
        shape = result.psf.computeShape(result.psf.getAveragePosition())
        self.assertFloatsAlmostEqual(shape.getIxx(), sigma**2, rtol=0.2)
        self.assertFloatsAlmostEqual(shape.getIyy(), sigma**2, rtol=0.2)


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
